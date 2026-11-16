import math
import json
import torch
import pickle
import torch.distributed as dist
from pathlib import Path
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from myvllm.models.qwen3 import Qwen3ForCausalLM
from myvllm.models.llama import LlamaForCausalLM
from myvllm.layers.sampler import SamplerLayer
from myvllm.engine.sequence import Sequence
from myvllm.engine.input_batch import build_prefill_batch
from myvllm.utils import *
from myvllm.config import resolve_dtype, resolve_kv_cache_dtype, seed_everything
from myvllm.utils.timing import StepTimings

class ModelRunner:
    def __init__(self, config: dict, rank: int, event: Event | list[Event]):
        self._standalone_store = None
        had_process_group = dist.is_initialized()
        try:
            self._initialize(config, rank, event)
        except BaseException:
            try:
                if not had_process_group and dist.is_initialized():
                    dist.destroy_process_group()
            finally:
                self._standalone_store = None
            raise

    def _initialize(self, config: dict, rank: int, event: Event | list[Event]):
        self.config = config
        self.event = event
        self.timings = None
        self._exited = False

        # set distributed config
        self.block_size = config['block_size']
        self.world_size = config['world_size']
        self.enforce_eager = config.get('enforce_eager', False)

        self.rank = rank
        seed_everything(config.get('seed', 0))
        model_dtype = resolve_dtype(config.get('dtype', 'float32'))
        self.kv_cache_dtype = resolve_kv_cache_dtype(config.get('kv_cache_dtype', 'auto'), model_dtype)
        torch.set_default_dtype(model_dtype)
        resources = config.get("_instance_resources")
        if resources is None:
            if self.world_size != 1:
                raise ValueError("Multi-rank ModelRunner requires engine instance resources")
            self._standalone_store = dist.TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False)
            # A failed process-group initialization may retain its store in the traceback.
            store = dist.TCPStore("127.0.0.1", self._standalone_store.port, is_master=False)
            shm_name = None
        else:
            port, shm_name = resources
            store = dist.TCPStore("127.0.0.1", port, is_master=False)
        dist.init_process_group('nccl', store=store, world_size=config['world_size'], rank=rank)
        torch.cuda.set_device(rank)

        # set model
        path_str = self.config['model_name_or_path']
        model_name = config.get('model_type', Path(path_str).name)
        match model_name:
            case 'Qwen3-0.6B' | 'Qwen3-14B':
                self.model = Qwen3ForCausalLM(
                    vocab_size=config['vocab_size'],
                    hidden_size=config['hidden_size'],
                    num_heads=config['num_heads'],
                    head_dim=config['head_dim'],
                    scale=config['scale'],
                    num_kv_heads=config['num_kv_heads'],
                    rms_norm_epsilon=config['rms_norm_epsilon'],
                    qkv_bias=config['qkv_bias'],
                    base=config['base'],
                    max_position=config['max_position'],
                    intermediate_size=config['intermediate_size'],
                    ffn_bias=config['ffn_bias'],
                    num_layers=config['num_layers'],
                    tie_word_embeddings=config['tie_word_embeddings'],
                    block_size=self.block_size,
                )
            case 'Llama-3.2-1B-Instruct':
                self.model = LlamaForCausalLM(
                    vocab_size=config['vocab_size'],
                    hidden_size=config['hidden_size'],
                    head_dim=config['head_dim'],
                    num_qo_heads=config['num_qo_heads'],
                    num_kv_heads=config['num_kv_heads'],
                    has_attn_bias=config['has_attn_bias'],
                    rms_norm_epsilon=config['rms_norm_epsilon'],
                    rope_base=config['rope_base'],
                    max_position_embeddings=config['max_position_embeddings'],
                    intermediate_size=config['intermediate_size'],
                    ffn_bias=config['ffn_bias'],
                    num_layers=config['num_layers'],
                    block_size=self.block_size,
                    tie_word_embeddings=config['tie_word_embeddings'],
                )
            case _:
                raise Exception(f"Unsupported model: {config['model_name_or_path']}")

        # Load weights in GPU (model moved to GPU before loading weights)
        self.model = self.model.cuda(rank)

        # Load pretrained weights if model_name_or_path is provided
        if config.get('model_name_or_path'):
            from myvllm.utils.loader import load_weights_from_checkpoint
            self.loaded_params = load_weights_from_checkpoint(self.model, config['model_name_or_path'],
                                         revision=config.get('model_revision'), cache_dir=config.get('cache_dir'))

        # Load weights in CPU (move the model to GPU after loading weights)
        # self.model = self.model.cuda(rank)

        self.sampler = SamplerLayer()

        # Store default dtype before it's needed in allocate_kv_cache
        self.default_dtype = torch.get_default_dtype()

        # Debug flag for first decode step
        self._first_decode = False

        # warm up model so that we know peak memory usage
        self.warmup_model()
        # allocate kv cache
        self.allocate_kv_cache()
        # capture cuda graph for decoding
        if not self.enforce_eager:
            self.capture_cudagraph()

        torch.set_default_device(f'cuda:{rank}')
        torch.set_default_dtype(self.default_dtype)

        if config.get("timing_path") and self.rank == 0:
            self.timings = StepTimings()

        # IMPORTANT: Attach shared memory and barrier AFTER all model initialization
        # This ensures both ranks complete warmup/allocation before rank 1 enters its event loop
        if self.world_size > 1:
            # Synchronize before attaching shared memory
            dist.barrier()
            self.shm = SharedMemory(name=shm_name)
            # Wait until all ranks have attached before issuing calls.
            dist.barrier()
            # Don't call self.loop() here - let the spawning code handle it
            # Otherwise we'll be stuck in an infinite loop during __init__

    # only use read when rank != 0
    def read_shm(self):
        assert self.world_size > 1 and self.rank != 0, "read_shm can only be called when world_size > 1 and rank != 0"
        self.event.wait()
        n = int.from_bytes(self.shm.buf[:4], 'little') # read length
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    # only use write when rank == 0
    def write_shm(self, method_name: str, args: tuple):
        assert self.world_size > 1 and self.rank == 0, "write_shm can only be called when world_size > 1 and rank == 0"
        # encode the length first
        # Flatten: (method_name, args) where args is a tuple -> (method_name, *args)
        data = pickle.dumps((method_name, *args))
        n = len(data)
        self.shm.buf[:4] = n.to_bytes(4, 'little')
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    # close shared memory, destroy process group, delete graphs
    def exit(self):
        if self._exited:
            return
        self._exited = True
        try:
            if self.world_size > 1:
                self.shm.close()
            if not self.enforce_eager:
                del self.graphs
                del self.graph_vars
            torch.cuda.synchronize()
            if self.timings:
                self.timings.flush()
        finally:
            try:
                # Check if process group exists before destroying
                if dist.is_initialized():
                    dist.destroy_process_group()
            finally:
                self._standalone_store = None
    
    # wait to read method and args from shared memory
    # execute the method with args
    # write results back to shared memory
    def loop(self):
        assert self.world_size > 1 and self.rank != 0, "loop can only be called when world_size > 1 and rank != 0"
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args) # Unpack args when calling
            if method_name == 'exit':
                break

    # will be called by both rank == 0 and rank != 0
    # given method name and args from shared memory
    # execute the method and return results
    def call(self, method_name: str, *args: dict):
        if self.world_size > 1 and self.rank == 0: # will be called in main engine
            self.write_shm(method_name, args)
        method = getattr(self, method_name, None)
        if method:
            return method(*args)
        raise ValueError(f"Unknown method: {method_name}")

    # cleanup memory
    # compute max number of sequence based on max token and max model length
    # run empty sequence to warm up the model
    # clear memory
    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_tokens = self.config['max_num_batched_tokens']
        max_model_length = self.config['max_model_length']
        seq_length = min(max_tokens, max_model_length)
        batch_size = min(self.config['max_num_sequences'], max_tokens // seq_length)
        seqs = [Sequence(token_ids=[0]*seq_length, block_size=self.config['block_size']) for _ in range(batch_size)]
        self.run(seqs, is_prefill=True)
        torch.cuda.empty_cache()

    # allocate kv cache memory blocks for model
    def allocate_kv_cache(self):
        # find all available memory
        free_mem, total_mem = torch.cuda.mem_get_info()
        total_free_mem = free_mem * self.config['gpu_memory_utilization']
        peak_mem_usage = torch.cuda.memory_stats()['allocated_bytes.all.peak']
        current_mem_usage = torch.cuda.memory_stats()['allocated_bytes.all.current']
        # reserve some room for peak memory usage during model execution
        available_mem = total_free_mem - (peak_mem_usage - current_mem_usage)
        
        # find parameters to compute kv cache size
        num_layers = self.config['num_layers']
        num_kv_heads = self.config['num_kv_heads'] // self.world_size
        head_dim = self.config['head_dim'] if 'head_dim' in self.config else self.config['hidden_size'] // self.config['num_heads']

        # check whether the current free memory can hold at least one block
        # compute the actual byte required of each block
        block_elements = self.block_size * 2 * num_layers * num_kv_heads
        data_block_bytes = block_elements * head_dim * self.kv_cache_dtype.itemsize
        scale_block_bytes = block_elements * torch.float32.itemsize if self.kv_cache_dtype == torch.float8_e4m3fn else 0
        block_bytes = data_block_bytes + scale_block_bytes
        num_available_kv_blocks = min(int(available_mem // block_bytes), self.config['max_cached_blocks'])
        assert num_available_kv_blocks >= 1, f'Not enough memory to hold at least one block of KV cache on rank {self.rank}'
        
        # Synchronize max_cached_blocks across all ranks.
        # Each rank independently computed num_available_kv_blocks from its own
        # free GPU memory. Ranks may differ slightly: rank-0 carries extra overhead
        # (NCCL buffers, process-group state) so it often has less free memory than
        # workers. Without sync, the scheduler (which runs only on rank-0) would use
        # rank-0's local value and could allocate more blocks than some rank can hold,
        # causing an OOM on that rank during KV cache writes.
        if self.world_size > 1:
            print(f"[Rank {self.rank}] Local max_cached_blocks: {num_available_kv_blocks}")
            per_rank_max_blocks_tensor = torch.tensor(
                num_available_kv_blocks,
                dtype=torch.long,
                device=f'cuda:{self.rank}'
            )
            # all_reduce with MIN: every rank learns the most conservative limit,
            # i.e. the block count that even the most memory-constrained rank can serve.
            # This single agreed-upon value is then stored in config so the Scheduler
            # (initialized afterwards on rank-0) never allocates more blocks than any
            # rank can physically hold.
            dist.all_reduce(per_rank_max_blocks_tensor, op=dist.ReduceOp.MIN)
            self.config['max_cached_blocks'] = per_rank_max_blocks_tensor.item()
        else:
            # Single GPU: no cross-rank sync needed; use the local value directly.
            self.config['max_cached_blocks'] = num_available_kv_blocks
        if self.config['max_model_length'] > self.config['max_cached_blocks'] * self.block_size:
            raise RuntimeError("Profiled KV cache cannot hold max_model_length; lower the model length or increase gpu_memory_utilization.")
        if self.rank == 0:
            print(f"[Rank 0] Global max_cached_blocks (min): {self.config['max_cached_blocks']}")

        if self.rank == 0:
            print(json.dumps({
                "event": "capacity", "total_gpu_bytes": total_mem,
                "free_before_kv_bytes": free_mem,
                "parameter_bytes": sum(p.numel() * p.element_size() for p in self.model.parameters()),
                "rope_cache_bytes": sum(b.numel() * b.element_size() for n, b in self.model.named_buffers() if n.endswith('cos_sin_cache')),
                "warmup_peak_allocated_bytes": peak_mem_usage,
                "current_allocated_bytes": current_mem_usage,
                "warmup_extra_bytes": peak_mem_usage - current_mem_usage,
                "model_dtype": str(self.default_dtype).removeprefix("torch."),
                "kv_cache_dtype": str(self.kv_cache_dtype).removeprefix("torch."),
                "data_block_bytes": data_block_bytes, "scale_block_bytes": scale_block_bytes,
                "block_bytes": block_bytes, "block_count": self.config['max_cached_blocks'],
                "kv_token_capacity": self.config['max_cached_blocks'] * self.block_size,
            }), flush=True)

        # allocate max possible kv cache for the model, instead for each sequence
        # this is the key for paged attention: one giant KV cache pool, divided into blocks
        # IMPORTANT: Use zeros() instead of empty() to avoid garbage values
        allocated_kv_cache = torch.zeros(2, self.config['num_layers'], self.config['max_cached_blocks'], self.block_size, num_kv_heads, head_dim, dtype=self.kv_cache_dtype, device=f'cuda:{self.rank}')
        self.kv_cache = allocated_kv_cache
        self.kv_cache_scales = None
        if self.kv_cache_dtype == torch.float8_e4m3fn:
            self.kv_cache_scales = torch.ones(2, self.config['num_layers'], self.config['max_cached_blocks'], self.block_size, num_kv_heads, dtype=torch.float32, device=f'cuda:{self.rank}')
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, 'k_cache') and hasattr(module, 'v_cache'):
                module.k_cache = allocated_kv_cache[0, layer_id]
                module.v_cache = allocated_kv_cache[1, layer_id]
                if self.kv_cache_scales is not None:
                    module.k_scale_cache = self.kv_cache_scales[0, layer_id]
                    module.v_scale_cache = self.kv_cache_scales[1, layer_id]
                layer_id += 1

    # given seqs
    # prepare the data needed for a prefill forward pass
    # taking prefix cache into consideration: 
    # input_ids, positions, cu_seqlens_q/k, slot_mapping (where to write new KV values), block_tables (where to read KV values)
    # cu_seqlens_q = [0, 3, 5, 9]
    #               │  │  │  │
    #               │  │  │  └─ end of seq3 (position 9)
    #               │  │  └──── end of seq2 (position 5)
    #               │  └─────── end of seq1 (position 3)
    #               └────────── start (position 0)
    def prepare_prefill(self, seqs: list[Sequence]) -> torch.Tensor:
        batch = build_prefill_batch(seqs, self.block_size)
        if self.timings:
            self.timings.h2d()
        input_ids = torch.tensor(batch["input_ids"], dtype=torch.long, pin_memory=True).cuda(non_blocking=True)
        slot_mapping_tensor = torch.tensor(batch["slot_mapping"], dtype=torch.long, pin_memory=True).cuda(non_blocking=True)

        set_context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor(batch["cu_seqlens_q"], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            cu_seqlens_k=torch.tensor(batch["cu_seqlens_k"], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            max_seqlen_q=batch["max_seqlen_q"],
            max_seqlen_k=batch["max_seqlen_k"],
            slot_mapping=slot_mapping_tensor,
            context_lens=None,
            block_tables=torch.tensor(batch["block_tables"], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True) if batch["block_tables"] else None,
            positions=torch.tensor(batch["positions"], dtype=torch.long, pin_memory=True).cuda(non_blocking=True),
            logit_indices=torch.tensor(batch["logit_indices"], dtype=torch.long, pin_memory=True).cuda(non_blocking=True),
        )
        return input_ids


    # prepare input data for decoding
    def prepare_decode(self, seqs: list[Sequence]) -> torch.Tensor:
        input_ids = []
        context_lens = []   
        slot_mappings = []  
        block_tables = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            context_lens.append(len(seq))
            slot_mappings.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        all_block_tables = [seq.block_table for seq in seqs]
        max_num_blocks = max(len(bt) for bt in all_block_tables)
        for i, seq in enumerate(seqs):
            block_table = seq.block_table + [-1]*(max_num_blocks - len(seq.block_table))
            block_tables.append(block_table)
        if self.timings:
            self.timings.h2d()
        input_ids = torch.tensor(input_ids, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)
        set_context(
            is_prefill=False,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=0,
            max_seqlen_k=0,
            slot_mapping=torch.tensor(slot_mappings, dtype=torch.long, pin_memory=True).cuda(non_blocking=True),
            context_lens=torch.tensor(context_lens, dtype=torch.long, pin_memory=True).cuda(non_blocking=True),
            block_tables=torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True) if block_tables else None,
            positions=torch.tensor([length - 1 for length in context_lens], dtype=torch.long, pin_memory=True).cuda(non_blocking=True),
        )
        return input_ids    

    # prepare the temperature
    def prepare_sample(self, seqs: list[Sequence]) -> None:
        return torch.tensor([seq.temperature for seq in seqs], dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)

    # when prefilling, directly compute model forward + logits
    # when decoding, use cuda graph execution to speed up
    # allocate input_ids, positions, slot_mapping, context_lens, block_tables, outputs
    # into graph_variable, and then replay the graph
    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        if is_prefill or self.enforce_eager:
            # For varlen prefill, keep input_ids as 1D (concatenated tokens)
            # Do NOT unsqueeze - flash_attn_varlen_func expects 1D input with cu_seqlens
            hidden_states = self.model(input_ids)
            logits = self.model.compute_logits(hidden_states)
        else:
            bs = input_ids.size(0)
            context = get_context()

            # finds smallest captured graph that fits the batch size
            graph = self.graphs[next(bs_ for bs_ in self.graphs.keys() if bs_ >= bs)]
            vars = self.graph_vars
            # copy input data into graph variables
            vars['input_ids'].zero_()
            vars['input_ids'][:bs].copy_(input_ids)
            vars['slot_mapping'].fill_(-1)
            vars['slot_mapping'][:bs].copy_(context.slot_mapping)
            vars["context_lens"].zero_()
            vars['context_lens'][:bs].copy_(context.context_lens)
            vars['positions'].zero_()
            vars['positions'][:bs].copy_(context.positions)
            vars["block_tables"].fill_(-1)
            vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            # replay the graph
            graph.replay()
            logits = self.model.compute_logits(vars['outputs'][:bs])

        return logits


    # prepare prefill
    # prepare sample
    # run model
    # sample logits
    # reset context
    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        if self.timings:
            self.timings.prepare()
        if is_prefill:
            input_ids = self.prepare_prefill(seqs)
        else:
            input_ids = self.prepare_decode(seqs)
        if self.timings:
            self.timings.end_gpu("h2d")
            self.timings.start_gpu("forward")
        transfers = getattr(self, "layer_transfers", None)
        active_transfers = [transfers[seq.seq_id] for seq in seqs if seq.seq_id in transfers] if transfers else []
        hooks = []
        if active_transfers:
            for layer_id, layer in enumerate(self.model.model.layers):
                def layer_ready(module, args, output, layer_id=layer_id):
                    for transfer in active_transfers:
                        transfer.on_layer(layer_id)
                hooks.append(layer.register_forward_hook(layer_ready))
        try:
            logits = self.run_model(input_ids, is_prefill)
        finally:
            for hook in hooks:
                hook.remove()
            # postprocess may release the source blocks for a terminal first token.
            # Socket sends only retain host slots after these D2H reads complete.
            for transfer in active_transfers:
                transfer.drain_copies()
        if self.timings:
            self.timings.end_gpu("forward")
        # only sample when rank == 0
        token_ids = None
        if self.rank == 0:
            if self.timings:
                self.timings.start_gpu("sample_h2d")
            sample_seqs = [seq for seq in seqs if seq.num_cached_tokens + seq.num_scheduled_tokens == len(seq)]
            temperature = self.prepare_sample(sample_seqs)
            if self.timings:
                self.timings.end_gpu("sample_h2d")
                self.timings.start_gpu("sampling")
            token_ids = self.sampler(logits, temperature)
            if self.timings:
                self.timings.end_gpu("sampling")
        reset_context()
        return token_ids

    # capture the CUDA graph:
    # pre-allocation at maximum sizes: allocated onece and reuse for all graphs
    # capture for different common batch sizes: [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
    # with torch.cuda.graph(graph, self.graph_pool):
    #        run model() and exact sequence of CUDA kernels for running self.model() will be captured
    # (later use graph.replay() to run the captured graph)
    @torch.inference_mode()
    def capture_cudagraph(self) -> None:
        max_bs = min(self.config['max_num_sequences'], self.config['max_num_batched_tokens'], self.config['max_cached_blocks'])
        max_len = self.config['max_model_length']
        max_num_blocks = math.ceil(max_len / self.block_size)
        # for decoding, input is always of shape (batch_size, 1)
        input_ids = torch.zeros(max_bs, dtype=torch.long, device=f'cuda:{self.rank}')
        # for paged attention
        # where to write new KV values in the cache
        slot_mapping = torch.full((max_bs,), -1, dtype=torch.long, device=f'cuda:{self.rank}')
        # how many tokens each sequence has processed
        context_lens = torch.zeros(max_bs, dtype=torch.long, device=f'cuda:{self.rank}')
        # where to read KV values in the cache
        block_tables = torch.full((max_bs, max_num_blocks), -1, dtype=torch.int32, device=f'cuda:{self.rank}')
        positions = torch.zeros(max_bs, dtype=torch.long, device=f'cuda:{self.rank}')
        # output hidden states; the LM head runs after replay
        outputs = torch.zeros(max_bs, self.config['hidden_size'], dtype=self.default_dtype, device=f'cuda:{self.rank}')

        # graphs to be captured for different batch sizes
        batch_sizes = sorted({size for size in [1, 2, 4, 8] + list(range(16, max_bs + 1, 16)) + [max_bs] if size <= max_bs})
        self.graphs = {}
        graph_pool = None
        capture_stream = torch.cuda.Stream(device=self.rank)

        for batch_size in reversed(batch_sizes):
            graph = torch.cuda.CUDAGraph()
            set_context(
                is_prefill=False,
                cu_seqlens_q=None,
                cu_seqlens_k=None,
                max_seqlen_q=0,
                max_seqlen_k=0,
                slot_mapping=slot_mapping[:batch_size],
                context_lens=context_lens[:batch_size],
                block_tables=block_tables[:batch_size],
                positions=positions[:batch_size],
            )
            capture_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(capture_stream):
                outputs[:batch_size] = self.model(input_ids[:batch_size])
            torch.cuda.current_stream().wait_stream(capture_stream)

            with torch.cuda.graph(graph, graph_pool, stream=capture_stream):
                outputs[:batch_size] = self.model(input_ids[:batch_size])
            if graph_pool is None:
                graph_pool = graph.pool()
            # store the captured graph
            self.graphs[batch_size] = graph

            # make sure that the capture is done before resetting and next capture
            torch.cuda.synchronize()
            reset_context()

        self.graphs = dict(sorted(self.graphs.items()))
        self.graph_vars = dict(
            input_ids=input_ids,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            positions=positions,
            outputs=outputs,
        )