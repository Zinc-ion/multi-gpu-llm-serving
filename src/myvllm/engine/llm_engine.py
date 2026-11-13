import atexit
import torch.distributed as dist
import time
import torch.multiprocessing as mp

from myvllm.engine.sequence import Sequence, SequenceStatus
from myvllm.engine.scheduler import Scheduler
from myvllm.engine.model_runner import ModelRunner
from myvllm.sampling_parameters import SamplingParams
from transformers import AutoTokenizer


def worker_process(config, rank, event):
    """Worker process function that initializes ModelRunner and enters loop."""
    # FIRST print before any other code
    import sys
    import os
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)  # Line buffering
    sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

    model_runner = ModelRunner(config, rank, event)
    model_runner.loop()


class LLMEngine:
    def __init__(self, config: dict):
        self.config = config
        self.request_ids = {}
        world_size = config.get("world_size", 1)
        ctx = mp.get_context("spawn")
        self.processes = []
        self.events = []
        for i in range(1, world_size):
            event = ctx.Event()
            process = ctx.Process(target=worker_process, args=(config, i, event))
            self.events.append(event)
            self.processes.append(process)
            process.start()
        # start the engine only on the master thread with rank = 0
        self.model_runner = ModelRunner(config, rank=0, event=self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.get("model_name_or_path", "gpt2"),
                                                       revision=config.get("tokenizer_revision"), cache_dir=config.get("cache_dir"))
        
        # scheduler needs to init after model_runner: when world_size > 1,
        # ModelRunner.__init__ calls dist.init_process_group() which is a
        # collective barrier — rank-0 blocks until all worker ranks have joined.
        # The scheduler should only be created after that rendezvous completes.
        # When world_size == 1 there is no barrier and no real dependency.
        self.scheduler = Scheduler(
            max_num_sequences=config.get("max_num_sequences", 16),
            max_num_batched_tokens=config.get("max_num_batched_tokens", 1024),
            max_cached_blocks=config.get("max_cached_blocks", 1024),
            block_size=config.get("block_size", 256),
            eos=config["eos"],
            max_model_length=config["max_model_length"]
        )

        atexit.register(self.exit)


    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for process in self.processes:
            process.join()

    # call scheduler to schedule the next batch
    # return scheduled sequences and whether it is for prefilling
    # call model_runner.run() to run the model
    # call postprocessor to process the outputs and update sequences and update block manager
    def step(self, on_tokens=None) -> tuple[list[tuple[int, list[int]]], int, bool]:
        timings = self.model_runner.timings
        if timings:
            timings.begin()
            schedule_start = time.perf_counter()
        scheduled_sequences, is_prefill = self.scheduler.schedule()
        if timings:
            timings.cpu["schedule_ms"] = (time.perf_counter() - schedule_start) * 1000
        num_processed_tokens = 0
        if not scheduled_sequences:
            return [], num_processed_tokens, is_prefill
        token_count = sum(len(seq) for seq in scheduled_sequences) if is_prefill else len(scheduled_sequences)
        # run the model
        outputs = self.model_runner.call("run", scheduled_sequences, is_prefill)
        # Move outputs to CPU and convert them to a list
        if timings:
            d2h_start = time.perf_counter()
        if outputs is not None:
            outputs = outputs.cpu().tolist()
        if timings:
            timings.cpu["d2h_ms"] = (time.perf_counter() - d2h_start) * 1000
            postprocess_start = time.perf_counter()
        # postprocess the outputs
        self.scheduler.postprocess(scheduled_sequences, outputs)
        if timings:
            timings.cpu["postprocess_ms"] = (time.perf_counter() - postprocess_start) * 1000
            timings.write(self.config["timing_path"], is_prefill, scheduled_sequences, token_count, self.request_ids)
        if on_tokens is not None:
            for seq, token in zip(scheduled_sequences, outputs):
                on_tokens(seq, token)

        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in scheduled_sequences if seq.is_finished]
        num_processed_tokens = sum(len(seq) for seq in scheduled_sequences) if is_prefill else len(scheduled_sequences)

        return outputs, num_processed_tokens, is_prefill


    # add prompt string to the waiting queue by first transforming it to Sequence object
    def add_prompt(self, prompt: str, sampling_params: SamplingParams) -> Sequence:
        seq = Sequence(token_ids=self.tokenizer.encode(prompt), block_size=self.config['block_size'],sampling_params=sampling_params)
        self.scheduler.add_sequence(seq)
        return seq

    # given a list of prompts
    # add_prompt for each prompt
    # call step until all sequences are finished
    # return the generated texts
    def generate(self, prompts: list[str], sampling_params: SamplingParams) -> dict:
        seqs = []
        try:
            for prompt in prompts:
                seqs.append(self.add_prompt(prompt, sampling_params))
        except Exception:
            for seq in seqs:
                self.scheduler.cancel(seq.seq_id)
            raise
        generated_tokens = {}
        while not self.scheduler.is_finished():
            start_t = time.time()
            outputs, num_processed_tokens, is_prefill = self.step()
            end_t = time.time()
            running_time = end_t - start_t + 1e-10
            if is_prefill:
                print(num_processed_tokens, 'number of processed tokens', num_processed_tokens/running_time, "tokens/sec during prefilling")
            else:
                print(num_processed_tokens, 'number of processed tokens', num_processed_tokens/running_time, "tokens/sec during decoding")
            generated_tokens.update({seq_id: tokens for seq_id, tokens in outputs})

        generated_tokens = [generated_tokens.get(seq.seq_id, []) for seq in seqs]
        finish_reasons = ['cancelled' if seq.status == SequenceStatus.CANCELLED else 'stop' for seq in seqs]
        output = {'text': [self.tokenizer.decode(tokens) for tokens in generated_tokens], 'token_ids': generated_tokens, 'finish_reason': finish_reasons}
        return output

    def cancel(self, seq_id: int) -> bool:
        return self.scheduler.cancel(seq_id)
