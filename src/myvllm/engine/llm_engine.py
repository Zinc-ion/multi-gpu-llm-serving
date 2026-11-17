import atexit
import fcntl
import os
import signal
import torch.distributed as dist
import time
import torch.multiprocessing as mp
from multiprocessing import parent_process
from multiprocessing.connection import wait

from myvllm.engine.sequence import Sequence, SequenceStatus
from myvllm.engine.scheduler import Scheduler
from myvllm.engine.model_runner import ModelRunner
from myvllm.engine.ipc import CommandSignal, CommandSizeError, CommandSlot, stop_workers, validate_timeout
from myvllm.config import resolve_fp8_prefill_compute_dtype
from myvllm.sampling_parameters import SamplingParams
from transformers import AutoTokenizer


def worker_process(config, rank, event):
    """Worker process function that initializes ModelRunner and enters loop."""
    # FIRST print before any other code
    import sys
    import os
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)  # Line buffering
    sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

    parent_fd = None
    flags = None
    model_runner = None
    try:
        parent_fd = os.dup(parent_process().sentinel)
        # Spawn's parent sentinel is an idle pipe until rank 0 exits. Kernel
        # notification also works while checkpoint loading holds the GIL.
        flags = fcntl.fcntl(parent_fd, fcntl.F_GETFL)
        fcntl.fcntl(parent_fd, fcntl.F_SETSIG, signal.SIGKILL)
        fcntl.fcntl(parent_fd, fcntl.F_SETOWN, os.getpid())
        fcntl.fcntl(parent_fd, fcntl.F_SETFL, flags | os.O_ASYNC)
        if wait([parent_fd], timeout=0):
            os._exit(1)
        model_runner = ModelRunner(config, rank, event)
        model_runner.loop()
    finally:
        error = sys.exception()
        cleanup = []
        if model_runner is not None:
            cleanup.append(("Runner shutdown", model_runner.exit))
        if flags is not None:
            cleanup.append(("Parent sentinel flags", lambda: fcntl.fcntl(parent_fd, fcntl.F_SETFL, flags)))
        if parent_fd is not None:
            cleanup.append(("Parent sentinel close", lambda: os.close(parent_fd)))
        cleanup.extend((("Command signal close", event.close),
                        ("Command slot close", config["_instance_resources"][1].close)))
        errors = []
        for name, action in cleanup:
            try:
                action()
            except BaseException as exc:
                if error is not None:
                    error.add_note(f"{name} failed: {exc!r}")
                else:
                    errors.append(exc)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("TP worker cleanup failed", errors)


class LLMEngine:
    _worker_exit_timeout_s = 30
    _worker_term_timeout_s = 10
    _worker_kill_timeout_s = 10

    def __init__(self, config: dict):
        config["fp8_prefill_compute_dtype"] = resolve_fp8_prefill_compute_dtype(
            config.get("fp8_prefill_compute_dtype", "float32"))
        validate_timeout(config.get("tp_command_timeout_s", 30), "tp_command_timeout_s")
        self.config = config
        self.request_ids = {}
        world_size = config.get("world_size", 1)
        self._store = None
        self._shm = None
        self.model_runner = None
        self.processes = []
        self.events = []
        self.worker_pids = []
        self.worker_exitcodes = []
        had_process_group = dist.is_initialized()
        try:
            self._store = dist.TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False)
            self._shm = CommandSlot(2**20) if world_size > 1 else None
            runner_config = config.copy()
            runner_config["_instance_resources"] = (
                self._store.port, self._shm,
            )
            ctx = mp.get_context("spawn")
            for i in range(1, world_size):
                event = CommandSignal(ctx)
                process = ctx.Process(target=worker_process, args=(runner_config, i, event))
                self.events.append(event)
                process.start()
                self.processes.append(process)
                event.close_worker_ends()
            # start the engine only on the master thread with rank = 0
            self.model_runner = ModelRunner(runner_config, rank=0, event=self.events)
            self.model_runner._worker_processes = self.processes
        except BaseException as error:
            try:
                if not had_process_group and dist.is_initialized():
                    dist.destroy_process_group()
            except Exception as exc:
                error.add_note(f"Process-group cleanup failed: {exc!r}")
            try:
                self._cleanup_resources(graceful=False)
            except Exception as exc:
                error.add_note(f"Engine resource cleanup failed: {exc!r}")
            raise
        config["max_cached_blocks"] = runner_config["max_cached_blocks"]
        self.model_runner.config = config
        try:
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
                max_model_length=config["max_model_length"],
                enable_chunked_prefill=config.get("enable_chunked_prefill", False),
            )
        except BaseException as error:
            try:
                self.exit()
            except Exception as exc:
                error.add_note(f"Engine shutdown failed: {exc!r}")
            raise

        atexit.register(self.exit)


    def exit(self):
        if self.model_runner is None:
            if self.processes or self.events or self._shm is not None or self._store is not None:
                self._cleanup_resources(graceful=False)
            return
        atexit.unregister(self.exit)
        runner, self.model_runner = self.model_runner, None
        error = None
        try:
            runner.call("exit")
        except BaseException as exc:
            error = exc
            try:
                runner.exit()
            except BaseException as exc:
                error.add_note(f"Runner shutdown failed: {exc!r}")
        try:
            self._cleanup_resources(graceful=error is None)
        except BaseException as exc:
            if error is None:
                error = exc
            else:
                error.add_note(f"Engine resource cleanup failed: {exc!r}")
        if error is not None:
            raise error

    def _cleanup_resources(self, graceful):
        errors = []
        try:
            stop_workers(self.processes, self._worker_exit_timeout_s,
                         self._worker_term_timeout_s, self._worker_kill_timeout_s,
                         graceful=graceful)
        except Exception as exc:
            errors.append(exc)
        if any(process.is_alive() for process in self.processes):
            if len(errors) == 1:
                raise errors[0]
            raise ExceptionGroup("Engine resource cleanup failed", errors)
        if self.processes and not self.worker_pids:
            self.worker_pids = [process.pid for process in self.processes]
            self.worker_exitcodes = [process.exitcode for process in self.processes]
        remaining = []
        for process in self.processes:
            try:
                process.close()
            except Exception as exc:
                errors.append(exc)
                remaining.append(process)
        self.processes[:] = remaining
        remaining = []
        for event in self.events:
            try:
                event.close()
            except Exception as exc:
                errors.append(exc)
                remaining.append(event)
        self.events[:] = remaining
        if self._shm:
            try:
                self._shm.close()
            except Exception as exc:
                errors.append(exc)
            else:
                self._shm = None
        self._store = None
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise ExceptionGroup("Engine resource cleanup failed", errors)

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
        token_count = sum(seq.num_scheduled_tokens for seq in scheduled_sequences)
        sampled_sequences = [
            seq for seq in scheduled_sequences
            if seq.num_cached_tokens + seq.num_scheduled_tokens == len(seq)
        ]
        # run the model
        try:
            outputs = self.model_runner.call("run", scheduled_sequences, is_prefill)
        except CommandSizeError:
            for seq in scheduled_sequences:
                self.cancel(seq.seq_id)
                self.request_ids.pop(seq.seq_id, None)
            raise
        # Move outputs to CPU and convert them to a list
        if timings:
            d2h_start = time.perf_counter()
            timings.start_gpu("d2h")
        if outputs is not None:
            outputs = outputs.cpu()
            if timings:
                timings.end_gpu("d2h")
            outputs = outputs.tolist()
        if timings:
            timings.cpu["d2h_host_ms"] = (time.perf_counter() - d2h_start) * 1000
            postprocess_start = time.perf_counter()
        # postprocess the outputs
        self.scheduler.postprocess(scheduled_sequences, outputs)
        if timings:
            timings.cpu["postprocess_ms"] = (time.perf_counter() - postprocess_start) * 1000
            timings.write(self.config["timing_path"], is_prefill, scheduled_sequences, token_count, self.request_ids)
        if on_tokens is not None:
            for seq, token in zip(sampled_sequences, outputs):
                on_tokens(seq, token)

        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in scheduled_sequences if seq.is_finished]
        num_processed_tokens = token_count

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
