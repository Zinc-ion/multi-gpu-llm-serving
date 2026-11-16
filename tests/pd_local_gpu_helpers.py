"""Independent tiny GPU P and D processes for deterministic mixed batches."""

import traceback

import torch

from myvllm.pd import DecodeCore, PrefillCore
from myvllm.sampling_parameters import SamplingParams
from pd_gpu_helpers import CaptureEngine


class MixedCaptureEngine(CaptureEngine):
    def __init__(self, config):
        super().__init__(config)
        self.batches = []
        call = self.model_runner.call

        def record(method, *args, **kwargs):
            if method == 'run':
                seqs, is_prefill = args
                self.batches.append({
                    'is_prefill': is_prefill,
                    'sequences': [(self.request_ids[s.seq_id], s.num_cached_tokens,
                                   s.num_cached_tokens + s.num_scheduled_tokens) for s in seqs],
                })
            return call(method, *args, **kwargs)
        self.model_runner.call = record

    def exit(self):
        if getattr(self, 'model_runner', None) is not None and hasattr(self, 'batches'):
            torch.save(self.batches, self.capture_path + '.batches.pt')
        super().exit()


def prefill_process(config, session, case, result):
    engine = None
    try:
        engine = CaptureEngine(config)
        core = PrefillCore(engine, 4, 1_000_000)
        h = core.open(session, case['id'], case['prompt'], SamplingParams(**case['sampling']))
        events = core.step()
        assert len(events) == 1 and not events[0].finished
        metadata = core.describe(h)
        payload = core.export(h)
        assert core.acknowledge(h)
        assert core.stats()['used_blocks'] == 0
        core.shutdown()
        result.send(('ok', metadata, payload))
    except BaseException:
        result.send(('error', traceback.format_exc()))
    finally:
        if engine is not None:
            engine.exit()
        result.close()


def decode_process(config, remote_case, local_case, channel):
    engine = None
    try:
        engine = MixedCaptureEngine(config)
        core = DecodeCore(engine, 4, 1_000_000)
        remote = core.open()
        channel.send(('ready', remote.session_id))
        metadata, payload = channel.recv()
        assert core.reserve(remote, metadata) and core.restore(remote, payload)
        assert core.activate(remote)
        local = core.open_local(local_case['id'], local_case['prompt'],
                                SamplingParams(**local_case['sampling']))
        output = {remote_case['id']: [], local_case['id']: []}
        while not core.scheduler.is_finished():
            for event in core.step():
                output[event.handle.request_id].append(event.token)
            stats = core.stats()
            assert stats['used_blocks'] <= stats['reserved_blocks'] <= stats['total_blocks']
        assert core.stats()['reserved_blocks'] == core.stats()['used_blocks'] == 0
        core.shutdown()
        engine.exit()
        engine = None
        channel.send(('ok', output))
    except BaseException:
        channel.send(('error', traceback.format_exc()))
    finally:
        if engine is not None:
            engine.exit()
        channel.close()
