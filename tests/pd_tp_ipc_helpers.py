"""CPU command/lifecycle boundary; ModelRunner owns the actual dispatch."""
import os
import signal

import torch
import torch.distributed as dist

from myvllm.engine.model_runner import ModelRunner
from myvllm.engine.ipc import read_command


class CPUCommandRunner(ModelRunner):
    def _initialize(self, config, rank, event):
        self.config, self.rank, self.event = config, rank, event
        self.world_size = config["world_size"]
        self.timings = None
        self._exited = False
        self.records = []
        self.tp_command_timeout_s = config.get("tp_command_timeout_s", 30)
        self.shm = config["_instance_resources"][1]
        if config.get("_gloo"):
            from datetime import timedelta

            store = dist.TCPStore("127.0.0.1", config["_instance_resources"][0],
                                  is_master=False)
            dist.init_process_group("gloo", store=store, rank=rank,
                                    world_size=self.world_size,
                                    timeout=timedelta(seconds=5))

    def record(self, value):
        self.records.append(value)
        if self.rank:
            message = ("record", value)
            if self.config.get("_rank_records"):
                message = (self.rank, *message)
            self.config["_records"].send(message)
        return value

    def collective(self):
        value = torch.tensor(self.rank + 1)
        dist.all_reduce(value)
        if self.rank:
            self.config["_records"].send(("collective", value.item()))
        return value.item()

    def exit(self):
        if self._exited:
            return
        self._exited = True
        try:
            if self.rank and "_exit_gate" in self.config:
                self.config["_exit_gate"].recv_bytes()
            if self.rank and self.rank == self.config.get("_exit_fail_rank"):
                self.config["_exit_fail_gate"].recv_bytes()
                os._exit(23)
            if not self.rank and "_exit_fail_release" in self.config:
                self.config["_exit_fail_release"].send_bytes(b"exit")
            if not self.rank and "_exit_io_error" in self.config:
                with open(self.config["_exit_io_error"], "w"):
                    pass
        finally:
            self.shm.close()
            if dist.is_initialized():
                dist.destroy_process_group()


def cpu_command_worker(config, rank, event):
    if config.get("_ignore_term"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    runner = CPUCommandRunner(config, rank, event)
    config["_ready"].set()
    if "_ready_by_rank" in config:
        config["_ready_by_rank"][rank].set()
    if config.get("_exit_before_read"):
        os._exit(17)
    try:
        if "_read_gate" in config:
            config["_read_gate"].wait()
        if "_read_gates" in config:
            config["_read_gates"][rank].wait()
        runner.loop()
    finally:
        runner.exit()
        event.close()


def waiting_command_reader(shm, event, ready):
    try:
        ready.send(True)
        read_command(shm, event, 30)
    finally:
        shm.close()
        event.close()
        ready.close()
