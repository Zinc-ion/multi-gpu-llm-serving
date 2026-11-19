"""Request-scoped registered host buffers for Mooncake TCP v2."""

from importlib.metadata import version
import os
import time


HOST = "127.0.0.1"
METADATA = "P2PHANDSHAKE"
VERSION = "0.3.14"


def configure():
    if version("mooncake-transfer-engine") != VERSION:
        raise RuntimeError(f"Mooncake TCP requires mooncake-transfer-engine=={VERSION}")
    # The classic engine can autodiscover RDMA even with protocol="tcp".
    os.environ["MC_FORCE_TCP"] = "1"
    os.environ["MC_USE_TENT"] = "0"
    from mooncake.engine import ScatterTransferCompletionStatus, TransferEngine

    return TransferEngine, ScatterTransferCompletionStatus.COMPLETED


class RegisteredSlots:
    def __init__(self, binding, slots):
        engine_type, self.completed = binding
        started = time.perf_counter()
        self.engine = engine_type()
        if self.engine.initialize(HOST, METADATA, "tcp", "") != 0:
            raise RuntimeError("Mooncake TCP initialization failed")
        self.engine_ms = (time.perf_counter() - started) * 1000
        self.segment = f"{HOST}:{self.engine.get_rpc_port()}"
        self.slots = slots
        started = time.perf_counter()
        for slot in slots:
            if self.engine.register_memory(slot.data_ptr(), slot.numel(), "cpu:0") != 0:
                self.close()
                raise RuntimeError("Mooncake host memory registration failed")
        self.register_ms = (time.perf_counter() - started) * 1000

    def destination(self):
        slot, = self.slots
        return {"segment": self.segment, "address": slot.data_ptr(), "capacity": slot.numel(),
                "engine_ms": self.engine_ms, "register_ms": self.register_ms}

    def write(self, index, peer, length):
        slot = self.slots[index]
        ticket = self.engine.submit_scatter_write(
            peer["segment"], [slot.data_ptr()], [slot.numel()],
            [peer["address"]], [peer["capacity"]], [[0]], [[0]], [[length]],
        )
        # A ticket may time out while native I/O still owns the source slot.
        # Keep it live until a drained status, including after Python cancellation.
        while not ticket.drained:
            ticket.drain(1000)
        if ticket.status != self.completed:
            raise ConnectionError(f"Mooncake TCP write failed: {ticket.status.name}")

    def close(self):
        # Destroy the private engine before releasing its registered memory.
        # On an aborted D request the peer may still be writing to the slab.
        self.engine = None
        self.slots = ()
