"""Owned-process failure paths of the TP2 GPU smoke, exercised without CUDA."""

from contextlib import contextmanager, suppress
import ctypes
import multiprocessing as mp
import os
from pathlib import Path
import signal
import time

import pytest

import test_pd_tp_gpu as smoke


def _bad_start(pid_file, child_pid_file, ready, stop):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    child = os.fork()
    if child == 0:
        Path(child_pid_file).write_text(str(os.getpid()))
        while True:
            signal.pause()
    deadline = time.monotonic() + 5
    while not Path(child_pid_file).exists():
        if time.monotonic() >= deadline:
            raise TimeoutError('descendant did not start')
        time.sleep(.01)
    Path(pid_file).write_text(str(os.getpid()))
    ready.send(('error', 'intentional startup failure'))
    ready.close()
    while True:
        signal.pause()


def _exit_after_stop(code, ready, stop):
    ready.send(('ready', os.getpid()))
    stop.recv()
    if code:
        os._exit(code)


def _ignore_stop_and_term(ready, stop):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ready.send(('ready', os.getpid()))
    while True:
        signal.pause()


def _exit_zero_on_term(ready, stop):
    def exit_zero(sig, frame):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, exit_zero)
    ready.send(('ready', os.getpid()))
    while True:
        signal.pause()


def _stuck_reference(visible, config, cases, ready):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    Path(config['pid_file']).write_text(str(os.getpid()))
    ready.send(('ok', os.getpid()))
    while True:
        signal.pause()


def _running(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text()
    except FileNotFoundError:
        return False
    return stat[stat.rfind(')') + 2:].split()[0] != 'Z'


@contextmanager
def _reap_test_descendant(pid_file):
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0
    assert libc.prctl(36, 1, 0, 0, 0) == 0
    try:
        yield
    finally:
        if pid_file.exists():
            pid = int(pid_file.read_text())
            if _running(pid):
                with suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGKILL)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                try:
                    reaped, _ = os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    break
                if reaped:
                    break
                time.sleep(.01)
        assert libc.prctl(36, previous.value, 0, 0, 0) == 0


@pytest.fixture
def short_timeouts(monkeypatch):
    monkeypatch.setattr(smoke, '_START_TIMEOUT', 30)
    monkeypatch.setattr(smoke, '_REFERENCE_TIMEOUT', .2)
    monkeypatch.setattr(smoke, '_STOP_TIMEOUT', .2)
    monkeypatch.setattr(smoke, '_TERM_TIMEOUT', .2)
    monkeypatch.setattr(smoke, '_KILL_TIMEOUT', 3)


def _cleanup(owners):
    with suppress(BaseException):
        smoke._stop_workers(owners)


def test_startup_failure_reaps_a_term_resistant_group(tmp_path, short_timeouts):
    pid_file = tmp_path / 'pid'
    child_pid_file = tmp_path / 'child-pid'
    owners = []
    with _reap_test_descendant(child_pid_file):
        try:
            with pytest.raises(AssertionError, match='intentional startup failure') as caught:
                smoke._start(mp.get_context('spawn'), _bad_start,
                             (str(pid_file), str(child_pid_file)), owners=owners)
            pid = int(pid_file.read_text())
            child = int(child_pid_file.read_text())
            assert not _running(pid), f'owned process {pid} survived startup failure'
            assert not _running(child), f'owned descendant {child} survived startup failure'
            assert owners[0]['exitcode'] == -signal.SIGKILL
            assert owners[0]['closed']
            assert owners[0]['signals'] == ['SIGTERM', 'SIGKILL']
            assert 'exit code -9' in ' '.join(caught.value.__notes__)
        finally:
            _cleanup(owners)


def test_all_groups_are_cleaned_without_overwriting_original_failure(short_timeouts):
    owners = []
    try:
        ctx = mp.get_context('spawn')
        normal = smoke._start(ctx, _exit_after_stop, (0,), owners=owners)
        stuck = smoke._start(ctx, _ignore_stop_and_term, (), owners=owners)
        failed = smoke._start(ctx, _exit_after_stop, (7,), owners=owners)
        pids = [item['pid'] for item in owners]
        with pytest.raises(ValueError, match='original failure') as caught:
            try:
                raise ValueError('original failure')
            finally:
                smoke._stop_workers(owners)
        notes = ' '.join(caught.value.__notes__)
        assert 'exit code 7' in notes and 'exit code -9' in notes
        assert [normal['exitcode'], stuck['exitcode'], failed['exitcode']] == [0, -signal.SIGKILL, 7]
        assert all(item['closed'] for item in owners)
        assert all(not _running(pid) for pid in pids)
        for item in owners:
            with pytest.raises(ValueError, match='closed'):
                item['process'].process.is_alive()
    finally:
        _cleanup(owners)


def test_graceful_timeout_is_reported_even_if_term_exits_zero(short_timeouts, monkeypatch):
    monkeypatch.setattr(smoke, '_TERM_TIMEOUT', 3)
    owners = []
    try:
        item = smoke._start(mp.get_context('spawn'), _exit_zero_on_term, (), owners=owners)
        with pytest.raises(RuntimeError, match='required signals'):
            smoke._stop_workers(owners)
        assert item['exitcode'] == 0
        assert item['signals'] == ['SIGTERM']
        assert item['closed'] and not _running(item['pid'])
    finally:
        _cleanup(owners)


def test_reference_timeout_preserves_assertion_and_cleans_process(tmp_path, short_timeouts, monkeypatch):
    pid_file = tmp_path / 'reference-pid'
    monkeypatch.setattr(smoke, 'reference_on_gpu', _stuck_reference)
    with pytest.raises(AssertionError, match='reference did not exit successfully') as caught:
        smoke._reference(mp.get_context('fork'), '0', {'pid_file': str(pid_file)}, [])
    assert not _running(int(pid_file.read_text()))
    assert 'exit code -9' in ' '.join(caught.value.__notes__)


def test_spawn_failure_closes_unstarted_process_and_pipe(short_timeouts):
    owners = []
    try:
        with pytest.raises((AttributeError, TypeError), match='pickl'):
            smoke._start(mp.get_context('spawn'), lambda *args: None, (), owners=owners)
        assert owners[0]['pid'] is None
        assert owners[0]['closed']
        assert owners[0]['process'].stdin.connection.closed
    finally:
        _cleanup(owners)
