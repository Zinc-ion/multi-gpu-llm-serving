"""Run the production controller with CPU-only child substitutions for tests."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import compare_pd

compare_pd.SCRIPT = ROOT / "tests/compare_pd_cpu_child.py"
original_capture = compare_pd.capture


def capture_with_cpu_children(args):
    for role in ("reference", "prefill", "decode"):
        setattr(args, role + "_gpu", "")
    return original_capture(args)


compare_pd.capture = capture_with_cpu_children
compare_pd.main()
