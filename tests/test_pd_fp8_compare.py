from copy import deepcopy

import pytest

from test_compare_runs import run
from tools.compare_runs import compare_runs


def test_legacy_auto_and_explicit_auto_match_but_fp8_is_not_lossless():
    legacy = run()
    auto = deepcopy(legacy)
    auto["metadata"]["config"]["kv_cache_dtype"] = "auto"
    assert compare_runs(legacy, auto)["tokens_identical"]
    fp8 = deepcopy(auto)
    fp8["metadata"]["config"]["kv_cache_dtype"] = "float8_e4m3fn"
    with pytest.raises(ValueError, match="kv_cache_dtype"):
        compare_runs(legacy, fp8)
    assert compare_runs(fp8, deepcopy(fp8))["tokens_identical"]
