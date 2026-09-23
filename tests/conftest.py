"""Suite-wide pytest configuration."""

import os
from pathlib import Path

import jax
import pytest

#: Compilation dominates test time, so compiled programs persist across runs and xdist
#: workers. JAX_COMPILATION_CACHE_DIR overrides the location.
if jax.config.jax_compilation_cache_dir is None:
    jax.config.update("jax_compilation_cache_dir", str(Path(__file__).parents[1] / ".jax_cache"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_compilation_cache_max_size", 1 << 30)

#: When set (as CI does), a skipped test fails the run. Optional-dependency skips exist
#: for users without the full backend set; the full test environment must run everything.
REQUIRE_NO_SKIPS = os.environ.get("PMRF_TESTS_NO_SKIPS") == "1"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if REQUIRE_NO_SKIPS and report.skipped and not hasattr(report, "wasxfail"):
        reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else report.longrepr
        report.outcome = "failed"
        report.longrepr = f"Skipped with PMRF_TESTS_NO_SKIPS=1: {reason}"
