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

#: The cache is pruned to this size, oldest entries first, once per run. JAX's own cap,
#: `jax_compilation_cache_max_size`, is not used: it rescans and reads every entry under a
#: lock on each write, which made the suite quadratic in the number of compiled programs.
CACHE_MAX_BYTES = 1 << 30

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


def pytest_sessionfinish(session):
    """Prunes the compilation cache in the main process, after any xdist workers."""
    if hasattr(session.config, "workerinput"):
        return
    # Access times left by JAX's cap in older caches; nothing reads them now.
    for path in Path(jax.config.jax_compilation_cache_dir).glob("*-atime"):
        path.unlink(missing_ok=True)
    entries = []
    for path in Path(jax.config.jax_compilation_cache_dir).glob("*-cache"):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        entries.append((stat.st_mtime, stat.st_size, path))
    excess = sum(size for _, size, _ in entries) - CACHE_MAX_BYTES
    for _, size, path in sorted(entries):
        if excess <= 0:
            break
        path.unlink(missing_ok=True)
        excess -= size
