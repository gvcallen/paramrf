"""Suite-wide pytest configuration."""

import json
import os
from collections import defaultdict
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

TESTS = Path(__file__).parent

#: Seconds per test file from a cold-cache run; only their proportions matter. `--shard`
#: balances files by these, and `--store-durations` refreshes them.
DURATIONS = TESTS / "durations.json"


def pytest_addoption(parser):
    group = parser.getgroup("pmrf")
    group.addoption(
        "--shard", metavar="K/N",
        help="Run the K-th of N shards (1-based). Whole files are balanced by durations.json, "
             "so a file's compiled programs are reused within one process.",
    )
    group.addoption(
        "--store-durations", action="store_true",
        help="Write the duration of each file run into durations.json.",
    )


def _test_file(config, nodeid):
    """The path of a test's file relative to `tests/`, as durations.json keys it."""
    path = config.rootpath / nodeid.split("::")[0]
    return path.relative_to(TESTS).as_posix() if path.is_relative_to(TESTS) else path.as_posix()


def _shards(files, durations, count):
    """Assigns each file to one of `count` shards, longest first to the least loaded.
    A file not in `durations` counts as the mean file."""
    default = sum(durations.values()) / len(durations) if durations else 1.0
    loads = [0.0] * count
    shard_of = {}
    for file in sorted(files, key=lambda f: (-durations.get(f, default), f)):
        shard = loads.index(min(loads))
        shard_of[file] = shard
        loads[shard] += durations.get(file, default)
    return shard_of


def pytest_collection_modifyitems(config, items):
    option = config.getoption("--shard")
    if option is None:
        return
    index, count = (int(part) for part in option.split("/"))
    if not 1 <= index <= count:
        raise pytest.UsageError(f"--shard={option}: K must be between 1 and N")
    durations = json.loads(DURATIONS.read_text()) if DURATIONS.exists() else {}
    files = {_test_file(config, item.nodeid) for item in items}
    shard_of = _shards(files, durations, count)
    selected, deselected = [], []
    for item in items:
        in_shard = shard_of[_test_file(config, item.nodeid)] == index - 1
        (selected if in_shard else deselected).append(item)
    config.hook.pytest_deselected(items=deselected)
    items[:] = selected


class _DurationRecorder:
    """Sums setup, call and teardown time per file. Under xdist the main process receives
    every worker's reports, and writes the file."""

    def __init__(self, config):
        self.config = config
        self.seconds = defaultdict(float)

    def pytest_runtest_logreport(self, report):
        self.seconds[_test_file(self.config, report.nodeid)] += report.duration

    def pytest_sessionfinish(self, session):
        if hasattr(self.config, "workerinput"):
            return
        durations = json.loads(DURATIONS.read_text()) if DURATIONS.exists() else {}
        durations.update({file: round(seconds, 1) for file, seconds in self.seconds.items()})
        DURATIONS.write_text(json.dumps(dict(sorted(durations.items())), indent=1) + "\n")


def pytest_configure(config):
    if config.getoption("--store-durations"):
        config.pluginmanager.register(_DurationRecorder(config), "pmrf-durations")


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
