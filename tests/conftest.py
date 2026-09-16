"""Suite-wide pytest configuration."""

import os

import pytest

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
