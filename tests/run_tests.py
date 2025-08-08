"""miniRL.tests.run_tests"""

from __future__ import annotations

import importlib
import io
import os
import pkgutil
import sys
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from datetime import datetime

# ANSI colors (fallback to plain text if not a TTY)
def _color(code: str, text: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"

GREEN = lambda s: _color("92", s)
RED = lambda s: _color("91", s)
YELLOW = lambda s: _color("93", s)
BOLD = lambda s: _color("1", s)
DIM = lambda s: _color("2", s)

@dataclass
class TestResult:
    name: str
    status: str  # PASS | FAIL | ERROR
    duration_s: float
    message: str = ""
    stdout: str = ""
    traceback: str = ""


def _discover() -> list[tuple[str, str]]:
    """Return list of (module_name, function_name) sorted by name."""
    discovered: list[tuple[str, str]] = []
    for _, modname, _ in pkgutil.iter_modules(["tests"]):
        if not modname.startswith("test_"):
            continue
        module = importlib.import_module(f"tests.{modname}")
        for name, obj in vars(module).items():
            if name.startswith("test_") and callable(obj):
                discovered.append((modname, name))
    return sorted(discovered)


def _run_test(modname: str, funcname: str, verbose: bool) -> TestResult:
    module = importlib.import_module(f"tests.{modname}")
    func = getattr(module, funcname)
    fqname = f"{modname}.{funcname}"

    stdout_buf = io.StringIO()
    t0 = time.perf_counter()
    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stdout_buf):
            func()
        dt = time.perf_counter() - t0
        return TestResult(name=fqname, status="PASS", duration_s=dt, stdout=stdout_buf.getvalue())
    except AssertionError as e:
        dt = time.perf_counter() - t0
        tb = traceback.format_exc()
        return TestResult(name=fqname, status="FAIL", duration_s=dt, message=str(e), stdout=stdout_buf.getvalue(), traceback=tb)
    except Exception as e:  # noqa: BLE001
        dt = time.perf_counter() - t0
        tb = traceback.format_exc()
        return TestResult(name=fqname, status="ERROR", duration_s=dt, message=f"{type(e).__name__}: {e}", stdout=stdout_buf.getvalue(), traceback=tb)


def run() -> int:
    verbose = ("-v" in sys.argv) or ("--verbose" in sys.argv) or (os.environ.get("TESTS_VERBOSE") == "1")

    header = f"miniRL tests | Python {sys.version.split()[0]} | {datetime.now().strftime('%H:%M:%S')}"
    print(BOLD(header))
    print(DIM(f"cwd: {os.getcwd()}\n"))

    tests = _discover()
    if not tests:
        print(YELLOW("No tests found under tests/test_*.py"))
        return 0

    # Compute padding for alignment
    max_name_len = max(len(f"{m}.{f}") for m, f in tests)

    results: list[TestResult] = []
    for modname, funcname in tests:
        res = _run_test(modname, funcname, verbose)
        results.append(res)

        name = res.name.ljust(max_name_len)
        ms = int(res.duration_s * 1000)

        if res.status == "PASS":
            status = GREEN("PASS")
            line = f"{status}  {name}  {DIM(f'({ms} ms)')}"
            print(line)
            if verbose and res.stdout.strip():
                print(DIM(res.stdout.rstrip()))
        elif res.status == "FAIL":
            status = RED("FAIL")
            line = f"{status}  {name}  {DIM(f'({ms} ms)')} - {res.message}"
            print(line)
        else:  # ERROR
            status = YELLOW("ERROR")
            line = f"{status} {name}  {DIM(f'({ms} ms)')} - {res.message}"
            print(line)

    # Summary
    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    errored = sum(1 for r in results if r.status == "ERROR")
    total_time = sum(r.duration_s for r in results)

    summary_parts = [GREEN(f"{passed} passed"), RED(f"{failed} failed") if failed else "0 failed", YELLOW(f"{errored} errors") if errored else "0 errors"]
    print()
    print(BOLD(f"=== {', '.join(summary_parts)} in {total_time:.2f}s ==="))

    # Failures detail
    if failed or errored:
        print()
        print(BOLD("Failures/Errors details:"))
        for r in results:
            if r.status in {"FAIL", "ERROR"}:
                print(BOLD(f"- {r.status} {r.name}"))
                if r.message:
                    print(f"  {r.message}")
                if r.stdout.strip():
                    print(DIM("  --- captured output ---"))
                    for line in r.stdout.rstrip().splitlines():
                        print(DIM(f"  {line}"))
                # print last lines of traceback for brevity
                if r.traceback:
                    print(DIM("  --- traceback ---"))
                    tb_lines = r.traceback.rstrip().splitlines()
                    tail = tb_lines[-6:] if len(tb_lines) > 6 else tb_lines
                    for line in tail:
                        print(DIM(f"  {line}"))

    return 1 if (failed or errored) else 0


if __name__ == '__main__':
    sys.exit(run())