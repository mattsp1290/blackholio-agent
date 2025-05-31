#!/usr/bin/env python3
"""
Test runner script for Blackholio Agent.

Provides convenient ways to run different test suites with appropriate options.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd: list, cwd: Path = None) -> int:
    """Run a command and return the exit code."""
    print(f"Running: {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=cwd)


def main():
    parser = argparse.ArgumentParser(description="Run tests for Blackholio Agent")
    parser.add_argument(
        "suite",
        nargs="?",
        default="all",
        choices=["all", "unit", "integration", "performance", "quick"],
        help="Test suite to run (default: all)"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Enable benchmark tests (disabled by default)"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        default=True,
        help="Generate coverage report (default: True)"
    )
    parser.add_argument(
        "--no-coverage",
        dest="coverage",
        action="store_false",
        help="Disable coverage report"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--failfast",
        "-x",
        action="store_true",
        help="Stop on first failure"
    )
    parser.add_argument(
        "--parallel",
        "-n",
        type=int,
        metavar="NUM",
        help="Run tests in parallel with NUM workers"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Include GPU tests"
    )
    parser.add_argument(
        "--markers",
        "-m",
        help="Run tests matching given mark expression"
    )
    parser.add_argument(
        "--keyword",
        "-k",
        help="Run tests matching given keyword expression"
    )
    
    args = parser.parse_args()
    
    # Base pytest command
    cmd = ["pytest"]
    
    # Add coverage options
    if args.coverage:
        cmd.extend([
            "--cov=src/blackholio_agent",
            "--cov-report=term-missing",
            "--cov-report=html"
        ])
    
    # Add benchmark option (only if pytest-benchmark is installed)
    try:
        import pytest_benchmark
        if args.benchmark:
            cmd.append("--benchmark-only")
        else:
            cmd.append("--benchmark-disable")
    except ImportError:
        # pytest-benchmark not installed, skip benchmark options
        pass
    
    # Add verbosity
    if args.verbose:
        cmd.append("-vv")
    else:
        cmd.append("-v")
    
    # Add failfast
    if args.failfast:
        cmd.append("-x")
    
    # Add parallel execution
    if args.parallel:
        cmd.extend(["-n", str(args.parallel)])
    
    # Add custom markers
    if args.markers:
        cmd.extend(["-m", args.markers])
    
    # Add keyword filter
    if args.keyword:
        cmd.extend(["-k", args.keyword])
    
    # Select test suite
    if args.suite == "unit":
        cmd.extend(["-m", "unit", "src/blackholio_agent/tests/unit"])
    elif args.suite == "integration":
        cmd.extend(["-m", "integration", "src/blackholio_agent/tests/integration"])
    elif args.suite == "performance":
        cmd.extend(["-m", "performance", "src/blackholio_agent/tests/performance"])
        if args.benchmark:
            cmd.append("--benchmark-compare")
    elif args.suite == "quick":
        # Quick tests exclude slow and performance tests
        cmd.extend(["-m", "not slow and not performance"])
    elif args.suite == "all":
        # Run all tests
        if not args.gpu:
            cmd.extend(["-m", "not requires_gpu"])
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Run tests
    exit_code = run_command(cmd)
    
    # Print coverage report location if generated
    if args.coverage and exit_code == 0:
        print("\nCoverage report generated at: htmlcov/index.html")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
