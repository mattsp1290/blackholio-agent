#!/usr/bin/env python3
"""
Test coverage analysis and reporting script for Blackholio Agent.

Generates comprehensive test coverage reports with visualizations.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
import json
import webbrowser
from typing import Dict, List, Tuple


def run_coverage_command(cmd: list, cwd: Path = None) -> Tuple[int, str]:
    """Run a command and return exit code and output."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


def generate_coverage_report(test_paths: List[str], 
                           output_dir: str = "coverage_report",
                           html: bool = True,
                           xml: bool = True,
                           json_report: bool = True,
                           open_browser: bool = True) -> Dict[str, float]:
    """Generate comprehensive coverage report."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Base coverage command
    cmd = [
        "pytest",
        "--cov=src/blackholio_agent",
        "--cov-branch",  # Include branch coverage
    ]
    
    # Add output formats
    if html:
        cmd.extend(["--cov-report", f"html:{output_path}/html"])
    if xml:
        cmd.extend(["--cov-report", f"xml:{output_path}/coverage.xml"])
    if json_report:
        cmd.extend(["--cov-report", f"json:{output_path}/coverage.json"])
    
    # Always include terminal report
    cmd.extend(["--cov-report", "term-missing:skip-covered"])
    
    # Add test paths
    cmd.extend(test_paths)
    
    # Run coverage
    exit_code, output = run_coverage_command(cmd)
    
    if exit_code != 0:
        print(f"Coverage run failed with exit code {exit_code}")
        print(output)
        return {}
    
    # Parse coverage results
    coverage_stats = {}
    if json_report:
        json_path = output_path / "coverage.json"
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
                coverage_stats = {
                    "total_percent": data["totals"]["percent_covered"],
                    "total_lines": data["totals"]["num_statements"],
                    "covered_lines": data["totals"]["covered_lines"],
                    "missing_lines": data["totals"]["missing_lines"],
                    "excluded_lines": data["totals"]["excluded_lines"]
                }
                
                # Get per-file stats
                coverage_stats["files"] = {}
                for file_path, file_data in data["files"].items():
                    rel_path = Path(file_path).relative_to(Path.cwd())
                    coverage_stats["files"][str(rel_path)] = {
                        "percent": file_data["summary"]["percent_covered"],
                        "missing_lines": file_data["summary"]["missing_lines"],
                        "covered_lines": file_data["summary"]["covered_lines"]
                    }
    
    # Open HTML report
    if html and open_browser:
        html_index = output_path / "html" / "index.html"
        if html_index.exists():
            webbrowser.open(f"file://{html_index.absolute()}")
    
    return coverage_stats


def analyze_coverage_gaps(coverage_stats: Dict) -> Dict[str, List[str]]:
    """Analyze coverage gaps and suggest improvements."""
    
    gaps = {
        "low_coverage_files": [],
        "uncovered_modules": [],
        "suggestions": []
    }
    
    # Find files with low coverage
    if "files" in coverage_stats:
        for file_path, stats in coverage_stats["files"].items():
            if stats["percent"] < 80:  # Less than 80% coverage
                gaps["low_coverage_files"].append(
                    f"{file_path}: {stats['percent']:.1f}% ({stats['missing_lines']} lines missing)"
                )
    
    # Add suggestions based on overall coverage
    total_percent = coverage_stats.get("total_percent", 0)
    if total_percent < 60:
        gaps["suggestions"].append("Critical: Overall coverage is below 60%. Focus on core functionality tests.")
    elif total_percent < 80:
        gaps["suggestions"].append("Warning: Coverage is below 80%. Add tests for edge cases and error handling.")
    elif total_percent < 90:
        gaps["suggestions"].append("Good coverage, but aim for 90%+ by adding integration tests.")
    else:
        gaps["suggestions"].append("Excellent coverage! Focus on behavior and performance tests.")
    
    return gaps


def generate_coverage_badge(coverage_percent: float, output_path: str = "coverage_badge.svg"):
    """Generate a coverage badge SVG."""
    
    # Determine color based on coverage
    if coverage_percent >= 90:
        color = "#4c1"  # Green
    elif coverage_percent >= 80:
        color = "#dfb317"  # Yellow
    elif coverage_percent >= 60:
        color = "#fe7d37"  # Orange
    else:
        color = "#e05d44"  # Red
    
    # SVG template
    svg_template = '''<svg xmlns="http://www.w3.org/2000/svg" width="114" height="20">
    <linearGradient id="b" x2="0" y2="100%">
        <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
        <stop offset="1" stop-opacity=".1"/>
    </linearGradient>
    <mask id="a">
        <rect width="114" height="20" rx="3" fill="#fff"/>
    </mask>
    <g mask="url(#a)">
        <path fill="#555" d="M0 0h63v20H0z"/>
        <path fill="{color}" d="M63 0h51v20H63z"/>
        <path fill="url(#b)" d="M0 0h114v20H0z"/>
    </g>
    <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
        <text x="31.5" y="15" fill="#010101" fill-opacity=".3">coverage</text>
        <text x="31.5" y="14">coverage</text>
        <text x="87.5" y="15" fill="#010101" fill-opacity=".3">{percent}%</text>
        <text x="87.5" y="14">{percent}%</text>
    </g>
</svg>'''
    
    svg_content = svg_template.format(color=color, percent=f"{coverage_percent:.1f}")
    
    with open(output_path, 'w') as f:
        f.write(svg_content)
    
    print(f"Coverage badge saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate test coverage reports")
    parser.add_argument(
        "--suite",
        nargs="+",
        default=["all"],
        choices=["all", "unit", "integration", "behavior", "performance"],
        help="Test suites to include in coverage"
    )
    parser.add_argument(
        "--output-dir",
        default="coverage_report",
        help="Output directory for coverage reports"
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Skip HTML report generation"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser after generating HTML report"
    )
    parser.add_argument(
        "--badge",
        action="store_true",
        help="Generate coverage badge"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze coverage gaps and provide suggestions"
    )
    parser.add_argument(
        "--minimum",
        type=float,
        default=80.0,
        help="Minimum coverage percentage required (default: 80.0)"
    )
    
    args = parser.parse_args()
    
    # Determine test paths
    test_paths = []
    test_base = Path("src/blackholio_agent/tests")
    
    if "all" in args.suite:
        test_paths = [str(test_base)]
    else:
        suite_map = {
            "unit": test_base / "unit",
            "integration": test_base / "integration",
            "behavior": test_base / "behavior",
            "performance": test_base / "performance"
        }
        for suite in args.suite:
            if suite in suite_map:
                test_paths.append(str(suite_map[suite]))
    
    # Generate coverage report
    print(f"Generating coverage report for: {', '.join(args.suite)}")
    coverage_stats = generate_coverage_report(
        test_paths,
        output_dir=args.output_dir,
        html=not args.no_html,
        xml=True,
        json_report=True,
        open_browser=not args.no_browser
    )
    
    if not coverage_stats:
        print("Failed to generate coverage report")
        return 1
    
    # Print summary
    print("\n" + "="*60)
    print("COVERAGE SUMMARY")
    print("="*60)
    print(f"Total Coverage: {coverage_stats['total_percent']:.1f}%")
    print(f"Lines Covered: {coverage_stats['covered_lines']}/{coverage_stats['total_lines']}")
    print(f"Lines Missing: {coverage_stats['missing_lines']}")
    
    # Analyze gaps if requested
    if args.analyze:
        print("\n" + "="*60)
        print("COVERAGE ANALYSIS")
        print("="*60)
        gaps = analyze_coverage_gaps(coverage_stats)
        
        if gaps["low_coverage_files"]:
            print("\nFiles with low coverage:")
            for file_info in gaps["low_coverage_files"][:10]:  # Show top 10
                print(f"  - {file_info}")
        
        if gaps["suggestions"]:
            print("\nSuggestions:")
            for suggestion in gaps["suggestions"]:
                print(f"  • {suggestion}")
    
    # Generate badge if requested
    if args.badge:
        badge_path = Path(args.output_dir) / "coverage_badge.svg"
        generate_coverage_badge(coverage_stats['total_percent'], str(badge_path))
    
    # Check minimum coverage
    if coverage_stats['total_percent'] < args.minimum:
        print(f"\n❌ Coverage {coverage_stats['total_percent']:.1f}% is below minimum {args.minimum}%")
        return 1
    else:
        print(f"\n✅ Coverage {coverage_stats['total_percent']:.1f}% meets minimum {args.minimum}%")
        return 0


if __name__ == "__main__":
    sys.exit(main())
