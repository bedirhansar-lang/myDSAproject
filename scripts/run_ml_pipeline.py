"""Run the complete ML pipeline and sync report figures.

Run from the repository root:

    python scripts/run_ml_pipeline.py

This script executes the ML modeling scripts in the correct order and then copies
all generated PNG figures into the folders used by ML/ML_detailed_report.ipynb.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

PIPELINE_SCRIPTS = [
    "scripts/modeling_baseline_commented.py",
    "scripts/hotel_normalization_robustness_commented.py",
    "scripts/modeling_fair_comparison_commented.py",
    "scripts/modeling_walk_forward_commented.py",
    "scripts/modeling_naive_benchmarks_commented.py",
    "scripts/sync_ml_report_figures.py",
]


def run_script(script_path: str) -> None:
    """Run one pipeline script and stop immediately if it fails."""
    print("=" * 80)
    print(f"Running: {script_path}")
    print("=" * 80)
    subprocess.run([sys.executable, script_path], cwd=REPO_ROOT, check=True)


def main() -> None:
    for script_path in PIPELINE_SCRIPTS:
        run_script(script_path)

    print("\nML pipeline completed successfully.")
    print("Report figures were synced into ML/Figures and ML/Naive_Benchmark.")


if __name__ == "__main__":
    main()
