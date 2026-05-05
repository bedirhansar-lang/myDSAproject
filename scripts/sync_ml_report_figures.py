"""Sync machine-learning plot outputs into the ML report figure folders.

Why this script exists:
The ML modeling scripts save their original outputs under model_outputs/...
The ML report reads clean, report-ready copies from:
- ML/Figures/
- ML/Naive_Benchmark/

Running this script after the ML scripts copies the generated PNG files into the
folders used by ML/ML_detailed_report.ipynb. This makes the report figures
reproducible without manually uploading or renaming images.
"""

from pathlib import Path
import shutil


REPO_ROOT = Path(__file__).resolve().parents[1]

REPORT_FIGURES_DIR = REPO_ROOT / "ML" / "Figures"
REPORT_NAIVE_DIR = REPO_ROOT / "ML" / "Naive_Benchmark"

REPORT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORT_NAIVE_DIR.mkdir(parents=True, exist_ok=True)


FIGURE_COPY_MAP = {
    # First-pass baseline ML plots
    REPO_ROOT / "model_outputs" / "actual_vs_pred_azura_deluxe.png":
        REPORT_FIGURES_DIR / "actual_vs_pred_azura_deluxe.png",
    REPO_ROOT / "model_outputs" / "actual_vs_pred_side_mare_hotel.png":
        REPORT_FIGURES_DIR / "actual_vs_pred_side_mare_hotel.png",

    # Hotel-wise normalization robustness plots
    REPO_ROOT / "model_outputs" / "hotel_normalization_robustness" / "actual_vs_pred_hotel_z_azura_deluxe.png":
        REPORT_FIGURES_DIR / "actual_vs_pred_hotel_z_azura_deluxe.png",
    REPO_ROOT / "model_outputs" / "hotel_normalization_robustness" / "actual_vs_pred_hotel_z_side_mare_hotel.png":
        REPORT_FIGURES_DIR / "actual_vs_pred_hotel_z_side_mare_hotel.png",

    # Fair same-window plots
    REPO_ROOT / "model_outputs" / "fair_same_window_comparison" / "actual_vs_pred_same_window_azura_deluxe.png":
        REPORT_FIGURES_DIR / "actual_vs_pred_same_window_azura_deluxe.png",
    REPO_ROOT / "model_outputs" / "fair_same_window_comparison" / "actual_vs_pred_same_window_side_mare_hotel.png":
        REPORT_FIGURES_DIR / "actual_vs_pred_same_window_side_mare_hotel.png",

    # Walk-forward validation plots
    REPO_ROOT / "model_outputs" / "walk_forward_validation" / "walk_forward_rmse_by_fold.png":
        REPORT_FIGURES_DIR / "walk_forward_rmse_by_fold.png",
    REPO_ROOT / "model_outputs" / "walk_forward_validation" / "walk_forward_mean_rmse_summary.png":
        REPORT_FIGURES_DIR / "walk_forward_mean_rmse_summary.png",

    # Naive benchmark plots
    REPO_ROOT / "model_outputs" / "naive_benchmark_comparison" / "same_window_rmse_with_naive_benchmarks.png":
        REPORT_NAIVE_DIR / "same_window_rmse_with_naive_benchmarks.png",
    REPO_ROOT / "model_outputs" / "naive_benchmark_comparison" / "walk_forward_rmse_with_naive_benchmarks.png":
        REPORT_NAIVE_DIR / "walk_forward_rmse_with_naive_benchmarks.png",
    REPO_ROOT / "model_outputs" / "naive_benchmark_comparison" / "walk_forward_mean_rmse_with_naive_benchmarks.png":
        REPORT_NAIVE_DIR / "walk_forward_mean_rmse_with_naive_benchmarks.png",
}


def main() -> None:
    copied = 0
    missing = []

    for source, destination in FIGURE_COPY_MAP.items():
        if source.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied += 1
            print(f"Copied: {source.relative_to(REPO_ROOT)} -> {destination.relative_to(REPO_ROOT)}")
        else:
            missing.append(source)
            print(f"Missing source: {source.relative_to(REPO_ROOT)}")

    print("\nSync complete.")
    print(f"Copied files: {copied}")
    print(f"Missing files: {len(missing)}")

    if missing:
        print("\nRun the relevant ML modeling scripts first to generate the missing plots.")


if __name__ == "__main__":
    main()
