"""
画 hami-core vGPU 方法对比图。

读取：
    test_comparison_detail.csv
    test_comparison_summary.csv

输出：
    avg_balance_score_bar.png
    avg_success_rate_bar.png
    avg_failure_rate_bar.png
    avg_objective_bar.png
    balance_score_test_batches.png
    success_rate_test_batches.png
    objective_test_batches.png
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_bar(summary_df: pd.DataFrame, y_col: str, output_dir: str):
    if y_col not in summary_df.columns:
        print(f"[skip] column not found: {y_col}")
        return

    plt.figure(figsize=(10, 6))
    plt.bar(summary_df["method"], summary_df[y_col])
    plt.xlabel("method")
    plt.ylabel(y_col)
    plt.title(y_col)
    plt.xticks(rotation=20)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    save_path = os.path.join(output_dir, f"{y_col}_bar.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[saved] {save_path}")


def plot_batch_curve(detail_df: pd.DataFrame, y_col: str, output_dir: str):
    if y_col not in detail_df.columns:
        print(f"[skip] column not found: {y_col}")
        return

    plt.figure(figsize=(12, 6))

    for method, group in detail_df.groupby("method"):
        group = group.sort_values("batch_id")

        plt.plot(
            group["batch_id"],
            group[y_col],
            marker="o",
            linewidth=1.5,
            label=method,
        )

    plt.xlabel("test batch id")
    plt.ylabel(y_col)
    plt.title(f"{y_col} over test batches")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)

    save_path = os.path.join(output_dir, f"{y_col}_test_batches.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[saved] {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Draw vGPU method comparison figures")

    parser.add_argument("--detail-path", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    detail_df = pd.read_csv(args.detail_path)
    summary_df = pd.read_csv(args.summary_path)

    plot_bar(summary_df, "avg_balance_score", args.output_dir)
    plot_bar(summary_df, "avg_success_rate", args.output_dir)
    plot_bar(summary_df, "avg_failure_rate", args.output_dir)
    plot_bar(summary_df, "avg_objective", args.output_dir)

    plot_batch_curve(detail_df, "balance_score", args.output_dir)
    plot_batch_curve(detail_df, "success_rate", args.output_dir)
    plot_batch_curve(detail_df, "failure_rate", args.output_dir)
    plot_batch_curve(detail_df, "objective", args.output_dir)


if __name__ == "__main__":
    main()