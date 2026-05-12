"""
实验 3：负载强度控制实验结果汇总脚本。

读取不同 target_load 目录下的：

    test_comparison_summary.csv

汇总输出：

    load_experiment_summary.csv
    avg_success_rate_vs_load.png
    avg_balance_score_vs_load.png
    avg_failure_rate_vs_load.png
    avg_objective_vs_load.png
    avg_allocated_count_vs_load.png
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def collect_results(root_dir: str, loads):
    all_rows = []

    for load in loads:
        exp_dir = os.path.join(root_dir, f"load_{load}")
        summary_path = os.path.join(exp_dir, "test_comparison_summary.csv")

        if not os.path.exists(summary_path):
            print(f"[skip] not found: {summary_path}")
            continue

        df = pd.read_csv(summary_path)
        df["target_load"] = float(load)
        df["exp_dir"] = exp_dir
        all_rows.append(df)

    if not all_rows:
        raise RuntimeError("no summary csv found")

    return pd.concat(all_rows, ignore_index=True)


def plot_metric(df: pd.DataFrame, y_col: str, output_dir: str):
    if y_col not in df.columns:
        print(f"[skip] column not found: {y_col}")
        return

    plt.figure(figsize=(10, 6))

    for method, group in df.groupby("method"):
        group = group.sort_values("target_load")
        plt.plot(
            group["target_load"],
            group[y_col],
            marker="o",
            linewidth=2,
            label=method,
        )

    plt.xlabel("target_load")
    plt.ylabel(y_col)
    plt.title(f"{y_col} under different load intensity")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)

    save_path = os.path.join(output_dir, f"{y_col}_vs_load.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[saved] {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect load-control experiment results"
    )

    parser.add_argument("--root-dir", required=True)

    parser.add_argument(
        "--loads",
        type=str,
        default="0.6,0.8,1.0,1.2,1.5",
    )

    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()

    loads = [x.strip() for x in args.loads.split(",") if x.strip()]

    os.makedirs(args.output_dir, exist_ok=True)

    df = collect_results(args.root_dir, loads)

    save_path = os.path.join(args.output_dir, "load_experiment_summary.csv")
    df.to_csv(save_path, index=False, encoding="utf-8")
    print(f"[saved] {save_path}")

    plot_metric(df, "avg_success_rate", args.output_dir)
    plot_metric(df, "avg_balance_score", args.output_dir)
    plot_metric(df, "avg_failure_rate", args.output_dir)
    plot_metric(df, "avg_objective", args.output_dir)
    plot_metric(df, "avg_allocated_count", args.output_dir)


if __name__ == "__main__":
    main()