"""
Draw figures for v5 mixed-load vGPU experiment.

v5 supports:
    - one Pod requesting multiple vGPUs
    - Pod-level success_rate
    - vGPU-level success_rate
    - balance_score
    - objective
    - allocated Pod / allocated vGPU counts

Input:
    DQN2/outputs_mixed_load_fixed_v5_eval/mixed_load_test_summary.csv
    DQN2/outputs_mixed_load_fixed_v5/vgpu_mixed_training_log.csv

Output:
    DQN2/outputs_mixed_load_fixed_v5_eval/summary_figures/*.png
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = [
    "dqn",
    "volcano-vgpu-binpack",
    "volcano-vgpu-spread",
    "random",
]


SUMMARY_METRICS = [
    "avg_success_rate",
    "avg_failure_rate",
    "avg_vgpu_success_rate",
    "avg_vgpu_failure_rate",
    "avg_balance_score",
    "avg_objective",
    "avg_allocated_count",
    "avg_failure_count",
    "avg_allocated_vgpu_count",
    "avg_failure_vgpu_count",
    "avg_total_vgpu_count",
]


TRAIN_METRICS = [
    "reward",
    "objective",
    "balance_score",
    "success_rate",
    "failure_rate",
    "allocated_count",
    "failure_count",
    "loss",
    "epsilon",
    "eval_objective",
    "eval_balance_score",
    "eval_success_rate",
    "eval_failure_rate",
    "best_eval_objective",
]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_summary(summary_path: str):
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"summary csv not found: {summary_path}")

    df = pd.read_csv(summary_path)
    return df


def normalize_summary_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["target_load"] = pd.to_numeric(df["target_load"], errors="coerce")

    for col in df.columns:
        if col != "method":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["target_load", "method"])

    method_rank = {m: i for i, m in enumerate(METHOD_ORDER)}
    df["method_rank"] = df["method"].map(method_rank).fillna(999)

    df = df.sort_values(["target_load", "method_rank", "method"])
    return df


def plot_metric_line(df: pd.DataFrame, metric: str, output_dir: str):
    if metric not in df.columns:
        print(f"[skip] missing column: {metric}")
        return

    plt.figure(figsize=(10, 6))

    for method in METHOD_ORDER:
        group = df[df["method"] == method].sort_values("target_load")

        if group.empty:
            continue

        plt.plot(
            group["target_load"],
            group[metric],
            marker="o",
            linewidth=2,
            label=method,
        )

    plt.xlabel("target_load")
    plt.ylabel(metric)
    plt.title(f"{metric} vs target_load")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)

    save_path = os.path.join(output_dir, f"{metric}_vs_load.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[saved] {save_path}")


def plot_metric_grouped_bar(df: pd.DataFrame, metric: str, output_dir: str):
    if metric not in df.columns:
        print(f"[skip] missing column: {metric}")
        return

    pivot = df.pivot_table(
        index="target_load",
        columns="method",
        values=metric,
        aggfunc="mean",
    )

    methods = [m for m in METHOD_ORDER if m in pivot.columns]
    loads = list(pivot.index)

    x = np.arange(len(loads))
    width = 0.8 / max(len(methods), 1)

    plt.figure(figsize=(12, 6))

    for i, method in enumerate(methods):
        offset = (i - (len(methods) - 1) / 2) * width

        plt.bar(
            x + offset,
            pivot[method].values,
            width=width,
            label=method,
        )

    plt.xlabel("target_load")
    plt.ylabel(metric)
    plt.title(f"{metric} comparison under different load intensity")
    plt.xticks(x, [str(v) for v in loads])
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    save_path = os.path.join(output_dir, f"{metric}_grouped_bar.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[saved] {save_path}")


def plot_success_balance_pair(df: pd.DataFrame, output_dir: str):
    """
    画两个最关键指标：
        success_rate 越高越好
        balance_score 越低越好
    """
    for metric in ["avg_success_rate", "avg_vgpu_success_rate", "avg_balance_score"]:
        plot_metric_line(df, metric, output_dir)
        plot_metric_grouped_bar(df, metric, output_dir)


def plot_dqn_improvement(df: pd.DataFrame, output_dir: str):
    """
    DQN 相对最强 baseline 的提升。

    success/objective/allocated 越高越好；
    failure/balance 越低越好。
    """
    records = []

    for load, group in df.groupby("target_load"):
        dqn = group[group["method"] == "dqn"]

        if dqn.empty:
            continue

        dqn = dqn.iloc[0]
        baseline = group[group["method"] != "dqn"]

        if baseline.empty:
            continue

        row = {"target_load": load}

        if "avg_success_rate" in group.columns:
            row["success_rate_gain"] = (
                dqn["avg_success_rate"] - baseline["avg_success_rate"].max()
            )

        if "avg_vgpu_success_rate" in group.columns:
            row["vgpu_success_rate_gain"] = (
                dqn["avg_vgpu_success_rate"] - baseline["avg_vgpu_success_rate"].max()
            )

        if "avg_objective" in group.columns:
            row["objective_gain"] = (
                dqn["avg_objective"] - baseline["avg_objective"].max()
            )

        if "avg_balance_score" in group.columns:
            row["balance_score_reduction"] = (
                baseline["avg_balance_score"].min() - dqn["avg_balance_score"]
            )

        if "avg_failure_rate" in group.columns:
            row["failure_rate_reduction"] = (
                baseline["avg_failure_rate"].min() - dqn["avg_failure_rate"]
            )

        if "avg_vgpu_failure_rate" in group.columns:
            row["vgpu_failure_rate_reduction"] = (
                baseline["avg_vgpu_failure_rate"].min() - dqn["avg_vgpu_failure_rate"]
            )

        if "avg_allocated_count" in group.columns:
            row["allocated_pod_gain"] = (
                dqn["avg_allocated_count"] - baseline["avg_allocated_count"].max()
            )

        if "avg_allocated_vgpu_count" in group.columns:
            row["allocated_vgpu_gain"] = (
                dqn["avg_allocated_vgpu_count"] - baseline["avg_allocated_vgpu_count"].max()
            )

        records.append(row)

    if not records:
        return

    gain_df = pd.DataFrame(records).sort_values("target_load")

    gain_path = os.path.join(output_dir, "dqn_vs_best_baseline_gain.csv")
    gain_df.to_csv(gain_path, index=False, encoding="utf-8")
    print(f"[saved] {gain_path}")

    for metric in gain_df.columns:
        if metric == "target_load":
            continue

        plt.figure(figsize=(10, 6))

        plt.plot(
            gain_df["target_load"],
            gain_df[metric],
            marker="o",
            linewidth=2,
        )

        plt.axhline(0, linestyle="--", linewidth=1)
        plt.xlabel("target_load")
        plt.ylabel(metric)
        plt.title(f"DQN {metric} over best baseline")
        plt.grid(True, linestyle="--", alpha=0.4)

        save_path = os.path.join(output_dir, f"dqn_{metric}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"[saved] {save_path}")


def smooth_series(series: pd.Series, window: int):
    if window <= 1:
        return series

    return series.rolling(window=window, min_periods=1).mean()


def plot_training_log(train_log: str, output_dir: str, smooth_window: int):
    if not train_log or not os.path.exists(train_log):
        print(f"[skip] training log not found: {train_log}")
        return

    df = pd.read_csv(train_log)

    if "episode" not in df.columns:
        print("[skip] training log has no episode column")
        return

    train_fig_dir = os.path.join(output_dir, "training_curves")
    ensure_dir(train_fig_dir)

    for metric in TRAIN_METRICS:
        if metric not in df.columns:
            continue

        x = pd.to_numeric(df["episode"], errors="coerce")
        y = pd.to_numeric(df[metric], errors="coerce")

        valid = (~x.isna()) & (~y.isna())

        if valid.sum() == 0:
            continue

        x = x[valid]
        y = y[valid]
        y_smooth = smooth_series(y, smooth_window)

        plt.figure(figsize=(10, 6))
        plt.plot(x, y, alpha=0.35, label=f"raw {metric}")
        plt.plot(x, y_smooth, linewidth=2, label=f"smoothed {metric}")
        plt.xlabel("episode")
        plt.ylabel(metric)
        plt.title(f"{metric} training curve")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)

        save_path = os.path.join(train_fig_dir, f"{metric}_curve.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"[saved] {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Draw v5 mixed-load multi-vGPU experiment figures"
    )

    parser.add_argument(
        "--summary-path",
        type=str,
        default="DQN2/outputs_mixed_load_fixed_v5_eval/mixed_load_test_summary.csv",
    )

    parser.add_argument(
        "--train-log",
        type=str,
        default="DQN2/outputs_mixed_load_fixed_v5/vgpu_mixed_training_log.csv",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="DQN2/outputs_mixed_load_fixed_v5_eval/summary_figures",
    )

    parser.add_argument(
        "--smooth-window",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    ensure_dir(args.output_dir)

    summary_df = load_summary(args.summary_path)
    summary_df = normalize_summary_df(summary_df)

    summary_save_path = os.path.join(args.output_dir, "mixed_load_summary_for_plot.csv")
    summary_df.to_csv(summary_save_path, index=False, encoding="utf-8")
    print(f"[saved] {summary_save_path}")

    for metric in SUMMARY_METRICS:
        plot_metric_line(summary_df, metric, args.output_dir)
        plot_metric_grouped_bar(summary_df, metric, args.output_dir)

    plot_dqn_improvement(summary_df, args.output_dir)

    plot_training_log(
        train_log=args.train_log,
        output_dir=args.output_dir,
        smooth_window=args.smooth_window,
    )


if __name__ == "__main__":
    main()
