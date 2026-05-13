"""
Draw figures for fixed mixed-load vGPU experiment.

Input:
    DQN2/outputs_mixed_load_fixed_eval/mixed_load_test_summary.csv
    DQN2/outputs_mixed_load_fixed_eval/vgpu_mixed_training_log.csv  可选

Output:
    DQN2/outputs_mixed_load_fixed_eval/summary_figures/*.png
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
    "avg_balance_score",
    "avg_objective",
    "avg_allocated_count",
    "avg_failure_count",
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


def load_summary(summary_path: str, root_dir: str, loads):
    """
    优先读取 combined summary；
    如果 combined summary 不存在，则从 eval_load_x/test_comparison_summary.csv 汇总。
    """
    if summary_path and os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        return df

    rows = []

    for load in loads:
        path = os.path.join(
            root_dir,
            f"eval_load_{load}",
            "test_comparison_summary.csv",
        )

        if not os.path.exists(path):
            print(f"[skip] not found: {path}")
            continue

        df = pd.read_csv(path)
        rows.append(df)

    if not rows:
        raise FileNotFoundError("no summary csv found")

    return pd.concat(rows, ignore_index=True)


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

    for method, group in df.groupby("method"):
        group = group.sort_values("target_load")

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


def plot_dqn_improvement(df: pd.DataFrame, output_dir: str):
    """
    画 DQN 相对最强 baseline 的提升。
    success/objective 越高越好；
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

        best_success = baseline["avg_success_rate"].max()
        best_objective = baseline["avg_objective"].max()
        best_balance = baseline["avg_balance_score"].min()
        best_failure = baseline["avg_failure_rate"].min()

        records.append(
            {
                "target_load": load,
                "success_rate_gain": dqn["avg_success_rate"] - best_success,
                "objective_gain": dqn["avg_objective"] - best_objective,
                "balance_score_reduction": best_balance - dqn["avg_balance_score"],
                "failure_rate_reduction": best_failure - dqn["avg_failure_rate"],
            }
        )

    if not records:
        return

    gain_df = pd.DataFrame(records).sort_values("target_load")
    gain_path = os.path.join(output_dir, "dqn_vs_best_baseline_gain.csv")
    gain_df.to_csv(gain_path, index=False, encoding="utf-8")
    print(f"[saved] {gain_path}")

    for metric in [
        "success_rate_gain",
        "objective_gain",
        "balance_score_reduction",
        "failure_rate_reduction",
    ]:
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

        y = pd.to_numeric(df[metric], errors="coerce")
        x = pd.to_numeric(df["episode"], errors="coerce")

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
        description="Draw fixed mixed-load vGPU experiment figures"
    )

    parser.add_argument(
        "--root-dir",
        type=str,
        default="DQN2/outputs_mixed_load_fixed_eval",
    )

    parser.add_argument(
        "--summary-path",
        type=str,
        default="DQN2/outputs_mixed_load_fixed_eval/mixed_load_test_summary.csv",
    )

    parser.add_argument(
        "--train-log",
        type=str,
        default="DQN2/outputs_mixed_load_fixed/vgpu_mixed_training_log.csv",
    )

    parser.add_argument(
        "--loads",
        type=str,
        default="0.6,0.8,1.0,1.2,1.5",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="DQN2/outputs_mixed_load_fixed_eval/summary_figures",
    )

    parser.add_argument(
        "--smooth-window",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    ensure_dir(args.output_dir)

    loads = [x.strip() for x in args.loads.split(",") if x.strip()]

    summary_df = load_summary(
        summary_path=args.summary_path,
        root_dir=args.root_dir,
        loads=loads,
    )

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
