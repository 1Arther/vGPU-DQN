#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_summary(path):
    df = pd.read_csv(path)

    if "target_load" not in df.columns:
        raise ValueError("summary csv must contain target_load column")

    if "method" not in df.columns:
        raise ValueError("summary csv must contain method column")

    df["target_load"] = df["target_load"].astype(float)

    return df


def filter_random_loads(df, loads):
    loads = [round(float(x), 1) for x in loads]

    filtered = df[df["target_load"].round(1).isin(loads)].copy()
    filtered["target_load"] = filtered["target_load"].round(1)

    return filtered


def filter_hard_cases(df, random_loads):
    random_loads = [round(float(x), 1) for x in random_loads]

    hard_df = df[~df["target_load"].round(1).isin(random_loads)].copy()
    hard_df["target_load"] = hard_df["target_load"].round(1)

    return hard_df


def plot_metric(df, metric, output_dir, title=None):
    if metric not in df.columns:
        print(f"[skip] no column: {metric}")
        return

    pivot = df.pivot_table(
        index="target_load",
        columns="method",
        values=metric,
        aggfunc="mean",
    ).sort_index()

    plt.figure(figsize=(8, 5))

    for method in pivot.columns:
        plt.plot(
            pivot.index,
            pivot[method],
            marker="o",
            label=method,
        )

    plt.xlabel("Target Load")
    plt.ylabel(metric)
    plt.title(title or metric)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{metric}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[saved] {save_path}")


def plot_objective_gain(df, output_dir):
    if "avg_objective" not in df.columns:
        print("[skip] no column: avg_objective")
        return

    rows = []

    for load, group in df.groupby("target_load"):
        dqn_rows = group[group["method"].str.contains("dqn", case=False, na=False)]
        baseline_rows = group[~group["method"].str.contains("dqn", case=False, na=False)]

        if dqn_rows.empty or baseline_rows.empty:
            continue

        dqn_obj = float(dqn_rows["avg_objective"].max())
        best_baseline_obj = float(baseline_rows["avg_objective"].max())

        rows.append(
            {
                "target_load": load,
                "dqn_objective": dqn_obj,
                "best_baseline_objective": best_baseline_obj,
                "gain": dqn_obj - best_baseline_obj,
            }
        )

    if not rows:
        return

    gain_df = pd.DataFrame(rows).sort_values("target_load")

    gain_csv = os.path.join(output_dir, "dqn_vs_best_baseline_gain.csv")
    gain_df.to_csv(gain_csv, index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(
        gain_df["target_load"],
        gain_df["gain"],
        marker="o",
    )
    plt.axhline(0, linestyle="--", linewidth=1)

    plt.xlabel("Target Load")
    plt.ylabel("Objective Gain")
    plt.title("DQN objective gain over best baseline")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "dqn_objective_gain.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[saved] {gain_csv}")
    print(f"[saved] {save_path}")


def export_hard_case_table(detail_path, output_dir, random_loads):
    """
    从 detail csv 里导出 hard case 表。
    这里用 scenario_id 识别 hard case。
    如果 detail 里没有 scenario_id，就退化成按 target_load 过滤。
    """

    if not detail_path or not os.path.exists(detail_path):
        print("[skip] detail csv not found, cannot export hard case table")
        return

    df = pd.read_csv(detail_path)

    if "scenario_id" in df.columns:
        hard_df = df[df["scenario_id"].astype(str).str.contains("hard", case=False, na=False)].copy()
    elif "target_load" in df.columns:
        random_loads = [round(float(x), 1) for x in random_loads]
        df["target_load"] = df["target_load"].astype(float).round(1)
        hard_df = df[~df["target_load"].isin(random_loads)].copy()
    else:
        print("[skip] no scenario_id or target_load in detail csv")
        return

    if hard_df.empty:
        print("[skip] no hard case rows found in detail csv")
        return

    keep_cols = [
        "scenario_id",
        "method",
        "target_load",
        "actual_load",
        "memory_load",
        "core_load",
        "num_gpus",
        "num_pods",
        "allocated_count",
        "failure_count",
        "allocated_vgpu_count",
        "failure_vgpu_count",
        "total_vgpu_count",
        "success_rate",
        "failure_rate",
        "vgpu_success_rate",
        "vgpu_failure_rate",
        "balance_score",
        "objective",
    ]

    keep_cols = [c for c in keep_cols if c in hard_df.columns]
    hard_df = hard_df[keep_cols].sort_values(["scenario_id", "method"])

    out_csv = os.path.join(output_dir, "hard_case_detail_table.csv")
    hard_df.to_csv(out_csv, index=False)

    # 再做一个按 scenario_id + method 的简化透视表
    pivot_metrics = [
        "success_rate",
        "vgpu_success_rate",
        "failure_count",
        "failure_vgpu_count",
        "balance_score",
        "objective",
    ]

    pivot_metrics = [m for m in pivot_metrics if m in hard_df.columns]

    for metric in pivot_metrics:
        pivot = hard_df.pivot_table(
            index="scenario_id",
            columns="method",
            values=metric,
            aggfunc="mean",
        )

        pivot_path = os.path.join(output_dir, f"hard_case_{metric}_pivot.csv")
        pivot.to_csv(pivot_path)

        print(f"[saved] {pivot_path}")

    print(f"[saved] {out_csv}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--summary",
        type=str,
        default="DQN2/outputs_mixed_load_fixed_v9_homo_real_hard_eval/mixed_load_test_summary.csv",
    )

    parser.add_argument(
        "--detail",
        type=str,
        default="DQN2/outputs_mixed_load_fixed_v9_homo_real_hard_eval/mixed_load_test_detail.csv",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="DQN2/outputs_mixed_load_fixed_v9_homo_real_hard_eval/formal_figures",
    )

    parser.add_argument(
        "--random-loads",
        type=str,
        default="0.6,0.8,1.0,1.2,1.5,1.8",
    )

    args = parser.parse_args()

    ensure_dir(args.output_dir)

    random_loads = [
        float(x.strip())
        for x in args.random_loads.split(",")
        if x.strip()
    ]

    df = load_summary(args.summary)

    random_df = filter_random_loads(df, random_loads)
    hard_summary_df = filter_hard_cases(df, random_loads)

    random_summary_path = os.path.join(args.output_dir, "random_load_summary.csv")
    hard_summary_path = os.path.join(args.output_dir, "hard_case_summary_from_load_bucket.csv")

    random_df.to_csv(random_summary_path, index=False)
    hard_summary_df.to_csv(hard_summary_path, index=False)

    print(f"[saved] {random_summary_path}")
    print(f"[saved] {hard_summary_path}")

    print(f"random rows: {len(random_df)}")
    print(f"hard/load-bucket rows: {len(hard_summary_df)}")

    metrics = [
        "avg_success_rate",
        "avg_vgpu_success_rate",
        "avg_failure_rate",
        "avg_vgpu_failure_rate",
        "avg_allocated_count",
        "avg_allocated_vgpu_count",
        "avg_failure_count",
        "avg_failure_vgpu_count",
        "avg_balance_score",
        "avg_objective",
    ]

    for metric in metrics:
        plot_metric(
            random_df,
            metric,
            args.output_dir,
            title=f"{metric} on random test loads",
        )

    plot_objective_gain(random_df, args.output_dir)

    export_hard_case_table(
        detail_path=args.detail,
        output_dir=args.output_dir,
        random_loads=random_loads,
    )


if __name__ == "__main__":
    main()