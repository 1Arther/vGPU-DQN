#!/usr/bin/env python3
"""Generate paper figures from CSV result files.

This script is the reproducible source for the figures used in the paper.
It intentionally uses short method labels for readability and keeps the
per-load figures consistent with the final evaluation load range.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = 'DQN2/paper_results'
FIG = os.path.join(BASE, 'figures')
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 180,
    'savefig.bbox': 'tight',
})

METHODS = [
    'DQN',
    'Random',
    'Binpack(mem-desc)',
    'Spread(mem-asc)',
    'Index-desc',
    'Greedy-balance',
    'Greedy-objective',
]
SHORT = {
    'DQN': 'DQN',
    'Random': 'Random',
    'Binpack(mem-desc)': 'Binpack',
    'Spread(mem-asc)': 'Spread',
    'Index-desc': 'Index',
    'Greedy-balance': 'Greedy-B',
    'Greedy-objective': 'Greedy-O',
}
COLORS = {
    'DQN': '#2563eb',
    'Random': '#9ca3af',
    'Binpack(mem-desc)': '#6b7280',
    'Spread(mem-asc)': '#059669',
    'Index-desc': '#d97706',
    'Greedy-balance': '#7c3aed',
    'Greedy-objective': '#dc2626',
}
ABLATION_ORDER = [
    'base_totaldelta_nojob',
    'splitdelta_nojob',
    'jobfeatures',
    'multiscore',
    'final_trainrerank',
]
ABLATION_LABELS = ['Base', 'Split', '+Job', '+Multi', '+Rerank']


def set_tight_ylim(ax, values, pad_ratio=0.10, include_zero=False):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return
    ymin = float(arr.min())
    ymax = float(arr.max())
    if include_zero:
        ymin = min(ymin, 0.0)
        ymax = max(ymax, 0.0)
    span = ymax - ymin
    pad = max(abs(ymax) * pad_ratio, 0.03) if span <= 1e-9 else span * pad_ratio
    ax.set_ylim(ymin - pad, ymax + pad)


def save(fig, name):
    fig.savefig(os.path.join(FIG, f'{name}.png'))
    fig.savefig(os.path.join(FIG, f'{name}.pdf'))
    plt.close(fig)


def ordered_overall():
    overall = pd.read_csv(os.path.join(BASE, 'baseline_overall_summary.csv'))
    return overall.set_index('method').loc[METHODS].reset_index()


def plot_overall(overall):
    # Main paper figure: separate higher-is-better and lower-is-better metrics.
    x = np.arange(len(overall))
    labels = [SHORT[m] for m in overall['method']]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    width = 0.34
    for off, key, name in [(-width / 2, 'success', 'Success'), (width / 2, 'objective', 'Objective')]:
        axes[0].bar(x + off, overall[key], width=width, label=name,
                    color=[COLORS[m] for m in overall['method']], alpha=0.85 if key == 'success' else 0.55)
    axes[0].set_title('Higher is better')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha='right')
    axes[0].grid(axis='y', alpha=0.25, linestyle='--')
    axes[0].legend(frameon=False)
    axes[1].bar(x, overall['balance'], width=0.52,
                color=[COLORS[m] for m in overall['method']], alpha=0.85)
    axes[1].set_title('Lower is better')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha='right')
    axes[1].grid(axis='y', alpha=0.25, linestyle='--')
    fig.tight_layout()
    save(fig, 'overall_baseline_grouped')

    # Supporting single-metric figures.
    for metric, ylabel, title in [
        ('objective', 'Objective (higher is better)', 'Overall Objective'),
        ('success', 'Success rate (higher is better)', 'Overall Allocation Success'),
        ('balance', 'Balance score (lower is better)', 'Overall Balance Score'),
    ]:
        fig, ax = plt.subplots(figsize=(7.0, 3.4))
        ax.bar(labels, overall[metric], color=[COLORS[m] for m in overall['method']], alpha=0.9)
        ax.set_ylabel(ylabel)
        set_tight_ylim(ax, overall[metric], pad_ratio=0.14)
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.25, linestyle='--')
        save(fig, f'overall_{metric}')


def plot_per_load():
    # Keep figures consistent with the final paper setting: target load 0.5-1.3.
    load_min, load_max = 0.5, 1.3
    for metric_file, ylabel, name in [
        ('per_load_objective.csv', 'Objective (higher is better)', 'per_load_objective'),
        ('per_load_success_rate.csv', 'Success rate (higher is better)', 'per_load_success'),
        ('per_load_balance_score.csv', 'Balance score (lower is better)', 'per_load_balance'),
        ('per_load_inter_gpu_balance_score.csv', 'Inter-GPU balance (lower is better)', 'per_load_inter_balance'),
        ('per_load_intra_gpu_balance_score.csv', 'Intra-GPU balance (lower is better)', 'per_load_intra_balance'),
    ]:
        df = pd.read_csv(os.path.join(BASE, metric_file))
        df = df[(df['load_bucket'] >= load_min) & (df['load_bucket'] <= load_max)].copy()
        fig, ax = plt.subplots(figsize=(7.0, 3.6))
        x = df['load_bucket']
        for method in METHODS:
            if method not in df.columns:
                continue
            ax.plot(x, df[method], marker='o', linewidth=2.4 if method == 'DQN' else 1.2,
                    color=COLORS[method], label=SHORT[method], alpha=1.0 if method == 'DQN' else 0.78)
        values = df[[m for m in METHODS if m in df.columns]].to_numpy().reshape(-1)
        set_tight_ylim(ax, values, pad_ratio=0.10)
        ax.set_xlabel('Target load')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.split('(')[0].strip() + ' across Load Levels')
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.legend(ncol=3, frameon=False)
        save(fig, name)

    margin = pd.read_csv(os.path.join(BASE, 'per_load_vs_best_baseline.csv'))
    margin = margin[(margin['load'] >= load_min) & (margin['load'] <= load_max)].copy()
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    vals = margin['objective_margin']
    bar_colors = ['#2563eb' if v >= 0 else '#dc2626' for v in vals]
    ax.axhline(0, color='black', linewidth=0.8)
    ax.bar(margin['load'].map(lambda v: f'{v:.1f}'), vals, color=bar_colors, edgecolor='#374151', linewidth=0.7)
    set_tight_ylim(ax, vals, pad_ratio=0.18, include_zero=True)
    ax.set_xlabel('Target load')
    ax.set_ylabel('Objective margin vs. best baseline')
    ax.set_title('Per-load objective margin')
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    save(fig, 'per_load_objective_margin')


def plot_ablation():
    ab = pd.read_csv(os.path.join(BASE, 'ablation_summary.csv'))
    ab = ab.set_index('variant').loc[ABLATION_ORDER].reset_index()
    for metric, ylabel, name in [
        ('objective', 'Overall objective (higher is better)', 'ablation_objective'),
        ('lowmid_objective', 'Low/mid objective (higher is better)', 'ablation_lowmid_objective'),
        ('conflict_objective', 'Conflict objective (higher is better)', 'ablation_conflict_objective'),
        ('conflict_success', 'Conflict success (higher is better)', 'ablation_conflict_success'),
    ]:
        fig, ax = plt.subplots(figsize=(5.6, 3.2))
        ax.plot(ABLATION_LABELS, ab[metric], marker='o', linewidth=2.0, color='#2563eb')
        set_tight_ylim(ax, ab[metric], pad_ratio=0.14)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis='y', alpha=0.25, linestyle='--')
        save(fig, name)


def plot_multiseed():
    ms = pd.read_csv(os.path.join(BASE, 'multiseed_final_summary.csv'))
    # Do not mix lower-is-better balance with higher-is-better metrics in this figure.
    keep = [
        ('success', 'Success'),
        ('objective', 'Objective'),
        ('lowmid_objective', 'Low/mid obj.'),
        ('conflict_success', 'Conflict success'),
        ('conflict_objective', 'Conflict obj.'),
    ]
    data = ms.set_index('metric')
    vals = [float(data.loc[k, 'mean']) for k, _ in keep]
    errs = [float(data.loc[k, 'std']) for k, _ in keep]
    labels = [name for _, name in keep]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    x = np.arange(len(labels))
    ax.bar(x, vals, yerr=errs, capsize=4, color='#60a5fa', edgecolor='#374151', linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right')
    ax.set_ylabel('Value (higher is better)')
    ax.set_title('Multi-seed summary')
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    save(fig, 'multiseed_final_mean_std')


def plot_conflict(overall):
    x = np.arange(len(overall))
    labels = [SHORT[m] for m in overall['method']]
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    width = 0.34
    ax.bar(x - width / 2, overall['conflict_success'], width=width, label='Conflict success',
           color=[COLORS[m] for m in overall['method']], alpha=0.85)
    ax.bar(x + width / 2, overall['conflict_objective'], width=width, label='Conflict objective',
           color=[COLORS[m] for m in overall['method']], alpha=0.50)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right')
    ax.set_ylabel('Value (higher is better)')
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    ax.legend(frameon=False)
    save(fig, 'conflict_objective_success')


def main():
    overall = ordered_overall()
    plot_overall(overall)
    plot_per_load()
    plot_ablation()
    plot_multiseed()
    plot_conflict(overall)
    print('Generated paper figures in', FIG)


if __name__ == '__main__':
    main()
