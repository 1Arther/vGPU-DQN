import os
import textwrap
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

colors = {
    'DQN': '#d62728',
    'Random': '#8c8c8c',
    'Binpack(mem-desc)': '#7f7f7f',
    'Spread(mem-asc)': '#1f77b4',
    'Index-desc': '#2ca02c',
    'Greedy-balance': '#9467bd',
    'Greedy-objective': '#ff7f0e',
}
methods = ['DQN','Random','Binpack(mem-desc)','Spread(mem-asc)','Index-desc','Greedy-balance','Greedy-objective']



def set_tight_ylim(ax, values, pad_ratio=0.10, include_zero=False):
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return
    ymin = float(arr.min())
    ymax = float(arr.max())
    if include_zero:
        ymin = min(ymin, 0.0)
        ymax = max(ymax, 0.0)
    span = ymax - ymin
    if span <= 1e-9:
        pad = max(abs(ymax) * pad_ratio, 0.05)
    else:
        pad = span * pad_ratio
    ax.set_ylim(ymin - pad, ymax + pad)

def save(fig, name):
    fig.savefig(os.path.join(FIG, f'{name}.png'))
    fig.savefig(os.path.join(FIG, f'{name}.pdf'))
    plt.close(fig)

# 1. Overall baseline comparison.
overall = pd.read_csv(os.path.join(BASE, 'baseline_overall_summary.csv'))
overall = overall.set_index('method').loc[methods].reset_index()
for metric, ylabel, title in [
    ('objective', 'Objective (higher is better)', 'Overall Objective'),
    ('success', 'Success rate (higher is better)', 'Overall Allocation Success'),
    ('balance', 'Balance score (lower is better)', 'Overall Balance Score'),
]:
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    bar_colors = [colors[m] for m in overall['method']]
    ax.bar(range(len(overall)), overall[metric], color=bar_colors, alpha=0.9)
    ax.set_xticks(range(len(overall)))
    ax.set_xticklabels([textwrap.fill(m, 12) for m in overall['method']], rotation=0)
    ax.set_ylabel(ylabel)
    set_tight_ylim(ax, overall[metric], pad_ratio=0.14)
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.25)
    for i, v in enumerate(overall[metric]):
        ax.text(i, v, f'{v:.3f}', ha='center', va='bottom' if v >= 0 else 'top', fontsize=8)
    save(fig, f'overall_{metric}')

# Combined overall metrics grouped bar.
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
for ax, metric, title in zip(axes, ['objective','success','balance'], ['Objective','Success rate','Balance score']):
    ax.bar(range(len(overall)), overall[metric], color=[colors[m] for m in overall['method']])
    set_tight_ylim(ax, overall[metric], pad_ratio=0.14)
    ax.set_title(title)
    ax.set_xticks(range(len(overall)))
    ax.set_xticklabels([textwrap.fill(m, 10) for m in overall['method']], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.25)
fig.tight_layout()
save(fig, 'overall_baseline_grouped')

# 2. Per-load line plots.
for metric_file, ylabel, name in [
    ('per_load_objective.csv', 'Objective (higher is better)', 'per_load_objective'),
    ('per_load_success_rate.csv', 'Success rate (higher is better)', 'per_load_success'),
    ('per_load_balance_score.csv', 'Balance score (lower is better)', 'per_load_balance'),
    ('per_load_inter_gpu_balance_score.csv', 'Inter-GPU balance (lower is better)', 'per_load_inter_balance'),
    ('per_load_intra_gpu_balance_score.csv', 'Intra-GPU balance (lower is better)', 'per_load_intra_balance'),
]:
    df = pd.read_csv(os.path.join(BASE, metric_file))
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x = df['load_bucket']
    for m in methods:
        if m not in df.columns:
            continue
        ax.plot(x, df[m], marker='o', linewidth=2.6 if m == 'DQN' else 1.4,
                color=colors[m], label=m, alpha=1.0 if m == 'DQN' else 0.78)
    set_tight_ylim(ax, df[[m for m in methods if m in df.columns]].to_numpy().reshape(-1), pad_ratio=0.10)
    ax.set_xlabel('Actual load bucket')
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel.split('(')[0].strip() + ' across Load Levels')
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    save(fig, name)

# 3. DQN margin against best baseline.
margin = pd.read_csv(os.path.join(BASE, 'per_load_vs_best_baseline.csv'))
fig, ax = plt.subplots(figsize=(8.2, 4.2))
bar_colors = ['#d62728' if v >= 0 else '#4b5563' for v in margin['objective_margin']]
ax.axhline(0, color='black', linewidth=0.8)
ax.bar(margin['load'].astype(str), margin['objective_margin'], color=bar_colors)
set_tight_ylim(ax, margin['objective_margin'], pad_ratio=0.18, include_zero=True)
ax.set_xlabel('Actual load bucket')
ax.set_ylabel('Objective margin vs. best baseline')
ax.set_title('DQN Objective Margin by Load')
ax.grid(axis='y', alpha=0.25)
for i, v in enumerate(margin['objective_margin']):
    ax.text(i, v, f'{v:+.3f}', ha='center', va='bottom' if v >= 0 else 'top', fontsize=8)
save(fig, 'per_load_objective_margin')

# 4. Ablation.
ab = pd.read_csv(os.path.join(BASE, 'ablation_summary.csv'))
order_ab = ['base_totaldelta_nojob','splitdelta_nojob','jobfeatures','multiscore','final_trainrerank']
ab = ab.set_index('variant').loc[order_ab].reset_index()
label_ab = ['Base\n(total delta)', 'Split\ndelta', 'Job\nfeatures', 'Multi-score\nckpt', 'Final\ntrain rerank']
for metric, ylabel, name in [
    ('objective', 'Overall objective', 'ablation_objective'),
    ('lowmid_objective', 'Low/mid-load objective', 'ablation_lowmid_objective'),
    ('conflict_objective', 'Conflict-job objective', 'ablation_conflict_objective'),
    ('conflict_success', 'Conflict-job success rate', 'ablation_conflict_success'),
]:
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(range(len(ab)), ab[metric], marker='o', linewidth=2.2, color='#d62728')
    set_tight_ylim(ax, ab[metric], pad_ratio=0.14)
    ax.set_xticks(range(len(ab)))
    ax.set_xticklabels(label_ab)
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + ' in Ablation Study')
    ax.grid(True, alpha=0.25)
    for i, v in enumerate(ab[metric]):
        ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    save(fig, name)

# 5. Multi-seed mean/std.
ms = pd.read_csv(os.path.join(BASE, 'multiseed_final_summary.csv'))
metrics_plot = ['success','objective','balance','lowmid_objective','conflict_success','conflict_objective']
sub = ms[ms['metric'].isin(metrics_plot)].set_index('metric').loc[metrics_plot].reset_index()
pretty = ['Success','Objective','Balance','Low/mid obj.','Conflict success','Conflict obj.']
fig, ax = plt.subplots(figsize=(8.4, 4.2))
ax.bar(range(len(sub)), sub['mean'], yerr=sub['std'], capsize=4, color='#d62728', alpha=0.88)
set_tight_ylim(ax, sub['mean'] + sub['std'], pad_ratio=0.14)
ax.set_xticks(range(len(sub)))
ax.set_xticklabels(pretty, rotation=20, ha='right')
ax.set_ylabel('Mean ± std over seeds')
ax.set_title('Multi-seed Stability of Final DQN')
ax.grid(axis='y', alpha=0.25)
for i, (mean, std) in enumerate(zip(sub['mean'], sub['std'])):
    ax.text(i, mean + std + 0.01, f'{mean:.3f}\n±{std:.3f}', ha='center', va='bottom', fontsize=8)
save(fig, 'multiseed_final_mean_std')

# 6. Conflict-focused comparison.
fig, ax = plt.subplots(figsize=(8.4, 4.2))
width = 0.35
x = np.arange(len(overall))
ax.bar(x - width/2, overall['conflict_objective'], width=width, color=[colors[m] for m in overall['method']], alpha=0.85, label='Objective')
ax.bar(x + width/2, overall['conflict_success'], width=width, color=[colors[m] for m in overall['method']], alpha=0.45, label='Success')
set_tight_ylim(ax, list(overall['conflict_objective']) + list(overall['conflict_success']), pad_ratio=0.12)
ax.set_xticks(x)
ax.set_xticklabels([textwrap.fill(m, 10) for m in overall['method']], rotation=45, ha='right')
ax.set_ylabel('Conflict-job metrics')
ax.set_title('Conflict-job Performance')
ax.grid(axis='y', alpha=0.25)
ax.legend()
save(fig, 'conflict_objective_success')

print('generated figures:')
for f in sorted(os.listdir(FIG)):
    print(f)
