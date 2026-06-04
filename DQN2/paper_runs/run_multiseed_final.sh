#!/usr/bin/env bash
set -euo pipefail
TRAIN="DQN2/data_mixed_load_preload_v15_lowmid_conflict/train_scenarios.jsonl"
VAL="DQN2/data_mixed_load_preload_v15_lowmid_conflict/val_scenarios.jsonl"
TEST="DQN2/data_mixed_load_preload_v15_lowmid_conflict/test_scenarios.jsonl"
for seed in 1 2 3; do
  out="DQN2/paper_runs/final_seed_${seed}"
  eval_out="DQN2/paper_runs/final_seed_${seed}_eval"
  echo "=== final seed $seed train ==="
  python3 DQN2/algorithm/train_vgpu_mixed_job_hardcase.py \
    --seed "$seed" \
    --train-path "$TRAIN" \
    --val-path "$VAL" \
    --output-dir "$out" \
    --episodes 5000 \
    --hidden-dim 512 \
    --batch-size 128 \
    --buffer-size 60000 \
    --lr 2e-4 \
    --gamma 0.9 \
    --success-weight 2.0 \
    --balance-weight 1.0 \
    --failure-weight 2.0 \
    --inter-balance-weight 1.0 \
    --intra-balance-weight 1.0 \
    --delta-inter-balance-weight 2.0 \
    --delta-intra-balance-weight 3.0 \
    --checkpoint-conflict-objective-weight 0.3 \
    --checkpoint-lowmid-objective-weight 0.2 \
    --checkpoint-conflict-intra-weight 0.2 \
    --checkpoint-lowmid-load-threshold 1.0 \
    --train-action-rerank \
    --action-rerank-topk 8 \
    --action-rerank-balance-weight 0.5 \
    --action-rerank-q-weight 1.0 \
    --eval-interval 100 \
    --target-update-interval 20 \
    --print-interval 200 \
    --log-save-interval 200 \
    --early-stop-patience 10 \
    --min-episodes 1200 \
    --disable-hard-cases > "${out}.log"
  echo "=== final seed $seed eval ==="
  python3 DQN2/algorithm/test_vgpu_mixed_homo_real.py \
    --model-path "$out/vgpu_dqn_mixed_best.pth" \
    --test-path "$TEST" \
    --output-dir "$eval_out" \
    --hidden-dim 512 \
    --inter-balance-weight 1.0 \
    --intra-balance-weight 1.0 \
    --delta-inter-balance-weight 2.0 \
    --delta-intra-balance-weight 3.0 \
    --action-rerank-topk 32 \
    --action-rerank-balance-weight 1.0 \
    --action-rerank-q-weight 1.0 > "${eval_out}.log"
done
python3 - <<'PY'
import os, pandas as pd
paths={
 'seed_42_existing':'DQN2/outputs_paper_v16_final_with_greedy/mixed_load_test_detail.csv',
 'seed_1':'DQN2/paper_runs/final_seed_1_eval/mixed_load_test_detail.csv',
 'seed_2':'DQN2/paper_runs/final_seed_2_eval/mixed_load_test_detail.csv',
 'seed_3':'DQN2/paper_runs/final_seed_3_eval/mixed_load_test_detail.csv',
}
rows=[]
for name,path in paths.items():
    df=pd.read_csv(path)
    g=df[df.method=='dqn']
    c=g[g.workload_type=='mixed_conflict']
    l=g[g.actual_load<=1.0]
    rows.append({
      'seed':name,
      'success':g.success_rate.mean(), 'objective':g.objective.mean(), 'balance':g.balance_score.mean(), 'inter':g.inter_gpu_balance_score.mean(), 'intra':g.intra_gpu_balance_score.mean(),
      'lowmid_success':l.success_rate.mean(), 'lowmid_objective':l.objective.mean(), 'lowmid_balance':l.balance_score.mean(),
      'conflict_success':c.success_rate.mean(), 'conflict_objective':c.objective.mean(), 'conflict_balance':c.balance_score.mean(), 'conflict_intra':c.intra_gpu_balance_score.mean(),
    })
raw=pd.DataFrame(rows)
raw.to_csv('DQN2/paper_runs/multiseed_final_raw.csv', index=False)
summary=[]
for col in raw.columns:
    if col == 'seed': continue
    summary.append({'metric':col, 'mean':raw[col].mean(), 'std':raw[col].std(ddof=1)})
sumdf=pd.DataFrame(summary)
sumdf.to_csv('DQN2/paper_runs/multiseed_final_summary.csv', index=False)
print('RAW')
print(raw.round(4).to_string(index=False))
print('\nSUMMARY')
print(sumdf.round(4).to_string(index=False))
PY
