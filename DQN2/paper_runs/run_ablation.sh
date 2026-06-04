#!/usr/bin/env bash
set -euo pipefail
BASE_TRAIN="DQN2/data_mixed_load_preload_v15_lowmid_conflict/train_scenarios.jsonl"
BASE_VAL="DQN2/data_mixed_load_preload_v15_lowmid_conflict/val_scenarios.jsonl"
BASE_TEST="DQN2/data_mixed_load_preload_v15_lowmid_conflict/test_scenarios.jsonl"
COMMON_TRAIN="--train-path $BASE_TRAIN --val-path $BASE_VAL --episodes 5000 --hidden-dim 512 --batch-size 128 --buffer-size 60000 --lr 2e-4 --gamma 0.9 --success-weight 2.0 --balance-weight 1.0 --failure-weight 2.0 --eval-interval 100 --target-update-interval 20 --print-interval 200 --log-save-interval 200 --early-stop-patience 10 --min-episodes 1200 --disable-hard-cases --seed 42"
COMMON_TEST="--test-path $BASE_TEST --hidden-dim 512 --inter-balance-weight 1.0 --intra-balance-weight 1.0 --action-rerank-topk 32 --action-rerank-balance-weight 1.0 --action-rerank-q-weight 1.0"
run_train_eval() {
  name="$1"; shift
  train_extra="$1"; shift
  test_extra="$1"; shift
  out="DQN2/paper_runs/ablation_${name}"
  eval_out="DQN2/paper_runs/ablation_${name}_eval"
  echo "=== train $name ==="
  python3 DQN2/algorithm/train_vgpu_mixed_job_hardcase.py $COMMON_TRAIN --output-dir "$out" $train_extra > "${out}.log"
  echo "=== eval $name ==="
  python3 DQN2/algorithm/test_vgpu_mixed_homo_real.py --model-path "$out/vgpu_dqn_mixed_best.pth" --output-dir "$eval_out" $COMMON_TEST $test_extra > "${eval_out}.log"
}
run_train_eval "base_totaldelta_nojob" "--disable-job-features --delta-balance-weight 2.0 --checkpoint-conflict-objective-weight 0 --checkpoint-lowmid-objective-weight 0 --checkpoint-conflict-intra-weight 0" "--disable-job-features --delta-balance-weight 2.0"
run_train_eval "splitdelta_nojob" "--disable-job-features --delta-inter-balance-weight 2.0 --delta-intra-balance-weight 3.0 --checkpoint-conflict-objective-weight 0 --checkpoint-lowmid-objective-weight 0 --checkpoint-conflict-intra-weight 0" "--disable-job-features --delta-inter-balance-weight 2.0 --delta-intra-balance-weight 3.0"
run_train_eval "jobfeatures" "--delta-inter-balance-weight 2.0 --delta-intra-balance-weight 3.0 --checkpoint-conflict-objective-weight 0 --checkpoint-lowmid-objective-weight 0 --checkpoint-conflict-intra-weight 0" "--delta-inter-balance-weight 2.0 --delta-intra-balance-weight 3.0"
run_train_eval "multiscore" "--delta-inter-balance-weight 2.0 --delta-intra-balance-weight 3.0 --checkpoint-conflict-objective-weight 0.3 --checkpoint-lowmid-objective-weight 0.2 --checkpoint-conflict-intra-weight 0.2" "--delta-inter-balance-weight 2.0 --delta-intra-balance-weight 3.0"
# final variant reuses the trained v16 model to keep ablation focused and avoid duplicated compute.
python3 DQN2/algorithm/test_vgpu_mixed_homo_real.py \
  --model-path DQN2/outputs_mixed_load_preload_v16_jobfeatures_multiscore_trainrerank/vgpu_dqn_mixed_best.pth \
  --output-dir DQN2/paper_runs/ablation_final_trainrerank_eval \
  $COMMON_TEST --delta-inter-balance-weight 2.0 --delta-intra-balance-weight 3.0 > DQN2/paper_runs/ablation_final_trainrerank_eval.log
python3 - <<'PY'
import glob, os, pandas as pd
rows=[]
labels={
 'ablation_base_totaldelta_nojob_eval':'base_totaldelta_nojob',
 'ablation_splitdelta_nojob_eval':'splitdelta_nojob',
 'ablation_jobfeatures_eval':'jobfeatures',
 'ablation_multiscore_eval':'multiscore',
 'ablation_final_trainrerank_eval':'final_trainrerank',
}
for path in sorted(glob.glob('DQN2/paper_runs/ablation_*_eval/mixed_load_test_detail.csv')):
    label=labels.get(os.path.basename(os.path.dirname(path)), os.path.basename(os.path.dirname(path)))
    df=pd.read_csv(path)
    g=df[df.method=='dqn']
    c=g[g.workload_type=='mixed_conflict']
    l=g[g.actual_load<=1.0]
    rows.append({
      'variant':label,
      'success':g.success_rate.mean(), 'objective':g.objective.mean(), 'balance':g.balance_score.mean(), 'inter':g.inter_gpu_balance_score.mean(), 'intra':g.intra_gpu_balance_score.mean(),
      'lowmid_success':l.success_rate.mean(), 'lowmid_objective':l.objective.mean(), 'lowmid_balance':l.balance_score.mean(),
      'conflict_success':c.success_rate.mean(), 'conflict_objective':c.objective.mean(), 'conflict_balance':c.balance_score.mean(), 'conflict_intra':c.intra_gpu_balance_score.mean(),
    })
out=pd.DataFrame(rows)
out.to_csv('DQN2/paper_runs/ablation_summary.csv', index=False)
print(out.round(4).to_string(index=False))
PY
