# vGPU-DQN Paper Reproduction Guide

This repository snapshot contains the simulation code, final dataset, trained final model, evaluation outputs, paper figures, ablation results, and multi-seed results for the vGPU scheduling experiments.

## 1. Environment

The experiments were run on Linux with Python 3 and PyTorch. The training script automatically uses CUDA when available.

Recommended working directory:

```bash
cd /home/zbs/vGPU-DQN
```

Core scripts:

```text
DQN2/algorithm/vgpu_dqn_sim.py
DQN2/algorithm/generate_mixed_load_fixed_homogeneous.py
DQN2/algorithm/train_vgpu_mixed_job_hardcase.py
DQN2/algorithm/test_vgpu_mixed_homo_real.py
```

## 2. Important Paths

Final dataset:

```text
DQN2/data_mixed_load_preload_v15_lowmid_conflict/
```

Final trained model:

```text
DQN2/outputs_mixed_load_preload_v16_jobfeatures_multiscore_trainrerank/vgpu_dqn_mixed_best.pth
```

Final evaluation with all baselines, including greedy baselines:

```text
DQN2/outputs_paper_v16_final_with_greedy/
```

Paper-ready result tables and figures:

```text
DQN2/paper_results/
DQN2/paper_results/figures/
```

Ablation and multi-seed runs:

```text
DQN2/paper_runs/
```

Formal experiment section draft:

```text
DQN2/paper_results/experiment_section_formal.md
```

## 3. Dataset Generation

The final dataset has already been generated and kept in the repository snapshot. To regenerate it, run:

```bash
python3 DQN2/algorithm/generate_mixed_load_fixed_homogeneous.py \
  --output-dir DQN2/data_mixed_load_preload_v15_lowmid_conflict \
  --loads 0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3 \
  --train-loads 0.5,0.6,0.7,0.8,0.9,1.0 \
  --train-per-load 750 \
  --val-per-load 90 \
  --test-per-load 90 \
  --min-gpus 4 \
  --max-gpus 8 \
  --min-pods 8 \
  --max-pods 24 \
  --enable-existing-load \
  --existing-load-min 0.05 \
  --existing-load-max 0.35 \
  --existing-pod-count-min 0 \
  --existing-pod-count-max 5 \
  --train-conflict-ratio 0.45 \
  --eval-conflict-ratio 0.10 \
  --hard-case-splits train \
  --train-hard-repeat 30 \
  --strict-actual-load \
  --actual-load-tolerance 0.12 \
  --max-generate-attempts 1000
```

The dataset includes pre-existing GPU load and mixed-conflict jobs. The conflict jobs contain memory-heavy, core-heavy, and balanced Pods within the same Job.

## 4. Final Model Training

The final model uses:

- Job-level input features.
- Split inter/intra balance-delta reward.
- Multi-objective checkpoint score.
- Train-time lightweight action reranking.

Training command:

```bash
python3 DQN2/algorithm/train_vgpu_mixed_job_hardcase.py \
  --seed 42 \
  --train-path DQN2/data_mixed_load_preload_v15_lowmid_conflict/train_scenarios.jsonl \
  --val-path DQN2/data_mixed_load_preload_v15_lowmid_conflict/val_scenarios.jsonl \
  --output-dir DQN2/outputs_mixed_load_preload_v16_jobfeatures_multiscore_trainrerank \
  --episodes 12000 \
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
  --print-interval 100 \
  --log-save-interval 200 \
  --early-stop-patience 20 \
  --min-episodes 1800 \
  --disable-hard-cases
```

## 5. Final Evaluation

The final evaluation uses one unified model and test-time mild reranking:

```bash
python3 DQN2/algorithm/test_vgpu_mixed_homo_real.py \
  --model-path DQN2/outputs_mixed_load_preload_v16_jobfeatures_multiscore_trainrerank/vgpu_dqn_mixed_best.pth \
  --test-path DQN2/data_mixed_load_preload_v15_lowmid_conflict/test_scenarios.jsonl \
  --output-dir DQN2/outputs_paper_v16_final_with_greedy \
  --hidden-dim 512 \
  --inter-balance-weight 1.0 \
  --intra-balance-weight 1.0 \
  --delta-inter-balance-weight 2.0 \
  --delta-intra-balance-weight 3.0 \
  --action-rerank-topk 32 \
  --action-rerank-balance-weight 1.0 \
  --action-rerank-q-weight 1.0
```

This evaluation compares DQN against:

```text
Random
Binpack(mem-desc)
Spread(mem-asc)
Index-desc
Greedy-balance
Greedy-objective
```

## 6. Paper Figure Generation

Paper-ready figures are generated from the CSV tables in `DQN2/paper_results`:

```bash
python3 DQN2/paper_results/generate_paper_figures.py
```

Output figures are saved as both PNG and PDF:

```text
DQN2/paper_results/figures/
```

The script uses adaptive y-axis ranges so that differences are visible in paper plots. For margin plots, the zero line is retained.

## 7. Ablation Experiments

Ablation runs are scripted in:

```text
DQN2/paper_runs/run_ablation.sh
```

Run:

```bash
bash DQN2/paper_runs/run_ablation.sh
```

Main ablation variants:

```text
base_totaldelta_nojob
splitdelta_nojob
jobfeatures
multiscore
final_trainrerank
```

Ablation summary:

```text
DQN2/paper_results/ablation_summary.csv
```

## 8. Multi-seed Experiments

Multi-seed final-model runs are scripted in:

```text
DQN2/paper_runs/run_multiseed_final.sh
```

Run:

```bash
bash DQN2/paper_runs/run_multiseed_final.sh
```

Multi-seed summaries:

```text
DQN2/paper_results/multiseed_final_raw.csv
DQN2/paper_results/multiseed_final_summary.csv
```

Current multi-seed result:

```text
objective = 0.4114 ± 0.0196
success   = 0.7290 ± 0.0038
balance   = 0.5047 ± 0.0101
conflict_objective = 0.8127 ± 0.0355
conflict_success   = 0.8476 ± 0.0093
```

## 9. Main Paper Results

Overall comparison:

```text
DQN2/paper_results/baseline_overall_summary.csv
```

Per-load objective:

```text
DQN2/paper_results/per_load_objective.csv
```

DQN margin against the best baseline per load:

```text
DQN2/paper_results/per_load_vs_best_baseline.csv
```

Key overall result:

| Method | Success | Objective | Balance |
|---|---:|---:|---:|
| DQN | 0.7338 | 0.4405 | 0.4946 |
| Random | 0.7257 | 0.3627 | 0.5400 |
| Binpack(mem-desc) | 0.7278 | 0.3371 | 0.5741 |
| Spread(mem-asc) | 0.7222 | 0.3725 | 0.5163 |
| Index-desc | 0.7277 | 0.3455 | 0.5651 |
| Greedy-balance | 0.6973 | 0.3332 | 0.4560 |
| Greedy-objective | 0.7099 | 0.3660 | 0.4737 |

The DQN method achieves the best overall objective and success rate. Greedy-balance achieves the lowest balance score but sacrifices allocation success, resulting in a lower objective.

## 10. Notes and Limitations

This is a simulation-based study. It does not currently model:

- Cross-node scheduling.
- GPU topology.
- NUMA effects.
- Communication overhead.
- MIG partitioning.
- Large-scale real production traces.

These limitations should be stated in the paper.
