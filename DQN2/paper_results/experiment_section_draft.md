# 实验与结果分析（草稿）

## 1. 实验目的

本文实验旨在验证所提出的基于 DQN 的 vGPU 调度策略在混合负载场景下的有效性。与仅考虑任务是否能够成功分配的调度策略不同，本文将调度目标建模为分配成功率与负载均衡的联合优化问题。具体而言，调度器不仅需要尽可能提高 Pod/vGPU 的分配成功率，还需要降低 GPU 间负载差异，以及降低单个 GPU 内部显存使用率与算力使用率之间的不匹配程度。

实验关注以下问题：

1. DQN 调度策略相比 random、binpack、spread、index-desc 以及 greedy 类启发式方法，是否能够获得更高的综合调度收益。
2. DQN 是否能在不同负载水平下保持稳定优势。
3. Job-level 资源形状特征、多目标 checkpoint 选择、训练期 rerank 等模块分别带来怎样的贡献。
4. 在多随机种子下，最终方法是否具有稳定性。

## 2. 仿真环境与工作负载

实验基于 vGPU 调度仿真环境进行。每个 scenario 表示一个节点上的 GPU 资源状态以及一个待调度 Job 的 Pod 批次。每张 GPU 同时具有显存容量和 core 容量，Pod 申请一个或多个 vGPU slice，每个 slice 具有显存需求和 core 需求。调度动作需要决定将某个 Pod 的 vGPU slice 放置到哪些 GPU 上。

仿真环境考虑以下约束：

- 一个 Pod 可以申请多个 vGPU。
- 同一个 Pod 的多个 vGPU slice 可以分配到同一节点内的多张 GPU 上。
- 单个 vGPU slice 必须同时满足显存和 core 资源约束。
- 本实验不考虑跨节点调度、GPU 拓扑、通信开销和 NUMA 影响。

为了提高实验场景的真实性，数据集中加入了节点已有负载，即 GPU 在调度当前 Job 之前已经存在一定显存/core 占用。已有负载包含 balanced、memory-heavy、core-heavy 和 light 等类型，用于模拟非空节点和资源碎片化现象。

工作负载覆盖 actual load 约 0.4 至 1.4 的不同压力区间。其中，训练集重点覆盖低/中负载场景，并混入一定比例的 conflict workload。Conflict workload 指同一个 Job 内同时包含高显存低 core Pod、低显存高 core Pod 和 balanced Pod。这类负载可以检验调度策略是否能够识别 Pod 之间的资源互补关系，从而改善 GPU 内部显存/core 均衡。

最终使用的数据集路径为：

```text
DQN2/data_mixed_load_preload_v15_lowmid_conflict/
```

## 3. 评价指标

本文使用以下指标评价调度效果。

### 3.1 分配成功率

Pod 分配成功率定义为成功分配的 Pod 数量占 Job 中 Pod 总数的比例：

```text
success_rate = allocated_pod_count / total_pod_count
```

同时记录 failure_rate、vGPU success rate 和 vGPU failure rate，用于描述分配失败情况。

### 3.2 GPU 间负载均衡

GPU 间负载均衡衡量不同 GPU 之间显存使用率和 core 使用率的离散程度：

```text
inter_balance = std(memory_usage_ratio_across_gpus)
              + std(core_usage_ratio_across_gpus)
```

该值越低，说明不同 GPU 之间负载越均衡。

### 3.3 GPU 内部负载均衡

GPU 内部负载均衡衡量同一 GPU 内显存使用率和 core 使用率之间的差异：

```text
intra_balance = mean(abs(memory_usage_ratio - core_usage_ratio))
```

该值越低，说明单张 GPU 内部显存和 core 使用更加匹配，资源碎片化程度更低。

### 3.4 综合目标函数

综合目标函数同时考虑分配成功率、负载均衡和失败率：

```text
objective = success_weight * success_rate
          - balance_weight * balance_score
          - failure_weight * failure_rate
```

其中：

```text
balance_score = inter_balance + intra_balance
```

在实验中，success_weight 和 failure_weight 均设置为 2.0，balance_weight 设置为 1.0。Objective 越高表示调度综合效果越好。

## 4. 对比方法

本文将 DQN 调度策略与以下 baseline 进行比较：

- Random：随机选择可行 GPU。
- Binpack(mem-desc)：优先选择已使用显存较多的 GPU，倾向于将任务集中放置。
- Spread(mem-asc)：优先选择已使用显存较少的 GPU，倾向于将任务分散放置。
- Index-desc：按照 GPU index 从高到低扫描，是系统自带的设备扫描策略。
- Greedy-balance：每一步枚举可行动作，选择分配后 balance_score 最低的动作。
- Greedy-objective：每一步枚举可行动作，综合考虑当前 objective 和后续 Pod 的可行性。

其中 greedy-balance 和 greedy-objective 是较强的启发式 baseline，用于检验 DQN 是否不仅优于简单规则，也能与一步 look-ahead 贪心方法竞争。

## 5. DQN 方法实现

DQN 的动作定义为：

```text
action = (gpu_idx, pod_idx)
```

其中 pod_idx 表示当前选择调度的 Pod，gpu_idx 表示该 Pod 的 anchor GPU。当 Pod 需要多个 vGPU slice 时，系统以 anchor GPU 为起点，补充分配剩余 GPU，并保证所有 vGPU slice 均满足显存和 core 约束。

状态输入包含三类特征：

1. GPU 状态特征：显存使用率、core 使用率、剩余显存、剩余 core、Pod 数量、显存/core gap 等。
2. Pod 状态特征：Pod 的显存需求、core 需求、vGPU 数量、是否已分配、Pod 自身显存/core gap 等。
3. Job-level 资源形状特征：当前未分配 Pod 的显存/core 需求均值、方差、gap 均值、gap 最大值、memory-heavy Pod 比例、core-heavy Pod 比例和剩余 Pod 比例。

Job-level 特征不依赖 workload_type 标签，因此真实调度时不需要提前判断 Job 是普通型还是冲突型。模型只能根据当前 Job 的资源需求形状自行学习调度策略。

Reward 采用分配成功率与负载均衡改善量的组合。与只使用总 balance delta 的方法不同，最终模型将 GPU 间均衡改善和 GPU 内均衡改善拆开：

```text
reward = success_reward
       + delta_inter_weight * inter_balance_delta
       + delta_intra_weight * intra_balance_delta
       - failure_penalty
       - balance_penalty
```

最终模型使用统一 checkpoint 选择策略，而不是针对不同 workload 切换模型。Checkpoint score 定义为：

```text
score = all_objective
      + 0.3 * conflict_objective
      + 0.2 * lowmid_objective
      - 0.2 * conflict_intra_balance
```

该策略用于在训练阶段防止模型只优化平均性能而忽略冲突型 Job，但部署时仍然只使用一个统一模型。

此外，最终模型在训练期启用轻量 action rerank。DQN 首先给出 top-k 候选动作，然后在候选动作内进行一步仿真，根据分配后的 balance 进行重新排序。训练期 rerank 参数为：

```text
topk = 8
balance_weight = 0.5
q_weight = 1.0
```

测试/部署阶段采用温和 rerank：

```text
topk = 32
balance_weight = 1.0
q_weight = 1.0
```

## 6. 总体实验结果

总体对比结果见表 1。DQN 在所有方法中取得最高 objective、最高 success rate，并显著优于 random、binpack、spread、index-desc 以及 greedy 类方法。

表 1：总体 baseline 对比（对应 `baseline_overall_summary.csv`）

| 方法 | success | objective | balance | inter | intra |
|---|---:|---:|---:|---:|---:|
| DQN | 0.7338 | 0.4405 | 0.4946 | 0.2401 | 0.2545 |
| Random | 0.7257 | 0.3627 | 0.5400 | 0.2788 | 0.2612 |
| Binpack(mem-desc) | 0.7278 | 0.3371 | 0.5741 | 0.3125 | 0.2617 |
| Spread(mem-asc) | 0.7222 | 0.3725 | 0.5163 | 0.2551 | 0.2612 |
| Index-desc | 0.7277 | 0.3455 | 0.5651 | 0.3007 | 0.2644 |
| Greedy-balance | 0.6973 | 0.3332 | 0.4560 | 0.2343 | 0.2217 |
| Greedy-objective | 0.7099 | 0.3660 | 0.4737 | 0.2458 | 0.2279 |

从结果可以看出，greedy-balance 的 balance_score 最低，但其 success rate 明显下降，导致 objective 低于 DQN。这说明单纯追求负载均衡会牺牲分配成功率。DQN 则在成功率和负载均衡之间取得了更好的折中。

对应图表：

- `figures/overall_baseline_grouped.pdf`
- `figures/overall_objective.pdf`
- `figures/overall_success.pdf`
- `figures/overall_balance.pdf`

## 7. 不同负载水平下的结果

不同 actual load 下的 objective 曲线见 `figures/per_load_objective.pdf`。从逐负载结果看，DQN 在 0.5 至 1.3 的大多数负载区间均优于最强 baseline，尤其在低/中负载区间具有稳定优势。

DQN 相比各负载下最强 baseline 的 objective margin 见 `figures/per_load_objective_margin.pdf`。在 load=0.5 至 1.3 区间，DQN 的 objective margin 大多为正。其中 load=0.5、0.6、0.8、1.0、1.1 和 1.3 的提升较明显。

需要注意的是，在极低负载 load=0.4 和极高负载 load=1.4 时，greedy 方法在局部 objective 上超过 DQN。这一现象是合理的：在极低负载下，调度空间较大，一步贪心可以较容易找到较优 balance；在极高负载下，容量约束强，局部可行性启发式可能更容易保住少数可分配任务。因此，DQN 的优势主要体现在更现实的中低负载和混合复杂场景，而不是所有极端负载点都严格最优。

对应图表：

- `figures/per_load_objective.pdf`
- `figures/per_load_success.pdf`
- `figures/per_load_balance.pdf`
- `figures/per_load_inter_balance.pdf`
- `figures/per_load_intra_balance.pdf`
- `figures/per_load_objective_margin.pdf`

## 8. 冲突型 Job 结果

Conflict workload 用于评估模型是否能够处理同一 Job 中同时存在 memory-heavy 和 core-heavy Pod 的情况。该场景下，调度策略需要识别资源互补关系，将不同资源偏斜类型的 Pod 放置到更合适的 GPU 上。

在 conflict workload 上，DQN 的表现显著优于所有 baseline：

```text
conflict_objective = 0.8398
conflict_success   = 0.8509
```

相比之下，Random、Binpack、Spread、Index-desc 的 conflict objective 分别为 0.5594、0.5991、0.5862 和 0.5963，Greedy-balance 和 Greedy-objective 分别为 0.4224 和 0.5566。说明 DQN 在冲突型 Job 上不仅保持了较高成功率，也取得了更好的综合收益。

对应图表：

- `figures/conflict_objective_success.pdf`

## 9. 消融实验

为分析各模块贡献，本文进行了消融实验。消融版本包括：

1. Base：使用总 balance delta reward，不使用 Job-level 特征。
2. Split-delta：将 inter 和 intra balance 改善量拆开，不使用 Job-level 特征。
3. Job features：加入 Job-level 资源形状特征。
4. Multi-score：使用多目标 checkpoint score。
5. Final：在上述基础上加入训练期 rerank。

消融结果见表 2。

表 2：消融实验结果（对应 `ablation_summary.csv`）

| 变体 | success | objective | balance | lowmid objective | conflict objective | conflict success |
|---|---:|---:|---:|---:|---:|---:|
| Base | 0.7326 | 0.4313 | 0.4991 | 0.7428 | 0.5808 | 0.7855 |
| Split-delta | 0.7339 | 0.4438 | 0.4919 | 0.7547 | 0.5799 | 0.7861 |
| Job features | 0.7267 | 0.4157 | 0.4912 | 0.7590 | 0.7043 | 0.8208 |
| Multi-score | 0.7296 | 0.4127 | 0.5058 | 0.7606 | 0.7689 | 0.8409 |
| Final | 0.7338 | 0.4405 | 0.4946 | 0.7782 | 0.8398 | 0.8509 |

结果表明，split-delta reward 有助于提升总体 objective 和 balance；Job-level 特征和 multi-score checkpoint 对 conflict workload 的提升较明显；训练期 rerank 进一步提升了 conflict success 和 conflict objective，使最终模型在统一策略下兼顾整体性能和冲突场景性能。

对应图表：

- `figures/ablation_objective.pdf`
- `figures/ablation_lowmid_objective.pdf`
- `figures/ablation_conflict_objective.pdf`
- `figures/ablation_conflict_success.pdf`

## 10. 多随机种子稳定性

为了验证结果稳定性，最终方法在多个随机种子下重复训练与评估。多 seed 统计结果如下：

表 3：多 seed 结果（对应 `multiseed_final_summary.csv`）

| 指标 | mean | std |
|---|---:|---:|
| success | 0.7290 | 0.0038 |
| objective | 0.4114 | 0.0196 |
| balance | 0.5047 | 0.0101 |
| inter | 0.2466 | 0.0077 |
| intra | 0.2580 | 0.0059 |
| lowmid objective | 0.7562 | 0.0160 |
| conflict success | 0.8476 | 0.0093 |
| conflict objective | 0.8127 | 0.0355 |

结果表明，最终方法在多 seed 下具有较稳定表现，尤其 success rate 和 conflict success 的标准差较小，说明模型在不同初始化和探索序列下均能学习到有效调度策略。

对应图表：

- `figures/multiseed_final_mean_std.pdf`

## 11. 实验结论

综合以上结果，本文提出的 DQN vGPU 调度策略在仿真环境中取得了较好的效果。相比 random、binpack、spread、index-desc 和 greedy 类启发式方法，DQN 在总体 objective、分配成功率以及低/中负载场景上表现更优。在 conflict workload 上，DQN 的优势更加明显，说明 Job-level 资源形状特征和多目标训练策略能够帮助模型识别同一 Job 内部 Pod 的资源互补关系。

实验也表明，单纯追求负载均衡并不一定带来更高综合收益。例如 greedy-balance 的 balance_score 最低，但由于其分配成功率明显下降，最终 objective 不如 DQN。因此，在 vGPU 调度场景中，成功率与负载均衡之间存在权衡。DQN 的优势在于能够通过学习在二者之间取得更好的折中。

## 12. 局限性

本文实验仍存在以下局限：

1. 实验基于仿真环境，尚未在真实 Kubernetes/Volcano 集群中部署验证。
2. 当前仿真不考虑跨节点调度，只考虑单节点内 GPU 选择。
3. 仿真未考虑 GPU 拓扑、NUMA、通信开销、MIG 分区等系统因素。
4. 工作负载为合成/半真实 workload，尚未使用大规模真实集群 trace。
5. 在极低负载和极高负载场景下，greedy 方法可能局部超过 DQN，说明 DQN 并非在所有极端场景都严格最优。

后续工作可以结合真实集群 trace，进一步验证模型在真实调度系统中的泛化能力，并将跨节点调度、拓扑感知和通信开销纳入状态建模与 reward 设计中。
