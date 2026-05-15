# vGPU-DQN: GNN-DQN gRPC Service for Volcano/HAMi vGPU Scheduling

本仓库用于研究 vGPU 场景下的强化学习调度策略。当前 v6 分支主要完成了 **DQN 推理服务的 gRPC 化部署**，使 Volcano/HAMi scheduler 可以通过 gRPC 调用外部 Python GNN-DQN 模型，获得节点内 GPU 排序结果。

---

## 1. 本分支主要改动

v6 分支基于 v5 的 GNN-DQN 训练与测试代码，新增并完善了 DQN gRPC 推理服务。

核心目标是：

```text
Volcano scheduler
  -> Go gRPC client
  -> Python DQN gRPC server
  -> GNN-DQN model inference
  -> 返回 GPU ordered_indexes
  -> Volcano/HAMi 按 DQN 排序进行 vGPU 分配
2. 新增 gRPC 服务

新增目录：

DQN2/grpc/

主要文件包括：

DQN2/grpc/proto/dqn_scheduler.proto
DQN2/grpc/dqn_scheduler_server.py
DQN2/grpc/test_client.py
DQN2/grpc/dqn_scheduler_pb2.py
DQN2/grpc/dqn_scheduler_pb2_grpc.py

其中：

dqn_scheduler.proto：定义 DQN 推理服务接口；
dqn_scheduler_server.py：Python gRPC server，加载 v5 训练得到的 GNN-DQN checkpoint；
test_client.py：本地测试客户端，用于验证 gRPC 服务能否正常返回 GPU 排序；
*_pb2.py / *_pb2_grpc.py：由 proto 自动生成的 Python gRPC 文件。
3. gRPC 接口设计

服务定义：

service DQNScheduler {
  rpc Predict(PredictRequest) returns (PredictResponse);
}

请求 PredictRequest 包含：

node_name：节点名称；
pod_namespace / pod_name：当前调度 Pod；
mem_req：Pod 请求的 vGPU 显存；
core_req：Pod 请求的 vGPU core；
nums：Pod 请求的 vGPU 数量；
gpus：当前节点内所有 GPU 的状态。

每张 GPU 的状态包括：

index
uuid
total_mem
used_mem
used_core
used_num
number
device_type

响应 PredictResponse 返回：

ordered_indexes
selected_index
scores
fallback
reason

其中 ordered_indexes 是 DQN 给出的 GPU 排序，Volcano scheduler 会按该顺序尝试进行 vGPU 分配。

4. DQN 模型加载方式

v6 复用 v5 的 GNN-DQN 模型结构，而不是重新实现一个 MLP DQN。

服务端通过：

from DQN2.algorithm.vgpu_dqn_sim import create_agent

agent = create_agent(args)
agent.load(model_path)
agent.epsilon = 0.0

加载 v5 训练出的模型。

默认模型路径：

DQN2/outputs_mixed_load_fixed_v5/vgpu_dqn_mixed_best.pth

如果模型不存在，需要先重新训练：

python DQN2/algorithm/train_vgpu_mixed_from_file.py \
  --train-path DQN2/data_mixed_load_fixed_v5/train_scenarios.jsonl \
  --val-path DQN2/data_mixed_load_fixed_v5/val_scenarios.jsonl \
  --output-dir DQN2/outputs_mixed_load_fixed_v5 \
  --episodes 8000 \
  --hidden-dim 256 \
  --seed 42
5. 启动 DQN gRPC 服务

在 R5300 上执行：

cd /home/bszh/vGPU-DQN

python DQN2/grpc/dqn_scheduler_server.py \
  --model DQN2/outputs_mixed_load_fixed_v5/vgpu_dqn_mixed_best.pth \
  --host 0.0.0.0 \
  --port 50051 \
  --device cpu \
  --hidden-dim 256 \
  --fallback binpack

当前实验中，服务部署地址为：

172.16.20.32:50051

Volcano scheduler 侧通过：

deviceshare.GPUSelectPolicy: dqn
deviceshare.DQNGRPCEndpoint: 172.16.20.32:50051

调用该服务。

6. 本地 gRPC 测试

启动 server 后，另开终端执行：

cd /home/bszh/vGPU-DQN
python DQN2/grpc/test_client.py

成功输出示例：

ordered_indexes: [3, 2, 1, 0]
selected_index: 3
fallback: False
reason: dqn
score: index= 3 score= ...
score: index= 2 score= ...
score: index= 1 score= ...
score: index= 0 score= ...

其中：

fallback: False
reason: dqn

表示结果来自 DQN 模型推理，而不是 fallback 策略。

7. 与 Volcano/HAMi 的联调结果

在 Volcano/HAMi scheduler 中配置：

deviceshare.GPUSelectPolicy: dqn
deviceshare.DQNGRPCEndpoint: 172.16.20.32:50051

scheduler 日志中观察到：

DQNPolicy grpc endpoint=172.16.20.32:50051 pod=default/vgpu-test-x selected=... ordered=[...] fallback=false reason=dqn
GPUSelectPolicy=dqn reqMem=4096 reqCore=10 ordered indexes=[...]

说明：

Volcano scheduler 已经成功通过 gRPC 调用 Python DQN 服务；
DQN 服务返回了 GPU 排序结果；
fallback=false reason=dqn 表明结果来自 DQN 模型；
Volcano/HAMi 已按 DQN 返回的 GPU 顺序进入节点内 GPU 分配逻辑。
8. 当前实验结果
8.1 baseline 策略结果

在单节点 4 × RTX 4090 环境下，连续创建 8 个 vGPU Pod，每个 Pod 请求：

volcano.sh/vgpu-number: 1
volcano.sh/vgpu-memory: 4096
volcano.sh/vgpu-cores: 10

已观察到的规则策略分配结果：

Policy	GPU0	GPU1	GPU2	GPU3	行为特点
original	0	1	3	4	原始高 index 优先倾向
binpack	0	0	4	4	集中到少数 GPU
spread	2	2	2	2	均匀分散
random	4	2	1	1	随机偏斜
8.2 DQN 策略行为

DQN 策略已成功接入 Volcano scheduler，并能够根据当前 GPU 状态返回排序。

观察到的典型行为：

ordered=[3 2 1 0]

表示 DQN 当前倾向优先选择高 index GPU。

当 GPU3 不满足分配条件时，DQN 能返回：

ordered=[2 1 0 3]

说明 DQN 能识别 fit=false 的 GPU，并将不可用 GPU 放到排序后面。

9. 当前限制

当前 v6 分支仍有以下限制：

DQN gRPC 服务目前以 Python 进程方式运行，尚未容器化；
DQN 服务依赖预训练 checkpoint；
目前主要验证了 Volcano scheduler 与 DQN 服务的推理链路；
JCT、吞吐、GPU 利用率等性能指标仍需进一步系统实验；
当前 DQN 输入状态主要来自 Volcano/HAMi 调度账本，若宿主机存在非 HAMI 管理的 GPU 进程，scheduler 可能无法感知真实 GPU 负载；
当前接入点是节点内 GPU 排序，多节点联合调度仍需进一步扩展。
10. 后续计划

后续工作包括：

将 Python DQN gRPC 服务容器化；
通过 Kubernetes Service 暴露 DQN 推理服务；
增加 DQN 与 original/binpack/spread/random 的 JCT 对比；
增加真实 CUDA workload 测试；
扩展状态输入，加入任务类型、训练/推理类型、模型特征等；
探索多节点 vGPU 调度下的 DQN 策略。
11. License

本仓库用于 vGPU 调度研究实验。相关代码基于原有项目修改。
EOF


---

## 3. 检查修改内容

```bash
git status --short

find DQN2/grpc -maxdepth 3 -type f | sort

确认主要修改包括：

README.md
DQN2/grpc/...