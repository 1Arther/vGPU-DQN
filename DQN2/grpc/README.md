# DQN vGPU Scheduler gRPC Service

This folder packages the final v16 DQN scheduler as a gRPC service.

The service exposes two APIs:

- `Predict`: backward-compatible single-Pod API. It returns an ordered GPU list and the first selected GPU.
- `ScheduleJob`: batch Job API. It receives a node GPU list and multiple Pod requests, then returns one allocation per Pod. This is the preferred API for Job-level scheduling because the v16 model uses Job-level Pod-shape features.

## Model

Default model:

```text
DQN2/outputs_mixed_load_preload_v16_jobfeatures_multiscore_trainrerank/vgpu_dqn_mixed_best.pth
```

The service uses the same v16 simulation runtime as the paper experiments:

- GPU/Pod graph features.
- Job-level resource-shape features.
- DQN top-k balance-aware rerank.
- Multi-vGPU Pod allocation by anchor GPU.

## Generate gRPC Python Files

If the proto changes, regenerate Python stubs:

```bash
python3 -m grpc_tools.protoc \
  -I DQN2/grpc/proto \
  --python_out=DQN2/grpc \
  --grpc_python_out=DQN2/grpc \
  DQN2/grpc/proto/dqn_scheduler.proto
```

Install dependencies if needed:

```bash
python3 -m pip install --user grpcio grpcio-tools protobuf
```

## Start Service

```bash
cd /home/zbs/vGPU-DQN
python3 DQN2/grpc/dqn_scheduler_server.py \
  --model DQN2/outputs_mixed_load_preload_v16_jobfeatures_multiscore_trainrerank/vgpu_dqn_mixed_best.pth \
  --host 0.0.0.0 \
  --port 50051 \
  --hidden-dim 512
```

Optional rerank parameters:

```bash
--action-rerank-topk 32
--action-rerank-balance-weight 1.0
--action-rerank-q-weight 1.0
```

## Run Client Test

In another shell:

```bash
cd /home/zbs/vGPU-DQN
python3 DQN2/grpc/test_client.py
```

The client calls both `Predict` and `ScheduleJob`.

## Formal 4-Pod x 2-vGPU Test

This test creates a simple node with 4 empty GPUs and a Job containing 4 Pods. Each Pod requests 2 vGPUs, with each vGPU slice requiring 8192 MB memory and 25 GPU core.

Run:

```bash
cd /home/zbs/vGPU-DQN
python3 DQN2/grpc/test_formal_job_allocation.py
```

Expected behavior:

- All 4 Pods are allocated successfully.
- Each Pod receives exactly 2 GPU indexes.
- Total allocated vGPU slices = 8.
- No GPU exceeds memory/core/count capacity.

Observed allocation in the current model:

```text
job-4pod-2vgpu-0 -> [0, 1]
job-4pod-2vgpu-1 -> [2, 3]
job-4pod-2vgpu-2 -> [0, 1]
job-4pod-2vgpu-3 -> [2, 3]
```

Each GPU receives two slices, i.e. 16384 MB and 50 core, which is within the 49152 MB / 100 core capacity.

## Kubernetes Deployment

Build the inference image from the repository root:

```bash
cd /home/zbs/vGPU-DQN
IMAGE=vgpu-dqn-grpc:v8 DQN2/grpc/scripts/build_grpc_image.sh
```

Deploy the gRPC service into the Volcano namespace:

```bash
kubectl apply -k DQN2/grpc/k8s
kubectl rollout status deployment/dqn-scheduler-grpc -n volcano-system
```

In-cluster endpoint for Volcano scheduler:

```text
dqn-scheduler-grpc.volcano-system.svc.cluster.local:50051
```

If the scheduler runs in the same namespace, `dqn-scheduler-grpc:50051` is also enough.
