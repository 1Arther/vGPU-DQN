import os
import sys

import grpc

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

import dqn_scheduler_pb2 as pb2
import dqn_scheduler_pb2_grpc as pb2_grpc


def main():
    channel = grpc.insecure_channel("127.0.0.1:50051")
    stub = pb2_grpc.DQNSchedulerStub(channel)

    req = pb2.PredictRequest(
        node_name="t3dgq",
        pod_namespace="default",
        pod_name="vgpu-test",
        mem_req=4096,
        core_req=10,
        nums=1,
        gpus=[
            pb2.GPUState(
                index=0,
                uuid="GPU-7b94f673-ce00-719b-2c94-706baa032dae",
                total_mem=24564,
                used_mem=0,
                used_core=0,
                used_num=0,
                number=10,
                device_type="NVIDIA",
            ),
            pb2.GPUState(
                index=1,
                uuid="GPU-fb6aaeaf-9ca4-3c9a-7b3b-f56714c1f7cd",
                total_mem=24564,
                used_mem=0,
                used_core=0,
                used_num=0,
                number=10,
                device_type="NVIDIA",
            ),
            pb2.GPUState(
                index=2,
                uuid="GPU-28cf7186-0bdd-be9f-92e5-3c59b4e92c1e",
                total_mem=24564,
                used_mem=8192,
                used_core=20,
                used_num=2,
                number=10,
                device_type="NVIDIA",
            ),
            pb2.GPUState(
                index=3,
                uuid="GPU-27be0024-0498-4114-e6b9-b2d0db148309",
                total_mem=24564,
                used_mem=16384,
                used_core=40,
                used_num=4,
                number=10,
                device_type="NVIDIA",
            ),
        ],
    )

    resp = stub.Predict(req, timeout=30.0)

    print("ordered_indexes:", list(resp.ordered_indexes))
    print("selected_index:", resp.selected_index)
    print("fallback:", resp.fallback)
    print("reason:", resp.reason)

    for s in resp.scores:
        print("score:", "index=", s.index, "score=", s.score, "fit=", s.fit)


if __name__ == "__main__":
    main()