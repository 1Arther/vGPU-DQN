import os
import sys

import grpc

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

import dqn_scheduler_pb2 as pb2
import dqn_scheduler_pb2_grpc as pb2_grpc


def build_4gpu_node():
    return [
        pb2.GPUState(
            index=i,
            uuid=f"GPU-{i}",
            total_mem=49152,
            used_mem=0,
            used_core=0,
            used_num=0,
            number=10,
            device_type="NVIDIA",
        )
        for i in range(4)
    ]


def run_predict(stub):
    req = pb2.PredictRequest(
        node_name="node-a",
        pod_namespace="default",
        pod_name="single-2vgpu",
        mem_req=8192,
        core_req=25,
        nums=2,
        gpus=build_4gpu_node(),
    )

    resp = stub.Predict(req, timeout=30.0)
    print("Predict ordered_indexes:", list(resp.ordered_indexes))
    print("Predict selected_index:", resp.selected_index)
    print("Predict fallback:", resp.fallback)
    print("Predict reason:", resp.reason)


def run_schedule_job(stub):
    req = pb2.ScheduleJobRequest(
        node_name="node-a",
        gpus=build_4gpu_node(),
        pods=[
            pb2.PodRequest(
                pod_namespace="default",
                pod_name=f"job-4pod-2vgpu-{i}",
                mem_req=8192,
                core_req=25,
                nums=2,
            )
            for i in range(4)
        ],
    )

    resp = stub.ScheduleJob(req, timeout=30.0)
    print("ScheduleJob fallback:", resp.fallback)
    print("ScheduleJob reason:", resp.reason)

    for alloc in resp.allocations:
        print(
            "allocation:",
            alloc.pod_name,
            "success=",
            alloc.success,
            "selected=",
            list(alloc.selected_indexes),
            "reason=",
            alloc.reason,
        )


def main():
    channel = grpc.insecure_channel("127.0.0.1:50051")
    stub = pb2_grpc.DQNSchedulerStub(channel)

    run_predict(stub)
    run_schedule_job(stub)


if __name__ == "__main__":
    main()
