import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '../..'))
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import dqn_scheduler_pb2 as pb2
from dqn_scheduler_server import DQNSchedulerService, grpc_gpu_to_sim, grpc_pod_to_sim


def build_4gpu_node():
    return [
        pb2.GPUState(
            index=i,
            uuid=f'GPU-{i}',
            total_mem=49152,
            used_mem=0,
            used_core=0,
            used_num=0,
            number=10,
            device_type='NVIDIA',
        )
        for i in range(4)
    ]


def build_4pod_2vgpu_job():
    return [
        pb2.PodRequest(
            pod_namespace='default',
            pod_name=f'job-4pod-2vgpu-{i}',
            mem_req=8192,
            core_req=25,
            nums=2,
        )
        for i in range(4)
    ]


def main():
    service = DQNSchedulerService(
        model_path='DQN2/outputs_mixed_load_preload_v16_jobfeatures_multiscore_trainrerank/vgpu_dqn_mixed_best.pth',
        hidden_dim=512,
        device_name='cpu',
    )
    gpus = [grpc_gpu_to_sim(g) for g in build_4gpu_node()]
    pods = [grpc_pod_to_sim(p, i) for i, p in enumerate(build_4pod_2vgpu_job())]
    rows, fallback, reason = service._schedule(gpus, pods)

    print('fallback:', fallback)
    print('reason:', reason)
    for row in rows:
        print(row)

    assert not fallback, 'DQN should not need fallback in this simple feasible case'
    assert len(rows) == 4, f'expected 4 allocation rows, got {len(rows)}'
    assert all(r['success'] for r in rows), 'all four Pods should be allocated'
    assert all(len(r['selected_indexes']) == 2 for r in rows), 'each Pod should receive 2 vGPUs'

    usage = {idx: {'mem': 0, 'core': 0, 'num': 0} for idx in range(4)}
    for row in rows:
        for idx in row['selected_indexes']:
            usage[idx]['mem'] += 8192
            usage[idx]['core'] += 25
            usage[idx]['num'] += 1

    for idx, u in usage.items():
        assert u['mem'] <= 49152, f'GPU {idx} memory over-allocated: {u}'
        assert u['core'] <= 100, f'GPU {idx} core over-allocated: {u}'
        assert u['num'] <= 10, f'GPU {idx} vGPU count over-allocated: {u}'

    assert sum(u['num'] for u in usage.values()) == 8, f'expected 8 total vGPU slices, got {usage}'
    print('formal 4pod x 2vGPU allocation test passed')


if __name__ == '__main__':
    main()
