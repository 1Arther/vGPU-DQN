import math
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


class InferredMLPDQN(nn.Module):
    """
    从 checkpoint 的 Linear weight/bias 自动重建一个 MLP DQN。
    适合 v5 这种普通 MLP DQN checkpoint。
    """

    def __init__(self, linear_params: List[Tuple[torch.Tensor, torch.Tensor]]):
        super().__init__()
        self.layers = nn.ModuleList()

        for weight, bias in linear_params:
            out_dim, in_dim = weight.shape
            layer = nn.Linear(in_dim, out_dim)
            with torch.no_grad():
                layer.weight.copy_(weight.float())
                if bias is not None:
                    layer.bias.copy_(bias.float())
                else:
                    layer.bias.zero_()
            self.layers.append(layer)

        if len(self.layers) == 0:
            raise ValueError("No linear layers found in checkpoint")

        self.input_dim = self.layers[0].in_features
        self.output_dim = self.layers[-1].out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = torch.relu(x)
        return x


def _torch_load(path: str, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _is_state_dict(obj) -> bool:
    if not isinstance(obj, dict):
        return False
    return any(torch.is_tensor(v) for v in obj.values())


def _extract_state_dict_or_model(ckpt):
    if isinstance(ckpt, nn.Module):
        return None, ckpt

    if not isinstance(ckpt, dict):
        raise ValueError(f"Unsupported checkpoint type: {type(ckpt)}")

    # 常见保存格式
    for key in [
        "model_state_dict",
        "q_network_state_dict",
        "policy_net_state_dict",
        "target_net_state_dict",
        "dqn_state_dict",
        "state_dict",
        "model",
        "net",
        "q_net",
        "policy_net",
    ]:
        if key in ckpt:
            obj = ckpt[key]
            if isinstance(obj, nn.Module):
                return None, obj
            if _is_state_dict(obj):
                return obj, None

    # checkpoint 本身就是 state_dict
    if _is_state_dict(ckpt):
        return ckpt, None

    raise ValueError(
        "Cannot find model/state_dict in checkpoint. "
        f"Available keys: {list(ckpt.keys())[:30]}"
    )


def _find_bias_for_weight(weight_key: str, state_dict: Dict[str, torch.Tensor]):
    prefix = weight_key.rsplit(".", 1)[0]
    candidates = [
        prefix + ".bias",
        weight_key.replace(".weight", ".bias"),
        weight_key.replace("weight", "bias"),
    ]

    weight = state_dict[weight_key]
    out_dim = weight.shape[0]

    for key in candidates:
        if key in state_dict:
            bias = state_dict[key]
            if torch.is_tensor(bias) and bias.ndim == 1 and bias.shape[0] == out_dim:
                return bias

    return None


def _infer_linear_params(state_dict: Dict[str, torch.Tensor]):
    linear_params = []

    # PyTorch state_dict 保持插入顺序，通常就是网络层顺序
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        if value.ndim != 2:
            continue

        # 只取 Linear 的 weight
        if "weight" not in key:
            continue

        bias = _find_bias_for_weight(key, state_dict)
        linear_params.append((value.detach().cpu(), None if bias is None else bias.detach().cpu()))

    if not linear_params:
        # 有些 state_dict 的 key 不叫 weight，兜底：取所有二维矩阵
        for key, value in state_dict.items():
            if torch.is_tensor(value) and value.ndim == 2:
                linear_params.append((value.detach().cpu(), None))

    return linear_params


def load_dqn_model(model_path: str, device: str = "cpu"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"DQN checkpoint not found: {model_path}")

    ckpt = _torch_load(model_path, device)
    state_dict, model = _extract_state_dict_or_model(ckpt)

    if model is not None:
        model = model.to(device)
        model.eval()

        # 尝试推断 input/output dim
        input_dim = getattr(model, "input_dim", None)
        output_dim = getattr(model, "output_dim", None)

        return model, input_dim, output_dim

    linear_params = _infer_linear_params(state_dict)
    model = InferredMLPDQN(linear_params).to(device)
    model.eval()

    return model, model.input_dim, model.output_dim


def compute_fit(gpu: dict, mem_req: int, core_req: int) -> bool:
    total_mem = int(gpu.get("total_mem", 0))
    used_mem = int(gpu.get("used_mem", 0))
    used_core = int(gpu.get("used_core", 0))
    used_num = int(gpu.get("used_num", 0))
    number = int(gpu.get("number", 0))

    if number > 0 and used_num >= number:
        return False

    if total_mem - used_mem < int(mem_req):
        return False

    if used_core + int(core_req) > 100:
        return False

    if int(core_req) == 100 and used_num > 0:
        return False

    if used_core == 100 and int(core_req) == 0:
        return False

    return True


def build_state_vector(
    gpus: List[dict],
    mem_req: int,
    core_req: int,
    nums: int,
    input_dim: int,
    num_gpus: int = 4,
) -> np.ndarray:
    """
    把 Volcano/HAMI 的 GPU 状态转成 DQN 输入向量。

    注意：
    1. 这里是部署适配器。
    2. 如果你 v5 训练时有专门的 state encoder，可以后续把这里替换成训练时完全一致的 encoder。
    3. 当前实现会自动 pad/truncate 到 checkpoint 的 input_dim。
    """

    by_index = {int(g["index"]): g for g in gpus}

    max_mem = max([int(g.get("total_mem", 1)) for g in gpus] + [int(mem_req), 1])
    num_gpus = max(num_gpus, max(by_index.keys(), default=0) + 1)

    feats = []

    for idx in range(num_gpus):
        g = by_index.get(idx)
        if g is None:
            feats.extend([0.0] * 8)
            continue

        total_mem = max(int(g.get("total_mem", 1)), 1)
        used_mem = int(g.get("used_mem", 0))
        used_core = int(g.get("used_core", 0))
        used_num = int(g.get("used_num", 0))
        number = int(g.get("number", 10)) or 10

        free_mem = max(total_mem - used_mem, 0)
        free_core = max(100 - used_core, 0)
        fit = 1.0 if compute_fit(g, mem_req, core_req) else 0.0

        feats.extend(
            [
                used_mem / total_mem,
                free_mem / total_mem,
                used_core / 100.0,
                free_core / 100.0,
                used_num / max(number, 1),
                total_mem / max_mem,
                idx / max(num_gpus - 1, 1),
                fit,
            ]
        )

    # Pod 请求特征
    feats.extend(
        [
            int(mem_req) / max_mem,
            int(core_req) / 100.0,
            int(nums) / max(num_gpus, 1),
            float(num_gpus),
        ]
    )

    arr = np.asarray(feats, dtype=np.float32)

    if input_dim is not None and input_dim > 0:
        if arr.shape[0] < input_dim:
            arr = np.pad(arr, (0, input_dim - arr.shape[0]), mode="constant")
        elif arr.shape[0] > input_dim:
            arr = arr[:input_dim]

    return arr


def q_values_to_gpu_scores(
    q_values: torch.Tensor,
    num_gpus: int,
    action_mapping: str = "block",
) -> Dict[int, float]:
    """
    把 DQN action Q 值映射成每张 GPU 的分数。

    v5 仿真里 action space 可能不是简单的 4 个 GPU，而是：
    gpu_idx × 其他动作维度。
    所以这里支持两种映射：

    block:
        action_id // (action_dim / num_gpus) -> gpu_idx
        适合动作按 GPU 分块编码。

    mod:
        action_id % num_gpus -> gpu_idx
        适合动作按 GPU 交错编码。
    """

    if q_values.ndim > 1:
        q_values = q_values.squeeze(0)

    q_values = q_values.detach().cpu().float()
    action_dim = int(q_values.numel())

    scores = {i: -float("inf") for i in range(num_gpus)}

    if action_dim == num_gpus:
        for i in range(num_gpus):
            scores[i] = float(q_values[i].item())
        return scores

    if action_mapping == "mod":
        for action_id in range(action_dim):
            gpu_idx = action_id % num_gpus
            scores[gpu_idx] = max(scores[gpu_idx], float(q_values[action_id].item()))
        return scores

    # default: block
    block_size = int(math.ceil(action_dim / float(num_gpus)))
    for action_id in range(action_dim):
        gpu_idx = min(action_id // block_size, num_gpus - 1)
        scores[gpu_idx] = max(scores[gpu_idx], float(q_values[action_id].item()))

    return scores


def fallback_order(gpus: List[dict], mem_req: int, core_req: int, mode: str = "binpack"):
    feasible = []
    infeasible = []

    for g in gpus:
        if compute_fit(g, mem_req, core_req):
            feasible.append(g)
        else:
            infeasible.append(g)

    if mode == "spread":
        feasible.sort(
            key=lambda g: (
                int(g.get("used_num", 0)),
                int(g.get("used_mem", 0)),
                int(g.get("used_core", 0)),
                -int(g.get("index", 0)),
            )
        )
    elif mode == "original":
        feasible.sort(key=lambda g: -int(g.get("index", 0)))
    else:
        # binpack
        feasible.sort(
            key=lambda g: (
                -int(g.get("used_mem", 0)),
                -int(g.get("used_core", 0)),
                -int(g.get("used_num", 0)),
                -int(g.get("index", 0)),
            )
        )

    infeasible.sort(key=lambda g: -int(g.get("index", 0)))

    return [int(g["index"]) for g in feasible + infeasible]