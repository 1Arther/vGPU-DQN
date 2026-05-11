import grpc
from concurrent import futures
import scheduler_pb2
import scheduler_pb2_grpc
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random


# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========== 你的模型 / Agent / get_state / update_node_resources / is_task_allocatable 等函数 ==========

class LargeBipartiteGNN(nn.Module):
    def __init__(self, node_in_dim, task_in_dim, hidden_dim=256, num_layers=2):
        super().__init__()
        self.num_layers = num_layers

        self.node_linears = nn.ModuleList()
        self.task_linears = nn.ModuleList()

        # 第一层: 输入 (node_in_dim + task_in_dim)
        self.node_linears.append(nn.Linear(node_in_dim + task_in_dim, hidden_dim))
        self.task_linears.append(nn.Linear(node_in_dim + task_in_dim, hidden_dim))

        # 后续 (num_layers-1) 层: 输入 (hidden_dim + hidden_dim) = 2 * hidden_dim
        for _ in range(num_layers - 1):
            self.node_linears.append(nn.Linear(2 * hidden_dim, hidden_dim))
            self.task_linears.append(nn.Linear(2 * hidden_dim, hidden_dim))

        self.act = nn.ReLU()
        self.out_dim = hidden_dim

    def forward(self, node_feats, task_feats, adj):
        node_emb = node_feats
        task_emb = task_feats

        for layer_idx in range(self.num_layers):
            node_deg = torch.clamp(adj.sum(dim=1, keepdim=True), min=1e-6)
            task_deg = torch.clamp(adj.sum(dim=0, keepdim=True), min=1e-6)

            node_agg = (adj.unsqueeze(-1) * task_emb.unsqueeze(0)).sum(dim=1) / node_deg
            task_agg = (adj.transpose(0, 1).unsqueeze(-1) * node_emb.unsqueeze(0)).sum(dim=1) / task_deg.transpose(0, 1)

            # 判断本层是第 0 层还是其他层
            if layer_idx == 0:
                # (N, node_in_dim) + (N, task_in_dim) => (N, node_in_dim + task_in_dim)
                node_in = torch.cat([node_emb, node_agg], dim=-1)
                task_in = torch.cat([task_emb, task_agg], dim=-1)
            else:
                # (N, hidden_dim) + (N, hidden_dim) => (N, 2*hidden_dim)
                node_in = torch.cat([node_emb, node_agg], dim=-1)
                task_in = torch.cat([task_emb, task_agg], dim=-1)

            node_out = self.node_linears[layer_idx](node_in)
            task_out = self.task_linears[layer_idx](task_in)

            node_out = self.act(node_out)
            task_out = self.act(task_out)

            node_emb = node_out
            task_emb = task_out

        return node_emb, task_emb

class LargeQValueMLP(nn.Module):
    """
    将 (node_embed, task_embed) 拼接后，通过多层MLP输出1个 Q 值。
    这里网络较深，以追求更强的拟合能力。
    """

    def __init__(self, embed_dim=256, hidden_dim=256):
        super(LargeQValueMLP, self).__init__()
        # 输入维度 = node_embed_dim + task_embed_dim = 128+128=256(若二者相同)
        in_dim = 2 * embed_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 最终输出 Q 的标量
        )

    def forward(self, node_emb, task_emb):
        """
        node_emb: (N, embed_dim)
        task_emb: (M, embed_dim)
        需要对 (N*M) 个对儿算 Q
        返回 Q_mat: (N,M)
        """
        N, edim = node_emb.size()
        M, edim_t = task_emb.size()
        # 维度检查
        assert edim == edim_t, "node/task embedding维度不一致，请检查LargeBipartiteGNN输出"

        # 扩展维度再拼接
        node_expand = node_emb.unsqueeze(1).expand(N, M, edim)
        task_expand = task_emb.unsqueeze(0).expand(N, M, edim)

        combo = torch.cat([node_expand, task_expand], dim=-1)  # (N,M, 2*edim)
        combo_flat = combo.view(N * M, 2 * edim)

        q_flat = self.mlp(combo_flat)  # (N*M, 1)
        Q_mat = q_flat.view(N, M)
        return Q_mat

class GNNBasedDQNAgent:
    def __init__(self,
                 node_in_dim=4,  # 节点特征维度
                 task_in_dim=4,  # 任务特征维度
                 gnn_hidden_dim=128,
                 gnn_num_layers=2,
                 q_hidden_dim=256,
                 lr=1e-3,
                 gamma=0.9,
                 epsilon=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.995,
                 buffer_size=2000,
                 alpha=0.6  # PER参数
                 ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # 构造 GNN 编码器
        self.gnn_encoder = LargeBipartiteGNN(
            node_in_dim=node_in_dim,
            task_in_dim=task_in_dim,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_num_layers
        ).to(device)

        # 构造 Q-value MLP
        self.q_net = LargeQValueMLP(
            embed_dim=self.gnn_encoder.out_dim,
            hidden_dim=q_hidden_dim
        ).to(device)

        # 构造 target 网络
        self.target_gnn = copy.deepcopy(self.gnn_encoder).to(device)
        self.target_q_net = copy.deepcopy(self.q_net).to(device)

    def update_target_model(self):
        self.target_gnn.load_state_dict(self.gnn_encoder.state_dict())
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def save_model(self, path):
        torch.save({
            'gnn_encoder': self.gnn_encoder.state_dict(),
            'q_net': self.q_net.state_dict()
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=device)
        self.gnn_encoder.load_state_dict(checkpoint['gnn_encoder'])
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.target_gnn.load_state_dict(checkpoint['gnn_encoder'])
        self.target_q_net.load_state_dict(checkpoint['q_net'])
        self.gnn_encoder.eval()
        self.q_net.eval()
        self.target_gnn.eval()
        self.target_q_net.eval()
        self.epsilon = 0.0  # 测试时不探索

    def act(self, graph_data, valid_mask):
        """
        graph_data: (node_feats, task_feats, adjacency)
        valid_mask: (N,M) bool, 表示该 (i,j) 是否可行
        """
        # epsilon-greedy
        if random.random() < self.epsilon:
            valid_idx = torch.nonzero(valid_mask)
            if valid_idx.size(0) > 0:
                rand_i = random.randint(0, valid_idx.size(0) - 1)
                node_idx, task_idx = valid_idx[rand_i].tolist()
            else:
                # 没有可用动作
                node_idx, task_idx = (0, 0)
            return (node_idx, task_idx)
        else:
            node_feats, task_feats, adj = graph_data
            node_feats_t = torch.tensor(node_feats, dtype=torch.float32, device=device)
            task_feats_t = torch.tensor(task_feats, dtype=torch.float32, device=device)
            adj_t = torch.tensor(adj, dtype=torch.float32, device=device)

            with torch.no_grad():
                node_emb, task_emb = self.gnn_encoder(node_feats_t, task_feats_t, adj_t)
                Q_mat = self.q_net(node_emb, task_emb)  # (N,M)
                # mask无效动作
                Q_mat[~valid_mask] = -1e9
                # 选 max
                max_flat_idx = torch.argmax(Q_mat)
                max_i = max_flat_idx // Q_mat.size(1)
                max_j = max_flat_idx % Q_mat.size(1)
                return (max_i.item(), max_j.item())

####################################
# 5) 构建二部图: Node特征 + Task特征
####################################
def build_graph(nodes_info, tasks_info, allocations):
    """
    nodes_info: [{'cpu_total','gpu_total','cpu','gpu',...},...]
    tasks_info: [{'task_id','cpu_demand','gpu_demand','allocated_node_id',...},...]
    allocations: dict {task_id -> node_idx}, 或者自行管理

    特征:
     - Node: [cpu_usage, mem_usage, gpu_usage, cpu_left, mem_left, gpu_left]
     - Task: [cpu_demand, mem_demand, gpu_demand, allocated_node_id]
             (若未分配则 -1)
    adjacency[i,j] = 1 表示 node i 可分配给 task j (资源足够 & 未分配)
    """
    N = len(nodes_info)
    M = len(tasks_info)

    node_feats = []
    for i, node in enumerate(nodes_info):
        # 计算利用率
        cpu_usage = 0.0
        if node['cpu_total'] > 0:
            cpu_usage = (node['cpu_total'] - node['cpu']) / node['cpu_total']
        mem_usage = 0.0
        if node['memory_total'] > 0:
            mem_usage = (node['memory_total'] - node['memory']) / node['memory_total']
        gpu_usage = 0.0
        if node['gpu_total'] > 0:
            gpu_usage = (node['gpu_total'] - node['gpu']) / node['gpu_total']

        node_feats.append([
            cpu_usage,
            mem_usage,
            gpu_usage,
            node['cpu'],  # cpu剩余量
            node['memory'],  # mem剩余量
            node['gpu']  # gpu剩余量
        ])
    node_feats = np.array(node_feats, dtype=np.float32)

    task_feats = []
    for j, t in enumerate(tasks_info):
        # 若已有分配则写对应的 node_id, 否则 -1
        alloc_nid = -1
        if t['task_id'] in allocations:
            alloc_nid = allocations[t['task_id']]

        task_feats.append([
            t['cpu_demand'],
            t['memory_demand'],
            t['gpu_demand'],
            float(alloc_nid)
        ])
    task_feats = np.array(task_feats, dtype=np.float32)

    # adjacency
    adj = np.zeros((N, M), dtype=np.float32)
    for i in range(N):
        node = nodes_info[i]
        for j in range(M):
            task = tasks_info[j]
            # 如果该 task_id 已经分配，就不再可行
            if task['task_id'] in allocations:
                continue
            # 资源足够则可行
            if (node['cpu'] >= task['cpu_demand'] and
                    node['memory'] >= task['memory_demand'] and
                    node['gpu'] >= task['gpu_demand']):
                adj[i, j] = 1.0

    return node_feats, task_feats, adj

def is_task_allocatable(node, task):
    return node['cpu'] >= task['cpu_demand'] and node['memory'] >= task['memory_demand'] and node['gpu'] >= task['gpu_demand']
def update_node_resources(node, task, revert=False):
    factor = -1 if revert else 1
    node['cpu'] -= factor * task['cpu_demand']
    node['memory'] -= factor * task['memory_demand']
    node['gpu'] -= factor * task['gpu_demand']

class DQNServiceServicer(scheduler_pb2_grpc.DQNServiceServicer):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        print("[*] DQNServiceServicer for batch inference initialized")

    def PredictBatch(self, request, context):
        """
        一次性调度一整批任务:
         - request.nodes: NodeInfo[]
         - request.tasks: TaskInfo[]
        返回:
         - BatchInferenceReply: 包含每个 task_id 对应的节点分配结果
        """
        # 1) 将proto中的 NodeInfo / TaskInfo 转成Python结构
        nodes_info = []
        for node in request.nodes:
            nodes_info.append({
                'node_id': node.node_id,
                'cpu': node.cpuRem,
                'memory': node.memoryRem,
                'gpu': node.gpuRem,
                'cpu_total': node.cpuRem,
                'memory_total': node.memoryRem,
                'gpu_total': node.gpuRem
            })
        tasks_info = []
        for task in request.tasks:
            tasks_info.append({
                'task_id': task.task_id,
                'cpu_demand': task.cpu_demand,
                'memory_demand': task.memory_demand,
                'gpu_demand': task.gpu_demand
            })

        # 2) 初始化分配相关
        allocations = {}
        allocated_tasks = set()
        # 这里的 node_start / task_start / get_state 等函数，需要你整合
        # 1) 使用 Build_graph 构建图
        graph_data = build_graph(nodes_info, tasks_info, allocations)

        # 反复分配直到全部任务被分配或无可分配动作
        while len(allocated_tasks) < len(tasks_info):
            node_feats, task_feats, adjacency = build_graph(nodes_info, tasks_info, allocations)
            adj_t = torch.tensor(adjacency, dtype=torch.float32, device=device)
            valid_mask = (adj_t > 0)  # (N,M) bool Tensor

            if not valid_mask.any():
                break  # 无可用动作

            action = self.agent.act((node_feats, task_feats, adjacency), valid_mask)
            node_idx, task_idx = action

            # 校验可分配
            if not valid_mask[node_idx, task_idx]:
                # 无效动作
                break
            # 分配
            the_task = tasks_info[task_idx]
            update_node_resources(nodes_info[node_idx], the_task)
            allocations[the_task['task_id']] = node_idx
            allocated_tasks.add(task_idx)

        # 3) 整理结果
        task_ids = []
        node_ids = []
        for t in tasks_info:
            t_id = t['task_id']
            task_ids.append(t_id)
            node_ids.append(allocations.get(t_id, -1))

        # 4) 返回
        reply = scheduler_pb2.BatchInferenceReply(task_ids=task_ids, node_ids=node_ids)
        return reply

def serve():
    # 创建 agent，并设置与训练时相同的超参数
    agent = GNNBasedDQNAgent(
        node_in_dim=6,  # 你训练时指定了 [cpu_usage, mem_usage, gpu_usage, cpu_left, mem_left, gpu_left]
        task_in_dim=4,  # 你训练时指定了 [cpu_demand, mem_demand, gpu_demand, allocated_node_id]
        gnn_hidden_dim=512,
        gnn_num_layers=2,
        q_hidden_dim=512,
        lr=1e-3,
        gamma=0.9,
        epsilon=0.0,  # 测试时一般关掉探索
        epsilon_min=0.0,
        epsilon_decay=1.0,  # 不再衰减
        buffer_size=2000
    )
    agent.load_model("best_model_LB0120.pth")  # 加载你的训练好模型

    # 启动 gRPC
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    scheduler_pb2_grpc.add_DQNServiceServicer_to_server(
        DQNServiceServicer(agent), server
    )
    server.add_insecure_port('[::]:50051')
    server.start()
    print("[*] gRPC server started on 50051")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
