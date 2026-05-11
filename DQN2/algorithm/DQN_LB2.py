import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from DQN.init import parse_node_info, parse_tasks_info

# 初始变量
node_start = []
task_start = []

REWARD = 0
LEARN_FREQ = 1

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha

    def __len__(self):
        return len(self.buffer)

    def add(self, error, sample):
        # 确保优先级非零
        max_priority = max(self.priorities.max() if self.buffer else 1.0, 1e-5)

        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)  # 如果缓冲区未满，直接追加
        else:
            self.buffer[self.pos] = sample  # 缓冲区满时，覆盖最旧的样本

        self.priorities[self.pos] = max(max_priority, error)  # 更新对应位置的优先级
        self.pos = (self.pos + 1) % self.capacity  # 更新写入位置指针

    def sample(self, batch_size, beta=0.4, random_sample_ratio=0.2):
        random_sample_size = int(batch_size * random_sample_ratio)
        priority_sample_size = batch_size - random_sample_size

        # 从优先级较高的样本中采样
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]

        # 避免优先级为 0 的问题
        if priorities.sum() == 0:
            probabilities = np.ones_like(priorities) / len(priorities)  # 平均分配概率
        else:
            probabilities = priorities ** self.alpha
            probabilities /= probabilities.sum()

        # 采样高优先级样本
        priority_indices = np.random.choice(len(self.buffer), priority_sample_size, p=probabilities)

        # 随机采样
        random_indices = np.random.choice(len(self.buffer), random_sample_size)

        # 合并两部分采样
        indices = np.concatenate([priority_indices, random_indices])
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_errors):
        for idx, error in zip(batch_indices, batch_errors):
            self.priorities[idx] = error


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


# ------------------------------
# 2. 更复杂的 Q值 MLP (LargeQValueMLP)
# ------------------------------
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


# ------------------------------
# 3. GNN + QValue DQN Agent
# ------------------------------
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

        # 优先经验回放
        self.memory = PrioritizedReplayBuffer(capacity=buffer_size, alpha=alpha)

        # 优化器 (同时更新 gnn + q_net)
        self.optimizer = optim.Adam(
            list(self.gnn_encoder.parameters()) + list(self.q_net.parameters()),
            lr=lr
        )

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

    def remember(self, graph_data, action, reward, next_graph_data, done):
        """
        graph_data = (node_feats, task_feats, adjacency)
        action = (node_idx, task_idx)
        """
        node_feats, task_feats, adj = graph_data
        node_feats_t = torch.tensor(node_feats, dtype=torch.float32, device=device)
        task_feats_t = torch.tensor(task_feats, dtype=torch.float32, device=device)
        adj_t = torch.tensor(adj, dtype=torch.float32, device=device)

        with torch.no_grad():
            # current Q
            node_emb, task_emb = self.gnn_encoder(node_feats_t, task_feats_t, adj_t)
            Q_mat = self.q_net(node_emb, task_emb)
            current_q = Q_mat[action[0], action[1]].item()

            # next Q
            if not done:
                n_node_feats, n_task_feats, n_adj = next_graph_data
                n_node_feats_t = torch.tensor(n_node_feats, dtype=torch.float32, device=device)
                n_task_feats_t = torch.tensor(n_task_feats, dtype=torch.float32, device=device)
                n_adj_t = torch.tensor(n_adj, dtype=torch.float32, device=device)

                n_node_emb, n_task_emb = self.target_gnn(n_node_feats_t, n_task_feats_t, n_adj_t)
                Q_mat_next = self.target_q_net(n_node_emb, n_task_emb)
                max_next_q = Q_mat_next.max().item()
            else:
                max_next_q = 0.0

            target_q = reward + self.gamma * max_next_q
            td_error = abs(current_q - target_q)

        # 存
        sample = (graph_data, action, reward, next_graph_data, done)
        self.memory.add(td_error, sample)

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

    def replay(self, batch_size, beta=0.4):
        if len(self.memory) < batch_size:
            return 0.0

        samples, indices, weights = self.memory.sample(batch_size, beta=beta)
        weights_t = torch.tensor(weights, dtype=torch.float32, device=device).unsqueeze(1)

        losses = []
        self.optimizer.zero_grad()

        for i, (graph_data, action, reward, next_graph_data, done) in enumerate(samples):
            node_feats, task_feats, adj = graph_data
            node_feats_t = torch.tensor(node_feats, dtype=torch.float32, device=device)
            task_feats_t = torch.tensor(task_feats, dtype=torch.float32, device=device)
            adj_t = torch.tensor(adj, dtype=torch.float32, device=device)

            node_emb, task_emb = self.gnn_encoder(node_feats_t, task_feats_t, adj_t)
            Q_mat = self.q_net(node_emb, task_emb)
            current_q = Q_mat[action[0], action[1]]

            if done:
                target_q = torch.tensor(reward, dtype=torch.float32, device=device)
            else:
                n_node_feats, n_task_feats, n_adj = next_graph_data
                n_node_feats_t = torch.tensor(n_node_feats, dtype=torch.float32, device=device)
                n_task_feats_t = torch.tensor(n_task_feats, dtype=torch.float32, device=device)
                n_adj_t = torch.tensor(n_adj, dtype=torch.float32, device=device)

                with torch.no_grad():
                    node_emb_next, task_emb_next = self.target_gnn(n_node_feats_t, n_task_feats_t, n_adj_t)
                    Q_mat_next = self.target_q_net(node_emb_next, task_emb_next)
                    max_next_q = Q_mat_next.max()
                target_q = reward + self.gamma * max_next_q

            td_error = current_q - target_q
            loss = (td_error ** 2) * weights_t[i]
            losses.append(loss)

        total_loss = torch.mean(torch.stack(losses))
        total_loss.backward()
        nn.utils.clip_grad_norm_(list(self.gnn_encoder.parameters()) + list(self.q_net.parameters()), max_norm=1.0)
        self.optimizer.step()

        # 更新优先级
        td_errors = [abs(l.item()) for l in losses]
        self.memory.update_priorities(indices, td_errors)

        return total_loss.item()

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min


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

def calculate_cost(nodes_info, tasks_info, allocations):
    # 计算负载均衡
    cpu_usages = []
    memory_usages = []
    gpu_usages = []
    resource_balance = 0.0
    for i in range(len(nodes_info)):
        node = nodes_info[i]
        # n = [item for item in node_start if item['node_id'] == node['node_id']]  # 查找匹配的节点信息
        n = next((item for item in node_start if item['node_id'] == node['node_id']), None)

        cpu_usage = (n['cpu'] - node['cpu']) / n['cpu'] if n['cpu'] > 0 else 0
        memory_usage = (n['memory'] - node['memory']) / n['memory'] if n['memory'] > 0 else 0
        gpu_usage = (n['gpu'] - node['gpu']) / n['gpu'] if n['gpu'] > 0 else 0
        resources = []
        cpu_usages.append(cpu_usage)
        memory_usages.append(memory_usage)
        gpu_usages.append(gpu_usage)
        resources.append(cpu_usage)
        resources.append(memory_usage)
        # 增加加权标准差
        resource_balance += np.std(resources) if resources else 0

    # 负载均衡计算：标准差越小，负载越均衡
    load_balance = (np.std(np.array(cpu_usages)) + np.std(np.array(memory_usages)) + np.std(np.array(gpu_usages))) / 3
    resource_balance /= len(nodes_info)

    # 负载均衡作为奖励的负值，并加入加权标准差的惩罚
    total_cost = -50 * load_balance - 50 * resource_balance

    return total_cost

def update_node_resources(node, task, revert=False):
    factor = -1 if revert else 1
    node['cpu'] -= factor * task['cpu_demand']
    node['memory'] -= factor * task['memory_demand']
    node['gpu'] -= factor * task['gpu_demand']


def is_task_allocatable(node, task):
    return node['cpu'] >= task['cpu_demand'] and node['memory'] >= task['memory_demand'] and node['gpu'] >= task['gpu_demand']

def get_task_batches(tasks_info):
    """根据任务的Arrival Time属性将任务分批"""
    batches = {}
    for task in tasks_info:
        arrival_time = task['arrival_time']
        if arrival_time not in batches:
            batches[arrival_time] = []
        batches[arrival_time].append(task)
    return [batches[t] for t in sorted(batches.keys())]

def can_schedule_tasks(nodes, tasks):
    """检查给定任务集合是否能在当前节点配置下调度"""
    total_cpu = sum(task['cpu_demand'] for task in tasks)
    total_memory = sum(task['memory_demand'] for task in tasks)
    total_gpu = sum(task['gpu_demand'] for task in tasks)

    # 计算集群总资源
    available_cpu = sum(node['cpu_total'] for node in nodes)
    available_memory = sum(node['memory_total'] for node in nodes)
    available_gpu = sum(node['gpu_total'] for node in nodes)

    # 如果任务总资源需求超过集群资源，就无法调度
    if total_cpu <= available_cpu and total_memory <= available_memory and total_gpu <= available_gpu:
        return True
    return False

def get_feasible_task_batch(tasks_info, nodes_info, min_tasks=3, max_tasks=10):
    """从任务列表中随机选取一定数量的任务，并确保其可调度"""
    while True:
        # 随机选择任务的数量
        num_tasks = random.randint(min_tasks, max_tasks)
        # 随机从任务池中选取num_tasks个任务
        selected_tasks = random.sample(tasks_info, num_tasks)

        # 检查这些任务是否能在当前节点配置下调度
        if can_schedule_tasks(nodes_info, selected_tasks):
            return selected_tasks

if __name__ == "__main__":
    nodes_info = parse_node_info('../data/nodes_info.txt')
    tasks_info = parse_tasks_info('../data/100Job/tasks_info.txt')

    # 给节点增加 cpu_total, memory_total
    for node in nodes_info:
        node['cpu_total'] = node['cpu']
        node['memory_total'] = node['memory']
        node['gpu_total'] = node['gpu']

    task_dict = {task['task_id']: task for task in tasks_info}
    node_start = copy.deepcopy(nodes_info)
    task_start = copy.deepcopy(tasks_info)
    print(tasks_info)
    batch_tasks = get_task_batches(tasks_info)
    print(batch_tasks)

    # 3) 初始化GNN-DQN代理
    agent = GNNBasedDQNAgent(
        node_in_dim=6,  # 对应 build_graph 里 node特征[cpu利用率、内存利用率、gpu利用率、cpu剩余量、内存剩余量、gpu剩余量]
        task_in_dim=4,  # 对应 task特征 = [cpu_demand, memory_demand, gpu_demand, alloc_id]
        gnn_hidden_dim=512,
        gnn_num_layers=2,
        q_hidden_dim=512,
        lr=1e-3,
        gamma=0.9,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        buffer_size=2000
    )

    model_path = 'dqn_model_LB.pth'
    f = False
    if f:
        try:
            agent.load_model(model_path)
            print("Model loaded successfully.")
        except FileNotFoundError:
            print("No pre-trained model found. Starting training from scratch.")

    if not f:
        with open('../data/100Job/output_LB_0120.txt', 'w') as file:  # 在训练开始时打开文件，并保持打开状态
            episodes = 5000
            batch_size = 32
            step = 0
            best_reward = -float('inf')
            for e in range(episodes):
                episode_reward = 0
                episode_loss = 0
                # # 重置节点
                # num_nodes_this_round = random.randint(3, 10)
                # # 从 node_start (所有节点) 中随机挑选 num_nodes_this_round 个节点
                # original_nodes = copy.deepcopy(
                #     random.sample(node_start, num_nodes_this_round)
                # )
                # # original_nodes = copy.deepcopy(nodes_info)

                for batch in batch_tasks:
                    flag = True
                    allocations = {}
                    allocated_tasks = set()  # 记录已分配的任务
                    # 重置节点
                    num_nodes_this_round = random.randint(3, 10)
                    # 从 node_start (所有节点) 中随机挑选 num_nodes_this_round 个节点
                    original_nodes = copy.deepcopy(
                        random.sample(node_start, num_nodes_this_round)
                    )
                    # original_nodes = copy.deepcopy(nodes_info)
                    batch = get_feasible_task_batch(batch, original_nodes, 3, min(10,len(batch)))
                    # print(f"节点数量：{len(original_nodes)} 任务数量：{len(batch)}")
                    # 构建初始图
                    graph_data = build_graph(original_nodes, batch, allocations)
                    batch_memory = []  # 记录当前批次的经验
                    step += 1
                    while len(allocated_tasks) < len(batch):  # 循环直到所有任务被分配
                        node_feats, task_feats, adjacency = graph_data
                        node_feats_t = torch.tensor(node_feats, dtype=torch.float32, device=device)
                        task_feats_t = torch.tensor(task_feats, dtype=torch.float32, device=device)
                        adj_t = torch.tensor(adjacency, dtype=torch.float32, device=device)

                        valid_mask = (adj_t > 0)  # 这是个 Tensor
                        valid_idx = torch.nonzero(valid_mask)  # 不会报错

                        # 若无任何可分配动作 => break
                        if not valid_mask.any():
                            # 无可行动作
                            print("=======无可行动作=======")
                            flag = False
                            break

                        action = agent.act(graph_data, valid_mask)
                        node_idx, task_idx = action
                        step_r = 0
                        if valid_mask[node_idx, task_idx]:
                            # print(f"任务{task_idx} ——》节点{node_idx}")
                            # 分配资源
                            node = original_nodes[node_idx]
                            task = batch[task_idx]
                            update_node_resources(node, task)
                            allocated_tasks.add(task_idx)
                            # 更新 allocations
                            t_id = batch[task_idx]['task_id']
                            allocations[t_id] = node_idx
                            # 计算奖励
                            # 这里可以传 {task_id: node_idx}, 也可以一次分完再算
                            step_r = calculate_cost(original_nodes, batch,
                                                    allocations={task['task_id']: node_idx})
                            # print(step_r)
                            done = False
                        else:
                            flag = False
                            print("=============惩罚")
                            step_r = -100

                        # next_graph
                        next_graph_data = build_graph(original_nodes, batch, allocations)

                        # 存储
                        batch_memory.append((graph_data, action, step_r, next_graph_data, done))
                        done = len(allocated_tasks) == len(batch) or not flag  # 判断批次是否完成
                        # 更新 graph_data
                        graph_data = next_graph_data

                    # 批次奖励计算与更新
                    if not flag:
                        episode_reward -= 100
                        print("=============惩罚")
                    else:
                        episode_reward += calculate_cost(original_nodes, batch, allocations)

                    # 存储经验并更新模型
                    for state, action, reward, next_state, done in batch_memory:
                        agent.remember(state, action, reward, next_state, done)

                    # 还原节点状态
                    for task_id, node_id in allocations.items():
                        task = task_dict.get(task_id, None)
                        node = original_nodes[node_id]
                        if task:
                            update_node_resources(node, task, revert=True)
                # 如果有足够的经验，进行经验回放
                if len(agent.memory) > batch_size and step % 1 == 0:
                    loss = agent.replay(batch_size)
                    episode_loss += loss
                    # 更新目标网络
                    agent.update_target_model()
                # 将奖励、损失、Epsilon写入文件
                # episode_reward /= len(batch_tasks)
                file.write(
                    f"Episode: {e + 1}/{episodes}, Reward: {episode_reward}, Loss: {episode_loss / len(batch_tasks)}, Epsilon: {agent.epsilon}\n")
                # 输出到控制台
                print(
                    f"Epoch {e + 1}/{episodes}, Reward: {episode_reward}, Loss: {episode_loss / len(batch_tasks)}, Epsilon: {agent.epsilon}")
                # 更新Epsilon值
                if agent.epsilon > agent.epsilon_min:
                    agent.epsilon *= agent.epsilon_decay
                else:
                    agent.epsilon = agent.epsilon_min

                if episode_reward > best_reward:
                    best_reward = episode_reward
                    agent.save_model("best_model_LB0129.pth")
            # 训练完成后保存模型
            agent.save_model("dqn_model_LB.pth")

