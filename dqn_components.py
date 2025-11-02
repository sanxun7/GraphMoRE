import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from torch_geometric.nn import GCNConv, global_mean_pool

class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """存储经验 (s, a, r, s', done)"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """随机采样一批经验"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为tensor
        states = torch.stack(states)
        if isinstance(actions[0], torch.Tensor):
            actions = torch.stack(actions)
        else:
            actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class DQNNetwork(nn.Module):
    """DQN Q网络：输入状态（拓扑特征），输出每个动作的Q值"""
    def __init__(self, in_dim, hidden_dim, out_dim, num_experts, sample_hop, device):
        super(DQNNetwork, self).__init__()
        self.device = device
        self.num_experts = num_experts
        self.sample_hop = sample_hop
        
        # 状态编码器（拓扑特征提取）
        self.encoder1 = GCNConv(in_dim, hidden_dim)
        self.encoder2 = GCNConv(hidden_dim, out_dim)
        self.pooling = global_mean_pool
        
        # Q值网络：状态 -> Q值（每个专家选择动作的Q值）
        state_dim = out_dim * len(sample_hop)
        # 动作空间：每个专家独立选择（0/1），共2^num_experts种组合
        # 简化：输出每个专家的Q值，然后可以用softmax或独立sigmoid选择
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)  # 每个专家的Q值
        )
    
    def encode_state(self, subgraph_x, subgraph_edge_index, subgraph_batch):
        """编码状态：从子图提取拓扑特征"""
        x = []
        for i in range(len(subgraph_x)):
            x_scale = self.encoder1(subgraph_x[i], subgraph_edge_index[i])
            x_scale = self.encoder2(x_scale, subgraph_edge_index[i])
            x_scale = self.pooling(x_scale, subgraph_batch[i])
            x.append(x_scale)
        x = torch.cat(x, -1)  # 拼接多分辨率特征
        return x
    
    def forward(self, subgraph_x, subgraph_edge_index, subgraph_batch):
        """前向传播：输出每个专家动作的Q值"""
        state = self.encode_state(subgraph_x, subgraph_edge_index, subgraph_batch)
        q_values = self.q_network(state)  # [num_nodes, num_experts]
        return q_values


class GatingDQN(nn.Module):
    """基于DQN的门控网络"""
    def __init__(self, in_dim, hidden_dim, out_dim, num_experts, configs, device):
        super(GatingDQN, self).__init__()
        self.device = device
        self.configs = configs
        self.num_experts = num_experts
        
        # Q网络
        self.q_network = DQNNetwork(
            in_dim, hidden_dim, out_dim, num_experts, 
            configs.sample_hop, device
        ).to(device)
        
        # 目标Q网络（用于稳定训练）
        self.target_q_network = DQNNetwork(
            in_dim, hidden_dim, out_dim, num_experts,
            configs.sample_hop, device
        ).to(device)
        
        # 初始化目标网络
        self.update_target_network(tau=1.0)
        
        # 缓存
        self.dis_edge = None
        self.dis = None
    
    def update_target_network(self, tau=0.01):
        """软更新目标网络参数"""
        for target_param, param in zip(self.target_q_network.parameters(), 
                                       self.q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    
    def get_q_values(self, subgraph_x, subgraph_edge_index, subgraph_batch, 
                     use_target=False):
        """获取Q值"""
        network = self.target_q_network if use_target else self.q_network
        return network(subgraph_x, subgraph_edge_index, subgraph_batch)
    
    def select_action(self, subgraph_x, subgraph_edge_index, subgraph_batch, 
                     epsilon=0.0, deterministic=False):
        """选择动作：epsilon-greedy策略"""
        q_values = self.get_q_values(subgraph_x, subgraph_edge_index, subgraph_batch)
        
        if deterministic:
            # 确定性策略：选择Q值最高的专家组合
            # 使用softmax将Q值转换为权重分布
            expert_weights = F.softmax(q_values, dim=-1)
            return expert_weights, q_values
        
        # Epsilon-greedy策略
        if random.random() < epsilon:
            # 随机探索：随机选择专家权重
            expert_weights = torch.softmax(torch.randn_like(q_values), dim=-1)
        else:
            # 利用：基于Q值选择
            expert_weights = F.softmax(q_values, dim=-1)
        
        return expert_weights, q_values
    
    def compute_distortion(self, expert_weights, embeddings, dis_shortest, 
                          emb_dim, edge_index):
        """计算嵌入失真度"""
        if (self.dis_edge is None or self.dis_edge.shape != edge_index.shape or 
            torch.any(self.dis_edge != edge_index)):
            self.dis_edge = edge_index
            edges = [(edge_index[0][i].item(), edge_index[1][i].item()) 
                    for i in range(edge_index.size(1))]
            self.dis = torch.tensor([dis_shortest[edge] for edge in edges]).to(self.device)
        
        diff = (embeddings[edge_index[0]] - embeddings[edge_index[1]])**2
        diff = diff.reshape(diff.shape[0], diff.shape[1]//emb_dim, emb_dim).sum(dim=2)
        weights = F.softmax(expert_weights[edge_index[0]] * expert_weights[edge_index[1]], dim=1)
        dis = torch.sum(diff * weights, -1)
        distortion = torch.abs((dis/self.dis) - 1)
        distortion = torch.mean(distortion)
        return distortion
    
    def forward(self, subgraph_x, subgraph_edge_index, subgraph_batch, 
               embeddings=None, dis_shortest=None, emb_dim=None, 
               edge_index=None, epsilon=0.0, deterministic=False):
        """
        前向传播：
        - 训练模式：返回expert_weights和loss_distortion
        - 评估模式：只返回expert_weights
        """
        if embeddings is None:
            # 评估模式：只返回权重
            expert_weights, _ = self.select_action(
                subgraph_x, subgraph_edge_index, subgraph_batch, 
                epsilon=epsilon, deterministic=deterministic
            )
            return expert_weights
        
        # 训练模式：选择动作并计算失真
        expert_weights, q_values = self.select_action(
            subgraph_x, subgraph_edge_index, subgraph_batch, 
            epsilon=epsilon, deterministic=deterministic
        )
        
        loss_distortion = self.compute_distortion(
            expert_weights, embeddings, dis_shortest, emb_dim, edge_index
        )
        
        return expert_weights, loss_distortion, q_values

