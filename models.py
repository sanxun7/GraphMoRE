import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from geoopt.manifolds.stereographic.math import project
from geoopt.manifolds.stereographic import StereographicExact
from geoopt import ManifoldTensor
from geoopt import ManifoldParameter
from backbone import GCN, GAT, GraphSAGE
import geoopt
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, to_scipy_sparse_matrix
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
from torch_geometric.nn.inits import zeros
import networkx as nx
import pickle, os, random
from collections import deque
import copy
seed = 3047
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

class FermiDiracDecoder(nn.Module):
    def __init__(self, r, t):
        super(FermiDiracDecoder, self).__init__()
        self.r = r
        self.t = t

    def forward(self, dist):
        probs = torch.sigmoid((self.r - dist) / self.t)
        return probs
    
class kappaLinear(nn.Module):
    def __init__(self, manifold, in_dim: int, out_dim: int, dropout: float=0.0, use_bias: bool=True):
        super(kappaLinear, self).__init__()
        self.manifold = manifold
        self.dropout = dropout
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(out_dim, in_dim))
        self.bias = nn.Parameter(torch.Tensor(out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        res = self.manifold.mobius_matvec(drop_weight, x, project=True)
        if self.use_bias:
            bias = self.manifold.proju(self.manifold.origin(self.bias.shape), self.bias)
            kappa_bias = self.manifold.expmap0(bias, project=True)
            res = self.manifold.mobius_add(res, kappa_bias, project=True)
        return res

class kappaGCNConv(MessagePassing):
    def __init__(self, k, in_dim: int, out_dim: int, learnable=True):
        super().__init__(aggr='add')
        self.manifold = geoopt.Stereographic(k=k, learnable=learnable)
        self.lin = kappaLinear(manifold = self.manifold, in_dim=in_dim, out_dim=out_dim, use_bias=True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        edge_index, _ = add_self_loops(edge_index)
        x = self.lin(x)

        x_tan0 = self.manifold.logmap0(x)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        out = self.propagate(edge_index, x=x_tan0, norm=norm)
        out = self.manifold.expmap0(out, project=True)
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class Encoder(nn.Module):
    def __init__(self, k, in_dim: int, hidden_dim: int, out_dim: int, learnable: bool = True):
        super(Encoder, self).__init__()
        self.manifold = geoopt.Stereographic(k=k, learnable=learnable)
        self.encoder1 = kappaGCNConv(k, in_dim, hidden_dim, learnable=learnable)
        self.encoder2 = kappaGCNConv(k, hidden_dim, out_dim, learnable=learnable)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = self.manifold.proju(self.manifold.origin(x.shape), x)
        x = self.manifold.expmap0(x, project=True)
        h = self.encoder1(x, edge_index)
        z = self.encoder2(h, edge_index)
        return z

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = self.manifold.proju(self.manifold.origin(x.shape), x)
        x = self.manifold.expmap0(x, project=True)
        h = self.encoder1(x, edge_index)
        z = self.encoder2(h, edge_index)
        z = self.manifold.logmap0(z)
        return z

class Gating(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_experts: int, noisy_gating = False, configs = None):
        super(Gating, self).__init__()
        self.encoder1 = GCNConv(in_dim, hidden_dim)
        self.encoder2 = GCNConv(hidden_dim, out_dim)
        self.pooling = global_mean_pool
        self.classifier = nn.Linear(out_dim*len(configs.sample_hop), num_experts, bias=True)
        self.num_experts = num_experts
        self.configs = configs
        self.dis_edge = None
        self.dis = None

    def forward(self, subgraph_x, subgraph_edge_index, subgraph_batch, embeddings = None, dis_shortest = None, emb_dim = None, edge_index = None, feature = None):
        x = []
        for i in range(len(subgraph_x)):
            x_scale = self.encoder1(subgraph_x[i], subgraph_edge_index[i])
            x_scale = self.encoder2(x_scale, subgraph_edge_index[i])
            x_scale = self.pooling(x_scale, subgraph_batch[i])
            x.append(x_scale)
        x = torch.cat(x, -1)
        out = self.classifier(x)
        temperature = 1.0
        out = F.softmax(out / temperature, dim=-1)
        if embeddings == None:
            return out

        loss_distortion = self.compute_distortion(out, embeddings, dis_shortest, emb_dim, edge_index)
        return out, loss_distortion

    def compute_distortion(self, expert_weights, embeddings, dis_shortest, emb_dim, edge_index):
        if self.dis_edge == None or self.dis_edge.shape != edge_index.shape or torch.any(self.dis_edge != edge_index):
            self.dis_edge = edge_index
            edges = [(edge_index[0][i].item(), edge_index[1][i].item()) for i in range(edge_index.size(1))]
            self.dis = torch.tensor([dis_shortest[edge] for edge in edges]).cuda()

        diff = (embeddings[edge_index[0]] - embeddings[edge_index[1]])**2
        diff = diff.reshape(diff.shape[0], diff.shape[1]//emb_dim, emb_dim).sum(dim=2)
        weights = F.softmax(expert_weights[edge_index[0]] * expert_weights[edge_index[1]], dim=1)   
        dis = torch.sum(diff * weights, -1)
        distortion = torch.abs((dis/self.dis)-1)
        distortion = torch.mean(distortion)
        loss_distortion = distortion
        return loss_distortion



class Experts(nn.Module):
    def __init__(self, init_curvs, in_dim: int, hidden_dim: int, out_dim: int, learnable=True, num_factors_cls = None):
        super(Experts, self).__init__()
        self.experts = nn.ModuleList()
        num_factors = len(init_curvs)
        for curv in init_curvs:
            if curv == 0:
                self.experts.append(Encoder(0,in_dim,hidden_dim,out_dim,learnable=False))
            else:
                self.experts.append(Encoder(curv,in_dim,hidden_dim,out_dim,learnable))
        self.norm1 = nn.LayerNorm(num_factors * out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        embeds = []
        for expert in self.experts:
            embed = expert(x, edge_index)
            embeds.append(embed)
        embeds = torch.concat(embeds, -1)
        return self.norm1(embeds)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, dataset = None):
        embeds = []
        for expert in self.experts:
            embed = expert.encode(x, edge_index)
            embeds.append(embed)
        embeds = torch.concat(embeds, -1)
        return embeds

class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """存储经验元组"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """随机采样一批经验"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """GNN参数化的Q网络"""
    def __init__(self, in_dim, hidden_dim, out_dim=2, backbone='gcn', n_heads=8):
        super(QNetwork, self).__init__()
        self.backbone = backbone
        
        if backbone == 'gcn':
            self.gnn1 = GCNConv(in_dim, hidden_dim)
            self.gnn2 = GCNConv(hidden_dim, hidden_dim)
        elif backbone == 'gat':
            self.gnn1 = GATConv(in_dim, hidden_dim, heads=n_heads, concat=False)
            self.gnn2 = GATConv(hidden_dim, hidden_dim, heads=n_heads, concat=False)
        elif backbone == 'sage':
            self.gnn1 = SAGEConv(in_dim, hidden_dim)
            self.gnn2 = SAGEConv(hidden_dim, hidden_dim)
        else:
            raise NotImplementedError(f"Backbone {backbone} not supported")
        
        # Q值输出层：对每个节点输出Q值（动作：0=排除, 1=包含）
        self.q_head = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, edge_index):
        """
        Args:
            x: 节点特征 [num_nodes, in_dim]
            edge_index: 边索引 [2, num_edges]
        Returns:
            q_values: Q值 [num_nodes, 2] (每行是[Q(排除), Q(包含)])
        """
        # GNN编码
        x = F.relu(self.gnn1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.gnn2(x, edge_index))
        
        # 输出Q值
        q_values = self.q_head(x)  # [num_nodes, 2]
        return q_values


class Sampler():
    def __init__(self, method = "ego", sample_hop = [2,3], dataset = "Cora", configs = None):
        self.method = method
        self.sample_hop = sample_hop
        self.dataset = dataset
        self.configs = configs
        # RL components (lazy init)
        self.policy = None
        self.policy_optimizer = None
        self.prev_score = 0.0
        
        # DQN components (lazy init)
        self.q_network = None
        self.target_q_network = None
        self.q_optimizer = None
        self.replay_buffer = None
        self.step_count = 0
        self.train_step = 0
    def sample(self, feature, edge_index, task, embeddings=None, dis_shortest=None, training=True):
        if self.method == "ego":
            new_feature_list, new_edge_index_list, batch_list = [], [], []
            for k in self.sample_hop:
                new_feature, new_edge_index, batch = self.sample_ego(feature, edge_index, k_hop = k)
                new_feature_list.append(new_feature)
                new_edge_index_list.append(new_edge_index)
                batch_list.append(batch)
            return new_feature_list, new_edge_index_list, batch_list
        if self.method == "rl":
            # 使用DQN采样
            result = self.sample_rl_dqn(feature, edge_index, embeddings, dis_shortest, training)
            if isinstance(result, tuple) and len(result) == 4:
                # 如果返回经验，只返回前三个（兼容原有接口）
                return result[0], result[1], result[2]
            else:
                # 向后兼容旧的sample_rl方法
                return self.sample_rl(feature, edge_index)
        return None
    
    def sample_ego(self, feature, edge_index, k_hop):
        G = nx.Graph()
        G.add_nodes_from(range(feature.shape[0]))
        edges = [(edge_index[0][i].item(), edge_index[1][i].item()) for i in range(edge_index.size(1))]
        G.add_edges_from(edges)
        new_features = []
        new_edge_indices = []
        offset = 0 
        batches = []
        for node in G.nodes():
            subgraph = nx.ego_graph(G, node, radius=k_hop)
            subgraph_feature = feature[[node for node in subgraph.nodes]]
            new_features.append(subgraph_feature)
            new_node_indices = {node: idx + offset for idx, node in enumerate(subgraph.nodes)}
            subgraph_edge_index = torch.tensor(
                [[new_node_indices[u], new_node_indices[v]] for u, v in subgraph.edges()],
                dtype=torch.long
            ).t()
            new_edge_indices.append(subgraph_edge_index)
            offset += len(new_node_indices)  
            subgraph_batch = torch.tensor([node]*subgraph_feature.shape[0], dtype=torch.long)
            batches.append(subgraph_batch)

        new_feature = torch.cat(new_features, dim=0).cuda()
        new_edge_index = torch.cat(new_edge_indices, dim=1).cuda()
        batch = torch.cat(batches, dim=0).cuda()

        return new_feature, new_edge_index, batch

    def sample_rl(self, feature, edge_index):
        device = feature.device
        num_nodes = feature.shape[0]
        
        # For LP task, edge_index is actually edges; for NC it's edges too
        # We need full graph for ego sampling
        # Build full graph from edge_index
        G = nx.Graph()
        edges_py = [(edge_index[0][i].item(), edge_index[1][i].item()) for i in range(edge_index.size(1))]
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(edges_py)

        # Lazy init policy
        in_dim = feature.shape[1]
        if self.policy is None:
            self.policy = nn.Sequential(
                nn.Linear(in_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 2)
            ).to(device)
            self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)

        self.policy.train()
        epsilon = getattr(self.configs, 'epsilon', 0.1) if self.configs is not None else 0.1
        rl_steps = getattr(self.configs, 'rl_steps', 10) if self.configs is not None else 10

        # Match ego behavior: sample ego subgraph for each node
        new_features = []
        new_edge_indices = []
        batches = []
        offset = 0
        
        # Use average hop from self.sample_hop as k_hop
        k_hop = int(np.mean(self.sample_hop)) if len(self.sample_hop) > 0 else 2
        
        for node in G.nodes():
            subgraph = nx.ego_graph(G, node, radius=k_hop)
            sub_nodes = list(subgraph.nodes)
            
            # RL: decide whether to keep or shrink this ego subgraph
            if len(sub_nodes) < 2:
                sel_nodes = sub_nodes
            else:
                # Apply RL policy to select subset of nodes in ego subgraph
                sub_features = feature[sub_nodes]
                logits = self.policy(sub_features)
                probs = F.softmax(logits, dim=-1)
                include_scores = probs[:, 1]
                
                # Simple strategy: keep top-k nodes by policy score, or random with epsilon
                budget_sub = max(2, int(0.5 * len(sub_nodes)))
                if torch.rand(1).item() < epsilon:
                    pick_indices = torch.randperm(len(sub_nodes), device=device)[:budget_sub]
                else:
                    _, pick_indices = torch.topk(include_scores, k=budget_sub)
                
                sel_nodes_indices = pick_indices.cpu().tolist()
                sel_nodes = [sub_nodes[i] for i in sel_nodes_indices]
            
            # Build subgraph features and edges
            subgraph_feature = feature[sel_nodes]
            new_features.append(subgraph_feature)
            
            new_node_indices = {node: idx + offset for idx, node in enumerate(sel_nodes)}
            sub_edges = []
            for u, v in subgraph.subgraph(sel_nodes).edges():
                sub_edges.append([new_node_indices[u], new_node_indices[v]])
            
            if len(sub_edges) == 0:
                sub_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            else:
                sub_edge_index = torch.tensor(sub_edges, dtype=torch.long, device=device).t().contiguous()
            new_edge_indices.append(sub_edge_index)
            
            offset += len(sel_nodes)
            subgraph_batch = torch.tensor([node] * subgraph_feature.shape[0], dtype=torch.long, device=device)
            batches.append(subgraph_batch)

        # Concatenate all subgraphs
        new_feature = torch.cat(new_features, dim=0)
        new_edge_index = torch.cat(new_edge_indices, dim=1)
        batch = torch.cat(batches, dim=0)

        # Return in same format as ego: list of scales
        new_feature_list, new_edge_index_list, batch_list = [], [], []
        for k in self.sample_hop:
            new_feature_list.append(new_feature)
            new_edge_index_list.append(new_edge_index)
            batch_list.append(batch)
        
        return new_feature_list, new_edge_index_list, batch_list
    
    def _init_dqn(self, feature, device):
        """初始化DQN组件"""
        if self.q_network is None:
            in_dim = feature.shape[1]
            hidden_dim = getattr(self.configs, 'q_hidden_dim', 64)
            backbone = getattr(self.configs, 'q_backbone', 'gcn')
            n_heads = getattr(self.configs, 'n_heads', 8)
            
            self.q_network = QNetwork(in_dim, hidden_dim, out_dim=2, 
                                     backbone=backbone, n_heads=n_heads).to(device)
            self.target_q_network = QNetwork(in_dim, hidden_dim, out_dim=2,
                                            backbone=backbone, n_heads=n_heads).to(device)
            self.target_q_network.load_state_dict(self.q_network.state_dict())
            self.target_q_network.eval()
            
            q_lr = getattr(self.configs, 'q_lr', 1e-3)
            self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=q_lr)
            
            buffer_size = getattr(self.configs, 'replay_buffer_size', 10000)
            self.replay_buffer = ReplayBuffer(capacity=buffer_size)
    
    def _compute_reward(self, selected_nodes, center_node, subgraph_nodes, 
                       embeddings, dis_shortest, configs):
        """计算奖励：基于子图质量和下游任务性能"""
        if len(selected_nodes) < 2:
            return -0.5
        compactness_reward = 1.0 - (len(selected_nodes) / max(len(subgraph_nodes), 1))
        distortion_reward = 0.0
        if embeddings is not None and len(selected_nodes) > 1:
            try:
                selected_emb = embeddings[selected_nodes]
                if center_node in selected_nodes:
                    center_idx = selected_nodes.index(center_node)
                    dists = torch.norm(selected_emb - selected_emb[center_idx], dim=1).mean()
                    distortion_reward = 0.1 * (1.0 / (1.0 + dists.item()))
            except:
                pass
        diversity_reward = 0.1 if len(selected_nodes) >= 3 else 0.0
        total_reward = compactness_reward * 0.5 + distortion_reward * 0.3 + diversity_reward * 0.2
        return total_reward
    
    def sample_rl_dqn(self, feature, edge_index, embeddings=None, dis_shortest=None, 
                     training=True, collect_experience=True):
        """使用DQN进行采样"""
        device = feature.device
        num_nodes = feature.shape[0]
        self._init_dqn(feature, device)
        G = nx.Graph()
        edges_py = [(edge_index[0][i].item(), edge_index[1][i].item()) 
                   for i in range(edge_index.size(1))]
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(edges_py)
        epsilon = getattr(self.configs, 'epsilon', 0.1) if self.configs else 0.1
        rl_budget = getattr(self.configs, 'rl_budget', 0.2) if self.configs else 0.2
        k_hop = int(np.mean(self.sample_hop)) if len(self.sample_hop) > 0 else 2
        new_features = []
        new_edge_indices = []
        batches = []
        offset = 0
        experiences = []
        if training:
            self.q_network.train()
        else:
            self.q_network.eval()
        for node in G.nodes():
            subgraph = nx.ego_graph(G, node, radius=k_hop)
            sub_nodes = list(subgraph.nodes)
            if len(sub_nodes) < 2:
                sel_nodes = sub_nodes
                sel_actions = [1] * len(sub_nodes)
            else:
                sub_features = feature[sub_nodes]
                sub_edge_list = []
                for u, v in subgraph.edges():
                    u_idx = sub_nodes.index(u)
                    v_idx = sub_nodes.index(v)
                    sub_edge_list.append([u_idx, v_idx])
                if len(sub_edge_list) == 0:
                    sub_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                    for i in range(len(sub_nodes)):
                        sub_edge_list.append([i, i])
                else:
                    sub_edge_index = torch.tensor(sub_edge_list, dtype=torch.long, device=device).t().contiguous()
                    sub_edge_index, _ = add_self_loops(sub_edge_index, num_nodes=len(sub_nodes))
                with torch.set_grad_enabled(training):
                    q_values = self.q_network(sub_features, sub_edge_index)
                    q_include = q_values[:, 1]
                budget_sub = max(2, int(rl_budget * len(sub_nodes)))
                if training and torch.rand(1).item() < epsilon:
                    pick_indices = torch.randperm(len(sub_nodes), device=device)[:budget_sub]
                    sel_actions = [0] * len(sub_nodes)
                    for idx in pick_indices.cpu().tolist():
                        sel_actions[idx] = 1
                else:
                    _, pick_indices = torch.topk(q_include, k=min(budget_sub, len(sub_nodes)))
                    sel_actions = [0] * len(sub_nodes)
                    for idx in pick_indices.cpu().tolist():
                        sel_actions[idx] = 1
                sel_nodes_indices = [i for i, a in enumerate(sel_actions) if a == 1]
                sel_nodes = [sub_nodes[i] for i in sel_nodes_indices]
                if collect_experience and training:
                    state = {'features': sub_features.detach().cpu(), 'edge_index': sub_edge_index.detach().cpu(), 
                            'sub_nodes': sub_nodes, 'center_node': node}
                    action = torch.tensor(sel_actions, dtype=torch.float32)
                    reward = self._compute_reward(sel_nodes, node, sub_nodes, embeddings, dis_shortest, self.configs)
                    reward = torch.tensor(reward, dtype=torch.float32)
                    next_state = state.copy()
                    done = False
                    experiences.append((state, action, reward, next_state, done))
            if len(sel_nodes) > 0:
                subgraph_feature = feature[sel_nodes]
                new_features.append(subgraph_feature)
                new_node_indices = {node: idx + offset for idx, node in enumerate(sel_nodes)}
                sub_edges = []
                for u, v in subgraph.subgraph(sel_nodes).edges():
                    sub_edges.append([new_node_indices[u], new_node_indices[v]])
                if len(sub_edges) == 0:
                    sub_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                else:
                    sub_edge_index = torch.tensor(sub_edges, dtype=torch.long, device=device).t().contiguous()
                new_edge_indices.append(sub_edge_index)
                offset += len(sel_nodes)
                subgraph_batch = torch.tensor([node] * subgraph_feature.shape[0], dtype=torch.long, device=device)
                batches.append(subgraph_batch)
        if len(new_features) > 0:
            new_feature = torch.cat(new_features, dim=0)
            new_edge_index = torch.cat(new_edge_indices, dim=1) if len(new_edge_indices) > 0 else torch.zeros((2, 0), dtype=torch.long, device=device)
            batch = torch.cat(batches, dim=0)
        else:
            new_feature = torch.zeros((0, feature.shape[1]), device=device)
            new_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            batch = torch.zeros((0,), dtype=torch.long, device=device)
        new_feature_list, new_edge_index_list, batch_list = [], [], []
        for k in self.sample_hop:
            new_feature_list.append(new_feature)
            new_edge_index_list.append(new_edge_index)
            batch_list.append(batch)
        if collect_experience and training and len(experiences) > 0:
            for exp in experiences:
                self.replay_buffer.push(*exp)
        self.step_count += 1
        return new_feature_list, new_edge_index_list, batch_list, experiences
    
    def train_dqn(self, batch_size=32, gamma=0.99):
        """训练DQN：从经验回放缓冲区采样并更新Q网络"""
        if len(self.replay_buffer) < batch_size:
            return None
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        device = next(iter(self.q_network.parameters())).device
        batch_q_values = []
        batch_target_q_values = []
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            state_features = state['features'].to(device)
            state_edge_index = state['edge_index'].to(device)
            action = action.to(device)
            reward = reward.to(device).item()
            q_vals = self.q_network(state_features, state_edge_index)
            q_value = (q_vals * action.unsqueeze(1)).sum(dim=1).mean()
            batch_q_values.append(q_value)
            if not done:
                next_features = next_state['features'].to(device)
                next_edge_index = next_state['edge_index'].to(device)
                with torch.no_grad():
                    next_q_vals = self.target_q_network(next_features, next_edge_index)
                    next_q_value = next_q_vals[:, 1].max()
                    target_q_value = reward + gamma * next_q_value
            else:
                target_q_value = reward
            batch_target_q_values.append(torch.tensor(target_q_value, device=device))
        q_values = torch.stack(batch_q_values)
        target_q_values = torch.stack(batch_target_q_values)
        loss = F.mse_loss(q_values, target_q_values)
        self.q_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.q_optimizer.step()
        self.train_step += 1
        target_update_freq = getattr(self.configs, 'target_update_freq', 100) if self.configs else 100
        if self.train_step % target_update_freq == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())
        return loss.item()