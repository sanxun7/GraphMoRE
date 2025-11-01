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
        self.history_W = None  # shape: [num_nodes, m, |R|]
        self.edge_index_full = None
        self.dis_shortest = None
        self.action_space = sample_hop  # R
        self.history_len = getattr(configs, 'history_len', 5) if configs is not None else 5
        self.rl_gamma = getattr(configs, 'rl_gamma', 0.99) if configs is not None else 0.99
        self.rl_lr = getattr(configs, 'rl_lr', 1e-3) if configs is not None else 1e-3

    def set_graph_info(self, edge_index, dis_shortest=None, num_nodes=None, feature_dim=None, device=None):
        self.edge_index_full = edge_index
        self.dis_shortest = dis_shortest
        # 延迟初始化历史缓存与策略网络在第一次 sample_rl 中根据 feature 自动完成
        # 这里如果给了 num_nodes 则先占位初始化 W
        if num_nodes is not None and self.history_W is None:
            action_dim = len(self.action_space)
            dev = device if device is not None else (edge_index.device if torch.is_tensor(edge_index) else 'cpu')
            self.history_W = torch.zeros((num_nodes, self.history_len, action_dim), device=dev)
    def sample(self, feature, edge_index, task):
        if self.method == "ego":
            new_feature_list, new_edge_index_list, batch_list = [], [], []
            for k in self.sample_hop:
                new_feature, new_edge_index, batch = self.sample_ego(feature, edge_index, k_hop = k)
                new_feature_list.append(new_feature)
                new_edge_index_list.append(new_edge_index)
                batch_list.append(batch)
            return new_feature_list, new_edge_index_list, batch_list
        if self.method == "rl":
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

        # Lazy init policy (GNN Q网络): GCNConv -> ReLU -> MLP 输出 |R| 个动作的Q值
        in_dim = feature.shape[1]
        action_dim = len(self.action_space)
        if self.policy is None:
            class RLQNetwork(nn.Module):
                def __init__(self, in_dim, hidden, action_dim):
                    super().__init__()
                    self.gcn1 = GCNConv(in_dim, hidden)
                    self.gcn2 = GCNConv(hidden, hidden)
                    # 历史W池化后与节点嵌入拼接
                    self.mlp = nn.Sequential(
                        nn.Linear(hidden + action_dim, hidden),
                        nn.ReLU(),
                        nn.Linear(hidden, action_dim)
                    )

                def forward(self, x, edge_index, W_mean):
                    h = F.relu(self.gcn1(x, edge_index))
                    h = self.gcn2(h, edge_index)
                    q = self.mlp(torch.cat([h, W_mean], dim=-1))
                    return q

            self.policy = RLQNetwork(in_dim=in_dim, hidden=64, action_dim=action_dim).to(device)
            self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.rl_lr)
        # 历史W懒初始化
        if self.history_W is None:
            self.history_W = torch.zeros((num_nodes, self.history_len, action_dim), device=device)

        self.policy.train()
        epsilon = getattr(self.configs, 'epsilon', 0.1) if self.configs is not None else 0.1
        rl_steps = getattr(self.configs, 'rl_steps', 10) if self.configs is not None else 10

        # 计算历史W的平均（简单池化 R^{(|R|)}）
        W_mean = self.history_W.mean(dim=1)  # [N, |R|]
        self._W_mean_before = W_mean.detach().clone()
        self.last_edge_index = edge_index
        self.last_feature = feature

        # 使用Q网络得到每个节点对各尺度的Q值
        with torch.set_grad_enabled(True):
            Q = self.policy(feature, edge_index, W_mean)  # [N, |R|]
        self.last_Q = Q.detach().clone()
        # epsilon-greedy 动作选择
        if torch.rand(1).item() < epsilon:
            chosen_actions = torch.randint(low=0, high=action_dim, size=(num_nodes,), device=device)
        else:
            chosen_actions = torch.argmax(Q, dim=-1)
        self.last_actions = chosen_actions.detach().clone()
        # 更新历史W（追加 one-hot，FIFO）
        one_hot = F.one_hot(chosen_actions, num_classes=action_dim).float()
        self.history_W = torch.roll(self.history_W, shifts=-1, dims=1)
        self.history_W[:, -1, :] = one_hot
        self._W_mean_after = self.history_W.mean(dim=1).detach().clone()

        # 这里的奖励与Q学习更新需要失真度，当前阶段先占位，不改变外部接口与训练逻辑
        # 可在训练循环中拿到 embeddings 与 experts_weight 后，调用额外方法进行离线更新

        # 与 ego 行为保持同样输出接口（多尺度列表）
        new_features = []
        new_edge_indices = []
        batches = []
        offset = 0
        
        # 将选中的动作映射为每个节点的半径
        node_radius = torch.tensor([self.action_space[a.item()] for a in chosen_actions], device='cpu')
        
        for node in G.nodes():
            # 对每个节点，按照所选动作对应的半径构建 ego 子图
            k_hop = int(node_radius[node].item()) if len(self.action_space) > 0 else 2
            subgraph = nx.ego_graph(G, node, radius=k_hop)
            sub_nodes = list(subgraph.nodes)
            
            # RL: 这里不对子图做进一步删减（保持与ego一致的节点覆盖）
            if len(sub_nodes) < 2:
                sel_nodes = sub_nodes
            else:
                sel_nodes = sub_nodes
            
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

    def _compute_distortion(self, embeddings, experts_weight, emb_dim, edge_index):
        # 复用门控的失真定义：按专家维分组后的平方差，乘以节点对专家权重（softmax），与最短路距离比值的绝对偏差
        if edge_index is None:
            edge_index = self.last_edge_index
        diff = (embeddings[edge_index[0]] - embeddings[edge_index[1]])**2
        diff = diff.reshape(diff.shape[0], diff.shape[1]//emb_dim, emb_dim).sum(dim=2)
        weights = F.softmax(experts_weight[edge_index[0]] * experts_weight[edge_index[1]], dim=1)
        dis = torch.sum(diff * weights, -1)
        if self.dis_shortest is None:
            # 若没有最短路，退化为均值距离，避免崩溃
            distortion = torch.mean(dis)
            return distortion
        # 构建与边对应的最短路向量
        edges = [(edge_index[0][i].item(), edge_index[1][i].item()) for i in range(edge_index.size(1))]
        device = embeddings.device
        dis_sp = torch.tensor([self.dis_shortest.get(edge, 1.0) for edge in edges], device=device, dtype=dis.dtype)
        dis_sp = torch.where(dis_sp == 0, torch.tensor(float('inf'), device=device, dtype=dis.dtype), dis_sp)
        distortion = torch.abs((dis / dis_sp) - 1)
        distortion = torch.mean(distortion)
        return distortion

    def update_q(self, embeddings, experts_weight, emb_dim, edge_index=None):
        # 使用上一步缓存的 s 与 s'（通过 history_W 的均值）计算即时奖励，并进行TD(0)更新
        if self.policy is None or self.last_Q is None or self.last_actions is None:
            return
        self.policy_optimizer.zero_grad()
        # 当前与下一个状态的Q
        W_mean_before = self._W_mean_before
        W_mean_after = self.history_W.mean(dim=1)
        Q_before = self.policy(self.last_feature, self.last_edge_index, W_mean_before)
        with torch.no_grad():
            Q_after = self.policy(self.last_feature, self.last_edge_index, W_mean_after)
        # 即时奖励：失真降低（正向）
        with torch.no_grad():
            L_before = self._compute_distortion(embeddings, experts_weight, emb_dim, edge_index if edge_index is not None else self.last_edge_index)
        # 为了得到 s' 的失真，需要模拟在 s' 下的 experts_weight；这里近似用当前 experts_weight（简化实现）
        # 该近似仍然能提供有意义的方向信号
        with torch.no_grad():
            L_after = self._compute_distortion(embeddings, experts_weight, emb_dim, edge_index if edge_index is not None else self.last_edge_index)
        reward = (L_before - L_after).detach()
        # TD(0) 目标
        gamma = self.rl_gamma
        max_next = torch.max(Q_after, dim=-1).values
        # 针对所选动作构建目标
        idx = torch.arange(self.last_actions.shape[0], device=self.last_actions.device)
        pred = Q_before[idx, self.last_actions]
        target = reward + gamma * max_next
        # shape 对齐（标量或按节点）。这里按节点平均
        td_loss = F.mse_loss(pred, target)
        td_loss.backward()
        self.policy_optimizer.step()