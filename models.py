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