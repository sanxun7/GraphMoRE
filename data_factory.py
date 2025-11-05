import random
import torch
import networkx as nx
import numpy as np
import torch_geometric.data
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.datasets import Amazon, Planetoid
from torch_geometric.utils import to_networkx
from torch_geometric.utils import negative_sampling
import scipy.sparse as sp
import pickle as pkl
import os
import torch_geometric.transforms as T
from torch_geometric.transforms import RandomLinkSplit
import warnings
warnings.filterwarnings('ignore')
seed = 3047
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

def get_mask(idx, length):
    """Create mask.
    """
    mask = torch.zeros(length, dtype=torch.bool)
    mask[idx] = 1
    return mask

def load_data(root: str, data_name: str, split='public', **kwargs):
    if data_name in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(root=root, name=data_name, split=split)
        train_mask, val_mask, test_mask = dataset.data.train_mask, dataset.data.val_mask, dataset.data.test_mask
    elif data_name == "airport":
        dataset = Airport(root)
        train_mask, val_mask, test_mask = dataset.data.mask
    elif data_name == "photo":
        dataset = Amazon(root=root, name="Photo")
        labels = dataset.data.y.tolist()
        val_prop, test_prop = 0.15, 0.15
        val_mask, test_mask, train_mask = split_data(labels, val_prop, test_prop, seed=3047)
        mask = (train_mask, val_mask, test_mask)
        features = dataset.data.x
        num_features = dataset.num_features
        edge_index = dataset.data.edge_index.long()
        neg_edges = negative_sampling(edge_index)
        num_classes = dataset.num_classes
        labels = torch.tensor(labels)
        return features, num_features, labels, edge_index, neg_edges, mask, num_classes
    else:
        raise NotImplementedError
    mask = (train_mask, val_mask, test_mask)
    features = dataset.data.x
    num_features = dataset.num_features
    labels = dataset.data.y
    edge_index = dataset.data.edge_index.long()
    neg_edges = negative_sampling(edge_index)
    num_classes = dataset.num_classes
    return features, num_features, labels, edge_index, neg_edges, mask, num_classes

def load_synthetic_data(root: str, data_name: str):
    with open(f'{root}/{data_name}.pkl', 'rb') as f:
        G = pkl.load(f)
    with open(f'{root}/{data_name}_feature.pkl', 'rb') as f:
        features = pkl.load(f)
    features = torch.tensor(features).float()
    num_features = features.shape[-1]
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    neg_edges = negative_sampling(edge_index)
    perm = torch.randperm(edge_index.shape[-1])
    edge_index = edge_index[:, perm]
    perm = torch.randperm(neg_edges.shape[-1])
    neg_edges = neg_edges[:, perm]
    labels = torch.tensor([])
    mask = torch.tensor([])
    num_classes = None
    return features, num_features, labels, edge_index, neg_edges, mask, num_classes


def mask_edges(edge_index, neg_edges, val_prop, test_prop):
    n = len(edge_index[0])
    n_val = int(val_prop * n)
    n_test = int(test_prop * n)
    edge_val, edge_test, edge_train = edge_index[:, :n_val], edge_index[:, n_val:n_val + n_test], edge_index[:, n_val + n_test:]
    val_edges_neg, test_edges_neg = neg_edges[:, :n_val], neg_edges[:, n_val:n_test + n_val]
    train_edges_neg = torch.concat([neg_edges, edge_val, edge_test], dim=-1)
    return (edge_train, edge_val, edge_test), (train_edges_neg, val_edges_neg, test_edges_neg)

def mask_edges_random(edge_index, num_nodes, val_prop, test_prop, seed=3047):
    """
    使用 RandomLinkSplit 进行随机边划分（用于链接预测任务）
    
    Args:
        edge_index: 原始边索引 [2, num_edges]
        num_nodes: 节点数量
        val_prop: 验证集比例
        test_prop: 测试集比例
        seed: 随机种子
    
    Returns:
        pos_edges: (edge_train, edge_val, edge_test) 正边元组
        neg_edges: (train_edges_neg, val_edges_neg, test_edges_neg) 负边元组
    """
    # 创建临时 Data 对象用于 RandomLinkSplit
    temp_data = Data(edge_index=edge_index, num_nodes=num_nodes)
    
    # 设置随机种子
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # 使用 RandomLinkSplit 进行随机划分
    # is_undirected=True 表示无向图，add_negative_train_samples=False 表示不在训练集中自动添加负样本
    # neg_sampling_ratio=1.0 表示每个正样本对应一个负样本
    transform = RandomLinkSplit(
        num_val=val_prop,
        num_test=test_prop,
        is_undirected=True,
        add_negative_train_samples=False,
        neg_sampling_ratio=1.0,
        split_labels=False
    )
    
    train_data, val_data, test_data = transform(temp_data)
    
    # 提取训练集的边（只有正边）
    edge_train = train_data.edge_index
    
    # 从验证集和测试集中分离正边和负边
    # RandomLinkSplit 会在 edge_label_index 中包含所有边（正负样本），edge_label 包含标签
    val_pos_mask = val_data.edge_label.bool()
    val_pos_edges = val_data.edge_label_index[:, val_pos_mask]
    val_neg_edges = val_data.edge_label_index[:, ~val_pos_mask]
    
    test_pos_mask = test_data.edge_label.bool()
    test_pos_edges = test_data.edge_label_index[:, test_pos_mask]
    test_neg_edges = test_data.edge_label_index[:, ~test_pos_mask]
    
    # 生成训练集的负边
    # 需要排除所有已观察到的边（包括训练、验证、测试集的正边）
    all_pos_edges = torch.cat([edge_train, val_pos_edges, test_pos_edges], dim=1)
    train_edges_neg = negative_sampling(edge_index=all_pos_edges, num_nodes=num_nodes, num_neg_samples=edge_train.size(1))
    
    pos_edges = (edge_train, val_pos_edges, test_pos_edges)
    neg_edges = (train_edges_neg, val_neg_edges, test_neg_edges)
    
    return pos_edges, neg_edges

def bin_feat(feat, bins):
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()

def augment(adj, features, normalize_feats=True):
    deg = np.squeeze(np.sum(adj, axis=0).astype(int))
    deg[deg > 5] = 5
    deg_onehot = torch.tensor(np.eye(6)[deg], dtype=torch.float).squeeze()
    const_f = torch.ones(features.shape[0], 1)
    features = torch.cat((features, deg_onehot, const_f), dim=1)
    return features

def split_data(labels, val_prop, test_prop, seed):
    random.seed(seed)
    num_class = np.max(labels) + 1
    label_dict = dict()
    for i in range(num_class):
        label_dict[i] = []
    for i, l in enumerate(labels):
        label_dict[l].append(i)
    idx_train, idx_val, idx_test = [], [], []
    for i in range(num_class):
        random.shuffle(label_dict[i])
        num_val = round(val_prop * len(label_dict[i]))
        num_test = round(test_prop * len(label_dict[i]))
        idx_val += label_dict[i][:num_val]
        idx_test += label_dict[i][num_val:num_val + num_test]
        idx_train += label_dict[i][num_val + num_test:]
    return idx_val, idx_test, idx_train


class Airport(InMemoryDataset):
    def __init__(self, root):
        super(Airport, self).__init__()
        val_prop, test_prop = 0.15, 0.15
        graph = pkl.load(open(f"{root}/airport/airport.p", 'rb'))
        adj = nx.adjacency_matrix(graph).toarray()
        row, col = np.nonzero(adj)
        edge_index = np.concatenate([row[None], col[None]], axis=0)
        features = np.array([graph._node[u]['feat'] for u in graph.nodes()])
        features = augment(adj, torch.tensor(features).float())
        label_idx = 4
        labels = features[:, label_idx]
        features = features[:, :label_idx]
        labels = bin_feat(labels, bins=[7.0 / 7, 8.0 / 7, 9.0 / 7])

        idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, random.seed(3047))
        mask = (idx_train, idx_val, idx_test)

        self.data = torch_geometric.data.Data(x=features,
                                              edge_index=torch.tensor(edge_index),
                                              y=torch.tensor(labels),
                                              mask=mask)

        @property
        def num_features(self) -> int:
            return self.data.x.shape[-1]

        @property
        def raw_file_names(self):
            pass

        @property
        def processed_file_names(self):
            pass

        def download(self):
            pass

        def process(self):
            pass

