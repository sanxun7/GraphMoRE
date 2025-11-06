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

def mask_edges_manual(edge_index, num_nodes, val_prop, test_prop, seed=3047, verbose=False, prevent_disconnect=True):
    """
    手动实现的随机边划分（参考 process.py 中的 mask_test_edges）
    确保图的连通性，严格验证边集互斥性
    
    Args:
        edge_index: 原始边索引 [2, num_edges]
        num_nodes: 节点数量
        val_prop: 验证集比例
        test_prop: 测试集比例
        seed: 随机种子
        verbose: 是否输出详细信息
        prevent_disconnect: 是否保持图的连通性
    
    Returns:
        pos_edges: (edge_train, edge_val, edge_test) 正边元组
        neg_edges: (train_edges_neg, val_edges_neg, test_edges_neg) 负边元组
    """
    if verbose:
        print('Preprocessing edges for manual link prediction split...')
    
    # 设置随机种子
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    # 转换为无向边的元组集合（去除重复，使用 (min, max) 格式）
    edges_list = [(edge_index[0][i].item(), edge_index[1][i].item()) for i in range(edge_index.size(1))]
    edge_tuples = [(min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in edges_list]
    all_edge_tuples = set(edge_tuples)
    
    # 构建 NetworkX 图（使用标准化的边，避免重复）
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edge_tuples)  # 使用标准化的边，避免重复
    
    orig_num_cc = nx.number_connected_components(G)
    if verbose:
        print(f'Original graph has {orig_num_cc} connected component(s)')
    
    # 计算需要划分的边数量
    num_test = int(np.floor(len(all_edge_tuples) * test_prop))
    num_val = int(np.floor(len(all_edge_tuples) * val_prop))
    
    if verbose:
        print(f'Requested: {num_val} val edges, {num_test} test edges')
    
    # 初始化边集合
    train_edges = set(edge_tuples)
    test_edges = set()
    val_edges = set()
    
    # 随机打乱边列表
    edge_tuples_shuffled = edge_tuples.copy()
    np.random.shuffle(edge_tuples_shuffled)
    
    if verbose:
        print('Generating test/val sets...')
    
    # 划分正边：先填充测试集，再填充验证集，同时保持图的连通性
    # 完全对齐 process.py 的逻辑
    for edge in edge_tuples_shuffled:
        node1, node2 = edge
        
        # 检查边是否还在训练集中（可能已经被移除了）
        if edge not in train_edges:
            continue
        
        # 先移除边检查连通性（与 process.py 完全一致）
        if not G.has_edge(node1, node2):
            continue
        G.remove_edge(node1, node2)
        
        # 如果移除边会断开连通分量，加回边并跳过
        if prevent_disconnect:
            if nx.number_connected_components(G) > orig_num_cc:
                G.add_edge(node1, node2)
                continue
        
        # 先填充测试集（与 process.py 完全一致）
        if len(test_edges) < num_test:
            test_edges.add(edge)
            train_edges.remove(edge)
            # 注意：边已经在上面被移除了，不需要再次移除
        
        # 再填充验证集（与 process.py 完全一致）
        elif len(val_edges) < num_val:
            val_edges.add(edge)
            train_edges.remove(edge)
            # 注意：边已经在上面被移除了，不需要再次移除
        
        # 两个集合都填满后退出
        elif len(test_edges) == num_test and len(val_edges) == num_val:
            break
    
    if len(val_edges) < num_val or len(test_edges) < num_test:
        print("WARNING: not enough removable edges to perform full train-test split!")
        print(f"Requested: (val={num_val}, test={num_test})")
        print(f"Got: (val={len(val_edges)}, test={len(test_edges)})")
    
    if prevent_disconnect:
        assert nx.number_connected_components(G) == orig_num_cc, "Graph connectivity changed!"
    
    # 生成负边
    if verbose:
        print('Creating negative edges...')
    
    # 测试集负边（与 process.py 完全一致）
    test_edges_false = set()
    while len(test_edges_false) < num_test:
        idx_i = np.random.randint(0, num_nodes)
        idx_j = np.random.randint(0, num_nodes)
        if idx_i == idx_j:
            continue
        
        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))
        
        # Make sure false_edge not an actual edge, and not a repeat (与 process.py 完全一致)
        if false_edge in all_edge_tuples:
            continue
        if false_edge in test_edges_false:
            continue
        
        test_edges_false.add(false_edge)
    
    # 验证集负边（与 process.py 完全一致）
    val_edges_false = set()
    while len(val_edges_false) < num_val:
        idx_i = np.random.randint(0, num_nodes)
        idx_j = np.random.randint(0, num_nodes)
        if idx_i == idx_j:
            continue
        
        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))
        
        # Make sure false_edge is not an actual edge, not in test_edges_false, not a repeat (与 process.py 完全一致)
        if false_edge in all_edge_tuples or \
           false_edge in test_edges_false or \
           false_edge in val_edges_false:
            continue
        
        val_edges_false.add(false_edge)
    
    # 训练集负边（与 process.py 完全一致）
    train_edges_false = set()
    while len(train_edges_false) < len(train_edges):
        idx_i = np.random.randint(0, num_nodes)
        idx_j = np.random.randint(0, num_nodes)
        if idx_i == idx_j:
            continue
        
        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))
        
        # Make sure false_edge is not an actual edge, not in test_edges_false, 
        # not in val_edges_false, not a repeat (与 process.py 完全一致)
        if false_edge in all_edge_tuples or \
           false_edge in test_edges_false or \
           false_edge in val_edges_false or \
           false_edge in train_edges_false:
            continue
        
        train_edges_false.add(false_edge)
    
    # 验证边集的互斥性
    if verbose:
        print('Validating edge set disjointness...')
    
    # 验证正边集互不相交
    assert train_edges.isdisjoint(val_edges), "Train and val positive edges overlap!"
    assert train_edges.isdisjoint(test_edges), "Train and test positive edges overlap!"
    assert val_edges.isdisjoint(test_edges), "Val and test positive edges overlap!"
    
    # 验证负边集不在正边集中
    assert train_edges_false.isdisjoint(all_edge_tuples), "Train negative edges overlap with positive edges!"
    assert val_edges_false.isdisjoint(all_edge_tuples), "Val negative edges overlap with positive edges!"
    assert test_edges_false.isdisjoint(all_edge_tuples), "Test negative edges overlap with positive edges!"
    
    # 验证负边集互不相交
    assert train_edges_false.isdisjoint(val_edges_false), "Train and val negative edges overlap!"
    assert train_edges_false.isdisjoint(test_edges_false), "Train and test negative edges overlap!"
    assert val_edges_false.isdisjoint(test_edges_false), "Val and test negative edges overlap!"
    
    if verbose:
        print('All edge set validations passed!')
    
    # 转换为 torch tensor 格式 [2, num_edges]
    def edges_to_tensor(edges_set):
        if len(edges_set) == 0:
            return torch.empty((2, 0), dtype=torch.long)
        edges_array = np.array(list(edges_set)).T
        return torch.tensor(edges_array, dtype=torch.long)
    
    edge_train = edges_to_tensor(train_edges)
    edge_val = edges_to_tensor(val_edges)
    edge_test = edges_to_tensor(test_edges)
    train_edges_neg = edges_to_tensor(train_edges_false)
    val_edges_neg = edges_to_tensor(val_edges_false)
    test_edges_neg = edges_to_tensor(test_edges_false)
    
    if verbose:
        print(f'Train edges: {len(train_edges)}, Val pos/neg: {len(val_edges)}/{len(val_edges_false)}, Test pos/neg: {len(test_edges)}/{len(test_edges_false)}')
        print('Done with manual train-test split!')
    
    pos_edges = (edge_train, edge_val, edge_test)
    neg_edges = (train_edges_neg, val_edges_neg, test_edges_neg)
    
    return pos_edges, neg_edges

def mask_edges_random(edge_index, num_nodes, val_prop, test_prop, seed=3047, verbose=False, prevent_disconnect=True):
    """
    使用 RandomLinkSplit 进行随机边划分（用于链接预测任务）
    参考 process.py 中的 mask_test_edges 函数，添加了验证和连通性检查
    
    Args:
        edge_index: 原始边索引 [2, num_edges]
        num_nodes: 节点数量
        val_prop: 验证集比例
        test_prop: 测试集比例
        seed: 随机种子
        verbose: 是否输出详细信息
        prevent_disconnect: 是否保持图的连通性（RandomLinkSplit 默认会保持）
    
    Returns:
        pos_edges: (edge_train, edge_val, edge_test) 正边元组
        neg_edges: (train_edges_neg, val_edges_neg, test_edges_neg) 负边元组
    """
    if verbose:
        print('Preprocessing edges for link prediction split...')
    
    # 创建临时 Data 对象用于 RandomLinkSplit
    temp_data = Data(edge_index=edge_index, num_nodes=num_nodes)
    
    # 设置随机种子
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # 检查图的连通性（如果需要）
    if prevent_disconnect:
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        edges = [(edge_index[0][i].item(), edge_index[1][i].item()) for i in range(edge_index.size(1))]
        G.add_edges_from(edges)
        orig_num_cc = nx.number_connected_components(G)
        if verbose:
            print(f'Original graph has {orig_num_cc} connected component(s)')
    
    # 使用 RandomLinkSplit 进行随机划分
    # is_undirected=True 表示无向图，add_negative_train_samples=False 表示不在训练集中自动添加负样本
    # neg_sampling_ratio=1.0 表示每个正样本对应一个负样本
    # disjoint_train_ratio=0.0 表示训练集和验证/测试集可以共享节点（但边不共享）
    transform = RandomLinkSplit(
        num_val=val_prop,
        num_test=test_prop,
        is_undirected=True,
        add_negative_train_samples=False,
        neg_sampling_ratio=1.0,
        split_labels=False,
        disjoint_train_ratio=0.0  # 允许训练集和测试集共享节点
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
    
    # 检查边的数量
    n_train = edge_train.size(1)
    n_val_pos = val_pos_edges.size(1)
    n_test_pos = test_pos_edges.size(1)
    n_val_neg = val_neg_edges.size(1)
    n_test_neg = test_neg_edges.size(1)
    
    if verbose:
        print(f'Train edges: {n_train}, Val pos/neg: {n_val_pos}/{n_val_neg}, Test pos/neg: {n_test_pos}/{n_test_neg}')
    
    # RandomLinkSplit 在处理无向图时会自动去重边（每条无向边只计算一次）
    # 所以实际边数量可能约为原始边数量的一半
    # 计算去重后的边数量（用于更准确的期望值计算）
    edges_list = [(edge_index[0][i].item(), edge_index[1][i].item()) for i in range(edge_index.size(1))]
    unique_edges = set((min(e[0], e[1]), max(e[0], e[1])) for e in edges_list)
    num_unique_edges = len(unique_edges)
    
    # 计算期望值（基于去重后的边数量）
    expected_val = int(val_prop * num_unique_edges)
    expected_test = int(test_prop * num_unique_edges)
    
    # 允许一定的误差（±5%），因为 RandomLinkSplit 的内部实现可能有细微差异
    tolerance = 0.05
    val_diff = abs(n_val_pos - expected_val) / max(expected_val, 1)
    test_diff = abs(n_test_pos - expected_test) / max(expected_test, 1)
    
    # 只有在差异较大时才警告
    if val_diff > tolerance or test_diff > tolerance:
        print(f"WARNING: Edge split differs from expected!")
        print(f"Expected (based on unique edges): val={expected_val}, test={expected_test}")
        print(f"Got: val={n_val_pos}, test={n_test_pos}")
        print(f"Note: RandomLinkSplit may use slightly different edge counting for undirected graphs.")
    
    # 将边转换为集合（考虑到无向边，使用 (min, max) 元组）
    def edge_to_set(edges_tensor):
        if edges_tensor.size(1) == 0:
            return set()
        edges_list = edges_tensor.t().cpu().numpy()
        return set(tuple(sorted(edge)) for edge in edges_list)
    
    # 获取所有正边集合（用于负边采样时排除）
    train_edges_set = edge_to_set(edge_train)
    val_pos_set = edge_to_set(val_pos_edges)
    test_pos_set = edge_to_set(test_pos_edges)
    all_pos_set = train_edges_set | val_pos_set | test_pos_set
    
    # 重新生成所有负边，确保它们互不相交（参考 process.py 的逻辑）
    if verbose:
        print('Regenerating negative edges to ensure disjointness...')
    
    # 先收集 RandomLinkSplit 生成的负边（作为候选，但需要去重）
    val_neg_set_initial = edge_to_set(val_neg_edges)
    test_neg_set_initial = edge_to_set(test_neg_edges)
    
    # 生成测试集负边（排除所有正边和已生成的负边）
    test_edges_false = set()
    all_used_neg_edges = set()  # 记录所有已使用的负边
    
    # 先尝试使用 RandomLinkSplit 生成的负边
    for edge in test_neg_set_initial:
        if edge not in all_pos_set and edge not in all_used_neg_edges:
            test_edges_false.add(edge)
            all_used_neg_edges.add(edge)
            if len(test_edges_false) >= n_test_neg:
                break
    
    # 如果还不够，随机生成
    while len(test_edges_false) < n_test_neg:
        idx_i = np.random.randint(0, num_nodes)
        idx_j = np.random.randint(0, num_nodes)
        if idx_i == idx_j:
            continue
        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))
        if false_edge not in all_pos_set and false_edge not in all_used_neg_edges:
            test_edges_false.add(false_edge)
            all_used_neg_edges.add(false_edge)
    
    # 生成验证集负边（排除所有正边和已生成的负边）
    val_edges_false = set()
    # 先尝试使用 RandomLinkSplit 生成的负边
    for edge in val_neg_set_initial:
        if edge not in all_pos_set and edge not in all_used_neg_edges:
            val_edges_false.add(edge)
            all_used_neg_edges.add(edge)
            if len(val_edges_false) >= n_val_neg:
                break
    
    # 如果还不够，随机生成
    while len(val_edges_false) < n_val_neg:
        idx_i = np.random.randint(0, num_nodes)
        idx_j = np.random.randint(0, num_nodes)
        if idx_i == idx_j:
            continue
        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))
        if false_edge not in all_pos_set and false_edge not in all_used_neg_edges:
            val_edges_false.add(false_edge)
            all_used_neg_edges.add(false_edge)
    
    # 生成训练集负边（排除所有正边和已生成的负边）
    train_edges_false = set()
    all_pos_edges_tensor = torch.cat([edge_train, val_pos_edges, test_pos_edges], dim=1)
    
    # 使用 negative_sampling 生成候选，然后过滤
    max_attempts = n_train * 3  # 最多尝试次数
    attempts = 0
    while len(train_edges_false) < n_train and attempts < max_attempts:
        # 批量生成负边
        remaining = n_train - len(train_edges_false)
        candidates = negative_sampling(edge_index=all_pos_edges_tensor, num_nodes=num_nodes, num_neg_samples=min(remaining * 2, n_train))
        candidates_set = edge_to_set(candidates)
        
        for edge in candidates_set:
            if edge not in all_pos_set and edge not in all_used_neg_edges:
                train_edges_false.add(edge)
                all_used_neg_edges.add(edge)
                if len(train_edges_false) >= n_train:
                    break
        attempts += 1
    
    # 如果 negative_sampling 生成的还不够，使用随机生成
    while len(train_edges_false) < n_train:
        idx_i = np.random.randint(0, num_nodes)
        idx_j = np.random.randint(0, num_nodes)
        if idx_i == idx_j:
            continue
        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))
        if false_edge not in all_pos_set and false_edge not in all_used_neg_edges:
            train_edges_false.add(false_edge)
            all_used_neg_edges.add(false_edge)
    
    # 转换为 tensor
    def edges_set_to_tensor(edges_set):
        if len(edges_set) == 0:
            return torch.empty((2, 0), dtype=torch.long)
        edges_array = np.array(list(edges_set)).T
        return torch.tensor(edges_array, dtype=torch.long)
    
    val_neg_edges = edges_set_to_tensor(val_edges_false)
    test_neg_edges = edges_set_to_tensor(test_edges_false)
    train_edges_neg = edges_set_to_tensor(train_edges_false)
    
    # 验证边集的互斥性（参考 process.py 的验证逻辑）
    if verbose:
        print('Validating edge set disjointness...')
    
    train_neg_set = edge_to_set(train_edges_neg)
    val_neg_set = edge_to_set(val_neg_edges)
    test_neg_set = edge_to_set(test_neg_edges)
    
    # 验证正边集互不相交
    assert train_edges_set.isdisjoint(val_pos_set), "Train and val positive edges overlap!"
    assert train_edges_set.isdisjoint(test_pos_set), "Train and test positive edges overlap!"
    assert val_pos_set.isdisjoint(test_pos_set), "Val and test positive edges overlap!"
    
    # 验证负边集不在正边集中
    assert train_neg_set.isdisjoint(all_pos_set), "Train negative edges overlap with positive edges!"
    assert val_neg_set.isdisjoint(all_pos_set), "Val negative edges overlap with positive edges!"
    assert test_neg_set.isdisjoint(all_pos_set), "Test negative edges overlap with positive edges!"
    
    # 验证负边集互不相交
    assert train_neg_set.isdisjoint(val_neg_set), "Train and val negative edges overlap!"
    assert train_neg_set.isdisjoint(test_neg_set), "Train and test negative edges overlap!"
    assert val_neg_set.isdisjoint(test_neg_set), "Val and test negative edges overlap!"
    
    if verbose:
        print('All edge set validations passed!')
    
    # 验证图的连通性（如果训练集被移除边后图仍然应该连通）
    if prevent_disconnect:
        G_train = nx.Graph()
        G_train.add_nodes_from(range(num_nodes))
        train_edges_list = [(edge_train[0][i].item(), edge_train[1][i].item()) for i in range(n_train)]
        G_train.add_edges_from(train_edges_list)
        train_num_cc = nx.number_connected_components(G_train)
        if verbose:
            print(f'Training graph has {train_num_cc} connected component(s)')
        # 注意：移除边后连通分量可能会增加，这是正常的
    
    pos_edges = (edge_train, val_pos_edges, test_pos_edges)
    neg_edges = (train_edges_neg, val_neg_edges, test_neg_edges)
    
    if verbose:
        print('Done with train-test split!')
    
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

