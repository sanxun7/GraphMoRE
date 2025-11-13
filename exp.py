import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from models import *
from hppo_mrs import H_PPO_MRS
from backbone import GNNClassifier
# 兼容导入：优先使用本项目的 reward_mrs；如不可用，回落到 graph_level 或内置最小实现
try:
    from reward_mrs import Normalization, compute_adv_ret_one_step, compute_adv_ret_seq
except Exception:
    try:
        from graph_level.reward import Normalization  # type: ignore
    except Exception:
        class _RunningMeanStd:
            def __init__(self, shape):
                import numpy as _np
                self.n = 0
                self.mean = _np.zeros(shape)
                self.S = _np.zeros(shape)
                self.std = (_np.zeros(shape))
            def update(self, x):
                import numpy as _np
                x = _np.array(x)
                self.n += 1
                if self.n == 1:
                    self.mean = x
                    self.std = x
                else:
                    old_mean = self.mean.copy()
                    self.mean = old_mean + (x - old_mean) / self.n
                    self.S = self.S + (x - old_mean) * (x - self.mean)
                    self.std = (self.S / self.n) ** 0.5
        class Normalization:
            def __init__(self, shape):
                self.running_ms = _RunningMeanStd(shape)
            def __call__(self, x, update=True):
                if update:
                    self.running_ms.update(x)
                return (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)
    def compute_adv_ret_one_step(critic, state, reward, gamma=0.99):
        import torch as _torch
        with _torch.no_grad():
            v = critic(state)
        adv = reward - v
        ret = reward
        return adv, ret
    def compute_adv_ret_seq(critic, states_seq, rewards_seq, gamma=0.99, lam=0.95):
        import torch as _torch
        T = len(states_seq)
        with _torch.no_grad():
            values = [critic(s).detach() for s in states_seq]
            next_values = values[1:] + [values[-1]]
        adv = _torch.zeros_like(values[0])
        adv_list = []
        for t in reversed(range(T)):
            delta = rewards_seq[t] + gamma * next_values[t] - values[t]
            adv = delta + gamma * lam * adv
            adv_list.insert(0, adv)
        ret_list = [adv_list[t] + values[t] for t in range(T)]
        return _torch.cat(adv_list, dim=0), _torch.cat(ret_list, dim=0)
from utils import cal_accuracy, cal_F1, cal_AUC_AP, cal_shortest_dis
from data_factory import load_data, mask_edges, mask_edges_random, mask_edges_manual, load_synthetic_data
from logger import create_logger
from geoopt.optim import RiemannianAdam
import time
import os
from torch_geometric.utils import negative_sampling
import pickle

class Exp:
    def __init__(self, configs):
        self.configs = configs
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def train(self):
        logger = create_logger(self.configs.log_path)
        device = self.device
        if "synthetic" in self.configs.dataset:
            features, in_features, labels, edge_index, neg_edge, masks, n_classes = load_synthetic_data(self.configs.root_path, self.configs.dataset)
        else:
            features, in_features, labels, edge_index, neg_edge, masks, n_classes = load_data(self.configs.root_path, self.configs.dataset)
        edge_index = edge_index.to(device)
        neg_edge = neg_edge.to(device)
        features = features.to(device)
        labels = labels.to(device)
        self.masks = masks
        self.in_features = in_features
        self.configs.in_features = in_features
        self.n_classes = n_classes
        self.labels = labels
        self.edge_index = edge_index
        self.neg_edge = neg_edge
        self.features = features
        self.dis_shortest = cal_shortest_dis(self.edge_index)

        val_prop = 0.05
        test_prop = 0.1
        # 对于标准数据集（Cora, Citeseer, Pubmed, Coauthor-CS, Coauthor-Physics, github, airport, photo, computers, BlogCatalog, Flickr, Facebook），根据配置选择划分方式
        # 对于合成数据，使用顺序划分（保持原有逻辑）
        if "synthetic" in self.configs.dataset:
            self.pos_edges, self.neg_edges = mask_edges(self.edge_index, self.neg_edge, val_prop, test_prop)
        else:
            num_nodes = self.features.shape[0]
            edge_index_cpu = self.edge_index.cpu()
            
            # 根据配置选择划分方式
            split_method = getattr(self.configs, 'edge_split_method', 'random')  # 默认使用 RandomLinkSplit
            
            if split_method == 'manual':
                # 使用手动实现的划分（类似 process.py）
                logger.info("Using manual edge splitting method")
                pos_edges, neg_edges = mask_edges_manual(edge_index_cpu, num_nodes, val_prop, test_prop, 
                                                         seed=3047, verbose=False, prevent_disconnect=True)
            else:
                # 使用 RandomLinkSplit（默认）
                logger.info("Using RandomLinkSplit edge splitting method")
                pos_edges, neg_edges = mask_edges_random(edge_index_cpu, num_nodes, val_prop, test_prop, 
                                                          seed=3047, verbose=False, prevent_disconnect=True)
            
            # 将结果移回 device
            self.pos_edges = tuple(edges.to(device) for edges in pos_edges)
            self.neg_edges = tuple(edges.to(device) for edges in neg_edges)
        self.subgraph_sampler = Sampler(method = "ego", sample_hop = self.configs.sample_hop, dataset = self.configs.dataset, configs = self.configs)


        if self.configs.downstream_task == "NC":
            accs = []
            wf1s = []
            mf1s = []

        elif self.configs.downstream_task == "LP":
            aucs = []
            aps = []


        if self.configs.downstream_task == 'LP':
            self.subgraph_feature, self.subgraph_edge_index, self.subgraph_batch = self.subgraph_sampler.sample(self.features, self.pos_edges[0], "LP")
        else:
            self.subgraph_feature, self.subgraph_edge_index, self.subgraph_batch = self.subgraph_sampler.sample(self.features, self.edge_index, "NC")

        for exp_iter in range(self.configs.exp_iters):
            logger.info(f"\ntrain iters {exp_iter}")

            model = Experts(init_curvs=self.configs.init_curvs, in_dim=in_features, hidden_dim=self.configs.hidden_features, out_dim=self.configs.embed_features, learnable=True, num_factors_cls = self.configs.num_factors_cls).to(self.device)
            model_gating = Gating(in_dim=in_features, hidden_dim=self.configs.hidden_features, out_dim=self.configs.embed_features, num_experts=self.configs.num_factors, configs = self.configs).to(self.device)

            # 可选：初始化多分辨率 H-PPO 策略（仅接入，训练由上层按需组织）
            rl_agent = None
            rl_optim = None
            # 奖励标准化器（仅在使用 H-PPO 时初始化）
            self.rl_norm = None
            if getattr(self.configs, 'use_hppo', False):
                rl_agent = H_PPO_MRS(
                    emb_dim=self.configs.embed_features,
                    num_hops=len(self.configs.sample_hop),
                    ensemble_num=1,
                    penalty_alpha_d=getattr(self.configs, 'hppo_penalty_alpha_d', 0.0),
                    penalty_alpha_c=getattr(self.configs, 'hppo_penalty_alpha_c', 0.0),
                    eps_clip_d=getattr(self.configs, 'hppo_eps_clip_d', 0.2),
                    eps_clip_c=getattr(self.configs, 'hppo_eps_clip_c', 0.2),
                    coeff_critic=getattr(self.configs, 'hppo_coeff_critic', 0.5),
                    coeff_entropy_d=getattr(self.configs, 'hppo_coeff_ent_d', 1e-3),
                    max_norm_grad=getattr(self.configs, 'hppo_max_norm_grad', 5.0),
                    init_log_std=getattr(self.configs, 'hppo_init_log_std', -2.0),
                    device=self.device,
                )
                rl_optim = torch.optim.Adam([
                    {"params": rl_agent.actor_ds[0].parameters(), "lr": self.configs.hppo_actor_d_lr, "name": "actor_d"},
                    {"params": rl_agent.actor_cs[0].parameters(), "lr": self.configs.hppo_actor_c_lr, "name": "actor_c"},
                    {"params": rl_agent.critic.parameters(), "lr": self.configs.hppo_critic_lr, "name": "critic"},
                ])
                self.rl_norm = Normalization(shape=1)

            logger.info("--------------------------Training Start-------------------------")
            if self.configs.downstream_task == 'NC':
                test_auc, test_ap, _ = self.train_lp(model, model_gating, self.pos_edges, self.neg_edges, logger, rl_agent, rl_optim)
                best_val, test_acc, test_weighted_f1, test_macro_f1, best_epoch = self.train_cls(model, model_gating, logger, rl_agent, rl_optim)
                logger.info(f"best_epoch={best_epoch}")
                logger.info(
                    f"test_accuracy={test_acc.item() * 100: .2f}%")
                logger.info(
                    f"weighted_f1={test_weighted_f1 * 100: .2f}%, macro_f1={test_macro_f1 * 100: .2f}%")
                accs.append(test_acc.item())
                wf1s.append(test_weighted_f1)
                mf1s.append(test_macro_f1)

            elif self.configs.downstream_task == 'LP':
                test_auc, test_ap, best_epoch = self.train_lp(model, model_gating, self.pos_edges, self.neg_edges, logger, rl_agent, rl_optim)
                logger.info(f"best_epoch={best_epoch}")
                logger.info(
                    f"test_auc={test_auc * 100: .2f}%, test_ap={test_ap * 100: .2f}%")
                aucs.append(test_auc)
                aps.append(test_ap)
            else:
                raise NotImplementedError

        if self.configs.downstream_task == "NC":
            logger.info(f"----NC Task----")
            logger.info(f"test acc: {np.mean(accs)}~{np.std(accs)}")
            logger.info(f"test weighted-f1: {np.mean(wf1s)}~{np.std(wf1s)}")
            logger.info(f"test macro-f1: {np.mean(mf1s)}~{np.std(mf1s)}")
        elif self.configs.downstream_task == "LP":
            logger.info(f"----LP Task----")
            logger.info(f"test AUC: {np.mean(aucs)}~{np.std(aucs)}")
            logger.info(f"test AP: {np.mean(aps)}~{np.std(aps)}")

    def cal_cls_loss(self, model, edge_index, mask, features, labels):
        out = model(features, edge_index)
        loss = F.cross_entropy(out[mask], labels[mask])
        acc = cal_accuracy(out[mask], labels[mask])
        weighted_f1, macro_f1 = cal_F1(out[mask].detach().cpu(), labels[mask].detach().cpu())
        return loss, acc, weighted_f1, macro_f1

    def train_cls(self, model, model_gating, logger, rl_agent=None, rl_optim=None):
        """masks = (train, val, test)"""
        self.configs.coef_dis = 1e-4
        d = self.configs.num_factors_cls * self.configs.embed_features
        model_cls = GNNClassifier(backbone=self.configs.backbone, n_layers=2, in_features=self.in_features + d,
                                    hidden_features=self.configs.hidden_features_cls, out_features=self.n_classes,
                                    n_heads=self.configs.n_heads, drop_edge=self.configs.drop_edge_cls, 
                                    drop_node=self.configs.drop_cls).to(self.device)
        optimizer_cls = torch.optim.Adam(model_cls.parameters(), lr=self.configs.lr_cls, weight_decay=self.configs.w_decay_cls)
        r_optim = RiemannianAdam(model.parameters(), lr=self.configs.lr_Riemann, weight_decay=self.configs.w_decay, stabilize=100)
        optimizer_gating = torch.optim.Adam(model_gating.parameters(), lr=self.configs.lr_gating, weight_decay=self.configs.w_decay_gating)
        best_acc = 0.
        best_epoch = 0
        early_stop_count = 0
        for epoch in range(self.configs.epochs_cls + 1):
            if rl_agent is not None:
                rl_agent.train_or_eval('train')
            # RL 参数退火（对齐 graph_level）：lr、entropy coeff、log_std
            if rl_agent is not None and rl_optim is not None:
                total_epochs = max(1, self.configs.epochs_cls)
                descend_decay_frac = 1.0 - (epoch - 1) / total_epochs
                if getattr(self.configs, 'hppo_policy_decay', 'none') == 'down':
                    for pg in rl_optim.param_groups:
                        if pg.get('name') == 'actor_d':
                            pg['lr'] = descend_decay_frac * self.configs.hppo_actor_d_lr
                        elif pg.get('name') == 'actor_c':
                            pg['lr'] = descend_decay_frac * self.configs.hppo_actor_c_lr
                        elif pg.get('name') == 'critic':
                            pg['lr'] = descend_decay_frac * self.configs.hppo_critic_lr
                # 熵系数退火
                rl_agent.coeff_ent_d = self.configs.hppo_coeff_ent_d * descend_decay_frac
                # log_std 退火到 -5
                new_log_std = self.configs.hppo_init_log_std + (-5 - self.configs.hppo_init_log_std) * (epoch - 1) / total_epochs
                for i in range(rl_agent.ensemble_num):
                    rl_agent.actor_cs[i].log_std.data = torch.ones(self.configs.embed_features, device=self.device) * new_log_std
            now_time = time.time()
            model_cls.train()
            model.train()
            model_gating.train()
            optimizer_cls.zero_grad()
            r_optim.zero_grad()
            optimizer_gating.zero_grad()
            
            embeddings = model.encode(self.features, self.edge_index, self.configs.dataset)

            if rl_agent is not None:
                # ===== 多步序列收集（步数=尺度数） =====
                T = len(self.configs.sample_hop)
                # 初始 per-hop 表示（无 RL 干预）
                _, base_info = model_gating(self.subgraph_feature, self.subgraph_edge_index, self.subgraph_batch,
                                            rl_return_info=True)
                cur_reps = base_info['reps']  # List[(N,D)]
                states_seq, actions_d, actions_c, logps_d, logps_c, rewards_seq = [], [], [], [], [], []

                # 初始损失/失真（用于 r_0 = L0 - L1）
                ew_base0, prev_distortion = model_gating(
                    self.subgraph_feature, self.subgraph_edge_index, self.subgraph_batch,
                    embeddings, self.dis_shortest, self.configs.embed_features, self.edge_index,
                    override_reps_per_hop=cur_reps
                )
                ew_base0_rep = ew_base0.repeat_interleave(self.configs.embed_features, dim=1)
                feat_base0 = torch.concat([self.features, embeddings * ew_base0_rep], -1)
                prev_loss, _, _, _ = self.cal_cls_loss(model_cls, self.edge_index, self.masks[0], feat_base0, self.labels)

                for t in range(T):
                    # 状态（动作前）
                    state_t = rl_agent.build_state_from_reps(cur_reps)
                    # 采样动作并应用（不重复编码）
                    ew_t, loss_distortion, rl_info = model_gating(
                        self.subgraph_feature, self.subgraph_edge_index, self.subgraph_batch,
                        embeddings, self.dis_shortest, self.configs.embed_features, self.edge_index,
                        rl_agent=rl_agent, rl_return_info=True, rl_collect=True,
                        override_reps_per_hop=cur_reps
                    )
                    # 计算当前损失（仅任务损失用于奖励）
                    ew_t_rep = ew_t.repeat_interleave(self.configs.embed_features, dim=1)
                    feat_t = torch.concat([self.features, embeddings * ew_t_rep], -1)
                    loss_t, _, _, _ = self.cal_cls_loss(model_cls, self.edge_index, self.masks[0], feat_t, self.labels)

                    # 奖励：默认 r_t = prev_loss - loss_t；若配置包含失真，则计入 coef_dis * distortion
                    if getattr(self.configs, 'hppo_reward_with_distortion', False):
                        r_t = (prev_loss.detach() + self.configs.coef_dis * prev_distortion.detach()) - \
                              (loss_t.detach() + self.configs.coef_dis * loss_distortion.detach())
                    else:
                        r_t = (prev_loss.detach() - loss_t.detach())
                    # 标准化为 (N,) 向量（用同一标量或广播均可，这里按节点均值广播）
                    r_vec = torch.full((state_t.size(0),), float(self.rl_norm(r_t.item())), device=self.device, dtype=state_t.dtype)

                    states_seq.append(state_t.detach())
                    actions_d.append(rl_info['action_d'])
                    actions_c.append(rl_info['action_c'])
                    logps_d.append(rl_info['logprob_d'])
                    logps_c.append(rl_info['logprob_c'])
                    rewards_seq.append(r_vec)

                    # 下一步初始化
                    cur_reps = rl_info['next_reps']
                    prev_loss = loss_t
                    prev_distortion = loss_distortion

                # 计算优势/回报（GAE）并展平
                adv_all, ret_all = compute_adv_ret_seq(rl_agent.critic, states_seq, rewards_seq,
                                                       gamma=self.configs.hppo_gamma, lam=self.configs.hppo_lam)
                state_all = torch.cat(states_seq, dim=0)
                a_d_all = torch.cat(actions_d, dim=0)
                a_c_all = torch.cat(actions_c, dim=0)
                lp_d_all = torch.cat(logps_d, dim=0)
                lp_c_all = torch.cat(logps_c, dim=0)

                # PPO 更新
                experience = (state_all, a_d_all, lp_d_all, a_c_all, lp_c_all, adv_all, ret_all)
                args_like = {
                    'minibatch_size': self.configs.hppo_minibatch_size,
                    'policy_update_nums': self.configs.hppo_update_nums,
                    'target_kl_d': self.configs.hppo_target_kl_d,
                    'target_kl_c': self.configs.hppo_target_kl_c,
                }
                rl_agent.train_step(args_like, 0, experience, rl_optim)

                # 用最终（多步后）表示计算本轮训练指标（含失真正则）
                # prev_loss 即最后一步任务损失；对应的 features 即 feat_t
                loss = prev_loss + self.configs.coef_dis * loss_distortion
                _, acc, weighted_f1, macro_f1 = self.cal_cls_loss(model_cls, self.edge_index, self.masks[0], feat_t, self.labels)
            else:
                # 原路径（无 RL）
                experts_weight, loss_distortion = model_gating(
                    self.subgraph_feature, self.subgraph_edge_index, self.subgraph_batch,
                    embeddings, self.dis_shortest, self.configs.embed_features, self.edge_index
                )
                ew = experts_weight.repeat_interleave(self.configs.embed_features, dim=1)
                emb = embeddings * ew
                features = torch.concat([self.features, emb], -1)
                loss, acc, weighted_f1, macro_f1 = self.cal_cls_loss(model_cls, self.edge_index, self.masks[0], features, self.labels)
                loss = loss + self.configs.coef_dis * loss_distortion

            loss.backward()
            optimizer_cls.step()
            r_optim.step()
            optimizer_gating.step()
            logger.info(f"Epoch {epoch}: train_loss={loss.item()}, train_accuracy={acc}, time={time.time()-now_time}")

            # 备注：多步RL已在上方执行PPO更新；此处不再重复进行单步更新

            if epoch % self.configs.eval_freq == 0:
                model_cls.eval()
                model.eval()
                model_gating.eval()
                if rl_agent is not None:
                    rl_agent.train_or_eval('eval')

                embeddings = model.encode(self.features, self.edge_index)
                if rl_agent is not None:
                    T = len(self.configs.sample_hop)
                    _, base_info_val = model_gating(self.subgraph_feature, self.subgraph_edge_index, self.subgraph_batch,
                                                    rl_return_info=True)
                    cur_reps = base_info_val['reps']
                    experts_weight = None
                    for t in range(T):
                        ew_t, rl_info_val = model_gating(
                            self.subgraph_feature, self.subgraph_edge_index, self.subgraph_batch,
                            rl_agent=rl_agent, rl_return_info=True, rl_collect=False,
                            override_reps_per_hop=cur_reps
                        )
                        cur_reps = rl_info_val['next_reps']
                        experts_weight = ew_t
                else:
                    experts_weight = model_gating(
                        self.subgraph_feature, self.subgraph_edge_index, self.subgraph_batch,
                    )
                experts_weight = experts_weight.repeat_interleave(self.configs.embed_features, dim=1)
                embeddings = embeddings * experts_weight
                features = torch.concat([self.features, embeddings], -1)
                _, acc, weighted_f1, macro_f1 = self.cal_cls_loss(model_cls, self.edge_index, self.masks[1], features, self.labels)
                logger.info(f"Epoch {epoch}: val_accuracy={acc}, val_wf1={weighted_f1}, val_mf1={macro_f1}")
                if acc > best_acc:
                    best_acc = acc
                    best_epoch = epoch
                    early_stop_count = 0
                    # Test
                    _, test_acc, test_weighted_f1, test_macro_f1 = self.cal_cls_loss(model_cls, self.edge_index, self.masks[2], features, self.labels)
                else:
                    early_stop_count += 1
                    if early_stop_count > self.configs.patience_cls:
                        break
                if epoch < self.configs.min_epoch_cls:
                    early_stop_count = 0
        return best_acc, test_acc, test_weighted_f1, test_macro_f1, best_epoch

    def cal_lp_loss(self, embeddings, experts_weight, decoder, pos_edges, neg_edges):
        pos_diff = (embeddings[pos_edges[0]] - embeddings[pos_edges[1]])**2
        pos_diff = pos_diff.reshape(pos_diff.shape[0], pos_diff.shape[1]//self.configs.embed_features, self.configs.embed_features).sum(dim=2)
        pos_weights = F.softmax(experts_weight[pos_edges[0]] * experts_weight[pos_edges[1]], dim=1)   
        pos_scores = decoder(torch.sum(pos_diff * pos_weights, -1))

        neg_diff = (embeddings[neg_edges[0]] - embeddings[neg_edges[1]])**2
        neg_diff = neg_diff.reshape(neg_diff.shape[0], neg_diff.shape[1]//self.configs.embed_features, self.configs.embed_features).sum(dim=2)
        neg_weights = F.softmax(experts_weight[neg_edges[0]] * experts_weight[neg_edges[1]], dim=1)   
        neg_scores = decoder(torch.sum(neg_diff * neg_weights, -1))

        loss = F.binary_cross_entropy(pos_scores.clip(0.01, 0.99), torch.ones_like(pos_scores)) + \
                F.binary_cross_entropy(neg_scores.clip(0.01, 0.99), torch.zeros_like(neg_scores))
        label = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        preds = list(pos_scores.detach().cpu().numpy()) + list(neg_scores.detach().cpu().numpy())
        auc, ap = cal_AUC_AP(preds, label)
        return loss, auc, ap

    def train_lp(self, model, model_gating, pos_edges, neg_edges, logger, rl_agent=None, rl_optim=None):
        r_optim = RiemannianAdam(model.parameters(), lr=self.configs.lr_Riemann, weight_decay=self.configs.w_decay, stabilize=100)
        optimizer_gating = torch.optim.Adam(model_gating.parameters(), lr=self.configs.lr_gating, weight_decay=self.configs.w_decay_gating)
        
        decoder = FermiDiracDecoder(self.configs.r, self.configs.t).to(self.device)
        best_ap = 0
        best_epoch = 0
        early_stop_count = 0
        for epoch in range(self.configs.epochs_lp + 1):
            if rl_agent is not None:
                rl_agent.train_or_eval('train')
            # RL 参数退火（对齐 graph_level）：lr、entropy coeff、log_std
            if rl_agent is not None and rl_optim is not None:
                total_epochs = max(1, self.configs.epochs_lp)
                descend_decay_frac = 1.0 - (epoch - 1) / total_epochs
                if getattr(self.configs, 'hppo_policy_decay', 'none') == 'down':
                    for pg in rl_optim.param_groups:
                        if pg.get('name') == 'actor_d':
                            pg['lr'] = descend_decay_frac * self.configs.hppo_actor_d_lr
                        elif pg.get('name') == 'actor_c':
                            pg['lr'] = descend_decay_frac * self.configs.hppo_actor_c_lr
                        elif pg.get('name') == 'critic':
                            pg['lr'] = descend_decay_frac * self.configs.hppo_critic_lr
                rl_agent.coeff_ent_d = self.configs.hppo_coeff_ent_d * descend_decay_frac
                new_log_std = self.configs.hppo_init_log_std + (-5 - self.configs.hppo_init_log_std) * (epoch - 1) / total_epochs
                for i in range(rl_agent.ensemble_num):
                    rl_agent.actor_cs[i].log_std.data = torch.ones(self.configs.embed_features, device=self.device) * new_log_std
            t = time.time()
            model.train()
            model_gating.train()
            r_optim.zero_grad()
            optimizer_gating.zero_grad()

            embeddings = model(self.features, pos_edges[0])
            neg_edge_train = neg_edges[0][:, np.random.randint(0, neg_edges[0].shape[1], pos_edges[0].shape[1])]

            if rl_agent is not None:
                # ===== 多步序列收集 =====
                T = len(self.configs.sample_hop)
                _, base_info = model_gating(self.subgraph_feature, self.subgraph_edge_index, self.subgraph_batch,
                                            rl_return_info=True)
                cur_reps = base_info['reps']
                states_seq, actions_d, actions_c, logps_d, logps_c, rewards_seq = [], [], [], [], [], []

                # 初始 loss/失真
                ew_base0, prev_distortion = model_gating(
                    self.subgraph_feature, self.subgraph_edge_index, self.subgraph_batch,
                    embeddings, self.dis_shortest, self.configs.embed_features, pos_edges[0],
                    override_reps_per_hop=cur_reps
                )
                loss_prev, _, _ = self.cal_lp_loss(embeddings, ew_base0, decoder, pos_edges[0], neg_edge_train)

                for t in range(T):
                    state_t = rl_agent.build_state_from_reps(cur_reps)
                    ew_t, loss_distortion, rl_info = model_gating(
                        self.subgraph_feature, self.subgraph_edge_index, self.subgraph_batch,
                        embeddings, self.dis_shortest, self.configs.embed_features, pos_edges[0],
                        rl_agent=rl_agent, rl_return_info=True, rl_collect=True,
                        override_reps_per_hop=cur_reps
                    )
                    loss_t, _, _ = self.cal_lp_loss(embeddings, ew_t, decoder, pos_edges[0], neg_edge_train)
                    if getattr(self.configs, 'hppo_reward_with_distortion', False):
                        r_t = (loss_prev.detach() + self.configs.coef_dis * prev_distortion.detach()) - \
                              (loss_t.detach() + self.configs.coef_dis * loss_distortion.detach())
                    else:
                        r_t = (loss_prev.detach() - loss_t.detach())
                    r_vec = torch.full((state_t.size(0),), float(self.rl_norm(r_t.item())), device=self.device, dtype=state_t.dtype)

                    states_seq.append(state_t.detach())
                    actions_d.append(rl_info['action_d'])
                    actions_c.append(rl_info['action_c'])
                    logps_d.append(rl_info['logprob_d'])
                    logps_c.append(rl_info['logprob_c'])
                    rewards_seq.append(r_vec)

                    cur_reps = rl_info['next_reps']
                    loss_prev = loss_t
                    prev_distortion = loss_distortion

                adv_all, ret_all = compute_adv_ret_seq(rl_agent.critic, states_seq, rewards_seq,
                                                       gamma=self.configs.hppo_gamma, lam=self.configs.hppo_lam)
                state_all = torch.cat(states_seq, dim=0)
                a_d_all = torch.cat(actions_d, dim=0)
                a_c_all = torch.cat(actions_c, dim=0)
                lp_d_all = torch.cat(logps_d, dim=0)
                lp_c_all = torch.cat(logps_c, dim=0)

                experience = (state_all, a_d_all, lp_d_all, a_c_all, lp_c_all, adv_all, ret_all)
                args_like = {
                    'minibatch_size': self.configs.hppo_minibatch_size,
                    'policy_update_nums': self.configs.hppo_update_nums,
                    'target_kl_d': self.configs.hppo_target_kl_d,
                    'target_kl_c': self.configs.hppo_target_kl_c,
                }
                rl_agent.train_step(args_like, 0, experience, rl_optim)

                # 本轮训练 loss 使用最后一步的 loss_t（含失真正则）
                loss = loss_prev + self.configs.coef_dis * loss_distortion
                # 同步计算训练期 AUC/AP 以便日志打印
                _, auc, ap = self.cal_lp_loss(embeddings, ew_t, decoder, pos_edges[0], neg_edge_train)
            else:
                experts_weight, loss_distortion = model_gating(
                    self.subgraph_feature, self.subgraph_edge_index, self.subgraph_batch,
                    embeddings, self.dis_shortest, self.configs.embed_features, pos_edges[0]
                )
                loss, auc, ap = self.cal_lp_loss(embeddings, experts_weight, decoder, pos_edges[0], neg_edge_train)
                loss = loss + self.configs.coef_dis * loss_distortion
            loss.backward()
            r_optim.step()
            optimizer_gating.step()
            logger.info(f"Epoch {epoch}: train_loss={loss.item()}, train_AUC={auc}, train_AP={ap}, time={time.time() - t}")

            # 备注：多步RL已在上方执行PPO更新；此处不再重复进行单步更新
            if epoch % self.configs.eval_freq == 0:
                model.eval()
                model_gating.eval()
                if rl_agent is not None:
                    rl_agent.train_or_eval('eval')
                embeddings = model(self.features, pos_edges[0])
                if rl_agent is not None:
                    T = len(self.configs.sample_hop)
                    _, base_info_val = model_gating(self.subgraph_feature, self.subgraph_edge_index, self.subgraph_batch,
                                                    rl_return_info=True)
                    cur_reps = base_info_val['reps']
                    experts_weight = None
                    for t in range(T):
                        ew_t, rl_info_val = model_gating(
                            self.subgraph_feature, self.subgraph_edge_index, self.subgraph_batch,
                            rl_agent=rl_agent, rl_return_info=True, rl_collect=False,
                            override_reps_per_hop=cur_reps
                        )
                        cur_reps = rl_info_val['next_reps']
                        experts_weight = ew_t
                else:
                    experts_weight = model_gating(
                        self.subgraph_feature, self.subgraph_edge_index, self.subgraph_batch,
                    )

                _, auc, ap = self.cal_lp_loss(embeddings, experts_weight, decoder, pos_edges[1], neg_edges[1])
                logger.info(f"Epoch {epoch}: val_AUC={auc}, val_AP={ap}")
                if ap > best_ap:
                    best_ap = ap
                    best_epoch = epoch
                    early_stop_count = 0
                    # Test
                    _, test_auc, test_ap = self.cal_lp_loss(embeddings, experts_weight, decoder, pos_edges[2], neg_edges[2])
                else:
                    early_stop_count += 1
                    if early_stop_count > self.configs.patience_lp:
                        break
                if epoch < self.configs.min_epoch_lp:
                    early_stop_count = 0

        return test_auc, test_ap, best_epoch
            
        
