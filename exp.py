import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from models import *
from backbone import GNNClassifier
from utils import cal_accuracy, cal_F1, cal_AUC_AP, cal_shortest_dis
from data_factory import load_data, mask_edges, mask_edges_random, mask_edges_manual, load_synthetic_data
from logger import create_logger
from geoopt.optim import RiemannianAdam
from rl_agent import MultiScaleHPPOController
from reward_utils import RewardScaling
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
            
        self.rl_controller = None
        self.rl_prev_objective = None
        self.rl_prev_metric = None
        self.rl_reward_history = []
        
        # 混合精度训练支持
        self.use_amp = getattr(configs, 'use_amp', False)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # 多GPU支持（预留）
        self.use_multi_gpu = False
        self.device_ids = None

    def _init_rl_controller(self):
        """根据配置初始化H-PPO控制器。"""
        if getattr(self.configs, 'rl_enable', False):
            num_hops = len(self.configs.sample_hop) if isinstance(self.configs.sample_hop, list) else 1
            self.rl_controller = MultiScaleHPPOController(self.configs, num_hops, self.configs.num_factors, self.device)
            self.rl_prev_objective = None
            self.rl_prev_metric = None
            # Initialize reward scaler for normalization
            if getattr(self.configs, 'rl_use_reward_scaling', True):
                gamma = getattr(self.configs, 'rl_gamma', 0.99)
                self.rl_reward_scaler = RewardScaling(shape=1, gamma=gamma)
            else:
                self.rl_reward_scaler = None
        else:
            self.rl_controller = None
            self.rl_prev_objective = None
            self.rl_prev_metric = None
            self.rl_reward_scaler = None

    def _build_rl_state(self):
        """提取多尺度子图的简单统计量，作为强化学习的状态输入。"""
        if self.subgraph_feature is None:
            return None
        max_hop = max(self.configs.sample_hop) if isinstance(self.configs.sample_hop, list) else 1
        total_nodes = max(int(self.features.shape[0]), 1)
        state_vecs = []
        for idx, hop in enumerate(self.configs.sample_hop):
            feat = self.subgraph_feature[idx].detach()
            edges = self.subgraph_edge_index[idx].detach()
            node_cnt = max(feat.shape[0], 1)
            edge_cnt = edges.shape[1]
            mean_feat = feat.mean()
            std_feat = feat.std(unbiased=False) if feat.numel() > 1 else torch.tensor(0.0, device=feat.device, dtype=feat.dtype)
            vec = torch.stack([
                torch.tensor(hop / max_hop, device=self.device, dtype=feat.dtype),
                torch.tensor(node_cnt / total_nodes, device=self.device, dtype=feat.dtype),
                torch.tensor(edge_cnt / node_cnt, device=self.device, dtype=feat.dtype),
                mean_feat.to(self.device),
                std_feat.to(self.device)
            ])
            state_vecs.append(vec)
        state = torch.stack(state_vecs, dim=0).unsqueeze(0)
        if state.size(-1) > self.configs.rl_state_dim:
            state = state[..., :self.configs.rl_state_dim]
        elif state.size(-1) < self.configs.rl_state_dim:
            pad = torch.zeros(state.size(0), state.size(1), self.configs.rl_state_dim - state.size(-1), device=self.device, dtype=state.dtype)
            state = torch.cat([state, pad], dim=-1)
        return state

    def _compose_reward_signal(self, stats: dict):
        """根据配置生成用于奖励的信号（损失或准确率等指标）。"""
        reward_type = getattr(self.configs, 'rl_reward_type', 'loss')
        task = self.configs.downstream_task
        if reward_type == 'metric':
            if task == 'NC':
                cls_metric = self._to_float(stats.get('cls_val_metric', 0.0))
                lp_metric = self._to_float(stats.get('lp_val_metric', 0.0))
                return cls_metric + self.configs.rl_aux_lp_coef * lp_metric
            elif task == 'LP':
                return self._to_float(stats.get('lp_val_metric', 0.0))
        else:
            if task == 'NC':
                cls_loss = self._to_float(stats.get('cls_val_loss', 0.0))
                lp_loss = self._to_float(stats.get('lp_val_loss', 0.0))
                return cls_loss + self.configs.rl_aux_lp_coef * lp_loss
            elif task == 'LP':
                return self._to_float(stats.get('lp_val_loss', 0.0))
        return 0.0

    def _calc_reward(self, stats: dict):
        """根据当前配置计算奖励。"""
        reward_type = getattr(self.configs, 'rl_reward_type', 'loss')
        current_signal = self._compose_reward_signal(stats)
        if reward_type == 'metric':
            if self.rl_prev_metric is None:
                reward = 0.0
            else:
                reward = current_signal - self.rl_prev_metric
            self.rl_prev_metric = current_signal
        else:
            if self.rl_prev_objective is None:
                reward = 0.0
            else:
                reward = self.rl_prev_objective - current_signal
            self.rl_prev_objective = current_signal
        return reward

    def _save_reward_curve(self, logger):
        if not self.rl_reward_history:
            return
        try:
            indices = list(range(len(self.rl_reward_history)))
            rewards = []
            labels = []
            for item in self.rl_reward_history:
                if isinstance(item, tuple) and len(item) >= 3:
                    labels.append(item[0])
                    rewards.append(float(item[2]))
                else:
                    labels.append('step')
                    rewards.append(float(item))
            fig, ax = plt.subplots()
            ax.plot(indices, rewards, label='reward', color='tab:blue')
            ax.set_xlabel('step')
            ax.set_ylabel('reward')
            ax.set_title('RL Reward Curve')
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
            plot_path = os.path.splitext(self.configs.log_path)[0] + "_reward.png"
            fig.savefig(plot_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Reward curve saved to {plot_path}")
        except Exception as exc:
            logger.warning(f"Failed to save reward curve: {exc}")

    def _to_float(self, value):
        """统一将标量张量转为float。"""
        if torch.is_tensor(value):
            return value.detach().cpu().item()
        return float(value)

    def train(self):
        logger = create_logger(self.configs.log_path)
        self.rl_reward_history = []
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


        # 在采样阶段也使用混合精度以节省内存
        if self.use_amp:
            with torch.cuda.amp.autocast():
                if self.configs.downstream_task == 'LP':
                    self.subgraph_feature, self.subgraph_edge_index, self.subgraph_batch = self.subgraph_sampler.sample(self.features, self.pos_edges[0], "LP")
                else:
                    self.subgraph_feature, self.subgraph_edge_index, self.subgraph_batch = self.subgraph_sampler.sample(self.features, self.edge_index, "NC")
        else:
            if self.configs.downstream_task == 'LP':
                self.subgraph_feature, self.subgraph_edge_index, self.subgraph_batch = self.subgraph_sampler.sample(self.features, self.pos_edges[0], "LP")
            else:
                self.subgraph_feature, self.subgraph_edge_index, self.subgraph_batch = self.subgraph_sampler.sample(self.features, self.edge_index, "NC")
        self._init_rl_controller()

        for exp_iter in range(self.configs.exp_iters):
            logger.info(f"\ntrain iters {exp_iter}")

            model = Experts(init_curvs=self.configs.init_curvs, in_dim=in_features, hidden_dim=self.configs.hidden_features, out_dim=self.configs.embed_features, learnable=True, num_factors_cls = self.configs.num_factors_cls).to(self.device)
            model_gating = Gating(in_dim=in_features, hidden_dim=self.configs.hidden_features, out_dim=self.configs.embed_features, num_experts=self.configs.num_factors, configs = self.configs).to(self.device)

            if self.rl_controller is not None:
                rl_state = self._build_rl_state()
                hop_mask, curv_bias = self.rl_controller.act(rl_state)
                # 根据控制模式选择性地传递参数
                control_mode = getattr(self.configs, 'rl_control_mode', 'both')
                if control_mode == 'hop_only':
                    model_gating.set_rl_inputs(hop_mask=hop_mask.to(self.device), curvature_bias=None)
                elif control_mode == 'curv_only':
                    model_gating.set_rl_inputs(hop_mask=None, curvature_bias=curv_bias.to(self.device))
                else:  # both
                    model_gating.set_rl_inputs(hop_mask=hop_mask.to(self.device), curvature_bias=curv_bias.to(self.device))
            else:
                model_gating.set_rl_inputs(None, None)

            logger.info("--------------------------Training Start-------------------------")
            if self.configs.downstream_task == 'NC':
                # 检查是否跳过LP预训练
                if getattr(self.configs, 'skip_lp_pretrain', False):
                    logger.info("⚠️  Skipping LP pretraining, directly running NC task")
                    test_auc, test_ap, lp_val_loss, lp_val_metric = 0, 0, 0, 0
                else:
                    test_auc, test_ap, _, lp_val_loss, _, lp_val_metric = self.train_lp(model, model_gating, self.pos_edges, self.neg_edges, logger)
                best_acc, test_acc, test_weighted_f1, test_macro_f1, best_epoch, cls_val_loss, cls_test_loss, cls_val_metric = self.train_cls(model, model_gating, logger)
                logger.info(f"best_epoch={best_epoch}")
                logger.info(
                    f"test_accuracy={test_acc.item() * 100: .2f}%")
                logger.info(
                    f"weighted_f1={test_weighted_f1 * 100: .2f}%, macro_f1={test_macro_f1 * 100: .2f}%")
                accs.append(test_acc.item())
                wf1s.append(test_weighted_f1)
                mf1s.append(test_macro_f1)

            elif self.configs.downstream_task == 'LP':
                test_auc, test_ap, best_epoch, lp_val_loss, _, lp_val_metric = self.train_lp(model, model_gating, self.pos_edges, self.neg_edges, logger)
                logger.info(f"best_epoch={best_epoch}")
                logger.info(
                    f"test_auc={test_auc * 100: .2f}%, test_ap={test_ap * 100: .2f}%")
                aucs.append(test_auc)
                aps.append(test_ap)
            else:
                raise NotImplementedError

            if self.rl_controller is not None:
                stats = {'lp_val_loss': lp_val_loss, 'lp_val_metric': lp_val_metric}
                if self.configs.downstream_task == 'NC':
                    stats['cls_val_loss'] = cls_val_loss
                    stats['cls_val_metric'] = self._to_float(cls_val_metric if cls_val_metric is not None else best_acc)
                reward = self._calc_reward(stats)
                self.rl_controller.record_reward(reward)
                self.rl_reward_history.append(('iter', exp_iter, reward))

        if self.rl_controller is not None:
            self.rl_controller.finalize()
            self._save_reward_curve(logger)

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

    def train_cls(self, model, model_gating, logger):
        """masks = (train, val, test)"""
        self.configs.coef_dis = 1e-4
        d = self.configs.num_factors_cls * self.configs.embed_features
        model_cls = GNNClassifier(backbone=self.configs.backbone, n_layers=2, in_features=self.in_features + d,
                                    hidden_features=self.configs.hidden_features_cls, out_features=self.n_classes,
                                    n_heads=self.configs.n_heads, drop_edge=self.configs.drop_edge_cls, 
                                    drop_node=self.configs.drop_cls).to(self.device)
        
        # 多GPU包装分类器
        if self.use_multi_gpu and self.device_ids is not None:
            model_cls = nn.DataParallel(model_cls, device_ids=self.device_ids)
            
        optimizer_cls = torch.optim.Adam(model_cls.parameters(), lr=self.configs.lr_cls, weight_decay=self.configs.w_decay_cls)
        r_optim = RiemannianAdam(model.parameters(), lr=self.configs.lr_Riemann, weight_decay=self.configs.w_decay, stabilize=100)
        optimizer_gating = torch.optim.Adam(model_gating.parameters(), lr=self.configs.lr_gating, weight_decay=self.configs.w_decay_gating)
        best_acc = -1.
        best_epoch = 0
        early_stop_count = 0
        best_val_loss = float('inf')
        best_test_loss = float('inf')
        test_acc = torch.tensor(0., device=self.device)
        test_weighted_f1 = 0.0
        test_macro_f1 = 0.0
        prev_val_signal = None
        last_val_acc = None
        episode_done = False
        if self.rl_controller is not None:
            self.rl_controller.start_episode()
            # 初始退火设置（LP）
            descend = 1.0
            self.rl_controller.coeff_ent_d = self.configs.rl_entropy_coef * descend
            target = getattr(self.configs, "rl_target_log_std", -5.0)
            new_log_std = self.configs.rl_init_log_std + (target - self.configs.rl_init_log_std) * (1 - descend)
            for i in range(self.rl_controller.ensemble_num):
                self.rl_controller.actor_cs[i].log_std = torch.ones(self.rl_controller.num_curvatures, device=self.device) * new_log_std
        else:
            model_gating.set_rl_inputs(None, None)
        for epoch in range(self.configs.epochs_cls + 1):
            now_time = time.time()
            model_cls.train()
            model.train()
            model_gating.train()
            optimizer_cls.zero_grad()
            r_optim.zero_grad()
            optimizer_gating.zero_grad()

            if self.rl_controller is not None:
                # 线性退火（LP）
                descend = 1.0 - (epoch / max(self.configs.epochs_lp, 1))
                self.rl_controller.coeff_ent_d = self.configs.rl_entropy_coef * descend
                target = getattr(self.configs, "rl_target_log_std", -5.0)
                new_log_std = self.configs.rl_init_log_std + (target - self.configs.rl_init_log_std) * (epoch / max(self.configs.epochs_lp, 1))
                for i in range(self.rl_controller.ensemble_num):
                    self.rl_controller.actor_cs[i].log_std = torch.ones(self.rl_controller.num_curvatures, device=self.device) * new_log_std
                rl_state = self._build_rl_state()
                hop_mask, curv_bias = self.rl_controller.act(rl_state)
                # 根据控制模式选择性地传递参数
                control_mode = getattr(self.configs, 'rl_control_mode', 'both')
                if control_mode == 'hop_only':
                    model_gating.set_rl_inputs(hop_mask.to(self.device), None)
                elif control_mode == 'curv_only':
                    model_gating.set_rl_inputs(None, curv_bias.to(self.device))
                else:  # both
                    model_gating.set_rl_inputs(hop_mask.to(self.device), curv_bias.to(self.device))
            
            # 混合精度训练
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    embeddings = model.encode(self.features, self.edge_index, self.configs.dataset)
                    experts_weight, loss_distortion = model_gating(self.subgraph_feature, self.subgraph_edge_index, self.subgraph_batch, embeddings, self.dis_shortest, self.configs.embed_features, self.edge_index)
                    experts_weight = experts_weight.repeat_interleave(self.configs.embed_features, dim=1)
                    embeddings = embeddings * experts_weight
                    features = torch.concat([self.features, embeddings], -1)
                    loss, acc, weighted_f1, macro_f1 = self.cal_cls_loss(model_cls, self.edge_index, self.masks[0], features, self.labels)
                    loss = loss + self.configs.coef_dis * loss_distortion
                
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer_cls)
                self.scaler.step(optimizer_gating)
                self.scaler.update()
                # RiemannianAdam 不支持 scaler，单独处理
                r_optim.step()
            else:
                embeddings = model.encode(self.features, self.edge_index, self.configs.dataset)
                experts_weight, loss_distortion = model_gating(self.subgraph_feature, self.subgraph_edge_index, self.subgraph_batch, embeddings, self.dis_shortest, self.configs.embed_features, self.edge_index)
                experts_weight = experts_weight.repeat_interleave(self.configs.embed_features, dim=1)
                embeddings = embeddings * experts_weight
                features = torch.concat([self.features, embeddings], -1)
                loss, acc, weighted_f1, macro_f1 = self.cal_cls_loss(model_cls, self.edge_index, self.masks[0], features, self.labels)
                loss = loss + self.configs.coef_dis * loss_distortion
                
                loss.backward()
                optimizer_cls.step()
                r_optim.step()
                optimizer_gating.step()
            logger.info(f"Epoch {epoch}: train_loss={loss.item()}, train_accuracy={acc}, time={time.time()-now_time}")

            step_reward = 0.0
            done_flag = False
            if epoch % self.configs.eval_freq == 0:
                model_cls.eval()
                model.eval()
                model_gating.eval()

                with torch.no_grad():
                    if self.use_amp:
                        with torch.cuda.amp.autocast():
                            embeddings = model.encode(self.features, self.edge_index)
                            experts_weight, _ = model_gating(self.subgraph_feature, self.subgraph_edge_index, self.subgraph_batch, embeddings, self.dis_shortest, self.configs.embed_features, self.edge_index)
                            experts_weight = experts_weight.repeat_interleave(self.configs.embed_features, dim=1)
                            embeddings = embeddings * experts_weight
                            features = torch.concat([self.features, embeddings], -1)
                            val_loss, acc, weighted_f1, macro_f1 = self.cal_cls_loss(model_cls, self.edge_index, self.masks[1], features, self.labels)
                    else:
                        embeddings = model.encode(self.features, self.edge_index)
                        experts_weight, _ = model_gating(self.subgraph_feature, self.subgraph_edge_index, self.subgraph_batch, embeddings, self.dis_shortest, self.configs.embed_features, self.edge_index)
                        experts_weight = experts_weight.repeat_interleave(self.configs.embed_features, dim=1)
                        embeddings = embeddings * experts_weight
                        features = torch.concat([self.features, embeddings], -1)
                        val_loss, acc, weighted_f1, macro_f1 = self.cal_cls_loss(model_cls, self.edge_index, self.masks[1], features, self.labels)
                logger.info(f"Epoch {epoch}: val_accuracy={acc}, val_wf1={weighted_f1}, val_mf1={macro_f1}")
                reward_type = getattr(self.configs, 'rl_reward_type', 'loss')
                if reward_type == 'metric':
                    current_signal = self._to_float(acc)
                    if prev_val_signal is not None:
                        step_reward = current_signal - prev_val_signal
                else:
                    current_signal = self._to_float(val_loss)
                    if prev_val_signal is not None:
                        step_reward = prev_val_signal - current_signal
                prev_val_signal = current_signal
                last_val_acc = self._to_float(acc)
                if acc > best_acc:
                    best_acc = acc
                    best_epoch = epoch
                    early_stop_count = 0
                    # Test
                    test_loss, test_acc, test_weighted_f1, test_macro_f1 = self.cal_cls_loss(model_cls, self.edge_index, self.masks[2], features, self.labels)
                    best_val_loss = self._to_float(val_loss)
                    best_test_loss = self._to_float(test_loss)
                else:
                    early_stop_count += 1
                    if early_stop_count > self.configs.patience_cls:
                        done_flag = True
                if epoch < self.configs.min_epoch_cls:
                    early_stop_count = 0
            if self.rl_controller is not None:
                if epoch == self.configs.epochs_cls:
                    done_flag = True
                # Apply reward scaling if enabled
                scaled_reward = step_reward
                if self.rl_reward_scaler is not None:
                    scaled_reward = self.rl_reward_scaler(step_reward)
                    if isinstance(scaled_reward, np.ndarray):
                        scaled_reward = float(scaled_reward.item())
                    if done_flag:
                        self.rl_reward_scaler.reset()
                self.rl_controller.record_reward(scaled_reward, done_flag)
                self.rl_reward_history.append(('nc', epoch, scaled_reward))
                # 记录RL详细信息
                if epoch % 100 == 0:
                    logger.info(f"  [RL] step_reward={step_reward:.6f}, scaled={scaled_reward:.6f}, val_signal={prev_val_signal:.6f}")
            if done_flag:
                episode_done = True
                break
        if self.rl_controller is not None and not episode_done:
            self.rl_controller.end_episode()
        return best_acc, test_acc, test_weighted_f1, test_macro_f1, best_epoch, best_val_loss, best_test_loss, last_val_acc

    def cal_lp_loss(self, embeddings, experts_weight, decoder, pos_edges, neg_edges):
        pos_diff = (embeddings[pos_edges[0]] - embeddings[pos_edges[1]])**2
        pos_diff = pos_diff.reshape(pos_diff.shape[0], pos_diff.shape[1]//self.configs.embed_features, self.configs.embed_features).sum(dim=2)
        pos_weights = F.softmax(experts_weight[pos_edges[0]] * experts_weight[pos_edges[1]], dim=1)   
        pos_scores = decoder(torch.sum(pos_diff * pos_weights, -1))

        neg_diff = (embeddings[neg_edges[0]] - embeddings[neg_edges[1]])**2
        neg_diff = neg_diff.reshape(neg_diff.shape[0], neg_diff.shape[1]//self.configs.embed_features, self.configs.embed_features).sum(dim=2)
        neg_weights = F.softmax(experts_weight[neg_edges[0]] * experts_weight[neg_edges[1]], dim=1)   
        neg_scores = decoder(torch.sum(neg_diff * neg_weights, -1))

        # 禁用autocast以兼容binary_cross_entropy
        with torch.cuda.amp.autocast(enabled=False):
            pos_scores_float = pos_scores.float()
            neg_scores_float = neg_scores.float()
            loss = F.binary_cross_entropy(pos_scores_float.clip(0.01, 0.99), torch.ones_like(pos_scores_float)) + \
                    F.binary_cross_entropy(neg_scores_float.clip(0.01, 0.99), torch.zeros_like(neg_scores_float))
        label = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        preds = list(pos_scores.detach().cpu().numpy()) + list(neg_scores.detach().cpu().numpy())
        auc, ap = cal_AUC_AP(preds, label)
        return loss, auc, ap

    def train_lp(self, model, model_gating, pos_edges, neg_edges, logger):
        r_optim = RiemannianAdam(model.parameters(), lr=self.configs.lr_Riemann, weight_decay=self.configs.w_decay, stabilize=100)
        optimizer_gating = torch.optim.Adam(model_gating.parameters(), lr=self.configs.lr_gating, weight_decay=self.configs.w_decay_gating)
        
        decoder = FermiDiracDecoder(self.configs.r, self.configs.t).to(self.device)
        best_ap = -1
        best_epoch = 0
        early_stop_count = 0
        best_val_loss = float('inf')
        best_test_loss = float('inf')
        best_test_auc = 0.0
        best_test_ap = 0.0
        prev_val_signal = None
        last_val_auc = None
        episode_done = False
        if self.rl_controller is not None:
            self.rl_controller.start_episode()
            # 初始退火设置（NC）
            descend = 1.0
            self.rl_controller.coeff_ent_d = self.configs.rl_entropy_coef * descend
            target = getattr(self.configs, "rl_target_log_std", -5.0)
            new_log_std = self.configs.rl_init_log_std + (target - self.configs.rl_init_log_std) * (1 - descend)
            for i in range(self.rl_controller.ensemble_num):
                self.rl_controller.actor_cs[i].log_std = torch.ones(self.rl_controller.num_curvatures, device=self.device) * new_log_std
        else:
            model_gating.set_rl_inputs(None, None)
        for epoch in range(self.configs.epochs_lp + 1):
            t = time.time()
            model.train()
            model_gating.train()
            r_optim.zero_grad()
            optimizer_gating.zero_grad()

            if self.rl_controller is not None:
                # 线性退火（NC）
                descend = 1.0 - (epoch / max(self.configs.epochs_cls, 1))
                self.rl_controller.coeff_ent_d = self.configs.rl_entropy_coef * descend
                target = getattr(self.configs, "rl_target_log_std", -5.0)
                new_log_std = self.configs.rl_init_log_std + (target - self.configs.rl_init_log_std) * (epoch / max(self.configs.epochs_cls, 1))
                for i in range(self.rl_controller.ensemble_num):
                    self.rl_controller.actor_cs[i].log_std = torch.ones(self.rl_controller.num_curvatures, device=self.device) * new_log_std
                rl_state = self._build_rl_state()
                hop_mask, curv_bias = self.rl_controller.act(rl_state)
                # 根据控制模式选择性地传递参数
                control_mode = getattr(self.configs, 'rl_control_mode', 'both')
                if control_mode == 'hop_only':
                    model_gating.set_rl_inputs(hop_mask.to(self.device), None)
                elif control_mode == 'curv_only':
                    model_gating.set_rl_inputs(None, curv_bias.to(self.device))
                else:  # both
                    model_gating.set_rl_inputs(hop_mask.to(self.device), curv_bias.to(self.device))

            # 混合精度训练
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    embeddings = model(self.features, pos_edges[0])
                    experts_weight, loss_distortion = model_gating(self.subgraph_feature, self.subgraph_edge_index, self.subgraph_batch, embeddings, self.dis_shortest, self.configs.embed_features, pos_edges[0])
                    neg_edge_train = neg_edges[0][:, np.random.randint(0, neg_edges[0].shape[1], pos_edges[0].shape[1])]
                    loss, auc, ap = self.cal_lp_loss(embeddings, experts_weight, decoder, pos_edges[0], neg_edge_train)
                    loss = loss + self.configs.coef_dis * loss_distortion
                
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer_gating)
                self.scaler.update()
                # RiemannianAdam 不支持 scaler，单独处理
                r_optim.step()
            else:
                embeddings = model(self.features, pos_edges[0])
                experts_weight, loss_distortion = model_gating(self.subgraph_feature, self.subgraph_edge_index, self.subgraph_batch, embeddings, self.dis_shortest, self.configs.embed_features, pos_edges[0])
                neg_edge_train = neg_edges[0][:, np.random.randint(0, neg_edges[0].shape[1], pos_edges[0].shape[1])]
                loss, auc, ap = self.cal_lp_loss(embeddings, experts_weight, decoder, pos_edges[0], neg_edge_train)
                loss = loss + self.configs.coef_dis * loss_distortion
                
                loss.backward()
                r_optim.step()
                optimizer_gating.step()
            logger.info(f"Epoch {epoch}: train_loss={loss.item()}, train_AUC={auc}, train_AP={ap}, time={time.time() - t}")
            step_reward = 0.0
            done_flag = False
            if epoch % self.configs.eval_freq == 0:
                model.eval()
                model_gating.eval()
                
                with torch.no_grad():
                    if self.use_amp:
                        with torch.cuda.amp.autocast():
                            embeddings = model(self.features, pos_edges[0])
                            experts_weight, _ = model_gating(self.subgraph_feature, self.subgraph_edge_index, self.subgraph_batch, embeddings, self.dis_shortest, self.configs.embed_features, pos_edges[0])
                            val_loss, auc, ap = self.cal_lp_loss(embeddings, experts_weight, decoder, pos_edges[1], neg_edges[1])
                    else:
                        embeddings = model(self.features, pos_edges[0])
                        experts_weight, _ = model_gating(self.subgraph_feature, self.subgraph_edge_index, self.subgraph_batch, embeddings, self.dis_shortest, self.configs.embed_features, pos_edges[0])
                        val_loss, auc, ap = self.cal_lp_loss(embeddings, experts_weight, decoder, pos_edges[1], neg_edges[1])
                logger.info(f"Epoch {epoch}: val_AUC={auc}, val_AP={ap}")
                reward_type = getattr(self.configs, 'rl_reward_type', 'loss')
                if reward_type == 'metric':
                    current_signal = self._to_float(auc)
                    if prev_val_signal is not None:
                        step_reward = current_signal - prev_val_signal
                else:
                    current_signal = self._to_float(val_loss)
                    if prev_val_signal is not None:
                        step_reward = prev_val_signal - current_signal
                prev_val_signal = current_signal
                last_val_auc = self._to_float(auc)
                if ap > best_ap:
                    best_ap = ap
                    best_epoch = epoch
                    early_stop_count = 0
                    # Test
                    test_loss, test_auc, test_ap = self.cal_lp_loss(embeddings, experts_weight, decoder, pos_edges[2], neg_edges[2])
                    best_val_loss = self._to_float(val_loss)
                    best_test_loss = self._to_float(test_loss)
                    best_test_auc = test_auc
                    best_test_ap = test_ap
                else:
                    early_stop_count += 1
                    if early_stop_count > self.configs.patience_lp:
                        done_flag = True
                if epoch < self.configs.min_epoch_lp:
                    early_stop_count = 0

            if self.rl_controller is not None:
                if epoch == self.configs.epochs_lp:
                    done_flag = True
                # Apply reward scaling if enabled
                scaled_reward = step_reward
                if self.rl_reward_scaler is not None:
                    scaled_reward = self.rl_reward_scaler(step_reward)
                    if isinstance(scaled_reward, np.ndarray):
                        scaled_reward = float(scaled_reward.item())
                    if done_flag:
                        self.rl_reward_scaler.reset()
                self.rl_controller.record_reward(scaled_reward, done_flag)
                self.rl_reward_history.append(('lp', epoch, scaled_reward))
            if done_flag:
                episode_done = True
                break

        if self.rl_controller is not None and not episode_done:
            self.rl_controller.end_episode()

        return best_test_auc, best_test_ap, best_epoch, best_val_loss, best_test_loss, last_val_auc
