import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from models import *
from backbone import GNNClassifier
from utils import cal_accuracy, cal_F1, cal_AUC_AP, cal_shortest_dis
from data_factory import load_data, mask_edges, load_synthetic_data
from logger import create_logger
from geoopt.optim import RiemannianAdam
import time
import os
from torch_geometric.utils import negative_sampling
import pickle
from dqn_components import GatingDQN, ReplayBuffer
from reward_fn import RewardFunction
import random

class Exp:
    def __init__(self, configs):
        self.configs = configs
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
    
    def update_epsilon(self, epsilon, step_count):
        """更新epsilon值：支持固定值、线性衰减和指数衰减"""
        # 如果设置了固定epsilon值，直接返回
        if self.configs.epsilon_fixed is not None:
            return self.configs.epsilon_fixed
        
        # 线性衰减：在指定步数内从epsilon_start线性衰减到epsilon_end
        if self.configs.epsilon_decay_steps is not None:
            if step_count < self.configs.epsilon_decay_steps:
                epsilon = self.configs.epsilon_start - (self.configs.epsilon_start - self.configs.epsilon_end) * step_count / self.configs.epsilon_decay_steps
            else:
                epsilon = self.configs.epsilon_end
        else:
            # 指数衰减：原有的方式
            epsilon = max(self.configs.epsilon_end, epsilon * self.configs.epsilon_decay)
        return epsilon

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
        self.pos_edges, self.neg_edges = mask_edges(self.edge_index, self.neg_edge, val_prop, test_prop)
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
            
            # 使用DQN门控网络
            if self.configs.use_dqn:
                model_gating = GatingDQN(in_dim=in_features, hidden_dim=self.configs.hidden_features, 
                                        out_dim=self.configs.embed_features, num_experts=self.configs.num_factors, 
                                        configs=self.configs, device=self.device)
                # 创建经验回放缓冲区
                replay_buffer = ReplayBuffer(capacity=self.configs.replay_buffer_size)
                reward_fn = RewardFunction(coef_distortion=self.configs.coef_distortion_reward,
                                         coef_task=self.configs.coef_task_reward,
                                         device=self.device)
            else:
                model_gating = Gating(in_dim=in_features, hidden_dim=self.configs.hidden_features, 
                                    out_dim=self.configs.embed_features, num_experts=self.configs.num_factors, 
                                    configs=self.configs).to(self.device)
                replay_buffer = None
                reward_fn = None

            logger.info("--------------------------Training Start-------------------------")
            if self.configs.downstream_task == 'NC':
                if self.configs.use_dqn:
                    test_auc, test_ap, _ = self.train_lp_dqn(model, model_gating, self.pos_edges, self.neg_edges, 
                                                             replay_buffer, reward_fn, logger)
                    best_val, test_acc, test_weighted_f1, test_macro_f1, best_epoch = self.train_cls_dqn(
                        model, model_gating, replay_buffer, reward_fn, logger)
                else:
                    test_auc, test_ap, _ = self.train_lp(model, model_gating, self.pos_edges, self.neg_edges, logger)
                    best_val, test_acc, test_weighted_f1, test_macro_f1, best_epoch = self.train_cls(model, model_gating, logger)
                logger.info(f"best_epoch={best_epoch}")
                logger.info(
                    f"test_accuracy={test_acc.item() * 100: .2f}%")
                logger.info(
                    f"weighted_f1={test_weighted_f1 * 100: .2f}%, macro_f1={test_macro_f1 * 100: .2f}%")
                accs.append(test_acc.item())
                wf1s.append(test_weighted_f1)
                mf1s.append(test_macro_f1)

            elif self.configs.downstream_task == 'LP':
                if self.configs.use_dqn:
                    test_auc, test_ap, best_epoch = self.train_lp_dqn(model, model_gating, self.pos_edges, self.neg_edges,
                                                                       replay_buffer, reward_fn, logger)
                else:
                    test_auc, test_ap, best_epoch = self.train_lp(model, model_gating, self.pos_edges, self.neg_edges, logger)
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

    def train_cls(self, model, model_gating, logger):
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
            now_time = time.time()
            model_cls.train()
            model.train()
            model_gating.train()
            optimizer_cls.zero_grad()
            r_optim.zero_grad()
            optimizer_gating.zero_grad()
            
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

            if epoch % self.configs.eval_freq == 0:
                model_cls.eval()
                model.eval()
                model_gating.eval()

                embeddings = model.encode(self.features, self.edge_index)
                experts_weight = model_gating(self.subgraph_feature, self.subgraph_edge_index, self.subgraph_batch)
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

    def train_lp(self, model, model_gating, pos_edges, neg_edges, logger):
        r_optim = RiemannianAdam(model.parameters(), lr=self.configs.lr_Riemann, weight_decay=self.configs.w_decay, stabilize=100)
        optimizer_gating = torch.optim.Adam(model_gating.parameters(), lr=self.configs.lr_gating, weight_decay=self.configs.w_decay_gating)
        
        decoder = FermiDiracDecoder(self.configs.r, self.configs.t).to(self.device)
        best_ap = 0
        best_epoch = 0
        early_stop_count = 0
        for epoch in range(self.configs.epochs_lp + 1):
            t = time.time()
            model.train()
            model_gating.train()
            r_optim.zero_grad()
            optimizer_gating.zero_grad()

            embeddings = model(self.features, pos_edges[0])
            experts_weight, loss_distortion = model_gating(self.subgraph_feature, self.subgraph_edge_index, self.subgraph_batch, embeddings, self.dis_shortest, self.configs.embed_features, pos_edges[0])
            
            neg_edge_train = neg_edges[0][:, np.random.randint(0, neg_edges[0].shape[1], pos_edges[0].shape[1])]
            loss, auc, ap = self.cal_lp_loss(embeddings, experts_weight, decoder, pos_edges[0], neg_edge_train)
            loss = loss + self.configs.coef_dis * loss_distortion 
            loss.backward()
            r_optim.step()
            optimizer_gating.step()
            logger.info(f"Epoch {epoch}: train_loss={loss.item()}, train_AUC={auc}, train_AP={ap}, time={time.time() - t}")
            if epoch % self.configs.eval_freq == 0:
                model.eval()
                model_gating.eval()
                embeddings = model(self.features, pos_edges[0])
                experts_weight = model_gating(self.subgraph_feature, self.subgraph_edge_index, self.subgraph_batch)

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
    
    def train_cls_dqn(self, model, model_gating, replay_buffer, reward_fn, logger):
        """使用DQN训练节点分类任务"""
        self.configs.coef_dis = 1e-4
        d = self.configs.num_factors_cls * self.configs.embed_features
        model_cls = GNNClassifier(backbone=self.configs.backbone, n_layers=2, in_features=self.in_features + d,
                                    hidden_features=self.configs.hidden_features_cls, out_features=self.n_classes,
                                    n_heads=self.configs.n_heads, drop_edge=self.configs.drop_edge_cls, 
                                    drop_node=self.configs.drop_cls).to(self.device)
        
        optimizer_cls = torch.optim.Adam(model_cls.parameters(), lr=self.configs.lr_cls, weight_decay=self.configs.w_decay_cls)
        r_optim = RiemannianAdam(model.parameters(), lr=self.configs.lr_Riemann, weight_decay=self.configs.w_decay, stabilize=100)
        optimizer_dqn = torch.optim.Adam(model_gating.q_network.parameters(), lr=self.configs.lr_gating, weight_decay=self.configs.w_decay_gating)
        
        best_acc = 0.
        best_epoch = 0
        early_stop_count = 0
        epsilon = self.configs.epsilon_start
        step_count = 0
        
        for epoch in range(self.configs.epochs_cls + 1):
            now_time = time.time()
            model_cls.train()
            model.train()
            model_gating.train()
            optimizer_cls.zero_grad()
            r_optim.zero_grad()
            optimizer_dqn.zero_grad()
            
            # 获取状态（编码拓扑特征）
            with torch.no_grad():
                state_features = model_gating.q_network.encode_state(
                    self.subgraph_feature, self.subgraph_edge_index, self.subgraph_batch
                )
            
            # 选择动作（专家权重）
            embeddings = model.encode(self.features, self.edge_index, self.configs.dataset)
            experts_weight, loss_distortion, q_values = model_gating(
                self.subgraph_feature, self.subgraph_edge_index, self.subgraph_batch,
                embeddings, self.dis_shortest, self.configs.embed_features, self.edge_index,
                epsilon=epsilon
            )
            
            experts_weight = experts_weight.repeat_interleave(self.configs.embed_features, dim=1)
            embeddings = embeddings * experts_weight
            features = torch.concat([self.features, embeddings], -1)
            
            # 计算任务损失和性能
            loss, acc, weighted_f1, macro_f1 = self.cal_cls_loss(model_cls, self.edge_index, self.masks[0], features, self.labels)
            
            # 计算奖励
            num_train_nodes = self.masks[0].sum().item()
            train_acc = acc.item()
            
            # 为每个节点计算奖励（简化：使用全局奖励）
            reward = reward_fn.compute_reward(
                loss_distortion.item(), 
                (train_acc, weighted_f1, macro_f1),
                task_type='NC'
            )
            
            # 存储经验到回放缓冲区（每个节点一个经验）
            # 注意：这里简化处理，实际可以更精细地为每个节点存储单独的经验
            if len(replay_buffer) < replay_buffer.buffer.maxlen:
                # 创建下一状态（这里使用相同状态作为简化，实际应该是下一个epoch的状态）
                next_state = state_features.detach()
                done = 0  # 非终止状态
                action = experts_weight.detach()[:, :self.configs.num_factors].mean(dim=0)  # 简化：使用平均权重
                
                for node_idx in range(min(100, state_features.shape[0])):  # 限制存储数量
                    replay_buffer.push(
                        state_features[node_idx].detach().cpu(),
                        action.detach().cpu(),
                        reward / num_train_nodes,  # 归一化奖励
                        next_state[node_idx].detach().cpu(),
                        done
                    )
            
            # 先完成主损失的反向传播和参数更新
            total_loss = loss + self.configs.coef_dis * loss_distortion
            total_loss.backward()
            optimizer_cls.step()
            r_optim.step()
            
            # DQN训练：从回放缓冲区采样（在主损失更新后进行，避免梯度图冲突）
            if len(replay_buffer) >= self.configs.batch_size_dqn and step_count % self.configs.dqn_update_freq == 0:
                states, actions, rewards, next_states, dones = replay_buffer.sample(self.configs.batch_size_dqn)
                states = states.to(self.device)
                next_states = next_states.to(self.device)
                rewards = rewards.to(self.device)
                dones = dones.to(self.device)
                
                # 当前Q值
                current_q_values = model_gating.q_network.q_network(states)
                # 获取动作对应的Q值（简化：使用最大Q值）
                current_q = current_q_values.max(dim=1)[0]
                
                # 目标Q值
                with torch.no_grad():
                    next_q_values = model_gating.target_q_network.q_network(next_states)
                    next_q = next_q_values.max(dim=1)[0]
                    target_q = rewards + (1 - dones) * self.configs.gamma * next_q
                
                # DQN损失
                dqn_loss = F.mse_loss(current_q, target_q)
                
                optimizer_dqn.zero_grad()
                dqn_loss.backward()
                optimizer_dqn.step()
                
                # 更新目标网络
                if step_count % self.configs.target_update_freq == 0:
                    model_gating.update_target_network(tau=self.configs.tau)
            
            # Epsilon衰减
            step_count += 1
            epsilon = self.update_epsilon(epsilon, step_count)
            
            logger.info(f"Epoch {epoch}: train_loss={total_loss.item()}, train_accuracy={acc}, "
                       f"distortion={loss_distortion.item()}, reward={reward:.4f}, epsilon={epsilon:.4f}, "
                       f"time={time.time()-now_time}")
            
            if epoch % self.configs.eval_freq == 0:
                model_cls.eval()
                model.eval()
                model_gating.eval()
                
                embeddings = model.encode(self.features, self.edge_index)
                experts_weight = model_gating(self.subgraph_feature, self.subgraph_edge_index, 
                                            self.subgraph_batch, deterministic=True)
                experts_weight = experts_weight.repeat_interleave(self.configs.embed_features, dim=1)
                embeddings = embeddings * experts_weight
                features = torch.concat([self.features, embeddings], -1)
                
                _, acc, weighted_f1, macro_f1 = self.cal_cls_loss(model_cls, self.edge_index, 
                                                                  self.masks[1], features, self.labels)
                logger.info(f"Epoch {epoch}: val_accuracy={acc}, val_wf1={weighted_f1}, val_mf1={macro_f1}")
                
                if acc > best_acc:
                    best_acc = acc
                    best_epoch = epoch
                    early_stop_count = 0
                    _, test_acc, test_weighted_f1, test_macro_f1 = self.cal_cls_loss(
                        model_cls, self.edge_index, self.masks[2], features, self.labels)
                else:
                    early_stop_count += 1
                    if early_stop_count > self.configs.patience_cls:
                        break
                if epoch < self.configs.min_epoch_cls:
                    early_stop_count = 0
                    
        return best_acc, test_acc, test_weighted_f1, test_macro_f1, best_epoch
    
    def train_lp_dqn(self, model, model_gating, pos_edges, neg_edges, replay_buffer, reward_fn, logger):
        """使用DQN训练链接预测任务"""
        r_optim = RiemannianAdam(model.parameters(), lr=self.configs.lr_Riemann, weight_decay=self.configs.w_decay, stabilize=100)
        optimizer_dqn = torch.optim.Adam(model_gating.q_network.parameters(), lr=self.configs.lr_gating, weight_decay=self.configs.w_decay_gating)
        
        decoder = FermiDiracDecoder(self.configs.r, self.configs.t).to(self.device)
        best_ap = 0
        best_epoch = 0
        early_stop_count = 0
        epsilon = self.configs.epsilon_start
        step_count = 0
        
        for epoch in range(self.configs.epochs_lp + 1):
            t = time.time()
            model.train()
            model_gating.train()
            r_optim.zero_grad()
            optimizer_dqn.zero_grad()
            
            # 获取状态
            with torch.no_grad():
                state_features = model_gating.q_network.encode_state(
                    self.subgraph_feature, self.subgraph_edge_index, self.subgraph_batch
                )
            
            embeddings = model(self.features, pos_edges[0])
            experts_weight, loss_distortion, q_values = model_gating(
                self.subgraph_feature, self.subgraph_edge_index, self.subgraph_batch,
                embeddings, self.dis_shortest, self.configs.embed_features, pos_edges[0],
                epsilon=epsilon
            )
            
            neg_edge_train = neg_edges[0][:, np.random.randint(0, neg_edges[0].shape[1], pos_edges[0].shape[1])]
            loss, auc, ap = self.cal_lp_loss(embeddings, experts_weight, decoder, pos_edges[0], neg_edge_train)
            
            # 计算奖励
            reward = reward_fn.compute_reward(
                loss_distortion.item(),
                (auc, ap),
                task_type='LP'
            )
            
            # 存储经验
            if len(replay_buffer) < replay_buffer.buffer.maxlen:
                next_state = state_features.detach()
                done = 0
                action = experts_weight.detach()[:, :self.configs.num_factors].mean(dim=0)
                
                for node_idx in range(min(100, state_features.shape[0])):
                    replay_buffer.push(
                        state_features[node_idx].detach().cpu(),
                        action.detach().cpu(),
                        reward / pos_edges[0].shape[1],
                        next_state[node_idx].detach().cpu(),
                        done
                    )
            
            # 先完成主损失的反向传播和参数更新
            total_loss = loss + self.configs.coef_dis * loss_distortion
            total_loss.backward()
            r_optim.step()
            
            # DQN训练（在主损失更新后进行，避免梯度图冲突）
            if len(replay_buffer) >= self.configs.batch_size_dqn and step_count % self.configs.dqn_update_freq == 0:
                states, actions, rewards, next_states, dones = replay_buffer.sample(self.configs.batch_size_dqn)
                states = states.to(self.device)
                next_states = next_states.to(self.device)
                rewards = rewards.to(self.device)
                dones = dones.to(self.device)
                
                current_q_values = model_gating.q_network.q_network(states)
                current_q = current_q_values.max(dim=1)[0]
                
                with torch.no_grad():
                    next_q_values = model_gating.target_q_network.q_network(next_states)
                    next_q = next_q_values.max(dim=1)[0]
                    target_q = rewards + (1 - dones) * self.configs.gamma * next_q
                
                dqn_loss = F.mse_loss(current_q, target_q)
                
                optimizer_dqn.zero_grad()
                dqn_loss.backward()
                optimizer_dqn.step()
                
                if step_count % self.configs.target_update_freq == 0:
                    model_gating.update_target_network(tau=self.configs.tau)
            
            step_count += 1
            epsilon = self.update_epsilon(epsilon, step_count)
            
            logger.info(f"Epoch {epoch}: train_loss={total_loss.item()}, train_AUC={auc}, train_AP={ap}, "
                       f"distortion={loss_distortion.item()}, reward={reward:.4f}, epsilon={epsilon:.4f}, "
                       f"time={time.time() - t}")
            
            if epoch % self.configs.eval_freq == 0:
                model.eval()
                model_gating.eval()
                embeddings = model(self.features, pos_edges[0])
                experts_weight = model_gating(self.subgraph_feature, self.subgraph_edge_index, 
                                            self.subgraph_batch, deterministic=True)
                
                _, auc, ap = self.cal_lp_loss(embeddings, experts_weight, decoder, pos_edges[1], neg_edges[1])
                logger.info(f"Epoch {epoch}: val_AUC={auc}, val_AP={ap}")
                
                if ap > best_ap:
                    best_ap = ap
                    best_epoch = epoch
                    early_stop_count = 0
                    _, test_auc, test_ap = self.cal_lp_loss(embeddings, experts_weight, decoder, 
                                                            pos_edges[2], neg_edges[2])
                else:
                    early_stop_count += 1
                    if early_stop_count > self.configs.patience_lp:
                        break
                if epoch < self.configs.min_epoch_lp:
                    early_stop_count = 0
        
        return test_auc, test_ap, best_epoch
            
        