import torch
import torch.nn.functional as F
import numpy as np

class RewardFunction:
    """奖励函数：结合失真度和下游任务性能"""
    
    def __init__(self, coef_distortion=1.0, coef_task=1.0, device='cuda'):
        self.coef_distortion = coef_distortion
        self.coef_task = coef_task
        self.device = device
    
    def compute_distortion_reward(self, distortion_value):
        """
        基于嵌入失真度的奖励
        失真越小，奖励越大
        reward = -distortion 或 reward = 1 / (1 + distortion)
        """
        # 方案1：直接负值
        reward = -distortion_value
        
        # 方案2：归一化奖励（可选）
        # reward = 1.0 / (1.0 + distortion_value)
        
        return reward
    
    def compute_task_reward_nc(self, accuracy, weighted_f1=None, macro_f1=None):
        """
        节点分类任务奖励
        直接使用准确率作为奖励信号
        """
        reward = accuracy.item() if torch.is_tensor(accuracy) else accuracy
        
        # 可选：结合F1分数
        if weighted_f1 is not None:
            reward = 0.7 * reward + 0.3 * weighted_f1
        
        return reward
    
    def compute_task_reward_lp(self, auc, ap):
        """
        链接预测任务奖励
        使用AUC和AP的平均值
        """
        reward = (auc + ap) / 2.0
        return reward
    
    def compute_reward(self, distortion_value, task_metric, task_type='NC'):
        """
        综合奖励函数
        Args:
            distortion_value: 嵌入失真度（标量）
            task_metric: 任务性能指标（accuracy或(auc, ap)）
            task_type: 'NC' 或 'LP'
        """
        distortion_reward = self.compute_distortion_reward(distortion_value)
        
        if task_type == 'NC':
            if isinstance(task_metric, tuple):
                accuracy, weighted_f1, macro_f1 = task_metric
                task_reward = self.compute_task_reward_nc(accuracy, weighted_f1, macro_f1)
            else:
                task_reward = self.compute_task_reward_nc(task_metric)
        else:  # LP
            auc, ap = task_metric
            task_reward = self.compute_task_reward_lp(auc, ap)
        
        # 综合奖励
        total_reward = (self.coef_distortion * distortion_reward + 
                       self.coef_task * task_reward)
        
        return total_reward
    
    def compute_node_rewards(self, distortions_per_node, task_metrics_per_node=None, 
                            task_type='NC'):
        """
        为每个节点计算奖励
        Args:
            distortions_per_node: [num_nodes] 每个节点的失真度
            task_metrics_per_node: [num_nodes] 每个节点的任务性能（可选）
            task_type: 'NC' 或 'LP'
        """
        if task_metrics_per_node is None:
            # 仅基于失真度
            rewards = -distortions_per_node * self.coef_distortion
        else:
            # 结合任务性能
            if task_type == 'NC':
                task_rewards = torch.tensor([self.compute_task_reward_nc(m) 
                                           for m in task_metrics_per_node], 
                                           device=self.device)
            else:
                task_rewards = torch.tensor([self.compute_task_reward_lp(m[0], m[1]) 
                                           for m in task_metrics_per_node],
                                           device=self.device)
            
            distortion_rewards = -distortions_per_node * self.coef_distortion
            rewards = distortion_rewards + task_rewards * self.coef_task
        
        return rewards

