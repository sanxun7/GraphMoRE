import torch
import numpy as np
import os
import random
import argparse
from exp import Exp
from datetime import datetime
from logger import create_logger
from typing import Union

seed = 3047
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

parser = argparse.ArgumentParser(description='')

# Experiment settings
parser.add_argument('--downstream_task', type=str, default='NC',
                    choices=['NC', 'LP'])
parser.add_argument('--dataset', type=str)
parser.add_argument('--root_path', type=str, default='./datasets')
parser.add_argument('--in_features', type=int)
parser.add_argument('--eval_freq', type=int, default=1)
parser.add_argument('--exp_iters', type=int, default=10)
parser.add_argument('--version', type=str, default="Train")
parser.add_argument('--log_path', type=str)


# Riemannian Embeds
parser.add_argument('--num_factors', type=int, default=5)
parser.add_argument('--init_curvs', type=float, nargs='+', default=[-3,-1,0,1,3])
parser.add_argument('--backbone', type=str, default='gcn', choices=['gcn', 'gat', 'sage'])
parser.add_argument('--hidden_features', type=int, default=64)
parser.add_argument('--embed_features', type=int, default=32, help='dimensions of graph embedding')
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--lr_Riemann', type=float, default=0.01)
parser.add_argument('--w_decay', type=float, default=5e-4)
parser.add_argument('--n_heads', type=int, default=8, help='number of attention heads')

# Gating
parser.add_argument('--sample_hop', type=int, nargs='+',default=[2,3])
parser.add_argument('--lr_gating', type=float, default=0.01)
parser.add_argument('--w_decay_gating', type=float, default=5e-4)
parser.add_argument('--coef_dis', type=float, default=0.1)

# Reinforcement Learning (H-PPO)
parser.add_argument('--rl_enable', dest='rl_enable', action='store_true', help='开启强化学习的多尺度控制')
parser.add_argument('--no_rl', dest='rl_enable', action='store_false', help='关闭强化学习控制')
parser.set_defaults(rl_enable=True)
parser.add_argument('--rl_control_mode', type=str, default='both', 
                    choices=['both', 'hop_only', 'curv_only'],
                    help='强化学习控制模式: both(两者都控制), hop_only(只控制多分辨率采样), curv_only(只控制曲率空间选择)')
parser.add_argument('--rl_state_dim', type=int, default=5, help='强化学习状态长度')
parser.add_argument('--rl_actor_d_lr', type=float, default=5e-4, help='离散策略学习率')
parser.add_argument('--rl_actor_c_lr', type=float, default=5e-4, help='连续策略学习率')
parser.add_argument('--rl_critic_lr', type=float, default=1e-3, help='价值网络学习率')
parser.add_argument('--rl_eps_clip_d', type=float, default=0.2, help='离散策略裁剪阈值')
parser.add_argument('--rl_eps_clip_c', type=float, default=0.2, help='连续策略裁剪阈值')
parser.add_argument('--rl_entropy_coef', type=float, default=0.01, help='熵正则系数')
parser.add_argument('--rl_min_hop_scale', type=float, default=0.2, help='未被选中尺度的保底权重')
parser.add_argument('--rl_temperature', type=float, default=1.0, help='连续动作softmax温度')
parser.add_argument('--rl_max_action', type=float, default=1.0, help='连续动作范围')
parser.add_argument('--rl_init_log_std', type=float, default=0.0, help='连续策略初始对数方差')
parser.add_argument('--rl_target_log_std', type=float, default=-5.0, help='方差退火的目标对数方差')
parser.add_argument('--rl_batch_size', type=int, default=4, help='累计多少迭代后更新H-PPO')
parser.add_argument('--rl_reward_scale', type=float, default=1.0, help='奖励缩放系数')
parser.add_argument('--rl_aux_lp_coef', type=float, default=0.2, help='分类任务下LP奖励的混合比')
parser.add_argument('--rl_max_grad_norm', type=float, default=0.5, help='梯度裁剪阈值')
parser.add_argument('--rl_gamma', type=float, default=0.95, help='强化学习折扣因子')
parser.add_argument('--rl_lambda', type=float, default=0.9, help='GAE 的 lambda 系数')
parser.add_argument('--rl_ensemble_num', type=int, default=1, help='策略集成的子 actor 数量')
parser.add_argument('--rl_penalty_alpha_d', type=float, default=0.0, help='离散策略与集成分布的KL约束系数')
parser.add_argument('--rl_penalty_alpha_c', type=float, default=0.0, help='连续策略与集成分布的KL约束系数')
parser.add_argument('--rl_reward_type', type=str, default='loss', choices=['loss', 'metric'],
                    help='强化学习奖励类型：loss 使用验证损失下降量，metric 使用下游指标提升量')

# Link Prediction
parser.add_argument('--epochs_lp', type=int, default=5000)
parser.add_argument('--patience_lp', type=int, default=100)
parser.add_argument('--min_epoch_lp', type=int, default=200)
parser.add_argument('--t', type=float, default=1., help='for Fermi-Dirac decoder')
parser.add_argument('--r', type=float, default=2., help='Fermi-Dirac decoder')
parser.add_argument('--edge_split_method', type=str, default='random', 
                    choices=['random', 'manual'], 
                    help='Edge splitting method: random (RandomLinkSplit) or manual (process.py style)')

# Node Classification
parser.add_argument('--drop_cls', type=float, default=0.0)
parser.add_argument('--drop_edge_cls', type=float, default=0.0)
parser.add_argument('--hidden_features_cls', type=int, default=32)
parser.add_argument('--num_factors_cls', type=int, default=5)
parser.add_argument('--lr_cls', type=float, default=0.01)
parser.add_argument('--w_decay_cls', type=float, default=5e-4)
parser.add_argument('--epochs_cls', type=int, default=5000)
parser.add_argument('--patience_cls', type=int, default=100)
parser.add_argument('--min_epoch_cls', type=int, default=200)

# GPU
parser.add_argument('--gpu', type=int, default=0, help='GPU ID')


configs = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(configs.gpu)
configs.num_factors = len(configs.init_curvs)
configs.num_factors_cls = configs.num_factors

# 根据RL控制模式确定日志文件后缀
if not configs.rl_enable:
    rl_suffix = "origin"
elif configs.rl_control_mode == "both":
    rl_suffix = "RL"
elif configs.rl_control_mode == "hop_only":
    rl_suffix = "hop"
elif configs.rl_control_mode == "curv_only":
    rl_suffix = "curv"
else:
    rl_suffix = "RL"  # 默认

results_dir = f"./results/{configs.version}"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = f"{results_dir}/{configs.downstream_task}_{configs.backbone}_{configs.dataset}_{rl_suffix}_{timestamp}.log"

configs.log_path = log_path
if not os.path.exists("./results"):
    os.mkdir("./results")
if not os.path.exists(results_dir):
    os.mkdir(results_dir)


logger = create_logger(configs.log_path)
logger.info(configs)

# 打印清晰的RL控制模式信息
logger.info("="*80)
if not configs.rl_enable:
    logger.info("🔧 RL Control Mode: ORIGIN (No RL Control)")
    logger.info("   - Multi-resolution Sampling: Normal Mode")
    logger.info("   - Curvature Space Selection: Normal Mode")
elif configs.rl_control_mode == "both":
    logger.info("🤖 RL Control Mode: RL (Full RL Control)")
    logger.info("   - Multi-resolution Sampling: RL Controlled ✓")
    logger.info("   - Curvature Space Selection: RL Controlled ✓")
elif configs.rl_control_mode == "hop_only":
    logger.info("🎯 RL Control Mode: HOP (Multi-resolution Only)")
    logger.info("   - Multi-resolution Sampling: RL Controlled ✓")
    logger.info("   - Curvature Space Selection: Normal Mode")
elif configs.rl_control_mode == "curv_only":
    logger.info("📐 RL Control Mode: CURV (Curvature Only)")
    logger.info("   - Multi-resolution Sampling: Normal Mode")
    logger.info("   - Curvature Space Selection: RL Controlled ✓")
logger.info("="*80)

exp = Exp(configs)
exp.train()
torch.cuda.empty_cache()
