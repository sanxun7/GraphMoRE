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

# DQN Settings
parser.add_argument('--use_dqn', action='store_true', help='Use DQN for gating network')
parser.add_argument('--replay_buffer_size', type=int, default=10000, help='Size of replay buffer')
parser.add_argument('--batch_size_dqn', type=int, default=64, help='Batch size for DQN training')
parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for DQN')
parser.add_argument('--epsilon_start', type=float, default=1.0, help='Initial epsilon for epsilon-greedy')
parser.add_argument('--epsilon_end', type=float, default=0.01, help='Final epsilon for epsilon-greedy')
parser.add_argument('--epsilon_decay', type=float, default=0.995, help='Epsilon decay rate (used if epsilon_decay_steps is not set)')
parser.add_argument('--epsilon_decay_steps', type=int, default=None, help='Number of steps to decay from epsilon_start to epsilon_end (linear decay)')
parser.add_argument('--epsilon_fixed', type=float, default=None, help='Fixed epsilon value (if set, epsilon will not decay)')
parser.add_argument('--target_update_freq', type=int, default=10, help='Frequency of target network update')
parser.add_argument('--dqn_update_freq', type=int, default=1, help='Frequency of DQN update')
parser.add_argument('--tau', type=float, default=0.01, help='Soft update coefficient for target network')
parser.add_argument('--coef_distortion_reward', type=float, default=1.0, help='Coefficient for distortion reward')
parser.add_argument('--coef_task_reward', type=float, default=1.0, help='Coefficient for task performance reward')

# Link Prediction
parser.add_argument('--epochs_lp', type=int, default=5000)
parser.add_argument('--patience_lp', type=int, default=100)
parser.add_argument('--min_epoch_lp', type=int, default=200)
parser.add_argument('--t', type=float, default=1., help='for Fermi-Dirac decoder')
parser.add_argument('--r', type=float, default=2., help='Fermi-Dirac decoder')

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
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')


configs = parser.parse_args()
configs.num_factors = len(configs.init_curvs)
configs.num_factors_cls = configs.num_factors

results_dir = f"./results/{configs.version}"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = f"{results_dir}/{configs.downstream_task}_{configs.backbone}_{configs.dataset}_{timestamp}.log"

configs.log_path = log_path
if not os.path.exists("./results"):
    os.mkdir("./results")
if not os.path.exists(results_dir):
    os.mkdir(results_dir)


logger = create_logger(configs.log_path)
logger.info(configs)

exp = Exp(configs)
exp.train()
torch.cuda.empty_cache()
