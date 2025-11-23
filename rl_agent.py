import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from typing import Dict, Optional


def _init_linear(layer: nn.Linear, gain: float = 1.0) -> None:
    """对线性层做正交初始化，保证策略训练更加稳定。"""
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0.0)


class ActorDiscrete(nn.Module):
    """负责离散动作（选择多分辨率采样尺度）的策略网络。"""

    def __init__(self, input_dim: int):
        super().__init__()
        hidden = max(input_dim // 2, 1)
        hidden_mid = max(input_dim // 4, 1)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden_mid),
            nn.Tanh(),
            nn.Linear(hidden_mid, 1)
        )
        _init_linear(self.net[0])
        _init_linear(self.net[2])
        _init_linear(self.net[4], gain=0.01)

    def forward(self, state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        state: [B, max_hop, input_dim]
        mask:  [B, max_hop]，1表示无效位置，需要屏蔽
        """
        logits = self.net(state).squeeze(-1)
        logits = torch.where(mask == 0, logits, torch.full_like(logits, float("-inf")))
        return F.softmax(logits, dim=-1)

    def get_action_logprob(self, state: torch.Tensor, mask: torch.Tensor):
        probs = self.forward(state, mask)
        dist = Categorical(probs)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob

    def logprob_entropy(self, state: torch.Tensor, action: torch.Tensor, mask: torch.Tensor):
        probs = self.forward(state, mask)
        dist = Categorical(probs)
        return dist.log_prob(action), dist.entropy()


class ActorContinuous(nn.Module):
    """负责连续动作（调节曲率权重）的策略网络。"""

    def __init__(self, input_dim: int, action_dim: int, max_action: float, init_log_std: float):
        super().__init__()
        self.max_action = max_action
        hidden = max(input_dim // 2, 1)
        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim)
        )
        _init_linear(self.body[0])
        _init_linear(self.body[2])
        _init_linear(self.body[4], gain=0.01)
        self.log_std = torch.ones(action_dim) * init_log_std

    def forward(self, state: torch.Tensor, action_d: torch.Tensor) -> torch.Tensor:
        """
        根据离散动作选中的尺度，抽取对应的状态向量再生成均值。
        state: [B, max_hop, input_dim]
        action_d: [B]
        """
        latent = self.body(state)  # [B, max_hop, action_dim]
        index = action_d.view(action_d.size(0), 1, 1).expand(-1, 1, latent.size(-1))
        latent = torch.gather(latent, dim=1, index=index).squeeze(1)
        return torch.tanh(latent) * self.max_action

    def get_action_logprob(self, state: torch.Tensor, action_d: torch.Tensor, deterministic: bool = False):
        mean = self.forward(state, action_d)
        std = torch.exp(self.log_std.to(mean.device)).expand_as(mean)
        dist = Normal(mean, std)
        if deterministic:
            action = mean
        else:
            action = dist.rsample()
        logprob = torch.sum(dist.log_prob(action), dim=-1)
        return action, logprob

    def logprob_entropy(self, state: torch.Tensor, action_d: torch.Tensor, action_c: torch.Tensor):
        mean = self.forward(state, action_d)
        std = torch.exp(self.log_std.to(mean.device)).expand_as(mean)
        dist = Normal(mean, std)
        logprob = torch.sum(dist.log_prob(action_c), dim=-1)
        entropy = torch.sum(dist.entropy(), dim=-1)
        return logprob, entropy


class Critic(nn.Module):
    """状态价值网络。"""

    def __init__(self, input_dim: int, max_hop: int):
        super().__init__()
        feat_dim = max(input_dim * max_hop, 1)
        hidden = max(feat_dim // 4, 4)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1)
        )
        _init_linear(self.net[1])
        _init_linear(self.net[3])
        _init_linear(self.net[5])

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class TransitionBuffer:
    """简易轨迹缓存，按H-PPO的需求存储每一步信息。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.storage: Dict[str, list] = {
            "state": [],
            "mask": [],
            "action_d": [],
            "logprob_d": [],
            "action_c": [],
            "logprob_c": [],
            "value": [],
            "reward": [],
            "done": []
        }

    def push(self, item: Dict[str, torch.Tensor]):
        for k, v in item.items():
            if torch.is_tensor(v):
                self.storage[k].append(v.detach())
            else:
                self.storage[k].append(v)

    def size(self) -> int:
        return len(self.storage["reward"])

    def mark_last_done(self):
        if not self.storage["done"]:
            return
        last = self.storage["done"][-1]
        self.storage["done"][-1] = torch.ones_like(last)

    def as_tensors(self) -> Dict[str, torch.Tensor]:
        data = {}
        data["state"] = torch.cat(self.storage["state"], dim=0)
        data["mask"] = torch.cat(self.storage["mask"], dim=0)
        data["action_d"] = torch.cat(self.storage["action_d"], dim=0)
        data["logprob_d"] = torch.cat(self.storage["logprob_d"], dim=0)
        data["action_c"] = torch.cat(self.storage["action_c"], dim=0)
        data["logprob_c"] = torch.cat(self.storage["logprob_c"], dim=0)
        data["value"] = torch.cat(self.storage["value"], dim=0)
        data["reward"] = torch.cat(self.storage["reward"], dim=0)
        data["done"] = torch.cat(self.storage["done"], dim=0)
        return data


class MultiScaleHPPOController:
    """
    使用H-PPO控制多分辨率采样与曲率选择，整体结构参考graph_level工程。
    """

    def __init__(self, configs, num_hops: int, num_curvatures: int, device: torch.device):
        self.configs = configs
        self.device = device
        self.num_hops = num_hops
        self.num_curvatures = num_curvatures

        # ensemble actors
        self.ensemble_num = getattr(configs, "rl_ensemble_num", 1)
        self.penalty_alpha_d = getattr(configs, "rl_penalty_alpha_d", 0.0)
        self.penalty_alpha_c = getattr(configs, "rl_penalty_alpha_c", 0.0)
        self.actor_ds = nn.ModuleList([ActorDiscrete(configs.rl_state_dim).to(device) for _ in range(self.ensemble_num)])
        self.actor_cs = nn.ModuleList([ActorContinuous(configs.rl_state_dim, num_curvatures, configs.rl_max_action, configs.rl_init_log_std).to(device) for _ in range(self.ensemble_num)])
        self.critic = Critic(configs.rl_state_dim, num_hops).to(device)

        # separate optimizers per actor
        self.opt_d = [torch.optim.Adam(self.actor_ds[i].parameters(), lr=configs.rl_actor_d_lr) for i in range(self.ensemble_num)]
        self.opt_c = [torch.optim.Adam(self.actor_cs[i].parameters(), lr=configs.rl_actor_c_lr) for i in range(self.ensemble_num)]
        self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=configs.rl_critic_lr)

        self.gamma = getattr(configs, "rl_gamma", 0.95)
        self.lam = getattr(configs, "rl_lambda", 0.9)
        self.coeff_ent_d = configs.rl_entropy_coef

        self.buffer = TransitionBuffer()
        self.pending: Optional[Dict[str, torch.Tensor]] = None
        self.episode_steps = 0

    def act(self, state: torch.Tensor):
        state = state.to(self.device)
        mask = torch.zeros(state.size(0), self.num_hops, device=self.device)
        # use the first actor for interaction by default
        action_d, logprob_d = self.actor_ds[0].get_action_logprob(state, mask)
        action_c, logprob_c = self.actor_cs[0].get_action_logprob(state, action_d)
        value = self.critic(state).detach()

        hop_mask = self._build_hop_mask(action_d).detach()
        curvature_bias = self._build_curvature_bias(action_c).detach()

        self.pending = {
            "state": state.detach(),
            "mask": mask.detach(),
            "action_d": action_d.detach().unsqueeze(-1),
            "logprob_d": logprob_d.detach().unsqueeze(-1),
            "action_c": action_c.detach(),
            "logprob_c": logprob_c.detach().unsqueeze(-1),
            "value": value.detach().unsqueeze(-1)
        }
        return hop_mask, curvature_bias

    def start_episode(self):
        if self.pending is not None:
            self.record_reward(0.0, done=True)
        self.episode_steps = 0

    def record_reward(self, reward: float, done: bool = False):
        if self.pending is None:
            return
        dtype = self.pending["state"].dtype
        # Ensure reward tensor matches the shape of other tensors [batch_size, 1]
        batch_size = self.pending["state"].size(0)
        reward_tensor = torch.full((batch_size, 1), reward, device=self.device, dtype=dtype)
        done_tensor = torch.full((batch_size, 1), 1.0 if done else 0.0, device=self.device, dtype=dtype)
        entry = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in self.pending.items()}
        entry["reward"] = reward_tensor
        entry["done"] = done_tensor
        self.buffer.push(entry)
        self.pending = None
        self.episode_steps += 1
        if done and self.buffer.size() >= self.configs.rl_batch_size:
            self._update()

    def end_episode(self):
        if self.pending is not None:
            self.record_reward(0.0, done=True)
        else:
            self.buffer.mark_last_done()
            if self.buffer.size() >= self.configs.rl_batch_size:
                self._update()
        self.episode_steps = 0

    def finalize(self):
        if self.pending is not None:
            self.record_reward(0.0, done=True)
        if self.buffer.size():
            self._update()

    def _build_hop_mask(self, action_d: torch.Tensor) -> torch.Tensor:
        base = torch.full((action_d.size(0), self.num_hops), self.configs.rl_min_hop_scale, device=self.device)
        base.scatter_(1, action_d.unsqueeze(-1), 1.0)
        return base.mean(dim=0)

    def _build_curvature_bias(self, action_c: torch.Tensor) -> torch.Tensor:
        temp = max(self.configs.rl_temperature, 1e-4)
        bias = F.softmax(action_c / temp, dim=-1)
        return bias.mean(dim=0)

    def _ensemble_penalty_d(self, a_idx: int, state: torch.Tensor, mask: torch.Tensor):
        pis = []
        for i in range(self.ensemble_num):
            pi = self.actor_ds[i].forward(state, mask)
            pis.append(pi)
        with torch.no_grad():
            ens_max_pi = torch.stack(pis, dim=1).max(dim=1)[0]
            ens_pi = F.normalize(ens_max_pi + 1e-10, p=1, dim=-1)
        cur_pi = F.normalize(pis[a_idx] + 1e-10, p=1, dim=-1)
        kl_pen = F.kl_div(torch.log(cur_pi), ens_pi, reduction='none').sum(dim=-1)
        ens_action_d = torch.argmax(ens_max_pi, dim=-1)
        return kl_pen * self.penalty_alpha_d, ens_action_d

    def _ensemble_penalty_c(self, a_idx: int, state: torch.Tensor, action_d: torch.Tensor):
        means, stds = [], []
        for i in range(self.ensemble_num):
            mean = self.actor_cs[i].forward(state, action_d)
            std = torch.exp(self.actor_cs[i].log_std.to(mean.device).expand_as(mean))
            means.append(mean); stds.append(std)
        with torch.no_grad():
            ens_mean = torch.stack(means, dim=1).mean(dim=1)
            ens_std = torch.stack(stds, dim=1).mean(dim=1)
        cur_dist = Normal(means[a_idx], stds[a_idx])
        ens_dist = Normal(ens_mean, ens_std)
        kl_pen = torch.distributions.kl.kl_divergence(cur_dist, ens_dist).sum(dim=-1)
        return kl_pen * self.penalty_alpha_c

    def _compute_advantages(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor):
        rewards = rewards.squeeze(-1)
        values = values.squeeze(-1)
        dones = dones.squeeze(-1)
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        gae = torch.zeros(1, device=rewards.device)
        next_value = torch.zeros(1, device=rewards.device)
        running_return = torch.zeros(1, device=rewards.device)

        for t in reversed(range(rewards.size(0))):
            mask = 1.0 - dones[t]
            running_return = rewards[t] + self.gamma * running_return * mask
            returns[t] = running_return

            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            advantages[t] = gae
            next_value = values[t]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
        return advantages, returns

    def _update(self):
        data = self.buffer.as_tensors()
        advantages, returns = self._compute_advantages(data["reward"], data["value"], data["done"])
        
        # [Trick] Advantage normalization over the whole batch
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_samples = data["state"].size(0)
        minibatch_size = min(getattr(self.configs, "rl_minibatch_size", 128), total_samples)
        policy_update_nums = getattr(self.configs, "rl_policy_update_nums", 10)
        target_kl_d = getattr(self.configs, "rl_target_kl_d", None)
        target_kl_c = getattr(self.configs, "rl_target_kl_c", None)

        # ===== 分别更新每个子 actor，带 KL 约束与熵系数 =====
        for a_idx in range(self.ensemble_num):
            # Minibatch multi-epoch update with early stopping
            for update_epoch in range(policy_update_nums):
                # Randomly shuffle data for each epoch
                indices = torch.randperm(total_samples, device=self.device)
                kl_d_sum, kl_c_sum = 0.0, 0.0
                num_minibatches = 0
                
                for start_idx in range(0, total_samples, minibatch_size):
                    end_idx = min(start_idx + minibatch_size, total_samples)
                    mb_indices = indices[start_idx:end_idx]
                    
                    # Extract minibatch
                    mb_state = data["state"][mb_indices]
                    mb_action_d = data["action_d"][mb_indices].squeeze(-1)
                    mb_action_c = data["action_c"][mb_indices]
                    mb_mask = data["mask"][mb_indices]
                    mb_logprob_old_d = data["logprob_d"][mb_indices].squeeze(-1)
                    mb_logprob_old_c = data["logprob_c"][mb_indices].squeeze(-1)
                    mb_adv = advantages[mb_indices]
                    mb_ret = returns[mb_indices]
                    
                    # ====== Discrete actor loss ====== #
                    logprob_new_d, entropy_d = self.actor_ds[a_idx].logprob_entropy(mb_state, mb_action_d, mb_mask)
                    ratios_d = torch.exp(logprob_new_d - mb_logprob_old_d)
                    surr1_d = ratios_d * mb_adv
                    surr2_d = torch.clamp(ratios_d, 1 - self.configs.rl_eps_clip_d, 1 + self.configs.rl_eps_clip_d) * mb_adv
                    
                    ens_pen_d, ens_action_d = (0.0, None)
                    if self.ensemble_num > 1 and self.penalty_alpha_d > 0:
                        ens_pen_d, ens_action_d = self._ensemble_penalty_d(a_idx, mb_state, mb_mask)
                    
                    loss_d = -(torch.min(surr1_d, surr2_d)).mean() - self.coeff_ent_d * entropy_d.mean()
                    if isinstance(ens_pen_d, torch.Tensor):
                        loss_d = loss_d + ens_pen_d.mean()
                    
                    # Compute approx KL for early stopping
                    with torch.no_grad():
                        approx_kl_d = ((ratios_d - 1) - (logprob_new_d - mb_logprob_old_d)).mean()
                        kl_d_sum += approx_kl_d.item()
                    
                    self.opt_d[a_idx].zero_grad()
                    loss_d.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor_ds[a_idx].parameters(), self.configs.rl_max_grad_norm)
                    self.opt_d[a_idx].step()
                    
                    # ====== Continuous actor loss ====== #
                    logprob_new_c, entropy_c = self.actor_cs[a_idx].logprob_entropy(mb_state, mb_action_d, mb_action_c)
                    ratios_c = torch.exp(logprob_new_c - mb_logprob_old_c)
                    surr1_c = ratios_c * mb_adv
                    surr2_c = torch.clamp(ratios_c, 1 - self.configs.rl_eps_clip_c, 1 + self.configs.rl_eps_clip_c) * mb_adv
                    
                    ens_pen_c = 0.0
                    if self.ensemble_num > 1 and self.penalty_alpha_c > 0:
                        if ens_action_d is None:
                            with torch.no_grad():
                                probs_list = [self.actor_ds[i].forward(mb_state, mb_mask) for i in range(self.ensemble_num)]
                                max_probs = torch.stack(probs_list, dim=1).max(dim=1)[0]
                                ens_action_d = torch.argmax(max_probs, dim=-1)
                        ens_pen_c = self._ensemble_penalty_c(a_idx, mb_state, ens_action_d)
                    
                    loss_c = -(torch.min(surr1_c, surr2_c)).mean() - self.configs.rl_entropy_coef * entropy_c.mean()
                    if isinstance(ens_pen_c, torch.Tensor):
                        loss_c = loss_c + ens_pen_c.mean()
                    
                    # Compute approx KL for early stopping
                    with torch.no_grad():
                        approx_kl_c = ((ratios_c - 1) - (logprob_new_c - mb_logprob_old_c)).mean()
                        kl_c_sum += approx_kl_c.item()
                    
                    self.opt_c[a_idx].zero_grad()
                    loss_c.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor_cs[a_idx].parameters(), self.configs.rl_max_grad_norm)
                    self.opt_c[a_idx].step()
                    
                    # ====== Critic loss ====== #
                    values = self.critic(mb_state).squeeze(-1)
                    loss_v = F.mse_loss(values, mb_ret)
                    
                    self.opt_v.zero_grad()
                    loss_v.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.configs.rl_max_grad_norm)
                    self.opt_v.step()
                    
                    num_minibatches += 1
                
                # Check for early stopping based on KL divergence
                avg_kl_d = kl_d_sum / num_minibatches if num_minibatches > 0 else 0
                avg_kl_c = kl_c_sum / num_minibatches if num_minibatches > 0 else 0
                
                if target_kl_d is not None and avg_kl_d > target_kl_d:
                    break
                if target_kl_c is not None and avg_kl_c > target_kl_c:
                    break

        self.buffer.reset()
