import copy
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical


# =========================
# 多分辨率采样的 H-PPO 策略
# 说明：
# - 与 graph_level/agent.py 的 H_PPO 类似，但状态定义和动作语义适配“多分辨率（多尺度）采样”场景。
# - 离散动作：为每个节点选择一个尺度（即 sample_hop 中的某个 k）。
# - 连续动作：为所选尺度的表示添加一个连续向量（可视作“提示/偏置”），以调节该尺度的表示后再送入门控分类器。
# - 该文件仅提供策略结构与推理/训练接口骨架，训练循环需在上层（如 exp.py）中根据任务损失构建奖励并对策略进行更新。
# =========================


def orthogonal_init(layer: nn.Linear, gain: float = 2 ** 0.5):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


def xavier_init(layer: nn.Linear):
    nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain('relu'))
    nn.init.constant_(layer.bias, 0)


class ActorDiscreteMRS(nn.Module):
    """
    多分辨率-离散 Actor：根据每个节点在各尺度上的表示，输出选择每个尺度的概率。

    输入 state 的形状: (N, H, D)
      - N: 节点数
      - H: 尺度数（len(sample_hop)）
      - D: 每个尺度的表示维度（与 Gating.encoder2 的输出维度一致）
    输出: (N, H) 的每节点尺度分类概率
    """
    def __init__(self, emb_dim: int):
        super().__init__()
        # 对每个尺度的向量做共享 MLP 到 1 维 logit，再按尺度维度 softmax
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2), nn.Tanh(),
            nn.Linear(emb_dim // 2, emb_dim // 4), nn.Tanh(),
            nn.Linear(emb_dim // 4, 1)
        )
        orthogonal_init(self.mlp[0])
        orthogonal_init(self.mlp[2])
        orthogonal_init(self.mlp[4], gain=0.01)

    def forward(self, state: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # state: (N, H, D)
        N, H, D = state.shape
        logits = self.mlp(state)  # (N, H, 1)
        logits = logits.squeeze(-1)  # (N, H)
        if mask is not None:
            # mask==0 表示不可用的尺度（这里一般全可用，保留接口以防后续扩展）
            logits = torch.where(mask == 0, torch.full_like(logits, float('-inf')), logits)
        prob = F.softmax(logits, dim=-1)  # (N, H)
        return prob

    def get_action_logprob(self, state: torch.Tensor, mask: Optional[torch.Tensor] = None):
        prob = self.forward(state, mask)  # (N, H)
        dist = Categorical(prob)
        action_d = dist.sample()  # (N,)
        logprob = dist.log_prob(action_d)  # (N,)
        return action_d, logprob

    def logprob_entropy(self, state: torch.Tensor, action_d: torch.Tensor, mask: Optional[torch.Tensor] = None):
        prob = self.forward(state, mask)
        dist = Categorical(prob)
        return dist.log_prob(action_d), dist.entropy()


class ActorContinuousMRS(nn.Module):
    """
    多分辨率-连续 Actor：在选定的尺度上产生一个连续向量（与表示维度 D 相同），作为“提示/偏置”。

    注意：这里的 log_std 作为可学习参数可选，默认先固定值，便于稳定。
    """
    def __init__(self, emb_dim: int, init_log_std: float = -2.0):
        super().__init__()
        self.log_std = nn.Parameter(torch.ones(emb_dim) * init_log_std, requires_grad=False)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2), nn.Tanh(),
            nn.Linear(emb_dim // 2, emb_dim // 2), nn.Tanh(),
            nn.Linear(emb_dim // 2, emb_dim)
        )
        orthogonal_init(self.mlp[0])
        orthogonal_init(self.mlp[2])
        orthogonal_init(self.mlp[4], gain=0.01)

    def _select_state_by_action(self, state: torch.Tensor, action_d: torch.Tensor) -> torch.Tensor:
        # 从 (N, H, D) 的 state 中，依据每个节点的 action_d（尺度索引）挑选对应尺度的向量
        N, H, D = state.shape
        idx = action_d.view(N, 1, 1).expand(N, 1, D)  # (N,1,D)
        gathered = torch.gather(state, dim=1, index=idx).squeeze(1)  # (N, D)
        return gathered

    def forward(self, state: torch.Tensor, action_d: torch.Tensor) -> torch.Tensor:
        # 选中尺度的向量 -> MLP -> mean
        s = self._select_state_by_action(state, action_d)  # (N, D)
        mean = torch.tanh(self.mlp(s))  # (N, D)
        return mean

    def get_action_logprob(self, state: torch.Tensor, action_d: torch.Tensor, deterministic: bool = False):
        mean = self.forward(state, action_d)
        std = torch.exp(self.log_std.expand_as(mean))
        dist = Normal(mean, std)
        if deterministic:
            action = mean
        else:
            action = dist.sample()  # (N, D)
        logprob = torch.sum(dist.log_prob(action), dim=-1)  # (N,)
        return action, logprob

    def logprob_entropy(self, state: torch.Tensor, action_d: torch.Tensor, action_c: torch.Tensor):
        mean = self.forward(state, action_d)
        std = torch.exp(self.log_std.expand_as(mean))
        dist = Normal(mean, std)
        return torch.sum(dist.log_prob(action_c), dim=-1), torch.sum(dist.entropy(), dim=-1)


class CriticMRS(nn.Module):
    """
    价值网络：输入 (N, H, D) 的状态，先展平到 (N, H*D) 再做 MLP -> V(s)
    """
    def __init__(self, emb_dim: int, num_hops: int):
        super().__init__()
        input_dim = emb_dim * num_hops
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, max(16, input_dim // 20)), nn.ReLU(),
            nn.Linear(max(16, input_dim // 20), max(8, input_dim // 40)), nn.ReLU(),
            nn.Linear(max(8, input_dim // 40), 1)
        )
        xavier_init(self.mlp[0]); xavier_init(self.mlp[2]); xavier_init(self.mlp[4])

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.flatten(state)).squeeze(-1)  # (N,)


class H_PPO_MRS:
    """
    分辨率层面的 H-PPO 封装：
    - 维护一组离散/连续 actor 与一个 critic（可选集成数 ensemble_num>1）。
    - 提供推理（采样动作）与训练（根据优势函数/回报更新）接口。
    - 训练细节需由上层（exp.py）组织：构造 state、reward、advantage、ret 等。
    """
    def __init__(
        self,
        emb_dim: int,
        num_hops: int,
        ensemble_num: int = 1,
        penalty_alpha_d: float = 0.0,
        penalty_alpha_c: float = 0.0,
        eps_clip_d: float = 0.2,
        eps_clip_c: float = 0.2,
        coeff_critic: float = 0.5,
        coeff_entropy_d: float = 1e-3,
        max_norm_grad: float = 5.0,
        init_log_std: float = -2.0,
        device: Optional[torch.device] = None,
    ):
        self.emb_dim = emb_dim
        self.num_hops = num_hops
        self.ensemble_num = ensemble_num
        self.penalty_alpha_d = penalty_alpha_d
        self.penalty_alpha_c = penalty_alpha_c
        self.actor_ds = [copy.deepcopy(ActorDiscreteMRS(emb_dim)).to(device) for _ in range(ensemble_num)]
        self.actor_cs = [copy.deepcopy(ActorContinuousMRS(emb_dim, init_log_std)).to(device) for _ in range(ensemble_num)]
        self.critic = CriticMRS(emb_dim, num_hops).to(device)

        self.eps_clip_d = eps_clip_d
        self.eps_clip_c = eps_clip_c
        self.coeff_critic = coeff_critic
        self.coeff_ent_d = coeff_entropy_d
        self.max_norm_grad = max_norm_grad
        self.device = device

    def train_or_eval(self, mode: str):
        if mode == 'train':
            for i in range(self.ensemble_num):
                self.actor_ds[i].train(); self.actor_cs[i].train()
            self.critic.train()
        elif mode == 'eval':
            for i in range(self.ensemble_num):
                self.actor_ds[i].eval(); self.actor_cs[i].eval()
            self.critic.eval()
        else:
            raise ValueError("Invalid mode; use 'train' or 'eval'.")

    # ========= 状态构造与动作 ========= #
    @torch.no_grad()
    def build_state_from_reps(self, reps_per_hop: List[torch.Tensor]) -> torch.Tensor:
        """
        将各尺度的节点表示列表拼接为状态张量。
        参数：
          - reps_per_hop: 长度为 H 的列表，每个元素形状为 (N, D)
        返回：
          - state: (N, H, D)
        """
        # 断言维度一致
        assert len(reps_per_hop) == self.num_hops
        state = torch.stack(reps_per_hop, dim=1)  # (N, H, D)
        return state

    def sample_actions(self, a_idx: int, state: torch.Tensor, mask: Optional[torch.Tensor] = None, deterministic_c: bool = False):
        """
        基于第 a_idx 个子策略（若 ensemble_num>1），对每个节点采样离散/连续动作。
        返回：
          - action_d: (N,)  离散尺度索引
          - action_c: (N,D) 连续向量
          - logprob_d, logprob_c: (N,)
        """
        action_d, logprob_d = self.actor_ds[a_idx].get_action_logprob(state, mask)
        action_c, logprob_c = self.actor_cs[a_idx].get_action_logprob(state, action_d, deterministic=deterministic_c)
        return action_d, action_c, logprob_d, logprob_c

    @torch.no_grad()
    def eval_actions(self, state: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        推理时（不训练）根据集成策略得到更稳定的动作：
        - 离散：对各子策略的概率取最大后归一化再采样
        - 连续：对各子策略的均值取均值
        """
        pis = []
        for i in range(self.ensemble_num):
            pis.append(self.actor_ds[i].forward(state, mask))  # (N,H)
        ens_max_pi = torch.stack(pis, dim=1).max(dim=1)[0]  # (N,H)
        ens_prob = F.normalize(ens_max_pi + 1e-10, p=1, dim=-1)
        dist = Categorical(ens_prob)
        action_d = dist.sample()  # (N,)

        means = []
        for i in range(self.ensemble_num):
            means.append(self.actor_cs[i].forward(state, action_d))  # (N,D)
        action_c = torch.stack(means, dim=1).mean(dim=1)  # (N,D)
        return action_d, action_c

    # ========= 集成惩罚（对齐 graph_level） ========= #
    def _ensemble_penalty_d(self, a_idx: int, state: torch.Tensor, mask: Optional[torch.Tensor] = None):
        pis = []
        for i in range(self.ensemble_num):
            # 只有当前子策略需要梯度
            pi = self.actor_ds[i].forward(state, mask)
            pis.append(pi)
        with torch.no_grad():
            ens_max_pi = torch.stack(pis, dim=1).max(dim=1)[0]
            ens_pi = F.normalize(ens_max_pi + 1e-10, p=1, dim=-1)
        cur_pi = F.normalize(pis[a_idx] + 1e-10, p=1, dim=-1)
        kl_penalty_d = F.kl_div(torch.log(cur_pi), ens_pi, log_target=False, reduction='none').sum(dim=-1)
        ens_action_d = torch.argmax(ens_max_pi, dim=-1)
        return kl_penalty_d * self.penalty_alpha_d, ens_action_d

    def _ensemble_penalty_c(self, a_idx: int, state: torch.Tensor, action_d: torch.Tensor):
        means, stds = [], []
        for i in range(self.ensemble_num):
            m = self.actor_cs[i].forward(state, action_d)
            s = torch.exp(self.actor_cs[i].log_std.expand_as(m))
            means.append(m); stds.append(s)
        with torch.no_grad():
            ens_mean = torch.stack(means, dim=1).mean(dim=1)
            ens_std = torch.stack(stds, dim=1).mean(dim=1)
        cur_dist = Normal(means[a_idx], stds[a_idx])
        ens_dist = Normal(ens_mean, ens_std)
        kl_penalty_c = torch.distributions.kl.kl_divergence(cur_dist, ens_dist).sum(dim=-1)
        return kl_penalty_c * self.penalty_alpha_c

    # ========= 损失计算（训练时使用，接口骨架） ========= #
    def _actor_loss(self, a_idx: int, state: torch.Tensor, action_d: torch.Tensor, action_c: torch.Tensor,
                    adv: torch.Tensor, logprob_old_d: torch.Tensor, logprob_old_c: torch.Tensor):
        """
        PPO surrogate objective.
        说明：此处未加入 ensemble 惩罚项（如 KL 到集成分布），如需可参考 graph_level/agent.py 的 _ensemble_penalty_* 实现。
        """
        logprob_new_d, entropy_d = self.actor_ds[a_idx].logprob_entropy(state, action_d, mask=None)
        ratio_d = torch.exp(logprob_new_d - logprob_old_d)
        surr1_d = ratio_d * adv
        surr2_d = torch.clamp(ratio_d, 1 - self.eps_clip_d, 1 + self.eps_clip_d) * adv
        # 集成惩罚（可选）
        ens_pen_d = 0.0
        ens_action_d = action_d
        if self.ensemble_num > 1 and self.penalty_alpha_d > 0:
            ens_penalty_d, ens_action_d = self._ensemble_penalty_d(a_idx, state)
            ens_pen_d = ens_penalty_d
        actor_loss_d = -torch.min(surr1_d, surr2_d) - self.coeff_ent_d * entropy_d + (ens_pen_d if isinstance(ens_pen_d, torch.Tensor) else 0)

        logprob_new_c, _entropy_c = self.actor_cs[a_idx].logprob_entropy(state, action_d, action_c)
        ratio_c = torch.exp(logprob_new_c - logprob_old_c)
        surr1_c = ratio_c * adv
        surr2_c = torch.clamp(ratio_c, 1 - self.eps_clip_c, 1 + self.eps_clip_c) * adv
        ens_pen_c = 0.0
        if self.ensemble_num > 1 and self.penalty_alpha_c > 0:
            ens_penalty_c = self._ensemble_penalty_c(a_idx, state, ens_action_d)
            ens_pen_c = ens_penalty_c
        actor_loss_c = -torch.min(surr1_c, surr2_c) + (ens_pen_c if isinstance(ens_pen_c, torch.Tensor) else 0)

        # KL 指标（用于 target_kl 提前停止）
        with torch.no_grad():
            kl_d = torch.mean(torch.abs(logprob_new_d - logprob_old_d)).item()
            kl_c = torch.mean(torch.abs(logprob_new_c - logprob_old_c)).item()
        return actor_loss_d.mean(), actor_loss_c.mean(), kl_d, kl_c

    def _critic_loss(self, state: torch.Tensor, ret: torch.Tensor):
        value = self.critic(state).squeeze(-1)
        return F.mse_loss(ret, value) * self.coeff_critic

    def train_step(self,
                   args_like: Dict,
                   a_idx: int,
                   experience: Tuple[torch.Tensor, ...],
                   optim: torch.optim.Optimizer):
        """
        单次策略更新步骤（骨架）：
        参数：
          - args_like: 包含 minibatch_size, policy_update_nums, target_kl_* 等必要超参
          - experience: (state, action_d, logprob_d, action_c, logprob_c, adv, ret)
        """
        state, action_d, logprob_old_d, action_c, logprob_old_c, adv, ret = experience
        # 优势归一化（可选）
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        minibatch_size = int(args_like.get('minibatch_size', max(1, len(state) // 4)))
        policy_update_nums = int(args_like.get('policy_update_nums', 5))
        target_kl_d = args_like.get('target_kl_d', None)
        target_kl_c = args_like.get('target_kl_c', None)

        minibatch_num = max(1, int(len(state) / max(1, minibatch_size)))
        for _ in range(policy_update_nums):
            index = torch.randperm(len(state))
            kl_d_acc, kl_c_acc = 0.0, 0.0
            for m in range(minibatch_num):
                idx = index[m * minibatch_size:(m + 1) * minibatch_size]
                actor_loss_d, actor_loss_c, kl_d, kl_c = self._actor_loss(
                    a_idx, state[idx], action_d[idx], action_c[idx], adv[idx], logprob_old_d[idx], logprob_old_c[idx]
                )
                critic_loss = self._critic_loss(state[idx], ret[idx])
                loss = actor_loss_d + actor_loss_c + critic_loss
                optim.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor_ds[a_idx].parameters()) +
                    list(self.actor_cs[a_idx].parameters()) +
                    list(self.critic.parameters()),
                    norm_type=2, max_norm=self.max_norm_grad
                )
                optim.step()
                kl_d_acc += kl_d
                kl_c_acc += kl_c
            # target KL 早停（对齐 graph_level）
            if target_kl_d is not None and minibatch_num > 0:
                if kl_d_acc / minibatch_num > float(target_kl_d):
                    break
            if target_kl_c is not None and minibatch_num > 0:
                if kl_c_acc / minibatch_num > float(target_kl_c):
                    break

    # ========= 将动作应用到多尺度表示 ========= #
    @staticmethod
    def apply_actions_to_reps(reps_per_hop: List[torch.Tensor], action_d: torch.Tensor, action_c: torch.Tensor) -> List[torch.Tensor]:
        """
        在选定尺度上对每个节点的表示进行加性调整：rep[h, i] += action_c[i]
        输入：
          - reps_per_hop: 长度 H 的列表，每个张量为 (N, D)
          - action_d: (N,) 每个节点选择的尺度索引
          - action_c: (N,D) 连续向量
        输出：
          - 新的 reps_per_hop 列表（已就地拷贝修改）
        """
        N = action_d.shape[0]
        H = len(reps_per_hop)
        assert all(rep.shape[0] == N for rep in reps_per_hop)
        new_reps = [r.clone() for r in reps_per_hop]
        # 将每个节点的 action 应用到对应尺度
        for h in range(H):
            mask = (action_d == h).view(-1, 1)  # (N,1)
            if mask.any():
                new_reps[h][mask.expand_as(new_reps[h])] += action_c[mask.expand_as(action_c)]
        return new_reps
