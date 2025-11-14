# GraphMoRE 项目强化学习组件与流程详解

## 目录
1. [项目概述](#项目概述)
2. [核心组件](#核心组件)
3. [强化学习架构](#强化学习架构)
4. [训练流程](#训练流程)
5. [配置参数](#配置参数)

---

## 项目概述

本项目实现了一个基于**分层近端策略优化（H-PPO）**的多分辨率图采样框架，用于图节点分类（NC）和链接预测（LP）任务。核心思想是通过强化学习自适应地为每个节点选择最优的采样尺度（k-hop邻域），并对表示进行微调。

### 关键创新
- **多分辨率采样策略**：不同节点可能需要不同范围的邻域信息（2-hop、3-hop等）
- **分层动作空间**：
  - 离散动作：选择使用哪个尺度的表示（如选2-hop或3-hop）
  - 连续动作：优化选定表示的特征值（⚠️ 不是产生"2.5-hop"，而是调整向量数值）
- **端到端学习**：RL策略与图表示学习联合优化，奖励来自下游任务

---

## 核心组件

### 1. 策略网络（hppo_mrs.py）

#### 1.1 ActorDiscreteMRS - 离散动作策略
```python
输入: state (N, H, D)
  - N: 节点数
  - H: 尺度数（len(sample_hop)，如[2,3]则H=2）
  - D: 每个尺度的表示维度

输出: 概率分布 (N, H)
  - 为每个节点在各尺度上输出选择概率
```

**功能**：为每个节点选择一个最适合的采样尺度（如2-hop或3-hop）

**网络结构**：
- 3层MLP：D → D/2 → D/4 → 1
- 激活函数：Tanh
- 输出层对尺度维度做Softmax

#### 1.2 ActorContinuousMRS - 连续动作策略
```python
输入: 
  - state (N, H, D): 多尺度状态
  - action_d (N,): 离散动作（选定的尺度索引）

输出: 连续向量 (N, D)
  - 为选定尺度的表示生成"提示/偏置"向量
```

**功能**：在选定尺度的基础上，对节点表示进行精细调整

**网络结构**：
- 3层MLP：D → D/2 → D/2 → D
- 激活函数：Tanh
- 输出：高斯分布的均值，标准差为可学习参数

---

### ⚠️ 重要澄清：离散动作 vs 连续动作

**常见误解**：连续动作会产生"2.5-hop"这样的中间尺度？

**正确理解**：

| 维度 | 离散动作 | 连续动作 |
|------|---------|---------|
| **作用对象** | 子图的拓扑范围 | 表示向量的特征值 |
| **操作** | 选择2-hop还是3-hop | 调整32维向量的数值 |
| **是否改变hop数** | ✅ 选择hop数 | ❌ hop数不变 |
| **是否改变表示** | ✅ 切换到另一个表示 | ✅ 微调当前表示的数值 |

#### 详细说明

```python
# 第1步：采样（训练前完成，固定不变）
subgraph_2hop = sample_ego(graph, node, k_hop=2)  # 包含2跳内所有邻居
subgraph_3hop = sample_ego(graph, node, k_hop=3)  # 包含3跳内所有邻居
# ⚠️ 这些子图在整个训练过程中是固定的！hop数是离散的拓扑概念！

# 第2步：编码（GNN处理）
rep_2hop = GNN(subgraph_2hop)  # → [0.5, -0.3, 0.8, ..., 0.2]  (32维向量)
rep_3hop = GNN(subgraph_3hop)  # → [0.3,  0.1, -0.5, ..., 0.7]  (32维向量)

# 第3步：RL策略介入
# 离散动作：选择用哪个表示
action_d = 0  # 选择rep_2hop（基于2-hop子图的信息）

# 连续动作：生成调整向量
action_c = [+0.1, -0.05, +0.08, ..., +0.03]  # 32个调整值

# 第4步：应用调整
final_rep = rep_2hop + action_c
         = [0.5, -0.3, 0.8, ..., 0.2]    # 原始2-hop表示
         + [0.1, -0.05, 0.08, ..., 0.03] # 连续调整
         = [0.6, -0.35, 0.88, ..., 0.23] # 优化后的表示
         
# ✅ 这仍然是基于2-hop子图的表示（拓扑范围没变）
# ✅ 只是特征向量的32个数值被优化了（特征空间的调整）
# ❌ 不存在"2.1-hop"这种概念！hop数保持为2！
```

#### 形象比喻

```
🍕 离散动作 = 选择披萨尺寸
   - 小号（2-hop）：覆盖2个街区的食材范围
   - 大号（3-hop）：覆盖3个街区的食材范围
   ⚠️ 一旦选定，覆盖范围（hop数）就固定了！

🧂 连续动作 = 调整配料比例
   - 选了小号后：[芝士:50g, 番茄:30g, 肉:20g]
   - RL微调：    [+10g,    -5g,      +8g]
   - 最终配料：  [60g,     25g,      28g]
   ⚠️ 还是小号披萨（2-hop），只是配料量（特征值）调整了！
```

#### 为什么这样设计？

1. **离散动作**：快速决策粗粒度的信息范围（2-hop局部 vs 3-hop全局）
2. **连续动作**：在固定范围内优化表示质量
   - GNN编码可能不完美
   - 不同任务需要强调不同特征
   - 个性化调整每个节点的表示

**关键**：连续动作调整的是"表示向量的D个特征值"，不是"hop数"！

---

#### 1.3 CriticMRS - 价值网络
```python
输入: state (N, H, D)
输出: 状态价值 V(s) (N,)
```

**功能**：估计当前状态的价值，用于计算优势函数

**网络结构**：
- 先展平：(N, H, D) → (N, H*D)
- 3层MLP：H*D → H*D//20 → H*D//40 → 1
- 激活函数：ReLU

### 2. H_PPO_MRS 封装类

**职责**：
1. 管理Actor-Discrete、Actor-Continuous、Critic三个网络
2. 提供策略采样接口（训练/评估模式）
3. 实现PPO损失计算和梯度更新
4. 支持集成学习（ensemble_num > 1时）

**关键方法**：
- `build_state_from_reps()`: 将各尺度表示拼接为RL状态
- `sample_actions()`: 采样离散和连续动作（训练时）
- `eval_actions()`: 确定性策略（评估时，使用argmax）
- `apply_actions_to_reps()`: 将动作应用到表示上
- `train_step()`: 执行PPO参数更新

### 3. 门控网络（Gating in models.py）

```python
class Gating(nn.Module):
    """
    计算多尺度表示并与RL策略集成
    """
    def forward(..., rl_agent=None, rl_return_info=False, ...):
        # 1. 编码各尺度的图表示
        per_hop_reps = [...]  # List[(N,D)] 长度为H
        
        # 2. 可选：RL策略介入
        if rl_agent is not None:
            state = rl_agent.build_state_from_reps(per_hop_reps)
            action_d, action_c = rl_agent.sample_actions(...)
            per_hop_reps = rl_agent.apply_actions_to_reps(...)
        
        # 3. 拼接并分类为专家权重
        x_cat = torch.cat(per_hop_reps, dim=-1)
        experts_weight = softmax(classifier(x_cat))
        
        return experts_weight, loss_distortion
```

### 4. 奖励与优势计算（exp.py + reward_mrs.py）

#### 4.1 奖励标准化
```python
self.rl_norm = Normalization(shape=1)
# 使用运行均值和标准差标准化奖励
```

#### 4.2 奖励计算策略
```python
# 基础奖励：任务损失的下降量
r_t = prev_loss - loss_t

# 可选：加入失真正则（--hppo_reward_with_distortion）
if hppo_reward_with_distortion:
    r_t = (prev_loss + coef_dis * prev_distortion) - 
          (loss_t + coef_dis * loss_distortion)
```

#### 4.3 优势函数计算（GAE）
```python
def compute_adv_ret_seq(critic, states_seq, rewards_seq, gamma, lam):
    """
    广义优势估计（Generalized Advantage Estimation）
    
    参数:
        gamma: 折扣因子 (default: 0.99)
        lam: GAE参数 (default: 0.95)
    
    返回:
        adv: 优势函数 A(s,a)
        ret: 回报 R(s) = A(s,a) + V(s)
    """
    # TD误差: δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
    # GAE: A_t = Σ_{l=0}^∞ (γλ)^l * δ_{t+l}
```

---

## 强化学习架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        输入图数据                              │
│              (节点特征, 边索引, 标签)                          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              多尺度子图采样 (Sampler)                          │
│   采样不同k-hop邻域: [2-hop, 3-hop, ...]                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              门控网络 (Gating)                                 │
│  ┌───────────────────────────────────────┐                   │
│  │  1. 为每个尺度编码图表示               │                   │
│  │     per_hop_reps = [rep_2hop, rep_3hop]│                   │
│  │                                        │                   │
│  │  2. 构造RL状态 (N, H, D)              │                   │
│  │     state = stack(per_hop_reps)       │                   │
│  └───────────┬───────────────────────────┘                   │
│              │                                                │
│              ▼                                                │
│  ┌─────────────────────────────────┐                         │
│  │   H-PPO 策略 (rl_agent)         │                         │
│  │                                 │                         │
│  │  ┌─────────────────┐            │                         │
│  │  │ ActorDiscrete   │            │                         │
│  │  │ 选择尺度 a_d    │            │                         │
│  │  └────────┬────────┘            │                         │
│  │           │                     │                         │
│  │           ▼                     │                         │
│  │  ┌─────────────────┐            │                         │
│  │  │ ActorContinuous │            │                         │
│  │  │ 生成偏置 a_c    │            │                         │
│  │  └────────┬────────┘            │                         │
│  │           │                     │                         │
│  │           ▼                     │                         │
│  │  ┌─────────────────┐            │                         │
│  │  │ Critic          │            │                         │
│  │  │ 估计价值 V(s)   │            │                         │
│  │  └─────────────────┘            │                         │
│  └─────────────────────────────────┘                         │
│              │                                                │
│              ▼                                                │
│  ┌───────────────────────────────────────┐                   │
│  │  3. 应用动作到表示                     │                   │
│  │     new_reps[a_d] += a_c              │                   │
│  └───────────┬───────────────────────────┘                   │
│              │                                                │
│              ▼                                                │
│  ┌───────────────────────────────────────┐                   │
│  │  4. 分类为专家权重                     │                   │
│  │     experts_weight = softmax(MLP(...)) │                   │
│  └───────────────────────────────────────┘                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              专家混合 (Experts)                                │
│   多个不同曲率流形上的GCN编码器                                 │
│   embeddings = concat([expert_1, ..., expert_K])              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              加权融合                                          │
│   final_emb = embeddings * experts_weight                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              下游任务                                          │
│   - 节点分类 (NC): GNN分类器                                   │
│   - 链接预测 (LP): Fermi-Dirac解码器                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 训练流程

### 主训练循环（exp.py）

#### 节点分类任务（train_cls）

```python
for epoch in range(epochs_cls):
    # ===== 阶段1: 多步序列收集 =====
    T = len(sample_hop)  # 尺度数，如[2,3]则T=2
    states_seq, actions_d, actions_c = [], [], []
    logps_d, logps_c, rewards_seq = [], [], []
    
    # 初始化：计算基准表示（不含RL干预）
    _, base_info = model_gating(..., rl_return_info=True)
    cur_reps = base_info['reps']  # [(N,D), (N,D)]
    
    # 计算初始损失L0（用于奖励计算）
    ew_base0, prev_distortion = model_gating(..., override_reps_per_hop=cur_reps)
    feat_base0 = concat([features, embeddings * ew_base0])
    prev_loss, _, _, _ = cal_cls_loss(model_cls, ..., feat_base0, labels)
    
    # 多步采样循环
    for t in range(T):
        # Step 1: 构造状态
        state_t = rl_agent.build_state_from_reps(cur_reps)  # (N,H,D)
        
        # Step 2: 采样动作
        ew_t, loss_distortion, rl_info = model_gating(
            ...,
            rl_agent=rl_agent,
            rl_return_info=True,
            rl_collect=True,  # 训练模式
            override_reps_per_hop=cur_reps
        )
        # rl_info包含: action_d, action_c, logprob_d, logprob_c, next_reps
        
        # Step 3: 计算当前损失L_t
        feat_t = concat([features, embeddings * ew_t])
        loss_t, _, _, _ = cal_cls_loss(model_cls, ..., feat_t, labels)
        
        # Step 4: 计算奖励 r_t = L_{t-1} - L_t
        r_t = prev_loss.detach() - loss_t.detach()
        # 可选：加入失真正则
        if hppo_reward_with_distortion:
            r_t += coef_dis * (prev_distortion - loss_distortion)
        
        # Step 5: 标准化奖励并存储
        r_vec = torch.full((N,), rl_norm(r_t.item()), device=device)
        states_seq.append(state_t.detach())
        actions_d.append(rl_info['action_d'])
        actions_c.append(rl_info['action_c'])
        logps_d.append(rl_info['logprob_d'])
        logps_c.append(rl_info['logprob_c'])
        rewards_seq.append(r_vec)
        
        # Step 6: 更新到下一状态
        cur_reps = rl_info['next_reps']
        prev_loss = loss_t
        prev_distortion = loss_distortion
    
    # ===== 阶段2: 计算优势与回报 =====
    adv_all, ret_all = compute_adv_ret_seq(
        rl_agent.critic,
        states_seq,
        rewards_seq,
        gamma=0.99,
        lam=0.95
    )
    
    # ===== 阶段3: PPO策略更新 =====
    state_all = torch.cat(states_seq, dim=0)  # (N*T, H, D)
    a_d_all = torch.cat(actions_d, dim=0)     # (N*T,)
    a_c_all = torch.cat(actions_c, dim=0)     # (N*T, D)
    lp_d_all = torch.cat(logps_d, dim=0)      # (N*T,)
    lp_c_all = torch.cat(logps_c, dim=0)      # (N*T,)
    
    experience = (state_all, a_d_all, lp_d_all, a_c_all, lp_c_all, adv_all, ret_all)
    
    args_like = {
        'minibatch_size': 512,
        'policy_update_nums': 5,
        'target_kl_d': 0.01,  # 离散动作KL散度阈值
        'target_kl_c': 0.01,  # 连续动作KL散度阈值
    }
    
    rl_agent.train_step(args_like, 0, experience, rl_optim)
    
    # ===== 阶段4: 主任务优化 =====
    # 使用最后一步的损失作为训练目标
    loss = prev_loss + coef_dis * loss_distortion
    loss.backward()
    optimizer_cls.step()
    r_optim.step()  # Riemannian优化器（专家网络）
    optimizer_gating.step()
```

#### 链接预测任务（train_lp）

流程类似，主要区别：
1. 损失函数：使用 `cal_lp_loss` 替代 `cal_cls_loss`
2. 边采样：每个epoch随机采样负边
3. 评估指标：AUC和AP替代准确率

### PPO更新细节（hppo_mrs.py → train_step）

```python
def train_step(args_like, a_idx, experience, optim):
    state, action_d, logprob_old_d, action_c, logprob_old_c, adv, ret = experience
    
    # 优势归一化（减少方差）
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    
    for update_iter in range(policy_update_nums):  # 默认5次
        # 随机打乱数据
        index = torch.randperm(len(state))
        
        for minibatch in split(state, minibatch_size):  # 默认512
            # ===== 离散Actor损失 =====
            logprob_new_d, entropy_d = actor_d.logprob_entropy(state, action_d)
            ratio_d = torch.exp(logprob_new_d - logprob_old_d)
            
            # PPO clip目标
            surr1_d = ratio_d * adv
            surr2_d = torch.clamp(ratio_d, 1-eps_clip, 1+eps_clip) * adv
            actor_loss_d = -torch.min(surr1_d, surr2_d) - coeff_ent * entropy_d
            
            # ===== 连续Actor损失 =====
            logprob_new_c, entropy_c = actor_c.logprob_entropy(state, action_d, action_c)
            ratio_c = torch.exp(logprob_new_c - logprob_old_c)
            
            surr1_c = ratio_c * adv
            surr2_c = torch.clamp(ratio_c, 1-eps_clip, 1+eps_clip) * adv
            actor_loss_c = -torch.min(surr1_c, surr2_c)
            
            # ===== Critic损失 =====
            value = critic(state)
            critic_loss = MSE(value, ret) * coeff_critic
            
            # ===== 总损失与优化 =====
            loss = actor_loss_d + actor_loss_c + critic_loss
            optim.zero_grad()
            loss.backward()
            clip_grad_norm_(parameters, max_norm=5.0)
            optim.step()
            
            # 早停检查（基于KL散度）
            if kl_divergence > target_kl:
                break
```

### 评估流程

```python
# 评估模式：使用确定性策略
rl_agent.train_or_eval('eval')

T = len(sample_hop)
_, base_info_val = model_gating(..., rl_return_info=True)
cur_reps = base_info_val['reps']

# 多步推理（不采样，使用argmax）
for t in range(T):
    ew_t, rl_info_val = model_gating(
        ...,
        rl_agent=rl_agent,
        rl_return_info=True,
        rl_collect=False,  # 评估模式
        override_reps_per_hop=cur_reps
    )
    cur_reps = rl_info_val['next_reps']
    experts_weight = ew_t  # 使用最后一步的权重

# 计算最终嵌入
embeddings = embeddings * experts_weight
features = concat([features, embeddings], -1)

# 评估
_, val_acc, val_wf1, val_mf1 = cal_cls_loss(model_cls, ..., features, labels)
```

---

## 配置参数

### RL相关超参数（main.py）

```python
# 启用H-PPO
--use_hppo                    # 是否启用强化学习（默认False）

# PPO核心参数
--hppo_eps_clip_d 0.2         # 离散动作裁剪系数
--hppo_eps_clip_c 0.2         # 连续动作裁剪系数
--hppo_coeff_critic 0.5       # Critic损失系数
--hppo_coeff_ent_d 1e-3       # 离散动作熵正则系数

# 网络初始化
--hppo_init_log_std -2.0      # 连续动作初始log标准差
--hppo_max_norm_grad 5.0      # 梯度裁剪阈值

# 集成学习（可选）
--hppo_penalty_alpha_d 0.0    # 离散动作集成惩罚
--hppo_penalty_alpha_c 0.0    # 连续动作集成惩罚

# 学习率
--hppo_actor_d_lr 1e-3        # 离散Actor学习率
--hppo_actor_c_lr 1e-3        # 连续Actor学习率
--hppo_critic_lr 1e-3         # Critic学习率

# 训练策略
--hppo_minibatch_size 512     # 小批量大小
--hppo_update_nums 5          # 每次更新的epoch数
--hppo_target_kl_d 0.01       # 离散动作早停KL阈值
--hppo_target_kl_c 0.01       # 连续动作早停KL阈值
--hppo_policy_decay down      # 学习率衰减策略（none/down）

# 奖励与优势
--hppo_gamma 0.99             # 折扣因子
--hppo_lam 0.95               # GAE λ参数
--hppo_reward_with_distortion # 奖励是否包含失真正则
```

### 使用示例

```bash
# 节点分类 + H-PPO
python main.py \
    --downstream_task NC \
    --dataset Cora \
    --use_hppo \
    --hppo_actor_d_lr 1e-3 \
    --hppo_actor_c_lr 1e-3 \
    --hppo_critic_lr 1e-3 \
    --hppo_gamma 0.99 \
    --hppo_lam 0.95 \
    --hppo_reward_with_distortion \
    --sample_hop 2 3

# 链接预测 + H-PPO
python main.py \
    --downstream_task LP \
    --dataset Cora \
    --use_hppo \
    --hppo_minibatch_size 512 \
    --hppo_update_nums 5 \
    --hppo_policy_decay down \
    --sample_hop 2 3 4
```

---

## 关键设计决策

### 1. 为什么使用分层动作空间？

- **离散动作**：快速选择粗粒度的信息范围（2-hop局部 vs 3-hop全局）
  - 作用于**子图选择**：决定使用哪个尺度的表示
  - 输出：离散的尺度索引（0, 1, 2, ...）
  - 可解释性强："节点A选了3-hop邻域"
  
- **连续动作**：在选定尺度基础上优化表示向量
  - 作用于**特征空间**：调整D维表示向量的数值
  - 输出：连续的调整向量（例：[+0.1, -0.05, +0.08, ...]）
  - ⚠️ **不改变hop数**：2-hop还是2-hop，只是特征值被优化
  
- **优势**：
  - 粗细结合：离散决策定范围，连续优化提质量
  - 搜索高效：离散空间小（几个尺度），连续梯度平滑
  - 任务驱动：端到端学习最适合下游任务的表示

### 2. 多步序列收集的意义

- **步数 = 尺度数**：每个尺度都有机会被优化
- **奖励信号**：每步都能获得任务损失的即时反馈
- **渐进优化**：从粗到细逐步精化表示

### 3. 奖励设计

```python
# 基础版本：纯任务损失下降
r_t = L_{t-1} - L_t

# 增强版本：考虑失真正则（保持几何结构）
r_t = (L_{t-1} + α*D_{t-1}) - (L_t + α*D_t)
```

- **直观性**：损失下降 → 正奖励 → 强化该动作
- **稳定性**：通过Running Mean Std标准化

### 4. 与图任务的深度耦合

- **状态**：多尺度图表示（经过GNN编码）
- **动作**：直接作用于表示空间
- **奖励**：来自下游任务的监督信号
- **优势**：端到端学习，无需预训练

---

## 日志与调试

### 重要日志信息

```python
# exp.py中的关键日志
logger.info(f"Epoch {epoch}: train_loss={loss.item()}, train_accuracy={acc}")
logger.info(f"Epoch {epoch}: val_accuracy={acc}, val_wf1={wf1}, val_mf1={mf1}")
logger.info(f"best_epoch={best_epoch}, test_accuracy={test_acc}")

# 可以添加的调试信息（建议）
logger.debug(f"Reward at step {t}: {r_t.item():.4f}")
logger.debug(f"Advantage mean: {adv.mean():.4f}, std: {adv.std():.4f}")
logger.debug(f"KL divergence: d={kl_d:.4f}, c={kl_c:.4f}")
logger.debug(f"Action distribution: {action_d.bincount()}")
```

### 常见问题排查

1. **奖励始终为负**：检查 `rl_norm` 是否正确初始化
2. **策略不更新**：验证 `rl_optim` 是否包含所有参数
3. **KL散度过大**：降低学习率或增加 `eps_clip`
4. **训练不稳定**：增加 `minibatch_size` 或减少 `policy_update_nums`

---

## 总结

本项目实现了一个完整的**图上强化学习框架**，核心特点：

1. ✅ **分层PPO**：离散+连续动作空间
   - 离散动作：在预采样的子图中选择使用哪个尺度（2-hop/3-hop/...）
   - 连续动作：优化选定尺度表示的D维特征向量
   - ⚠️ **关键理解**：连续动作调整的是"特征值"，不是"hop数"！
   
2. ✅ **多分辨率采样**：自适应选择k-hop邻域
   - 采样在训练前完成（固定拓扑）
   - RL在特征空间工作（优化表示）
   
3. ✅ **端到端训练**：RL与图表示联合优化
   - 状态：多尺度图表示
   - 动作：尺度选择+表示微调
   - 奖励：下游任务损失的下降量
   
4. ✅ **双任务支持**：节点分类和链接预测

5. ✅ **灵活配置**：丰富的超参数选项

### 核心设计理念

```
子图采样（固定）  →  GNN编码  →  RL优化表示  →  下游任务
   2-hop子图          向量1        选择+微调       分类/链接预测
   3-hop子图          向量2            ↓
                                   最优表示
```

该框架可扩展到其他图学习任务，如图分类、社区发现等。

