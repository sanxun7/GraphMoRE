# GraphMoRE 强化学习组件与流程文档

## 目录
- [概述](#概述)
- [核心组件](#核心组件)
- [工作流程](#工作流程)
- [算法细节](#算法细节)
- [配置参数](#配置参数)
- [关键代码位置](#关键代码位置)

---

## 概述

本项目使用**分层近端策略优化（H-PPO, Hierarchical Proximal Policy Optimization）**算法，通过强化学习自适应地控制多尺度子图采样和曲率选择，以优化图表示学习的效果。

### 强化学习的作用

1. **多尺度采样控制**：动态选择不同跳数（hop）的子图采样尺度
2. **曲率权重调节**：自适应调整不同曲率空间的专家权重
3. **端到端优化**：根据下游任务（节点分类NC或链接预测LP）的性能反馈，自动优化模型参数

---

## 核心组件

### 1. MultiScaleHPPOController (`rl_agent.py`)

**主要职责**：实现H-PPO算法，协调离散和连续动作策略的学习。

**关键属性**：
- `actor_ds`: 离散动作策略网络集合（选择采样尺度）
- `actor_cs`: 连续动作策略网络集合（调节曲率权重）
- `critic`: 状态价值网络（评估状态价值）
- `buffer`: 轨迹缓存（存储经验数据）

**核心方法**：
- `act(state)`: 根据状态选择动作，返回 `hop_mask` 和 `curvature_bias`
- `record_reward(reward, done)`: 记录奖励并触发策略更新
- `start_episode()` / `end_episode()`: 管理训练回合

### 2. ActorDiscrete (`rl_agent.py`)

**功能**：离散动作策略网络，负责选择多分辨率采样尺度。

**输入**：
- `state`: `[B, max_hop, input_dim]` - 多尺度子图的状态表示
- `mask`: `[B, max_hop]` - 无效位置的掩码

**输出**：
- 每个尺度的选择概率分布

**网络结构**：
```
Linear(input_dim → hidden) → Tanh
→ Linear(hidden → hidden_mid) → Tanh  
→ Linear(hidden_mid → 1) → Softmax
```

### 3. ActorContinuous (`rl_agent.py`)

**功能**：连续动作策略网络，负责调节曲率空间的权重偏置。

**输入**：
- `state`: `[B, max_hop, input_dim]` - 多尺度子图的状态表示
- `action_d`: `[B]` - 离散动作（选中的尺度索引）

**输出**：
- `[B, num_curvatures]` - 每个曲率的权重偏置向量

**网络结构**：
```
Linear(input_dim → hidden) → Tanh
→ Linear(hidden → hidden) → Tanh
→ Linear(hidden → action_dim) → Tanh * max_action
```

**策略分布**：使用高斯分布，可学习的对数标准差 `log_std`

### 4. Critic (`rl_agent.py`)

**功能**：状态价值网络，评估当前状态的价值。

**输入**：
- `state`: `[B, max_hop, input_dim]` - 多尺度子图的状态表示

**输出**：
- 标量价值估计

**网络结构**：
```
Flatten → Linear(feat_dim → hidden) → ReLU
→ Linear(hidden → hidden//2) → ReLU
→ Linear(hidden//2 → 1)
```

### 5. TransitionBuffer (`rl_agent.py`)

**功能**：存储训练轨迹数据，支持H-PPO的批量更新。

**存储内容**：
- `state`: 状态
- `mask`: 掩码
- `action_d`: 离散动作
- `logprob_d`: 离散动作对数概率
- `action_c`: 连续动作
- `logprob_c`: 连续动作对数概率
- `value`: 价值估计
- `reward`: 奖励
- `done`: 回合结束标志

### 6. Gating 模块 (`models.py`)

**功能**：门控网络，整合RL输出的控制信号。

**关键方法**：
- `set_rl_inputs(hop_mask, curvature_bias)`: 接收RL控制信号
- `forward(...)`: 在计算专家权重时应用RL控制

**RL信号的应用**：
1. **hop_mask**: 在子图特征池化后，对每个尺度的特征进行加权
2. **curvature_bias**: 对专家权重进行偏置调整，然后归一化

---

## 工作流程

### 整体训练流程

```
初始化阶段
    ↓
[数据加载] → [子图采样] → [初始化RL控制器]
    ↓
┌─────────────────────────────────────────┐
│         实验迭代循环 (exp_iters)         │
│  ┌───────────────────────────────────┐  │
│  │  1. 构建RL状态 (_build_rl_state)  │  │
│  │  2. RL选择动作 (controller.act)   │  │
│  │  3. 设置Gating的RL输入             │  │
│  │  4. 训练下游任务模型               │  │
│  │  5. 计算奖励 (_calc_reward)       │  │
│  │  6. 记录奖励 (record_reward)      │  │
│  │  7. 更新RL策略 (自动触发)          │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
    ↓
[保存奖励曲线] → [输出最终结果]
```

### 详细步骤

#### 步骤1: 状态构建 (`_build_rl_state`)

从多尺度子图中提取统计特征作为RL状态：

```python
对于每个采样尺度 hop:
    - 尺度归一化: hop / max_hop
    - 节点比例: node_count / total_nodes  
    - 边密度: edge_count / node_count
    - 特征均值: mean(features)
    - 特征标准差: std(features)
```

**输出形状**: `[1, num_hops, rl_state_dim]`

#### 步骤2: 动作选择 (`controller.act`)

1. **离散动作**：ActorDiscrete根据状态选择采样尺度
   ```python
   action_d = Categorical(probs).sample()  # 选择尺度索引
   ```

2. **连续动作**：ActorContinuous根据选中的尺度生成曲率偏置
   ```python
   action_c = Normal(mean, std).rsample()  # 生成曲率权重偏置
   ```

3. **动作转换**：
   - `hop_mask`: 将离散动作转换为尺度权重掩码
     - 选中尺度权重 = 1.0
     - 未选中尺度权重 = `rl_min_hop_scale` (默认0.2)
   - `curvature_bias`: 将连续动作通过softmax转换为曲率权重偏置

#### 步骤3: 应用到模型 (`Gating.set_rl_inputs`)

RL控制信号被注入到Gating模块：

```python
# 在Gating.forward中：
for i, scale in enumerate(subgraph_scales):
    x_scale = pool(encode(subgraph[i]))
    if rl_hop_mask is not None:
        x_scale = x_scale * rl_hop_mask[i]  # 尺度加权
    x.append(x_scale)

expert_weights = softmax(classifier(x))
if rl_curv_bias is not None:
    expert_weights = expert_weights * rl_curv_bias  # 曲率偏置
    expert_weights = normalize(expert_weights)
```

#### 步骤4: 奖励计算 (`_calc_reward`)

根据下游任务性能计算奖励：

**奖励类型1: 基于损失 (loss)**
```python
if reward_type == 'loss':
    reward = prev_loss - current_loss  # 损失下降 = 正奖励
```

**奖励类型2: 基于指标 (metric)**
```python
if reward_type == 'metric':
    reward = current_metric - prev_metric  # 指标提升 = 正奖励
```

**任务特定奖励**：
- **节点分类 (NC)**:
  - `reward = cls_metric + rl_aux_lp_coef * lp_metric`
  - 同时考虑分类准确率和辅助链接预测指标
- **链接预测 (LP)**:
  - `reward = lp_metric` (AUC或AP)

#### 步骤5: 策略更新 (`_update`)

当缓冲区积累足够经验（`rl_batch_size`）时触发更新：

1. **计算优势函数 (GAE)**:
   ```python
   advantages = compute_gae(rewards, values, dones)
   returns = compute_returns(rewards, gamma)
   ```

2. **更新离散策略**:
   ```python
   ratios_d = exp(logprob_new - logprob_old)
   loss_d = -min(ratios_d * adv, clip(ratios_d) * adv) 
            - entropy_coef * entropy
            + ensemble_penalty  # 如果启用集成
   ```

3. **更新连续策略**:
   ```python
   ratios_c = exp(logprob_new - logprob_old)
   loss_c = -min(ratios_c * adv, clip(ratios_c) * adv)
            - entropy_coef * entropy
            + ensemble_penalty
   ```

4. **更新价值网络**:
   ```python
   loss_v = MSE(values, returns)
   ```

---

## 算法细节

### H-PPO (Hierarchical PPO)

本项目实现了分层PPO算法，特点：

1. **分层动作空间**：
   - 第一层：离散动作（选择尺度）
   - 第二层：连续动作（调节曲率，依赖于第一层选择）

2. **策略集成**：
   - 支持多个actor的集成学习
   - 通过KL散度惩罚保持集成一致性
   - 集成策略用于更稳定的动作选择

3. **熵退火**：
   - 训练初期：高熵（探索）
   - 训练后期：低熵（利用）
   - 通过线性退火 `log_std` 实现

### GAE (Generalized Advantage Estimation)

使用GAE计算优势函数，平衡偏差和方差：

```python
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
GAE_t = delta_t + (gamma * lambda) * GAE_{t+1}
```

**参数**：
- `gamma`: 折扣因子（默认0.95）
- `lambda`: GAE系数（默认0.9）

### 奖励设计

**增量奖励**：奖励 = 当前信号 - 前一步信号
- 鼓励性能持续改进
- 避免绝对数值的影响

**信号类型**：
- `loss`: 验证损失（越小越好）
- `metric`: 验证指标（越大越好，如准确率、AUC）

---

## 配置参数

### 基础配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rl_enable` | `True` | 是否启用强化学习 |
| `rl_state_dim` | `5` | 状态向量维度 |
| `rl_reward_type` | `'loss'` | 奖励类型：'loss' 或 'metric' |

### 网络结构

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rl_ensemble_num` | `1` | 策略集成中actor的数量 |
| `rl_max_action` | `1.0` | 连续动作的最大值 |
| `rl_init_log_std` | `0.0` | 连续策略初始对数标准差 |
| `rl_target_log_std` | `-5.0` | 方差退火的目标对数标准差 |

### 学习率

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rl_actor_d_lr` | `5e-4` | 离散策略学习率 |
| `rl_actor_c_lr` | `5e-4` | 连续策略学习率 |
| `rl_critic_lr` | `1e-3` | 价值网络学习率 |

### PPO超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rl_eps_clip_d` | `0.2` | 离散策略裁剪阈值 |
| `rl_eps_clip_c` | `0.2` | 连续策略裁剪阈值 |
| `rl_entropy_coef` | `0.01` | 熵正则化系数 |
| `rl_batch_size` | `4` | 累计多少步后更新策略 |
| `rl_max_grad_norm` | `0.5` | 梯度裁剪阈值 |

### 奖励相关

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rl_reward_scale` | `1.0` | 奖励缩放系数 |
| `rl_aux_lp_coef` | `0.2` | NC任务下LP奖励的混合比 |
| `rl_gamma` | `0.95` | 折扣因子 |
| `rl_lambda` | `0.9` | GAE的lambda系数 |

### 动作空间

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rl_min_hop_scale` | `0.2` | 未被选中尺度的保底权重 |
| `rl_temperature` | `1.0` | 连续动作softmax温度 |

### 集成学习

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rl_penalty_alpha_d` | `0.0` | 离散策略KL约束系数 |
| `rl_penalty_alpha_c` | `0.0` | 连续策略KL约束系数 |

---

## 关键代码位置

### 1. RL控制器初始化
- **文件**: `exp.py`
- **方法**: `_init_rl_controller()` (第30-40行)
- **调用时机**: 在子图采样后，训练开始前

### 2. 状态构建
- **文件**: `exp.py`
- **方法**: `_build_rl_state()` (第42-70行)
- **调用时机**: 每个训练epoch开始时

### 3. 动作选择与应用
- **文件**: `exp.py`, `models.py`
- **方法**: 
  - `controller.act()` (`rl_agent.py` 第209行)
  - `Gating.set_rl_inputs()` (`models.py` 第114-117行)
  - `Gating.forward()` (`models.py` 第119-140行)
- **调用时机**: 每个训练epoch的前向传播阶段

### 4. 奖励计算
- **文件**: `exp.py`
- **方法**: 
  - `_compose_reward_signal()` (第72-90行)
  - `_calc_reward()` (第92-108行)
- **调用时机**: 每个验证阶段后

### 5. 策略更新
- **文件**: `rl_agent.py`
- **方法**: `_update()` (第327-379行)
- **触发条件**: 
  - 缓冲区大小 >= `rl_batch_size`
  - 回合结束 (`done=True`)

### 6. 训练循环集成
- **文件**: `exp.py`
- **方法**: 
  - `train_cls()` (第277-397行) - 节点分类训练
  - `train_lp()` (第417-522行) - 链接预测训练
- **关键点**: 
  - 每个epoch调用 `controller.act()` 获取动作
  - 每个验证阶段计算奖励并调用 `record_reward()`
  - 支持熵退火（线性降低探索率）

### 7. 奖励曲线保存
- **文件**: `exp.py`
- **方法**: `_save_reward_curve()` (第110-135行)
- **调用时机**: 所有实验迭代结束后

---

## 训练示例

### 节点分类任务 (NC)

```python
# 在每个epoch中：
1. 构建RL状态（多尺度子图统计特征）
2. RL选择动作（hop_mask, curvature_bias）
3. 设置Gating的RL输入
4. 前向传播：
   - Experts生成多曲率嵌入
   - Gating计算专家权重（应用RL控制）
   - 拼接特征并分类
5. 反向传播更新模型
6. 验证阶段：
   - 计算验证准确率
   - 计算奖励（准确率提升）
   - 记录到RL控制器
7. 当缓冲区满时，更新RL策略
```

### 链接预测任务 (LP)

```python
# 流程类似，但：
- 奖励基于AUC/AP指标
- 不需要分类器，直接使用嵌入预测链接
```

---

## 注意事项

1. **状态维度对齐**：确保 `rl_state_dim` 与状态向量维度匹配，代码会自动padding或截断

2. **奖励缩放**：根据任务调整 `rl_reward_scale`，避免奖励过大或过小

3. **探索与利用平衡**：
   - 初始阶段：高熵（`rl_init_log_std=0.0`）
   - 后期阶段：低熵（`rl_target_log_std=-5.0`）
   - 通过线性退火实现平滑过渡

4. **批量更新**：`rl_batch_size` 控制策略更新频率，太小可能不稳定，太大可能延迟学习

5. **集成学习**：当 `rl_ensemble_num > 1` 时，需要设置 `rl_penalty_alpha_d` 和 `rl_penalty_alpha_c` 来约束集成一致性

---

## 总结

本项目的强化学习组件通过H-PPO算法实现了对多尺度图表示学习的自适应控制。核心思想是：

1. **状态**：多尺度子图的统计特征
2. **动作**：尺度选择和曲率权重调节
3. **奖励**：下游任务性能的改进量
4. **策略**：分层PPO算法，支持离散+连续动作空间

这种设计使得模型能够根据数据特性自动选择最优的采样尺度和曲率组合，从而提升图表示学习的性能。

