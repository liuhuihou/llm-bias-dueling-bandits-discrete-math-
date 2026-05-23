# 电影评分偏置修正：基于两两比较与自适应去偏的评分框架

## 1. 项目背景与动机

在线电影平台（如豆瓣、IMDb、Letterboxd）每天汇聚海量用户评分，但这些评分普遍受到以下三类系统性偏置的干扰：

**展示顺序偏置（Attention bias）**
比较两部电影时，用户倾向于对第一个看到的选项给出更高评价——即注意力锚定效应。

**从众效应偏置（Echo-chamber bias）**
用户评分受已有评分影响：当某部电影当前评分较高时，后续用户会给出更高评分，形成"好评雪球"效应。这正是课程要求中提到的典型案例。

**极化反馈偏置（Polarisation bias）**
当两部电影真实质量接近时，持极端意见的用户主导了投票，导致近似平局的比较结果被放大，无法稳定区分两者。

这三类偏置共同导致：平台最终呈现的评分排名可能严重偏离电影的真实质量顺序。

**项目目标**：建模上述偏置机制，设计能够在受污染的两两投票反馈下仍然稳定输出修正评分的算法。

---

## 2. 问题建模

### 2.1 两两比较框架

电影数量庞大，直接对每部电影收集独立评分效率低且受主观尺度影响大。本项目采用**两两比较（pairwise comparison）**作为核心反馈机制：系统每轮选择两部电影展示给一组用户，用户投票选出更偏好的一部，系统记录投票份额。

设候选电影集合为 `M = {1, 2, ..., K}`，真实偏好矩阵 `P` 满足：
```
P_ij = Pr(电影 i 在真实质量上优于电影 j)
P_ij + P_ji = 1,   P_ii = 0.5
```

若存在 `m*` 使 `P_{m*,j} > 0.5` 对所有 `j ≠ m*` 成立，则 `m*` 为最优电影（Condorcet winner）。

### 2.2 合成电影质量矩阵

每部电影被分配潜在质量分数 `u_i ~ N(0,1)`，两两偏好通过 logistic 函数生成：
```
P_ij = sigmoid((u_i - u_j) / τ)
```
再叠加电影级独立扰动 `ε_i ~ N(0, σ_base)`，使偏好矩阵不完全由一维质量决定。

### 2.3 偏置反馈模型

一次投票会话的完整生成过程（movie i 展示在前）：

```
歧义权重:   A_ij = clip(1 - |P_ij - 0.5| / τ_sel, 0, 1)
判断噪声:   P_noisy = clip(P_ij + N(0, σ*(1+A_ij)), p_min, p_max)
展示偏置:   Δ_pos = b_pos                                    （常数，偏向第一个展示项）
从众偏置:   Δ_peer = b_conf · tanh(3·(pop_i(t) - pop_j(t)))  （动态，随历史好评变化）
极化偏置:   Δ_sel  = 0.5·b_sel·A_ij·(0.65·social_cue + 0.35) （仅在近似平局时激活）
观测概率:   Q_ij(t) = clip(P_noisy + Δ_pos + Δ_peer + Δ_sel, p_min, p_max)
投票结果:   votes_i ~ Binomial(audience_size, Q_ij(t))
```

### 2.4 Regret 指标

设 `m*` 为最优电影，第 `t` 轮比较 `(i_t, j_t)` 的即时 regret：
```
r_t = max(P[m*, i_t] + P[m*, j_t] - 1, 0)
R_T = sum_{t=1}^T r_t
```

---

## 3. 算法设计

三个算法复杂度递增，分别对应"无优化"、"优化探索"和"显式去偏"三个层次。

### 3.1 RRE：轮询经验评分（Round-Robin Empirical）

**定位**：最简单基线，代表大多数平台的隐性行为。

**选对策略**：按固定顺序轮流遍历所有电影对，确保每对获得相同比较次数。

**偏好估计**：原始投票份额的经验均值，不做任何修正。

**输出评分**：
```
R_i = 1 + 9 · mean_j P_hat[i,j]
```

RRE 的问题：均匀探索效率较低；不修正任何偏置，评分结果直接受三类偏置污染。

---

### 3.2 UCB-R：基于置信上界的评分（UCB-based Rating）

**定位**：智能探索基线，分离"聪明探索"与"去偏"各自的贡献。

**核心思路**：最小化所有电影评分的总均方误差。对于电影 i，评分方差为：
```
Var(R_i) = (9/(K-1))^2 · sum_{j≠i} Var(P_hat[i,j])
```
最小化 sum_i Var(R_i) 等价于优先比较使 `Var(P_hat[i,j])` 最大的电影对。

**选对优先级**：
```
priority(i,j) = P_hat[i,j]·(1-P_hat[i,j]) / N_ij  +  α·log(t) / N_ij
```
- 第一项：方差估计，偏好估计越接近 0.5 优先级越高
- 第二项：UCB 探索奖励，比较次数越少优先级越高

**偏好估计**：原始投票份额（无修正），以便与 SBCR 形成对照。

---

### 3.3 SBCR：对称去偏评分（Symmetric Bias-Corrected Rating）

**定位**：本项目提出的完整方法，针对电影评分中三类偏置逐一设计修正。

#### 第一步：展示方向对称化（消除展示顺序偏置）

分方向记录投票：
- `fwd_wins[i,j]`：电影 i 在前（j 在后）时，i 的加权投票份额
- `fwd_counts[i,j]`：以上配置的总权重

对称化估计（两个方向都有记录时）：
```
P_sym[i,j] = 0.5 · (P_fwd[i,j] + (1 - P_rev[i,j]))
```

其中：
- `P_fwd[i,j] ≈ P_ij + b_pos + Δ_peer`（i 在前时的投票率）
- `1 - P_rev[i,j] ≈ P_ij - b_pos + Δ_peer`（j 在前时 i 的等效投票率）

两式相加除 2，`b_pos` 精确消除：
```
P_sym[i,j] = P_ij + Δ_peer(i,j,t)         ← 展示顺序偏置完全消除
```

当某对只有单方向记录时，退化为该单向估计（有偏但好于 0.5）。

#### 第二步：从众效应修正（超额流行度差距法）

**关键洞察**：如果没有从众效应，电影 i 的累计胜率应等于当前偏好估计所预测的质量分数：
```
arm_share_i  ≈  mean_j P_sym[i,j]    （无从众效应时）
```
任何超出这一预测的部分即为从众效应带来的虚假膨胀：
```
residual_i   = arm_share_i - mean_j P_sym[i,j]   ≈ 从众效应对 i 的平均贡献
B_peer[i,j]  = γ · (residual_i - residual_j)
P_corr[i,j]  = P_sym[i,j] - support · B_peer[i,j]
```

其中 `support = N_ij / (N_ij + min_count)` 是样本支持度，避免稀疏对子被过度修正。

**当 b_conf = 0 时**（无从众效应）：arm_share ≈ P_sym.mean(axis=1)，residual ≈ 0，B_peer ≈ 0。算法不会产生错误修正。

#### 第三步（第四步对应选对策略）：不对称性奖励 + 方差 UCB

选对优先级结合三项：
```
priority(i,j) = P_corr·(1-P_corr)/N + α·log(t)/N      # 方差-UCB（同 UCB-R 但用 P_corr）
              + sym_bonus / N  （仅当该对只记录了一个方向）  # 激励双向对称覆盖
              + 0.5·|B_peer[i,j]| / (N+1)                  # 从众估计不确定时多探索
```

不对称性奖励促使算法尽快完成每对电影的双向比较，最大化展示方向修正的效果。

---

## 4. 理论分析

### 4.1 展示顺序偏置消除的精确性

设展示顺序偏置为常数 `b_pos`：
```
E[P_fwd[i,j]]   = P_ij + b_pos + Δ_peer(i,j)
E[1-P_rev[i,j]] = P_ij - b_pos + Δ_peer(i,j)
E[P_sym[i,j]]   = P_ij + Δ_peer(i,j)           （b_pos 精确消除）
```
无论 b_pos 大小，只要正反两个方向各有足够样本，位置偏置均可精确消除。

### 4.2 从众效应修正的一致性

记 pop_i(T) 为最终累计好评率，则：
```
residual_i  = arm_share_i - expected_i
            ≈ b_conf · mean_j tanh(3·(pop_i(T) - pop_j(T)))
```

因此：
```
B_peer[i,j] = γ · (residual_i - residual_j)
            ≈ γ · b_conf · (mean peer boost on i - mean peer boost on j)
```

这是对配对从众效应的合理估计。当样本量 N → ∞ 时，B_peer → 真实从众偏置。

### 4.3 样本复杂度讨论

SBCR 的修正依赖双向对称观测，因此在初期（轮数较少）时方差高于不做修正的基线。这是**偏置修正的固有代价**：

- 早期（样本稀疏）：P_sym 的方差略高于 P_hat（每个方向的样本更少）
- 中后期（样本充足）：P_sym 精确消除位置偏置，P_corr 进一步减小从众效应带来的系统误差
- 不对称性奖励加速双向覆盖，缩短收敛时间

---

## 5. 偏置场景

| 场景名 | 启用偏置 | 现实对应 |
|---|---|---|
| `attention_bias` | 展示顺序偏置（强）| 搜索结果靠前的电影获得更多点击 |
| `echo_chamber` | 从众效应偏置（强）| 高评分电影持续获得更高分（"好评雪球"）|
| `polarisation` | 极化反馈偏置（强）| 质量接近的电影被极端意见用户主导 |
| `realistic` | 三者同时叠加（中等）| 真实平台的综合偏置环境 |

---

## 6. 三维对比评价体系

### 6.1 准确性（Accuracy）

| 指标 | 含义 |
|---|---|
| Rating MAE | 估计评分与真实评分的平均绝对误差（1–10 分制）|
| Rank Spearman ρ | 估计排名与真实排名的相关系数（越近 1 越好）|
| Top-1 accuracy | 最终识别出的最优电影是否与真实最优一致 |
| Mean rank error | 所有电影名次偏差的均值 |

### 6.2 效率（Efficiency）

| 指标 | 含义 |
|---|---|
| Cumulative regret | 算法偏离最优电影对的累计损失 |
| Final regret | 实验结束时的累计 regret |
| AUC regret | 累计 regret 曲线下面积，衡量全程整体效率 |

### 6.3 鲁棒性（Robustness）

| 指标 | 含义 |
|---|---|
| Relative final regret | 算法 final regret 相对同场景最佳算法的比例 |
| 跨场景方差 | 指标在四个偏置场景间的标准差（越小越稳定）|
| Decision flip rate | 偏置后观测概率跨越 0.5 边界的比例 |

---

## 7. 从偏好矩阵到星级评分

最终输出每部电影在 1–10 分制下的修正评分：
```
q_i      = mean_{j≠i} P_corr[i,j]    （平均胜率，衡量相对质量）
R_i      = 1 + 9 · q_i               （映射到 1–10 分制）
```

这是**相对评分**：反映该电影在当前候选集合中相对其他电影的质量。对于足够大的候选集，相对评分会逼近绝对质量排名。

---

## 8. 快速开始

```bash
pip install -r requirements.txt

# 快速调试（1000 轮，8 次 Monte Carlo）
python -m src.main --quick

# 完整实验（5000 轮，24 次 Monte Carlo）
python -m src.main

# 自定义规模
python -m src.main --horizon 8000 --runs 30
```

---

## 9. 输出文件

| 路径 | 内容 |
|---|---|
| `results/figures/regret_*.png` | 各偏置场景下的累计遗憾曲线 |
| `results/figures/movie_ratings_*.png` | 各场景下修正评分 vs 真实质量对比图 |
| `results/figures/robustness_comparison.png` | 各算法相对最终遗憾对比图 |
| `results/figures/bias_diagnostics_*.png` | 每个场景下的偏置诊断指标 |
| `results/raw_data/movie_ratings.csv` | 每部电影在每个场景下的修正评分（1–10）|
| `results/raw_data/bias_effect_summary.csv` | Rating MAE、Spearman ρ、Flip Rate 汇总表 |
| `results/raw_data/statistical_report.json` | 最终遗憾、AUC 和显著性检验结果 |
