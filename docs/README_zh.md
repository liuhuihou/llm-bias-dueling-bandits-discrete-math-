# llm-bias-dueling-bandits

面向 LLM 人类偏好评测偏置的 dueling bandits 实验框架。项目通过合成偏好矩阵和可配置的人类反馈偏置，比较 RUCB、BS-UCB 与 DBS-UCB 在不同偏置场景下的累计遗憾、最终遗憾和鲁棒性表现。

## 项目目标

在 LLM 输出偏好评测中，人工或类人工反馈常受到展示位置、群体从众和选择性反馈等因素影响。该项目将这些因素建模为 biased dueling environment，并评估偏置鲁棒算法能否在受污染反馈下更稳定地识别优选模型或策略。

核心流程包括：

1. 生成真实 pairwise preference matrix。
2. 在不同偏置机制下模拟人类二元比较反馈。
3. 运行多种 dueling bandit 算法。
4. 统计累计遗憾、AUC regret、最终遗憾和 paired permutation test。
5. 输出图表、CSV、JSON、Markdown 和 LaTeX 表格。

## 目录结构

```text
llm-bias-dueling-bandits/
├── config/
│   └── config.py                 # 实验规模、随机种子、算法参数和偏置场景
├── docs/
│   ├── project_report.tex         # 项目报告 LaTeX 源文件
│   ├── project_report.pdf         # 已编译报告
│   └── references.bib             # 参考文献
├── results/
│   ├── figures/                   # 实验图像输出
│   ├── logs/                      # 预留日志目录
│   └── raw_data/                  # CSV/JSON/Markdown/LaTeX 数据输出
├── src/
│   ├── main.py                    # 实验入口
│   ├── algorithms/                # RUCB、BS-UCB、DBS-UCB
│   ├── data/                      # 合成数据与 MT-Bench 预留加载器
│   └── utils/                     # 指标、绘图和报告生成工具
├── requirements.txt
└── README.md
```

## 环境要求

- Python 3.10 或更高版本
- NumPy
- Matplotlib

安装依赖：

```bash
pip install -r requirements.txt
```

建议使用虚拟环境：

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux：

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

如果 Windows 上的 `python` 命令指向 Microsoft Store 占位程序，可以直接调用虚拟环境解释器：

```powershell
.\.venv\Scripts\python.exe -m src.main --quick
```

## 快速开始

请在项目根目录运行命令。

快速调试实验：

```bash
python -m src.main --quick
```

完整默认实验：

```bash
python -m src.main
```

自定义实验规模：

```bash
python -m src.main --horizon 6000 --runs 30
```

参数说明：

| 参数 | 含义 | 默认值 |
|---|---|---:|
| `--quick` | 使用较小规模快速验证流程 | `False` |
| `--horizon` | 每次 Monte Carlo run 的比较轮数 | `config/config.py` 中的 `horizon` |
| `--runs` | 每个场景的 Monte Carlo 重复次数 | `config/config.py` 中的 `n_runs` |

`--horizon` 和 `--runs` 必须为正整数。

## 实验配置

主要配置在 `config/config.py`：

| 字段 | 说明 |
|---|---|
| `random_seed` | 全局随机种子 |
| `n_arms` | 候选 arm 数量，可对应待比较模型或策略数量 |
| `horizon` | 每次运行的 dueling 轮数 |
| `n_runs` | 每个场景下的 Monte Carlo 重复次数 |
| `preference_temperature` | 真实偏好矩阵生成温度 |
| `base_noise` | arm 级别偏好扰动强度 |
| `algorithms` | 参与比较的算法列表 |
| `dbs_params` | DBS-UCB 参数 |
| `rucb_params` | RUCB 参数 |
| `bsucb_params` | BS-UCB 参数 |
| `scenarios` | 偏置场景配置 |

## 偏置场景

项目默认包含四类 synthetic human-bias scenarios：

| 场景 | 机制 |
|---|---|
| `position_bias` | 用户更倾向选择第一个展示的候选项 |
| `conformity_bias` | 用户受公开胜率或流行度影响 |
| `selective_feedback` | 在接近五五开的比较中，反馈更容易被放大或扰动 |
| `mixed_bias` | 同时叠加位置偏置、从众偏置和选择性反馈偏置 |

## 算法

| 算法 | 文件 | 说明 |
|---|---|---|
| RUCB | `src/algorithms/baselines.py` | Relative UCB baseline |
| BS-UCB | `src/algorithms/baselines.py` | 基于使用频次和置信上界的 baseline |
| DBS-UCB | `src/algorithms/dbs_ucb.py` | 带偏置惩罚和不确定性探索的 debiasing-aware 方法 |

所有算法继承自 `BaseDuelingBanditAlgorithm`，核心接口为：

- `select_pair()`：选择本轮要比较的两个 arms。
- `update(i, j, winner)`：根据观测胜者更新胜负统计。
- `estimated_preferences()`：返回当前偏好估计矩阵。
- `confidence_radius()`：返回 UCB 置信半径。

## 算法思路与理论基础

### 问题建模

项目采用 dueling bandits 建模二元偏好学习问题。与普通 multi-armed bandit 不同，系统每轮不是直接观察某个 arm 的标量 reward，而是选择两个 arms 进行比较，并只观察二者谁胜出。

设候选集合为：

```text
A = {1, 2, ..., K}
```

真实偏好由矩阵 `P` 表示：

```text
P_ij = Pr(i beats j)
P_ij + P_ji = 1
P_ii = 0.5
```

其中 `P_ij > 0.5` 表示 arm `i` 在真实偏好上强于 arm `j`。如果存在一个 arm `a*` 满足：

```text
P_a*,j > 0.5, for all j != a*
```

则 `a*` 被称为 Condorcet winner。实验中的学习目标是在有限比较轮数内尽快把比较集中到优质 arms 上，降低累计 regret。

### 合成偏好矩阵

`SyntheticDuelingGenerator` 先为每个 arm 采样一个潜在 utility，然后通过 logistic 函数构造 pairwise preference：

```text
P_ij = sigmoid((u_i - u_j) / temperature)
```

随后加入 arm 级别的基础扰动 `base_noise`，使偏好矩阵不完全由一维 utility 决定。这样可以生成满足反对称约束的偏好矩阵，同时保留一定非理想性。

生成后满足：

```text
P_ji = 1 - P_ij
P_ii = 0.5
```

### 偏置反馈模型

真实偏好矩阵 `P` 不会直接被算法观察到。算法观察到的是受人类反馈偏置污染后的胜率 `Q_ij(t)`。在代码中，`BiasedDuelingEnvironment.observed_probability(i, j)` 对真实概率 `P_ij` 加入三类偏置：

位置偏置：

```text
Q_ij(t) = P_ij + b_pos
```

当 `i` 被放在第一个展示位置时，它会额外获得 `b_pos` 的胜率提升。这个机制模拟用户更容易选择先出现选项的倾向。

从众偏置：

```text
pop_i(t) = (public_wins_i + 1) / (public_duels_i + 2)
Q_ij(t) = P_ij + b_conf * (pop_i(t) - pop_j(t))
```

如果某个 arm 在公开历史中更受欢迎，它会在后续比较中获得额外优势。这个机制模拟排行榜、点赞数、已有评价等社会信号对人类判断的影响。

选择性反馈偏置：

```text
if |P_ij - 0.5| <= tau:
    with probability b_sel:
        Q_ij(t) = P_ij + 0.12 * sign(pop_i(t) - pop_j(t))
```

当两个 arms 的真实差距很小时，反馈更容易受到社会信号扰动。这个机制模拟近似平局场景中用户判断不稳定、容易被外部线索放大的情况。

最终观测概率会被裁剪到 `[0.02, 0.98]`，避免出现确定性反馈：

```text
Q_ij(t) = clip(Q_ij(t), 0.02, 0.98)
```

### 经验偏好估计

每个算法维护两个矩阵：

- `wins[i, j]`：arm `i` 战胜 arm `j` 的次数。
- `comparisons[i, j]`：arm `i` 与 arm `j` 被比较的次数。

经验偏好估计为：

```text
P_hat_ij(t) = wins[i, j] / comparisons[i, j]
```

如果某对 arms 尚未比较，默认估计为 `0.5`，表示未知状态下不偏向任何一方。

### UCB 置信上界

项目中所有算法都基于 optimism under uncertainty。核心思想是：对于观察次数少的 pair，偏好估计不确定性更高，因此应该给予更大的探索奖励。

代码中的置信半径为：

```text
r_ij(t) = sqrt(alpha * log(t) / max(N_ij(t), 1))
```

其中：

- `N_ij(t)` 是 pair `(i, j)` 的比较次数。
- `alpha` 控制探索强度。
- `log(t)` 使置信半径随时间缓慢增长。
- `1 / sqrt(N_ij(t))` 使频繁比较过的 pair 不确定性下降。

这类形式来自 Hoeffding concentration bound。对于 Bernoulli 比较反馈，经验均值会以高概率集中在真实均值附近：

```text
Pr(|P_hat_ij - P_ij| >= epsilon) <= 2 exp(-2 N_ij epsilon^2)
```

因此 UCB 方法使用：

```text
P_hat_ij(t) + r_ij(t)
```

作为 pairwise preference 的乐观估计，在探索不足时主动尝试不确定选项，在证据充分后逐渐转向利用。

### RUCB 原理

RUCB 是 dueling bandits 中经典的 relative upper confidence bound baseline。它的核心思路是寻找可能击败所有其他 arms 的 champion。

代码中的候选 champion 条件为：

```text
UCB_ij(t) >= 0.5, for all j
```

也就是 arm `i` 在乐观估计下仍可能不输给任何对手。若没有满足条件的 arm，则退化为从所有 arms 中选择。

选出 champion 后，RUCB 选择最可能击败 champion 的 opponent：

```text
opponent = argmax_j UCB_j,champion(t)
```

这样做的意义是用最强挑战者验证 champion 是否真的可靠。RUCB 在无偏、稳定反馈下有较好的理论基础，但当观测反馈被位置、从众或选择性反馈系统性污染时，它会把偏置当成真实偏好的一部分。

### BS-UCB 原理

BS-UCB 是项目中的均衡探索 baseline。它先选择历史使用次数最少的 arm：

```text
first = argmin_i usage_i(t)
```

然后为该 arm 选择 UCB 分数最高的挑战者：

```text
second = argmax_j P_hat_j,first(t) + beta * r_j,first(t)
```

这个方法的设计意图是避免某些 arms 长期缺少比较，提升覆盖率。它比纯随机探索更有方向性，但没有显式建模反馈偏置，因此在 biased environment 中仍可能受到系统性误导。

### DBS-UCB 原理

DBS-UCB 是项目提出的 debiasing-aware UCB 策略。它保留 UCB 的乐观探索，同时对疑似受偏置影响的 arm 加入逐渐衰减的惩罚项。

首先计算每个 arm 的平均经验强度：

```text
mean_strength_i(t) = average_j P_hat_ij(t)
```

再计算平均探索奖励：

```text
explore_bonus_i(t) = average_j r_ij(t)
```

项目使用一个简单的偏置签名：

```text
estimated_bias_i(t) = |mean_strength_i(t) - 0.5|
```

直觉是：在受偏置污染的早期反馈中，某些 arms 可能因为位置或流行度被异常推高，导致平均胜率快速偏离中性值。DBS-UCB 不直接相信这种偏离，而是在主选择分数中扣除一个偏置惩罚：

```text
lambda_t = bias_penalty / sqrt(t)

primary_score_i(t)
  = mean_strength_i(t)
    - lambda_t * estimated_bias_i(t)
    + explore_bonus_i(t)
```

第一只 arm 的选择规则为：

```text
first = argmax_i primary_score_i(t)
```

第二只 arm 用于挑战第一只 arm，优先选择不确定且接近胜负边界的对手：

```text
near_boundary_j(t) = 0.5 - |P_hat_j,first(t) - 0.5|
challenge_score_j(t) = r_j,first(t) + near_boundary_j(t)
second = argmax_j challenge_score_j(t)
```

这样做有两个目的：

1. 对高不确定 pair 保持探索。
2. 对接近 `0.5` 的关键 pair 追加比较，因为这些 pair 最可能改变排序判断。

### DBS-UCB 的理论直觉

DBS-UCB 的关键是 `lambda_t = lambda_0 / sqrt(t)`。这个偏置惩罚在早期较强，可以抑制由位置、从众或选择性反馈导致的异常乐观估计；随着样本增多，惩罚逐渐变弱，避免长期压制真实强 arm。

如果偏置签名满足：

```text
0 <= estimated_bias_i(t) <= 1
```

则累计偏置惩罚的量级满足：

```text
sum_{t=1}^T lambda_t * estimated_bias_i(t)
<= lambda_0 * sum_{t=1}^T 1 / sqrt(t)
<= 2 lambda_0 sqrt(T)
```

因此惩罚项是 sublinear 的：

```text
O(sqrt(T))
```

这意味着 DBS-UCB 的惩罚主要影响早期决策，不会在线性量级上支配长期学习过程。换句话说，它在早期更保守地抵抗偏置，在后期仍保留 UCB 类算法依赖数据收敛的性质。

### 算法理论如何投入实际

在真实评价系统中，算法并不能控制人类评审者内心如何形成判断。它能控制的是评价流程：选择哪些对象被比较、以什么顺序展示、如何记录胜负结果，以及如何把有偏反馈转换成更可靠的排序。

实际部署时，可以把每个待评价对象看作一个 arm，例如一篇论文、一个商品、一部电影或一个 LLM 回答。系统每轮选择两个 arms 给评审者比较，记录评审者选择的赢家，然后更新 `P_hat` 和置信半径。最终输出的不是简单投票平均值，而是基于 pairwise model 学到的 debiased preference ranking。

DBS-UCB 中的理论项对应到实际流程如下：

1. `P_hat_ij(t)` 对应当前系统认为 `i` 胜过 `j` 的经验概率。
2. `r_ij(t)` 对应证据不足程度，用来决定哪些 pair 还需要更多人工比较。
3. `estimated_bias_i(t)` 对应早期异常偏离的风险信号，提示该 arm 可能受到展示位置、流行度或选择性反馈影响。
4. `lambda_t = lambda_0 / sqrt(t)` 对应逐渐放松的保守校准策略：早期更谨慎，后期更多相信累积数据。
5. 最终 ranking 可以用于推荐排序、模型输出筛选、论文/商品/电影评价聚合等场景。

因此，这个项目的实际意义不是改变人的评分行为，而是在人的评分已经可能受偏置影响时，用算法设计评价流程和排序层，减少偏置对最终结果的影响。

### 从胜率矩阵转回评分

DBS-UCB 的直接输出是偏好估计矩阵 `P_hat`，其中：

```text
P_hat_ij(T) = 经过 T 轮比较后，系统估计 i 胜过 j 的概率
```

为了让结果回到常见评分系统，可以把每个对象对其他对象的平均胜率定义为修正后分数：

```text
q_i = average_{j != i} P_hat_ij(T)
```

其中 `q_i` 的取值在 0 到 1 之间，表示对象 `i` 在当前候选集合中平均战胜其他对象的概率。然后可以映射到常见评分尺度：

```text
百分制评分: rating_i = 100 * q_i

五星评分: stars_i = 1 + 4 * q_i
```

例如，如果某个 LLM 回答的修正后平均胜率为 `q_i = 0.80`，则可以得到：

```text
rating_i = 80
stars_i = 4.2
```

需要强调的是，这个评分是相对评分：它表示该对象在当前候选集合中相对于其他对象的表现，而不是一个脱离比较集合的绝对质量分数。因此，两两比较是中间建模方式，最终系统仍然可以输出用户熟悉的百分制分数或五星评分。

### Regret 理论

项目使用 strong regret。设 `a*` 是 Condorcet winner，则一轮比较 `(i_t, j_t)` 的即时 regret 定义为：

```text
r_t = (P_a*,i_t - 0.5) + (P_a*,j_t - 0.5)
```

代码中等价实现为：

```text
r_t = max(P[a*, i_t] + P[a*, j_t] - 1, 0)
```

累计 regret 为：

```text
R_T = sum_{t=1}^T r_t
```

如果算法频繁选择远离 Condorcet winner 的 arms，`R_T` 会增长更快；如果算法逐渐集中到最优或接近最优 arms，累计 regret 的斜率会下降。

在标准无偏 dueling bandits 中，UCB 类方法依赖置信区间逐步排除次优 arms，从而获得次线性 regret。当前项目的偏置环境比标准设定更难，因为观测概率 `Q_ij(t)` 不一定等于真实偏好 `P_ij`，并且从众偏置会随历史反馈变化。因此这里的 DBS-UCB 理论基础不是完整的无偏 regret bound，而是以下组合：

1. UCB 置信半径提供对 pairwise Bernoulli feedback 的高概率探索依据。
2. Condorcet winner 假设提供明确的最优 arm 目标。
3. Strong regret 提供偏离最优 arm 的损失度量。
4. 衰减偏置惩罚提供 `O(sqrt(T))` 的次线性影响，避免早期偏置信号长期主导。
5. Monte Carlo 实验与 paired permutation test 用于验证在合成偏置机制下的经验鲁棒性。

### 为什么偏置不能只当作随机噪声

普通随机噪声通常满足均值为零，随着样本增加会被平均掉。但本项目中的偏置具有结构性：

- 位置偏置会稳定偏向展示顺序靠前的 arm。
- 从众偏置会形成 popularity feedback loop。
- 选择性反馈会在真实差距很小时放大外部信号。

这类偏置可能使经验均值收敛到错误的观测概率 `Q_ij`，而不是真实偏好 `P_ij`。因此 DBS-UCB 的设计目标不是简单增加探索，而是在探索过程中显式削弱疑似偏置带来的早期优势。

### 参考理论来源

本项目的理论背景主要来自：

- K-armed dueling bandits：将学习反馈从标量 reward 扩展为 pairwise comparison。
- Relative UCB：用 pairwise upper confidence bound 寻找可能的 Condorcet winner。
- Hoeffding concentration：为 Bernoulli 反馈的经验均值置信半径提供依据。
- Human evaluation calibration：将人类评价中的噪声和偏置作为需要建模的信号，而不是简单丢弃。
- Paired permutation test：用于比较同一批随机实验下不同算法的最终 regret 差异。

## 输出文件

运行实验后会生成或覆盖以下文件：

| 路径 | 内容 |
|---|---|
| `results/figures/regret_position_bias.png` | 位置偏置场景累计遗憾曲线 |
| `results/figures/regret_conformity_bias.png` | 从众偏置场景累计遗憾曲线 |
| `results/figures/regret_selective_feedback.png` | 选择性反馈场景累计遗憾曲线 |
| `results/figures/regret_mixed_bias.png` | 混合偏置场景累计遗憾曲线 |
| `results/figures/robustness_comparison.png` | 各算法相对最终遗憾柱状图 |
| `results/raw_data/regret_summary.csv` | 每个场景、算法、轮次的均值和标准差 |
| `results/raw_data/robustness_table.csv` | 相对最终遗憾表 |
| `results/raw_data/statistical_report.json` | 最终遗憾、AUC 和显著性检验结果 |
| `results/raw_data/statistical_summary.md` | 显著性检验 Markdown 汇总 |
| `results/raw_data/robustness_table.tex` | 可插入论文的 LaTeX 表格 |

## 指标说明

- Instantaneous regret：相对于 best arm 的单轮 strong regret。
- Cumulative regret：单轮 regret 的累计和。
- Final regret：实验结束时的累计 regret。
- AUC regret：累计 regret 曲线下面积，用于衡量整个过程的整体损失。
- Relative final regret：算法最终 regret 相对于同场景最佳算法的比例，越低越好。
- Paired permutation test：比较 DBS-UCB 与 baseline 的 paired final regret 差异。

## 复现实验

默认配置已固定随机种子。复现实验时请确保：

1. 使用相同 Python 版本和依赖版本。
2. 不修改 `config/config.py` 中的随机种子、场景参数和算法参数。
3. 从项目根目录运行 `python -m src.main`。
4. 检查 `results/raw_data/statistical_report.json` 和 `results/figures/` 输出。

推荐记录环境：

```bash
python --version
pip freeze
```

## 扩展方式

新增算法：

1. 在 `src/algorithms/` 中新增算法类。
2. 继承 `BaseDuelingBanditAlgorithm`。
3. 实现 `select_pair()`。
4. 在 `src/main.py` 的 `algo_registry` 注册算法。
5. 在 `config/config.py` 的 `algorithms` 中加入算法名称。

新增偏置场景：

1. 在 `config/config.py` 的 `scenarios` 列表中增加配置。
2. 如需新偏置机制，扩展 `BiasedDuelingEnvironment.observed_probability()`。
3. 重新运行实验并检查 `results/figures/` 与 `results/raw_data/`。

接入真实数据：

- `src/data/mt_bench_loader.py` 目前是 MT-Bench 集成预留入口。
- 可以将真实 pairwise judgments 转换为偏好矩阵或 replay environment 后接入主实验流程。

## 常见问题

### 运行 `python -m src.main` 提示找不到 `src`

请确认当前目录是项目根目录，而不是 `src/` 目录。

### Windows 上 `python` 无法启动

可能是 PATH 指向 Microsoft Store 占位程序。建议激活 `.venv`，或直接运行：

```powershell
.\.venv\Scripts\python.exe -m src.main --quick
```

### Matplotlib 在无图形界面环境失败

项目绘图模块使用非交互式 `Agg` 后端，适合服务器、CI 和无 GUI 环境运行。

### 输出结果被覆盖

每次运行会覆盖 `results/figures/` 和 `results/raw_data/` 中同名文件。需要保留历史结果时，可在运行前复制或重命名 `results/` 下的输出目录。

## 已知限制

- 当前实验主要基于合成偏好矩阵，真实 LLM judge 或 MT-Bench judgment 尚未完整接入。
- DBS-UCB 的偏置估计是启发式设计，仍可继续做理论推导和消融实验。
- 目前没有独立测试套件，建议后续补充 unit tests 和 small regression tests。
- 默认结果目录为固定路径，多组实验管理可以继续扩展为带 timestamp 或 run id 的输出结构。
