# Poisson Denoising V1 计划（给 Codex 的实现说明）

## 1. 目标

这是第一版 **idea validation**。目标不是做最优方法，也不是先讲理论；目标是用 **最简单、最 naive、能跑起来的版本** 检查下面两个想法是否有信号：

- **Method 1**：局部统计驱动的动态 quantile filter
- **Method 2**：非局部 GLR 匹配 + 动态 quantile filter

本版 **不训练 CNN，不训练任何学习模型**。所谓 “quantile regression” 在这一版里统一实现为：

\[
\hat q_i \in \arg\min_q \sum_j w_{ij}\,\rho_{p_i}(Y_j-q)
\]

也就是 **加权经验 quantile**。这一步有闭式的离散求解方式：排序后找累计权重第一次超过 \(p_i\) 的值。

换句话说，V1 里：

- 不做神经网络
- 不做全局优化器
- 不做 block-level 共享回归
- **每个 pixel 单独输出一个估计值**，但这个估计值不是只用它自己，而是用它的局部/非局部样本池

---

## 2. Version 1 的固定实现选择

这些选择先固定，不要来回改：

1. **噪声模型**：只做纯 Poisson。
2. **算法工作域**：所有算法都在 **count 域** 上运行，不在归一化后的 \([0,1]\) 强度域上运行。
3. **Method 2 的匹配方式**：第一版只用 **GLR distance**，不实现 oracle，也不实现 paper 里的其他 similarity estimator。
4. **quantile 拟合算法**：统一用 **weighted empirical quantile**。
5. **每个 pixel 都单独估计**，不做 block-wise shared regression。
6. **边界处理**：统一用 `reflect padding`。
7. **概率裁剪**：所有 \(p_i\) 都做 clip，避免退化成 0 或 1。

默认 clip：

\[
p_i \leftarrow \min(0.95, \max(0.05, p_i))
\]

说明：这是第一版工程稳定化，不是理论定义的一部分。

---

## 3. 模拟数据与评测设置

### 3.1 干净图像

先用最简单的内置灰度图，统一 resize/crop 到小尺寸，保证 naive non-local loops 能跑：

- `camera`（128x128）
- `moon`（128x128）
- `shepp_logan_phantom`（128x128）

图像统一记为 \(x \in [0,1]^{H\times W}\)。

### 3.2 Poisson 噪声生成

设峰值 photon count 为 `peak`，例如：

```text
peak_list = [1, 2, 4, 8, 16, 32]
```

对每张干净图像生成 count 图：

\[
Z_i \sim \text{Poisson}(\text{peak} \cdot x_i)
\]

算法输入是整数 count 图 \(Z\)。

显示和评测时，把结果转回 intensity 域：

\[
Y^{noisy} = Z / \text{peak}, \qquad \hat x = \hat Z / \text{peak}
\]

### 3.3 随机重复

每个 `image × peak` 至少跑 5 个随机 seed：

```text
seeds = [0, 1, 2, 3, 4]
```

### 3.4 评测指标

至少输出：

- PSNR
- SSIM
- MAE
- runtime

所有指标都在 intensity 域上比较：`denoised / peak` vs `clean x`。

---

## 4. 必做 baseline

第一版只做下面几个 baseline：

1. **Noisy**（原始噪声图）
2. **3x3 mean filter**
3. **3x3 median filter**
4. **Method 1**
5. **Method 2**

先不要加别的复杂 baseline。

---

## 5. 公共工具函数（必须实现）

Codex 需要先实现这些通用函数。

### 5.1 `generate_poisson_counts(clean, peak, seed)`

输入：
- `clean`: float image in `[0,1]`
- `peak`: positive scalar
- `seed`

输出：
- `counts`: integer Poisson counts
- `noisy_intensity = counts / peak`

---

### 5.2 `local_mean(counts, win=3)`

输出每个像素的局部均值图。

---

### 5.3 `local_median(counts, win=3)`

输出每个像素的局部中值图。

---

### 5.4 `poisson_prob_strict_less(lam, threshold)`

返回：

\[
P(S < \text{threshold} \mid S \sim \text{Poisson}(\lambda))
\]

实现时使用：

\[
P(S < t) = P(S \le t-1) = F_{\text{Pois}(\lambda)}(t-1)
\]

要求：

- 调用 `scipy.stats.poisson.cdf`
- 支持数组输入
- threshold 如果小于等于 0，返回 0

---

### 5.5 `weighted_quantile(values, weights, q)`

这是整个项目里最重要的基础函数。

输入：
- `values`: 1D array
- `weights`: 1D nonnegative array, same length
- `q`: scalar in `[0,1]`

实现：
1. 按 `values` 升序排序
2. 权重归一化到和为 1
3. 累计权重 `cumsum`
4. 返回第一个使 `cumsum >= q` 的 `value`

这就是 V1 里的 “quantile regression / pinball loss minimizer”。

备注：
- `weights=None` 时等价于普通经验 quantile
- `q=0.5` 时就是 weighted median

---

### 5.6 `extract_patch(img, center, patch_size, padding='reflect')`

返回以某个位置为中心的 patch。

---

### 5.7 `glr_patch_distance(patch_a, patch_b)`

定义：

\[
d^{GLR}(a,b)=\sum_u \left[a_u\log a_u+b_u\log b_u-(a_u+b_u)\log\frac{a_u+b_u}{2}\right]
\]

实现要求：
- 输入是两个同尺寸 patch，元素为非负 count
- 约定 `0 * log(0) = 0`
- 返回一个标量距离

---

## 6. Method 1：纯局部动态 quantile filter

## 6.1 定义

对每个像素 \(i\)：

### Step 1. 局部统计

取 \(3\times 3\) 邻域 \(N_i\)：

\[
\mu_i^{loc}=\frac{1}{9}\sum_{j\in N_i} Z_j
\]

\[
m_i^{loc}=\operatorname{median}\{Z_j: j\in N_i\}
\]

### Step 2. 动态概率

定义窗口和尺度下的概率：

\[
p_i^{raw}=P\big(S<9m_i^{loc}\mid S\sim \text{Poisson}(9\mu_i^{loc})\big)
\]

实现时：

\[
p_i^{raw} = F_{\text{Pois}(9\mu_i^{loc})}(9m_i^{loc}-1)
\]

然后裁剪：

\[
p_i=\text{clip}(p_i^{raw}, 0.05, 0.95)
\]

### Step 3. quantile 拟合（第一版最简单实现）

**不要训练模型。**

直接把输出定义成：

\[
\hat Z_i = \arg\min_q \sum_{j\in N_i} \rho_{p_i}(Z_j-q)
\]

这等价于：

- 对 \(N_i\) 中的 9 个像素做 **普通经验 \(p_i\)-quantile**
- 权重全部相等

即：

\[
\hat Z_i = Q_{emp}(N_i; p_i)
\]

### Step 4. 输出

返回 denoised intensity：

\[
\hat x_i = \hat Z_i / \text{peak}
\]

## 6.2 默认参数

```text
local_stats_window = 3
fit_window = 3
p_clip = [0.05, 0.95]
```

## 6.3 实现说明

这是一个 **adaptive local quantile filter**。

它是第一版最 naive、最适合试 idea 的实现。

---

## 7. Method 2：GLR 非局部动态 quantile filter

## 7.1 定义

对每个像素 \(i\)：

### Step 1. 局部 Poisson 率预估

先算局部均值：

\[
\mu_i^{loc}=\frac{1}{9}\sum_{j\in N_i} Z_j
\]

其中 \(N_i\) 是 \(3\times 3\) 邻域。

### Step 2. 非局部 GLR 匹配

在搜索窗内找相似 patch。

默认参数：

```text
patch_size = 5
search_size = 21
top_k = 24
```

对搜索窗内每个候选位置 \(j\neq i\)：

1. 提取 `patch_size x patch_size` 的 raw count patch
2. 计算 `GLR distance`
3. 保留距离最小的 top-k 个候选

记保留集合为 \(\mathcal J_i\)。

### Step 3. 相似度权重

对 top-k 距离 \(d_{ij}\)，定义：

\[
h_i = \operatorname{median}\{d_{ij}: j\in \mathcal J_i\}+10^{-8}
\]

\[
w_{ij}=\exp(-d_{ij}/h_i)
\]

再归一化到和为 1。

### Step 4. 非局部稳健基准

取这些匹配 patch 的**中心像素 count**：

\[
v_j = Z_j,\qquad j\in \mathcal J_i
\]

然后定义 non-local weighted median：

\[
M_i^{NL}=\operatorname{wmed}\{(v_j,w_{ij})\}
\]

实现上直接调用：

```python
M_nl = weighted_quantile(values=v, weights=w, q=0.5)
```

### Step 5. 动态概率

定义：

\[
p_i^{raw}=P\big(S<9M_i^{NL}\mid S\sim \text{Poisson}(9\mu_i^{loc})\big)
\]

实现时：

\[
p_i^{raw}=F_{\text{Pois}(9\mu_i^{loc})}(9M_i^{NL}-1)
\]

然后裁剪：

\[
p_i=\text{clip}(p_i^{raw}, 0.05, 0.95)
\]

### Step 6. quantile 拟合（第一版最简单实现）

同样，**不要训练任何模型**。

直接定义：

\[
\hat Z_i = \arg\min_q \sum_{j\in \mathcal J_i} w_{ij}\,\rho_{p_i}(v_j-q)
\]

这等价于：

- 对 top-k 匹配到的中心像素值 \(v_j\)
- 以权重 \(w_{ij}\)
- 取 weighted empirical \(p_i\)-quantile

即：

```python
zhat_i = weighted_quantile(values=v, weights=w, q=p_i)
```

### Step 7. 输出

返回 denoised intensity：

\[
\hat x_i = \hat Z_i / \text{peak}
\]

## 7.2 默认参数

```text
local_stats_window = 3
patch_size = 5
search_size = 21
top_k = 24
p_clip = [0.05, 0.95]
```

## 7.3 实现说明

第一版只用：

- raw count patch
- GLR distance
- top-k nearest neighbors
- exponential weights
- center-pixel weighted median
- center-pixel weighted quantile output

不要做这些扩展：

- 不做 patch re-projection
- 不做 whole-patch aggregation
- 不做 oracle estimator
- 不做 uploaded paper 的 similarity estimator
- 不做预筛选
- 不做多尺度

---

## 8. 为什么 V1 不训练 CNN / 不训练真正的 QR 模型

因为第一版的目标只是：

> 验证 `动态 p_i` 是否能比固定 quantile（尤其是 0.5 / median）更好。

要验证这个问题，最小实现就是：

- Method 1：局部样本池上的 adaptive quantile
- Method 2：非局部样本池上的 adaptive quantile

这样更干净，也更容易 debug。

如果第一版都没有信号，再上 CNN 没有意义。

---

## 9. 每个 pixel 还是 block-wise？

第一版明确采用：

- **每个 pixel 单独估计一个输出值**
- 但是这个输出值来自一组样本池，不是只用该 pixel 自己

所以：

- Method 1：每个 pixel 对应一个 `3x3` 局部样本池
- Method 2：每个 pixel 对应一个 `top-k` 非局部样本池

**不要做 block-wise 共享回归。**

---

## 10. 第一版实验流程

## Phase 0：工具和 baseline

必须先完成：

1. Poisson 数据生成
2. 3x3 mean filter
3. 3x3 median filter
4. weighted quantile
5. GLR patch distance

然后做一个单图 sanity check：

- 图像：`camera` 128x128
- peak = 8
- seed = 0

先确认：

- mean / median baseline 能跑
- Method 1 / Method 2 输出图像不报错
- `p_map` 能可视化

## Phase 1：Method 1

跑下面这个表：

- images = `camera`, `moon`, `shepp_logan`
- peaks = `[1, 2, 4, 8, 16, 32]`
- seeds = `[0,1,2,3,4]`

保存：

- 平均 PSNR/SSIM/MAE
- `p_i` 直方图
- 代表性图像对比

## Phase 2：Method 2

同样跑完整网格。

特别保存：

- `M_NL` 图
- `p_i` 图
- top-k 匹配的可视化（至少对 5 个典型像素保存）

## Phase 3：比较

最核心比较：

1. Method 1 vs median filter
2. Method 2 vs Method 1
3. Method 2 vs mean / median baseline

---

## 11. 必须保存的输出文件

建议输出目录：

```text
outputs/
  metrics.csv
  summary_by_image_peak.csv
  figs/
    camera_peak8_seed0_comparison.png
    camera_peak8_seed0_pmap_method1.png
    camera_peak8_seed0_pmap_method2.png
    camera_peak8_seed0_match_vis_method2.png
  arrays/
    camera_peak8_seed0_method1.npy
    camera_peak8_seed0_method2.npy
    camera_peak8_seed0_pmap_method1.npy
    camera_peak8_seed0_pmap_method2.npy
```

`metrics.csv` 每行至少包含：

```text
image, peak, seed, method, psnr, ssim, mae, runtime_sec
```

---

## 12. 代码结构建议（给 Codex）

```text
project/
  plan.md
  run_demo.py
  run_experiments.py
  src/
    data.py
    utils.py
    metrics.py
    baselines.py
    method1.py
    method2.py
    visualize.py
  outputs/
```

### 12.1 `src/data.py`

实现：
- 读入内置图像
- resize/crop 到 128x128
- 生成 Poisson counts

### 12.2 `src/utils.py`

实现：
- reflect padding
- local mean
- local median
- weighted quantile
- poisson strict-less probability
- patch extraction
- GLR distance

### 12.3 `src/baselines.py`

实现：
- noisy passthrough
- 3x3 mean filter
- 3x3 median filter

### 12.4 `src/method1.py`

实现：
- `method1_denoise(counts, peak, loc_win=3, p_clip=(0.05, 0.95))`
- 返回：`denoised_counts`, `p_map`, `mu_loc_map`, `m_loc_map`

### 12.5 `src/method2.py`

实现：
- `method2_denoise(counts, peak, loc_win=3, patch_size=5, search_size=21, top_k=24, p_clip=(0.05, 0.95))`
- 返回：`denoised_counts`, `p_map`, `mu_loc_map`, `M_nl_map`

### 12.6 `run_demo.py`

只跑一个例子：

- `camera`
- `peak=8`
- `seed=0`

输出图像和 metrics。

### 12.7 `run_experiments.py`

循环：

- images
- peaks
- seeds
- methods

最后汇总成 `metrics.csv`。

---

## 13. 关键实现细节（必须遵守）

### 13.1 Method 1 的 quantile 求解

不要写成训练过程。

直接：

```python
values = 3x3 neighborhood values
weights = np.ones_like(values) / len(values)
zhat_i = weighted_quantile(values, weights, p_i)
```

### 13.2 Method 2 的 quantile 求解

也不要训练。

直接：

```python
values = topk matched center pixel values
weights = normalized similarity weights
zhat_i = weighted_quantile(values, weights, p_i)
```

### 13.3 GLR distance

数值实现时，写一个安全版：

```python
term(x) = x * log(x) if x > 0 else 0
```

### 13.4 速度

Method 2 的 naive loops 会很慢。

第一版建议：

- 先只跑 `128x128`
- 如果还慢，先只跑 `64x64`
- patch=5, search=21, top_k=24 不要再变大

### 13.5 输出类型

`denoised_counts` 可以是浮点数，也可以保持整数。第一版建议保持浮点即可。

---

## 14. 第一版最重要的 sanity checks

Codex 实现完成后，先检查：

1. `weighted_quantile(..., q=0.5)` 是否和普通 median 一致
2. Method 1 在 `p_i=0.5` 人工固定时，是否退化为 local median filter
3. Method 2 在 `p_i=0.5` 人工固定时，是否退化为 non-local weighted median
4. `p_map` 是否大量集中在 0 或 1；如果是，确认 clip 正常工作
5. 在 `peak=32` 时，Method 1 / Method 2 是否至少不比 raw noisy 更差

---

## 15. V1 之后再做的事（现在不要做）

这些内容都放到 V2，不要在第一版实现：

- 用 uploaded paper 的 similarity estimator 代替 GLR
- 用预平滑参考图做 non-local 预筛选
- whole-patch aggregation + re-projection
- patch-wise output 而不是 center-pixel output
- local linear quantile regression
- CNN / U-Net / shallow conv net
- Poisson thinning
- 参数自动选择
- 理论分析

---

## 16. 给 Codex 的一句话任务说明

请严格按照本文件实现一个 **最小可运行 Poisson denoising 实验框架**：

- 在 count 域模拟 Poisson 噪声
- 实现 mean / median baseline
- 实现 Method 1（local adaptive quantile filter）
- 实现 Method 2（GLR-based nonlocal adaptive quantile filter）
- quantile 求解统一使用 weighted empirical quantile
- 输出 PSNR / SSIM / MAE / runtime 和若干可视化

不要加入任何额外复杂化设计。
