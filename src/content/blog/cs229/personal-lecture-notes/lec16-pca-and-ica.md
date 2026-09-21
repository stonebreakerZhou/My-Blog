---
title: "CS229 : Lec 16 — 主成分分析 (PCA) 与独立成分分析 (ICA)"
description: CS229 Lecture 16 学习笔记。① PCA：找到数据最大方差方向，用于降维 / 可视化 / 压缩；② ICA：给定混合观测 x = As，找分离矩阵 W = A⁻¹ 使 s = Wx 还原独立源信号。
pubDate: 2026-09-21
series: cs229
subSeries: personal-lecture-notes
order: 16
categories:
  - CS229
  - PCA
  - ICA
  - Eigenvector
  - Cocktail Party Problem
---


> **TL;DR**:
> - **PCA**：找数据的**主轴方向**，把 $n$ 维数据投影到 $k$ 维子空间。这是一个**非概率模型**——只描述低维结构，不建模 $P(x)$，帮助我们更好地了解数据的分布特性。
> - **ICA**：鸡尾酒会问题——给定 $x = As$，找 $W = A^{-1}$ 使 $s = Wx$ 还原独立源。
> - **PCA vs ICA**：PCA 找正交方向、强调方差最大；ICA 找独立成分、强调非高斯。




## 引子

Lec 16 进入**降维**与**独立成分分析**：

- **PCA**（主成分分析）——找数据的低维结构，但**不建模 $P(x)$**
- **ICA**（独立成分分析）——找分离矩阵从混合观测中还原独立成分
- **四种无监督学习对比理解**：factor analysis / PCA / Mixture of Gaussians / K-Means

---





## 1. 主成分分析 (PCA)

### 1.1 动机

> 回顾 Lec 15 的 factor analysis 模型——它试图对高维空间中的 $P(x)$ 建模来进行异常值分析。而 PCA **不是概率模型**，**不建模 $P(x)$**——但它同样能让你**看出数据是否落在低维空间**。

### 1.2 一个例子

有一个**无标签**数据集 $\{x^{(1)}, \dots, x^{(m)}\} \in \mathbb{R}^n$。我们想把维度从 $n$ 降到 $k$（$k \ll n$）。

比如我们有一个 inch–centimeter 的二维数据集，由于长度单位之间可以转换，这个数据集实际上应当处于**一个一维的子空间**：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec16_PCA_eg1_dataset.webp" alt="PCA example dataset" width="80%" loading="lazy" decoding="async" /></div>

> PCA 算法要做的事就是**找到图中那个倾斜的一维维度方向**——那应该是数据变化的**主轴**。与它正交的维度上只有数据点采集时的噪声。把数据投影到这根轴上，二维数据就会变为一维。


### 1.3 预处理 (Pre-processing)

在 PCA 之前要预处理数据：
#### ① 零均值化 (Zero out mean)

$$
\begin{aligned}
\mu &= \frac{1}{m} \sum_{i=1}^{m} x^{(i)} \\
x^{(i)} &\leftarrow x^{(i)} - \mu
\end{aligned}
$$
#### ② 方差归一化到 1 (Standardize variance)

$$
\begin{aligned}
\sigma_j^2 &= \frac{1}{m} \sum_{i=1}^{m} (x_j^{(i)})^2 \\
x_j^{(i)} &\leftarrow x_j^{(i)} / \sigma_j
\end{aligned}
$$


### 1.4 几何直觉与图示

预处理后的数据集：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec16_PCA_pre-processed_dataset.webp" alt="pre-processed dataset" width="60%" loading="lazy" decoding="async" /></div>

> 看起来**绿线是一个相当好的变化轴（一维子空间）**——而红线是一个不好的子空间。**为什么？**

**理由 1**：如果把所有数据点正交投影到绿线和红线上——可以看到**所有数据点到绿线的距离平方和应当很小**。这可以是定义 PCA 的一种方式。

**理由 2**：如果只在绿/红两线上看投影点之间的位置关系——绿线上投影点之间**相隔较远**，红线上的投影点之间**挤成一团**。所以 PCA 也可以这样定义：**找到子空间，让数据点投影上去后尽可能保持分散**，保留更多的数据变异性。

> 事实上，上面两种直觉在**数学上是等价的**。

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec16_PCA_intuition_illustration.webp" alt="PCA intuition" width="60%" loading="lazy" decoding="async" /></div>


### 1.5 数学公式化

设 $\|u\| = 1$（单位方向向量），则 $x^{(i)}$ 在 $u$ 方向上的投影为：

$$
\text{Prj} = u^T x^{(i)}
$$

PCA 中我们要选 $u$ 来最大化数据点向量 $x$ 在 $u$ 上投影长度的平方和，即：

$$
\begin{aligned}
&\max_{u: \|u\|=1} \frac{1}{m} \sum_{i=1}^{m} (x^{(i)T} u)^2 \\
&= \max_{u: \|u\|=1} \frac{1}{m} \sum_{i=1}^{m} u^T x^{(i)} x^{(i)T} u \\
&= \max_{u: \|u\|=1} u^T \left( \frac{1}{m} \sum_{i=1}^{m} x^{(i)} x^{(i)T} \right) u
\end{aligned}
$$

> 注意此时 $x$ 的协方差矩阵是 $\Sigma_{xx}$ （这是个对称方阵）。

所以化简为：

$$
\max_{u: \|u\|=1} u^T \Sigma_{xx} u
$$


### 1.6 求解：$u$ 是 $\Sigma_{xx}$ 的主特征向量

> 为让上式最大化，最终选取的 **$u$ 就是 $\Sigma_{xx}$ 的主特征向量**。（注意：由于 $\Sigma_{xx}$ 对称，所以有一组正交基。我们找到的所有 $u_i$ 其实可以构成一组正交基，并张成一个低维子空间。）

**Lagrange 算数法证明**：

$$
\max_u u^T \Sigma_{xx} u, \quad \text{s.t.} \quad u^T u = 1
$$

构造 Lagrangian：

$$
\begin{aligned}
\mathcal{L}(u, \lambda) &= u^T \Sigma_{xx} u + \lambda(u^T u - 1) \\
\frac{\partial \mathcal{L}}{\partial u} &= (\Sigma_{xx}^T + \Sigma_{xx}) u + 2 \lambda u \\
\frac{\partial \mathcal{L}}{\partial \lambda} &= u^T u - 1
\end{aligned}
$$

令两个偏导数均为 0 并代入约束条件（且 $\Sigma_{xx} = \Sigma_{xx}^T$），于是得：

$$
\Sigma_{xx} u = -\lambda u
$$

即 $u$ 就是 $\Sigma_{xx}$ 的**特征向量**。

> **总结**：如果要用一个一维子空间近似数据，选的子空间方向就是对应的**特征向量方向**。



### 1.7 一般情况：要将数据降维到一个 $k$ 维子空间

若要把数据投影到 $k$ 维子空间，就把 $u_1, u_2, \dots, u_k$ 设为 $\Sigma_{xx}$ 的**前 $k$ 个特征向量**。

假设有一个非常高维的数据集：

$$
x^{(i)} \in \mathbb{R}^n \quad (\text{say } n = 1000)
$$

想把维度从 $n$ 降到 $k$（$k \ll n$），要找：

$$
u_1, u_2, \dots, u_k \in \mathbb{R}^n \quad (\text{say } k = 10)
$$

得到新表示：

$$
x^{(i)} \to \begin{pmatrix} u_1^T x^{(i)} \\ u_2^T x^{(i)} \\ \vdots \\ u_k^T x^{(i)} \end{pmatrix} = y^{(i)} \in \mathbb{R}^k
$$

> **现在的 $k$ 维空间是由 $k$ 个特征向量张成——原向量 $x$ 在每一维上的坐标就是该维对应特征向量上的投影大小**

> 所以现在我们不用 1000 个数来表示训练样本，而是用低维 $k$ 维子空间上的 $k$ 个坐标来降维化表示原始数据。

如果要从 $y^{(i)}$ 回到 $x^{(i)}$：

$$
x^{(i)} \approx y_1^{(i)} u_1 + y_2^{(i)} u_2 + \dots + y_k^{(i)} u_k \in \mathbb{R}^n
$$

> ⚠️ **待解决问题**：为什么数据预处理要减去均值并除以标准差？几何直观上的意义？


### 1.8 PCA应用

| #   | 应用               | 说明                                                                                                             |
| --- | ---------------- | -------------------------------------------------------------------------------------------------------------- |
| ①   | **可视化**          | 投影到 1D / 2D                                                                                                    |
| ②   | **压缩以提升 ML 效率**  | $x^{(i)} \in \mathbb{R}^{10000} \xrightarrow{\text{compress}} y^{(i)} \in \mathbb{R}^{10000}$ — 在低维数据上跑学习算法更高效 |
| ③   | **减少过拟合（该方法存疑）** | 可能用正则化来避免过拟合更合适                                                                                                |
| ④   | **异常检测 / 匹配**    | 曾用于人脸检测——把像素向量投影到低维，测两张图之间的欧氏距离。但现在已不太用                                                                        |


#### Rule of Thumb（使用 PCA 的经验法则）

> - 使用 PCA 前，**先考虑直接用原始数据**
> - 如果决定用 PCA，**测试集要用同一组训练集上找出的特征向量**
> - **每个特征向量的方向很不稳定**——试图解释方向的物理含义通常是幻觉，但**张成的子空间本身通常是稳定的**

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec16_unstable_eigenvectors.webp" alt="unstable eigenvectors" width="70%" loading="lazy" decoding="async" /></div>



### 1.9 <span style="color:red">四种无监督学习算法对比理解</span>

| 模型类型        | 最终建模出 $P(x)$（如异常检测）                                     | 非概率方法（如压缩、可视化）                             |
| ----------- | ------------------------------------------------------- | ------------------------------------------ |
| **寻找“子空间”** | <span style="color:red">**factor analysis**</span>      | <span style="color:red">**PCA**</span>     |
| **数据聚类**    | <span style="color:red">**Mixture of Gaussians**</span> | <span style="color:red">**K-means**</span> |


### 1.10 选 $k$（子空间维度）

若选：

$$
\frac{\lambda_1 + \dots + \lambda_k}{\lambda_1 + \dots + \lambda_k + \dots + \lambda_n} = c\%
$$

> 这就是 **"保留了 $c\%$ 的数据方差"**。通常取 $c\% = 0.90 / 0.95 / 0.98 \dots$

---







## 2. 独立成分分析 (ICA)

### 2.1 Cocktail Party 例子引入

> e.g. ICA 可用于**从混合样本中分离出不同的独立声音**。

假设有原始声源：

$$
s \in \mathbb{R}^n \quad (n \text{ speakers}), \quad s_j^{(i)} = \text{信号来自第 } j \text{ 个 speaker 在时刻 } i
$$

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec16_ICA_eg1_source.webp" alt="source signals" width="50%" loading="lazy" decoding="async" /></div>

> 注意：我们此时采样要保证两个样本**时间戳一致**。

而我们的观测数据是：

$$
x^{(i)} = A s^{(i)}, \quad x^{(i)} \in \mathbb{R}^n \quad (n \text{ microphones})
$$

因为每个麦克风实际上捕获的是来自同一时间不同 speaker 的声音的**线性组合**：

$$
x_j^{(i)} = \text{microphone } j \text{ 在时刻 } t \text{ 的收音}, \quad j = 1, \dots, n
$$

所谓线性组合在矩阵乘法上有所体现：

$$
x_j^{(i)} = \sum_k A_{jk} s_k^{(i)}
$$

在 ICA 中，我们的目标是找到一个分离矩阵：

$$
W = A^{-1}
$$

使得：

$$
s^{(i)} = W x^{(i)}
$$
这样我们就可以从观测 $x$ 出发来还原原始声源 $s$ 了。



### 2.2 ICA 目标

> 整个算法就是：给定数据 $x$，**找到矩阵 $W$**。

> 注意：标准 ICA 中我们要求 $s$ 与 $x$ 的**维数相同**——这样线性变换矩阵是个**方阵**，矩阵才有可能可逆！

**记号**：

$$
W = \begin{pmatrix} \text{---} w_1^T \text{---} \\ \text{---} w_2^T \text{---} \\ \vdots \\ \text{---} w_n^T \text{---} \end{pmatrix}
$$

所以 $s_j^{(i)} = w_j^T x^{(i)}$。



### 2.3 ICA 的直观图示

假设数据源是 2 个 speaker（每个时刻每个 speaker 发出的随机数在 $(-1, 1)$ 之间）：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec16_ICA_illustration_source.webp" alt="pre-processed source" width="60%" loading="lazy" decoding="async" /></div>

先回忆下 $s$ 怎么变成 $x$ （矩阵乘法所代表的每一维度值的线性组合）：

$$
x^{(i)} = A s^{(i)}
$$

即每个 $s$ 通过一个线性变换变成 $x$：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec16_ICA_illustration_observe.webp" alt="observed data" width="60%" loading="lazy" decoding="async" /></div>

> 所以实际中我们**观测到 $x$**，试图找到一个**线性变换把 $x$ 还原回 $s$**。



### 2.4 两个歧义 (Ambiguities)

ICA 有两个**不可消除**的歧义：

**① 轴序歧义 (Axis Ambiguity)**：从 $x$ 反向变换到 $s$ 时，我们**不知道 $s_1, s_2$ 的顺序**——所以还原出来时顺序是随机的。

**② 符号歧义 (Sign Ambiguity)**：线性变换中存在**正负翻转**。还原源信号时可能得到 $\pm s_1, \pm s_2$——但实际应用中**这并不重要**（声音正反不影响含义）。




这一讲关于 ICA 叙述至此而止，下一讲将把 ICA 讲完

---





## 参考资料

- [CS229 Lecture Note : PCA](https://cs229.stanford.edu/notes2021fall/cs229-notes10.pdf) — Lec 16 主讲义：PCA
- [CS229 Lecture Note : ICA](https://cs229.stanford.edu/notes2021fall/cs229-notes11.pdf) — Lec 16 讲义后半部分：ICA
- [CS229 2018](https://www.bilibili.com/video/BV1b4anzMEUv) — B 站上 CS229 Lec 16 讲课视频
- [Bishop, Pattern Recognition and Machine Learning, Ch.12](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) — Continuous Latent Variables（PCA 完整推导 + ICA 介绍）
- [Deep Learning](https://www.deeplearningbook.org/) — Goodfellow, Bengio, Courville；Ch.13 Linear Factor Models（含 PCA / FA / ICA）