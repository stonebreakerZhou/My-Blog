---
title: "CS229 : Lec 17 — 独立成分分析 (ICA) 续集"
description: CS229 Lecture 17 学习笔记。① ICA 的两个歧义；② ICA 预处理：均值归零 + 白化（方差归一）；③ ICA 的核心限制：输出 $s$ 必须非高斯；④ ICA 的 MLE 与随机梯度上升。
pubDate: 2026-09-22
series: cs229
subSeries: personal-lecture-notes
order: 18
categories:
  - CS229
  - ICA
  - Whiten
  - Sigmoid CDF
  - Non-Gaussianity
  - MLE
  - Cocktail Party Problem
---



> **TL;DR**:
> - **ICA 的两个歧义（ambiguities）**：置换歧义+ 放缩歧义。这两类歧义**无法消除**，但**实际应用中不重要**。
> - **预处理**：① 均值归零；② 白化（让协方差矩阵变为 $I$）。
> - **ICA 核心限制**：**输出 $s$ 必须非高斯分布**。
> - **密度变换公式**：$P_x(x) = P_s(Wx) \cdot |W|$
> - **选择 sigmoid 作为 CDF** 并进一步预先设定 ICA 步骤中 $P_s(s)$ 分布，再进行 ICA **MLE** 步骤推导。





## 引子

Lec 16 末尾提到 ICA 存在两个**不可消除**的歧义（置换 + 放缩）。Lec 17 把 ICA 讲完——核心围绕以下问题：

- **ICA 的两个歧义（补充证明）**：置换矩阵 + 行/列放缩
- **ICA 的预处理**：均值归零 + 白化
- **ICA 的核心限制**：为什么**输出必须非高斯**？如果输出服从 $\mathcal{N}(\vec{0}, I)$，任何正交矩阵 $R$ 都给出等价的 $W' = RW$
- **变换变量后概率密度之间的等式关系**：连续情形下 $P_x(x) = P_s(Wx) \cdot |W|$
- **如何选 $P_s(s)$**：sigmoid CDF（sigmoid PDF 有更胖的尾部）
- **MLE 推导**：随机梯度上升更新 $W$

---




## 1. ICA 两个歧义的补充说明

> 回顾 Lec 16 末尾：ICA 的解混矩阵 $W$ 满足 $s = Wx$。但 ICA 存在两个**不可消除**的歧义：

### 1.1 置换歧义 (Permutation Ambiguity)

> 对观测 $x$ 使用解混矩阵 $W$ 之后，我们所得到的输出 $s$ 的各个维度**无法区分顺序**。

证明：首先找一个置换矩阵 $P$（仅做行重排），再用这个作用到已经满足条件的解混矩阵 $W$ 上得到一个矩阵 $W'$ ：

$$
P = \begin{pmatrix}
0 & 1 & 0 \\
1 & 0 & 0 \\
0 & 0 & 1
\end{pmatrix}, \quad W' = PW
$$

> 如果 $W$ 已经是满足条件的解混矩阵，则 $W'$ 同样满足。也就是说，我们可以对 $W$ 进行任意的行变换，得到的仍是满足条件的解混矩阵。那么此时对应的输出 $s$ 同样会被行重排，所以我们得到的输出 $s$ 各个维度（也就是各行）之间无法区分顺序！


### 1.2 放缩歧义 (Scaling Ambiguity)

> 对观测作用 $W$ 后得到的 $s$ 各维度**可能被放缩**——在 ICA 中我们可以对已经得到的满足条件的 $W$ 的某一行进行放缩，得到的矩阵仍然是一个满足条件的解混矩阵。

> 此时，我们最后分离出来的独立成分（独立声源）$s$ （相应行）会被相应地缩放相应倍数（非零），但依然满足 ICA 目标，我们最后得到的输出始终有这种放缩歧义。

---





## 2. ICA 的预处理 (Pre-processing)

> 在正式进入 ICA 建模前，需要对数据做两个预处理步骤。

### 2.1 均值归零 (Zero out the mean)

$$
x \leftarrow x - \mathbb{E}[x]
$$

> 注意我们对数据均值归零后，由于 ICA 建模为 $s = Wx$，因此我们预计将分离出来的输出 $s$ 也将会被均值归零：

$$
\mathbb{E}[s] = \vec{0}
$$


### 2.2 白化 (Whiten / Standardize variance to 1)

$$
x \leftarrow x / \sigma_x
$$

> 这一步将观测数据方差归一。由于均值归零后的 $x$ 的协方差矩阵为 $\Sigma_x = \mathbb{E}[x x^T] \in \mathbb{R}^n$（symmetric）——目标是让 $\Sigma_x$ 变为单位矩阵 $I$。

先将协方差矩阵 $\Sigma_x$ 做特征值分解：

$$
\Sigma_x = U \Lambda U^T \quad (\Lambda \text{ is diagonal}, U \text{ is orthogonal})
$$

因此我们可以找到一个矩阵 $V$ 处理 $x$ 后将 $x$ 协方差矩阵变换为 $I$：

$$
V = \Lambda^{-1/2} U^T
$$

**验证**：

$$
\begin{aligned}
\text{Cov}(Vx) &= V \Sigma_x V^T \\
&= \Lambda^{-1/2} U^T \Sigma_x U \Lambda^{-1/2 \, T} \\
&= \Lambda^{-1/2} U^T U \Lambda U^T U \Lambda^{-1/2} \\
&= \Lambda^{-1/2} (U^T U) \Lambda (U^T U) \Lambda^{-1/2} \\
&= \Lambda^{-1/2} I \Lambda I \Lambda^{-1/2} \\
&= I
\end{aligned}
$$

> 所以对 $x$ 使用 $V$ 之后 $Vx$ 协方差矩阵变为 $I$。

> **结论**：原始观测数据 $x$ 经过上述 ① 均值归零 ② 白化 两个先续处理步骤之后，均值变为 $\vec{0}$，方差变为单位矩阵 $I$。

---






## 3. ICA 核心限制：输出必须非高斯 (non-Gaussian output)

### 3.1 ICA 目标回顾

> Goal : 寻找最优的解混矩阵 $W$，以从观测数据还原至原始独立声源：

$$
s = Wx
$$

> 假如我们现在已经对于最后的参数 $W$ 有一个估计取值，怎样知道这个估计是否好？这依赖于我们对于原始源的独立性先验假设：

$$
s_1, s_2, \dots, s_n \text{ are independent}
$$

> 这是我们最为核心的判别法则：对于估计的参数 $W$ 对应的输出 $s$，我们将根据其**各维度的独立性**判断当前的参数 $W$ 选取的好坏。


### 3.2 高斯情形下引出的问题

> 然而当前判据并不足够。考虑下面一种特殊情况：

> 假设选取一个 $W$ 以后，我们得到的 $s$ 各个维度服从一个多元正态分布 $s \sim \mathcal{N}(\vec{0}, I)$（注：由于前面对 $x$ 均值归零，故 $\mathbb{E}[s] = \vec{0}$；又由于 scaling ambiguity 所以我们可以不妨假设 $s$ 满足的多元高斯分布协方差矩阵为 $I$ ），此时我们能否判断当前选取的参数 $W$ 好坏？

> 首先，因为 $s \sim \mathcal{N}(\vec{0}, I)$，故此时输出的 $s$ 各维依然相互独立，满足我们最开始给出的判别方法。**那么此时选取的 $W$ 一定最好吗？**（即便已经抛开 permutation + scaling 所产生的变体）

> 但是，我们发现此时满足条件的参数矩阵 $W$ **具有无数多个**！（即便已经抛开 permutation + scaling 所产生的变体）下面会推出这无数多个 $W$ 。


### 3.3 为什么会推出有无数多个 $W'$ 

> 事实上，我们可以找到这些所有的参数矩阵 $W'$ 与现在我们已选择的这个参数矩阵 $W$ 之间的关系：

$$
W' = RW, \quad R \text{ 是任意的正交矩阵}
$$

**证明**：

> 若已有：

$$
s = Wx \sim \mathcal{N}(\vec{0}, I)
$$

那么更改参数矩阵后：

$$
\begin{aligned}
s' = W'x = RWx &\sim \mathcal{N}(R\vec{0}, RIR^T) \\
&\sim \mathcal{N}(\vec{0}, I)
\end{aligned}
$$

> 注意：**正交矩阵作用到一个 $\mathcal{N}(0, I)$ 的变量上面后得到的分布仍然是 $\mathcal{N}(0, I)$**！

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec17_standard_Guassian_contour.webp" alt="standard Gaussian contour" width="80%" loading="lazy" decoding="async" /></div>

> **补充理解**：正交矩阵只包含「翻转 + 旋转」这两种操作及其复合，而以二维正态分布为例，其等高线为圆——翻转、旋转后等高线不变，故输出的分布不变！

> 因此，如果输出 $s$ 满足多元高斯分布，那么我们可以在已有参数 $W$ 基础上选择**无数多种参数矩阵 $W'$**，这些 $W'$ 并不是 $W$ 的 permutation 或者 scaling 变体，但是它们作用于 $x$ 后得到的输出 $s$ 完全一样 ！因此此时参数的选取有无数种！**无法找到正确的那一类参数 $W$**！

> （补充知识：实际上我们除了独立性还有其他损失函数可以评判，但在此处高斯分布依然会导致损失评判恒为 0 而使得无法找到最优的参数 $W$）


### 3.4 核心限制

> 为避免上述情况产生，我们对于最后的输出 $s$ 加一条评判标准——**输出的 $s$ 不应满足多维高斯分布**！

---





## 4. CDF 与概率密度的关系

> 在 ICA 中，我们可以指定 $s$ 的 CDF（而非直接指定 PDF）来建模。

### 4.1 连续随机变量的 CDF

> **补充知识**：An equivalent way to represent the probability of the density of continuous random variables is via CDF：

$$
F(s) = P(S \leq s) \quad (S \text{ is a random variable, and } s \text{ is a constant})
$$

> **e.g.**：若 $S$ 是高斯随机变量，则其 CDF 是从 0 单调递增到 1 的函数。

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec17_CDF-to-PDF.webp" alt="CDF to PDF" width="100%" loading="lazy" decoding="async" /></div>

> （这其实就是概率论中密度函数与分布函数的关系）

> In ICA, instead of specifying a PDF for the source data, we're gonna choose a specified CDF that is **not a Gaussian density CDF**.


### 4.2 一个简单例子：$s \sim U(0, 1)$

> 假设：

$$
P_s(s) = \mathbf{1}\{0 \leq s \leq 1\} \quad (s \sim U(0, 1))
$$

假设：

$$
x = 2s \quad (\text{here } A = 2, W = 1/2, n = 1 \text{ (one-dimensional)})
$$

> 因此：

$$
x \sim U(0, 2), \quad P_x(x) = \frac{1}{2} \cdot \mathbf{1}\{0 \leq x \leq 2\}
$$

画出 $s$ 和 $x$ 的密度图：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec17_density_s-x.webp" alt="density from $s$ to $x$" width="70%" loading="lazy" decoding="async" /></div>



### 4.3 正确公式：$P_x(x) = P_s(Wx) \cdot |det(W)|$

> 一个**直觉上”感觉正确“**的方法是：

$$
P_x(x) = P_s(Wx) \quad (\text{because } Wx = s)
$$

> **但是，这是错的！** 这个式子只对**离散概率分布**成立，对**连续概率密度**不成立。


> **正确的公式**为：

$$
P_x(x) = P_s(Wx) \cdot |det(W)|
$$

其中 $|W|$ 是 $W$ 的行列式——这一项是为了保证分布仍归一化到 1。

**验证（在上面的简单例子中）**：

$$
\begin{aligned}
P_x(x) &= \frac{1}{2} \cdot \mathbf{1}\{0 \leq x \leq 2\} \\
&= \mathbf{1}\{0 \leq \tfrac{1}{2}x \leq 2\} \\
\text{and } \tfrac{1}{2} x = s \text{, so we're back to } P_s(s)
\end{aligned}
$$

---





## 5. 选择 $P_s(s)$ 分布具有的 CDF 为 Sigmoid 函数

### 5.1 输出 $s$ 的分布必须是非高斯

> **We need to choose a non-Gaussian for $P_s(s)$.**


### 5.2 我们想用 Sigmoid 作为 CDF

> **We can choose the sigmoid function for $F(s)$**：

$$
F(s) = P(S \leq s) = \frac{1}{1 + e^{-s}}
$$

> 对它求导后可以得到相应的 PDF。它比高斯密度有**更胖的尾部 (fatter tail)**，这种分布能更好地捕捉人声或其他自然现象——因为自然现象中存在大量极端值 (extreme outliers)。


### 5.3 其他可选 CDF

> **双侧指数分布 (Laplacian distribution)** 也是 $P_s(s)$ 的一个不错选择。

---





## 6. MLE 估计

> 现在我们进行 MLE 估计最优参数：MLE 实质就是给定观测数据 $x$，将数据概率值参数化为待估计的参数 $W$，随后最大化这个概率来估计参数 $W$ 的最优值。

### 6.1 写出 $P_x(x)$

> 因为 $s$ 各分量独立（ICA 核心假设），通过边缘分布写出联合分布的概率 ：

$$
P_s(s) = \prod_{i=1}^{n} P_s(s_i) \quad
(s \text{ is the vector of total voice sources}; \ n \text{ speakers are independent})
$$

MLE 需要将观测数据的概率值全部参数化为要估计的参数 $W$ ：

$$
\begin{aligned}
P_x(x) &= P_s(Wx) \cdot |det(W)| \\
&= \prod_{j=1}^{n} P_s(W_j^T x) \cdot |det(W)|
\end{aligned}
$$

> 所以 ICA 模型就是上式。这里我们选择 $P_s(\cdot)$ 为 sigmoid 对应的 PDF 概率分布，并将 $P_x(x)$ 表示为参数 $W$ 的函数。


### 6.2 对数似然

> Now the MLE：

$$
\ell(W) = \sum_{i=1}^{m} \log \left[ \left( \prod_{j} P_s(W_j^T x^{(i)}) \right) |det(W)| \right]
$$


### 6.3 随机梯度上升

> 我们可用**随机梯度上升**来最大化对数似然函数，更新参数 $W$ ：

$$
\nabla_W \ell(W) = \begin{pmatrix} 1 - 2g(W_1^T x) \\ \vdots \\ 1 - 2g(W_n^T x) \end{pmatrix} x^{(i)T} + (W^T)^{-1}
$$

> 其中 $g(\cdot)$ 是 sigmoid 函数。

---





## 7. Recap: ICA 完整算法流程

> ① 我们有整个训练集（观测 $x$）：

$$
x^{(1)}, \dots, x^{(m)}
$$

> 其中每个训练样本是一个麦克风录音（and we can split a certain timestamp out）。


（此时进行 均值归零 + 白化 预处理步骤）


> ② 我们现在**随机初始化解混矩阵 $W$**，然后跑**随机梯度上升**。当收敛时，我们得到最终的 $W$ 并用它来还原源信号 $s$ ：

$$
s = Wx
$$

---





## 参考资料

- [CS229 Lecture Note : ICA](https://cs229.stanford.edu/notes2021fall/cs229-notes11.pdf) — Lec 17 主讲义：ICA 完整推导 + MLE
- [CS229 2018](https://www.bilibili.com/video/BV1b4anzMEUv) — B 站上 CS229 Lec 17 讲课视频
- [Bishop, Pattern Recognition and Machine Learning, Ch.13](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) — Linear Latent Variable Models（含 ICA 的密度变换公式推导）
- [Deep Learning](https://www.deeplearningbook.org/) — Goodfellow, Bengio, Courville；Ch.13 Linear Factor Models（含 ICA / Sigmoid CDF 的尾部性质讨论）
