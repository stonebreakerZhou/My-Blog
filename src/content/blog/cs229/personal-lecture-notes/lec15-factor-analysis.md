---
title: "CS229 : Lec 15 — 因子分析模型 (Factor Analysis Model)"
description: CS229 Lecture 15 学习笔记。① EM 可看成 J(θ, Q) 的坐标上升法；② GMM 在 m≪n 时协方差矩阵奇异不可逆；③ 因子分析模型的假设以及 Factor Analysis 上的 EM 推导。
pubDate: 2026-09-18
series: cs229
subSeries: personal-lecture-notes
order: 16
categories:
  - CS229
  - Factor Analysis
  - EM Algorithm
  - Multivariate Gaussian
  - Latent Variable
  - Low-rank Decomposition
---


> **TL;DR**:
> - **EM 收敛性（另一视角）**：定义 $\mathcal{J}(\theta, Q)$，EM 实质就是**坐标上升**：① E-step 固定 $\theta$ 优化 $Q$；② M-step 固定 $Q$ 优化 $\theta$。
> - **GMM 在 $m \ll n$ 时的困境**：**协方差矩阵奇异不可逆**。
> - **Factor Analysis 模型**：$z \sim \mathcal{N}(0, I) \in \mathbb{R}^d$（$d < n$），$x | z \sim \mathcal{N}(\mu + \Lambda z, \Psi)$——$z$ 是**连续低维潜在因子**，$\Lambda \in \mathbb{R}^{n \times d}$ 映射到观测，$\Psi$ 对角噪声。
> - **FA 核心思想**：高维数据实际上落在**低维子空间附近**。
> - **FA 上的 EM**：E-step 算 $z^{(i)} | x^{(i)}$ 的后验高斯分布；M-step 对二次型求导更新参数。




## 引子

Lec 14 讲了 GMM + EM。Lec 15 进入**因子分析 (Factor Analysis)**——当**样本数 $m$ 远小于特征维度 $n$** 时，GMM 不再适用，需要换一种模型：

- **EM 收敛性（另一种视角）**：把 EM 看成坐标上升法
- **GMM 在 $m \ll n$ 的问题**：单高斯协方差矩阵奇异
- **因子分析 (Factor Analysis)**：用低维潜在因子 + 对角噪声建模高维数据
- **多元高斯的两个核心性质**：边缘 / 条件分布公式
- **FA 上的 EM 推导**：E-step + M-step
- **GMM vs 因子分析对比**：一张大表

---





## 1. EM 收敛性（另一种视角）

### 1.1 回顾 E-step 和 M-step

回顾 E-step——计算后验概率（即**最终结论选 $Q_i$ 分布为后验概率**）：

$$
w_j^{(i)} = Q_i(z^{(i)} = j) = P(z^{(i)} = j \mid x^{(i)}; \phi, \mu, \Sigma)
$$

M-step——做这个最大化：

$$
\max_{\phi, \mu, \Sigma} \sum_i \sum_{z^{(i)}} Q_i(z^{(i)}) \log \frac{P(x^{(i)}, z^{(i)}; \phi, \mu, \Sigma)}{Q_i(z^{(i)})}
$$

> 注意：$Q_i(z^{(i)})$（即 $w^{(i)}$）是**后验概率**；$P(z^{(i)} = j) = \phi_j$ 是**先验概率**。

代入 GMM 具体形式上面这个式子可再写为：

$$
\begin{aligned}
&\Rightarrow \max_{\phi, \mu, \Sigma} \sum_i \sum_{z^{(i)}} Q_i(z^{(i)}) \log \frac{P(x^{(i)} \mid z^{(i)}) \, P(z^{(i)})}{Q_i(z^{(i)})} \\
&\Rightarrow \max_{\phi, \mu, \Sigma} \sum_i \sum_{j=1}^{k} w_j^{(i)} \log \frac{\mathcal{N}(x^{(i)}; \mu_j, \Sigma_j) \cdot \phi_j}{w_j^{(i)}}
\end{aligned}
$$

### 1.2 EM 的另一种等价视角——坐标上升法

定义：

$$
\mathcal{J}(\theta, Q) = \sum_i \sum_{z^{(i)}} Q_i(z^{(i)}) \log \frac{P(x^{(i)}, z^{(i)}; \phi, \mu, \Sigma)}{Q_i(z^{(i)})}
$$

我们知道：

$$
\ell(\theta) \ge \mathcal{J}(\theta, Q) \quad \forall\, \theta, Q
$$

所以 EM 的另一种等价视角是：

> - **E-step**：对 $Q$ 最大化 $\mathcal{J}$
> - **M-step**：对 $\theta$ 最大化 $\mathcal{J}$

> **这正是坐标上升法 (coordinate ascent)**！E-step 先选 $Q$ 让 $\ell(\theta) = \mathcal{J}(\theta, Q)$（最优 $Q_i$ 是后验分布）；在 E-step 固定了 $Q$ 之后 M-step 再选 $\theta$ 最大化 $\mathcal{J}(\theta, Q)$。


### 1.3 收敛性论证

坐标上升法就是关于 EM 算法的**与 Jensen 不等式视角之外的另一种等价视角**，在这个视角下可以回答了 EM 为何收敛：

$$
\ell(\theta^{(t+1)}) \ge \mathcal{J}(\theta^{(t+1)}, Q^t) \ge \mathcal{J}(\theta^t, Q^t) = \ell(\theta^t)
$$

> 每次从 $\theta^t \to \theta^{(t+1)}$ 都会让 $\ell(\theta)$ **上升**，而它有上界（具体上界尚未严格说明），故**单调有上界必收敛**。

---





## 2. GMM 的问题（$m \ll n$）

### 2.1 一个例子

比如有一个 **2 维**特征数据集，100 个数据点。这样具有充足数据点的时候我们可以用 mixture of Gaussians 拟合：

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs229/Lec15_GMM_fit_eg.webp" alt="GMM fit example" width="70%" loading="lazy" decoding="async" /></div>

> 但是存在**不**适合用 GMM 而用因子分析的情况：$m \approx n$ 或 $m \ll n$（$n$ 是特征维度，$m$ 是样本数）


### 2.2 单高斯模型有什么问题？

#### ① 拟合单高斯

假如建模单高斯分布：

$$
x \sim \mathcal{N}(\mu, \Sigma)
$$

MLE 结果：

$$
\mu = \frac{1}{m} \sum_{i=1}^{m} x^{(i)}, \qquad \Sigma = \frac{1}{m} \sum_{i=1}^{m} (x^{(i)} - \mu)(x^{(i)} - \mu)^T
$$

$\Sigma$ 是 $m$ 个中心化向量的外积之和——而这 $m$ 个向量张成的空间维度最大为 $m$。

> 若 $m \le n$，**协方差矩阵 $\Sigma$ 奇异（不可逆）**——因为 $\text{rank}(\Sigma) \le m < n$。

此时再看高斯密度公式：

$$
\frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}} \exp\!\left(-\frac{1}{2} (\cdots) \Sigma^{-1} (\cdots)\right)
$$

> $|\Sigma| = 0$，$\Sigma^{-1}$ 不存在。


#### ② 一个简单的不能用高斯分布拟合的例子

比如 $m = 2, n = 2$：

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs229/Lec15_GMM_fit_problem_eg.webp" alt="GMM fit problem" width="70%" loading="lazy" decoding="async" /></div>

> 如果画拟合出来的高斯分布的等高线，这会是**无限细的一条线**（实际上我们拟合了一条直线给这些点）。



### 2.3 引出因子分析的思路

既然直接拟合单高斯不行（因为 $\Sigma$ 奇异），拟合 GMM（多个高斯混合）显然**更**不行。

> 那能否对 $\Sigma$ 加一些限制，让我们能够拟合单高斯？

#### Option 1：把 $\Sigma$ 限制为对角阵

$$
\Sigma = \begin{pmatrix} \sigma_1^2 & \dots & 0 \\ 0 & \sigma_2^2 & \dots \\ \vdots & \vdots & \sigma_n^2 \end{pmatrix}
$$

这相当于强制高斯等高线是**轴对齐的椭圆**。

MLE 后：

$$
\sigma_j^2 = \frac{1}{m} \sum_i (x_j^{(i)} - \mu_j)^2
$$

现在 $\Sigma$ 只有 $n$ 个参数。

> **问题**：这个建模约束假设**所有特征都不相关**——这通常**不合理**（会导致欠拟合）。


#### Option 2：把 $\Sigma$ 限制为 $\sigma^2 I$

现在只有一个参数 $\sigma$ 。此时 MLE 结果：

$$
\sigma^2 = \frac{1}{mn} \sum_i \sum_j (x_j^{(i)} - \mu_j)^2
$$

> 但是这个限制不仅假设**各特征相互独立**，而且**各自的方差还相等**——这无疑是更不合理的。**但它可以引出因子分析的核心思路**。

以上这个不合理的限制对应的模型假设是：

$$
x \sim \mathcal{N}(\mu, \Sigma) \quad \Rightarrow \quad x = \mu + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)
$$

> 这个假设意味着数据在所有维度上的变化都是**等价、独立**的，<u>即这个变化完全依赖于噪声</u>——但这个假设太强，**不合理**。


### 2.4 因子分析的核心动机

> 承接上面，进一步有个自然的想法：如果数据的变化**并不完全取决于噪声**，而同样来自某种**共同的隐藏因素**？这个因素不是噪声，而是一个**真实的、结构性的变化来源**。

于是我们可以将生成模型改成：

$$
x = \mu + \underbrace{\Lambda z}_{\text{结构部分}} + \underbrace{\epsilon}_{\text{噪声}}
$$

其中：
- $z \in \mathbb{R}^d$：低维**潜在因子**，驱动数据的共同变化
- $\Lambda$：把低维因子映射到高维观测
- $\epsilon \sim \mathcal{N}(0, \Psi)$：每个维度独立的噪声

> **Personal Insight**：这与矩阵的低秩分解紧密相关——低维数据驱动得到的高维观测，对应着参数个数的减少。FA 模型中低秩分解最直观的观察就在协方差矩阵的变化中：
> $$\Sigma = \Lambda \Lambda^T + \Psi$$
> 其中 $\Lambda \Lambda^T = \mathbb{R}^{n \times d} \times \mathbb{R}^{d \times n}$ 就是一个**低秩分解**！

> 因子分析想要捕获**一些相关性**，但又不至于陷入单高斯模型那种**不可逆**的困境。

---





## 3. Factor Analysis 模型

### 3.1 模型框架

$$
P(x, z) = P(x \mid z) \, P(z), \quad z \text{ 是隐变量}
$$

$$
\begin{aligned}
z &\sim \mathcal{N}(0, I), \quad z \in \mathbb{R}^d \; (d < n) \\
x \mid z &\sim \mathcal{N}(\mu + \Lambda z, \Psi)
\end{aligned}
$$

> 模型骨架与 GMM 一样，都属于是**生成式模型**相同的框架。


### 3.2 假设

在因子分析模型中，我们有这些参数：

$$
\mu \in \mathbb{R}^n, \quad \Lambda \in \mathbb{R}^{n \times d}, \quad \Psi \in \mathbb{R}^{n \times n} \; (\text{diagonal})
$$

此时我们建模 $x$ 的方式为 ：

$$
\begin{aligned}
x &= \mu + \Lambda z + \epsilon \;(\text{Gaussian noise}) \\
z &\sim \mathcal{N}(0, I), \quad \epsilon \sim \mathcal{N}(0, \Psi)
\end{aligned}
$$

这也等价于 ：

$$
x \mid z \sim \mathcal{N}(\mu + \Lambda z, \Psi)
$$

> **关于这个建模的直观**：
> 1. 我们相信有 $d$ 个**主要因子**驱动 $x$ 的取值——所以 $x$ 是 $z \in \mathbb{R}^d$ 的线性函数。
> 2. 所有样本 $x$ 上的**噪声相互独立**——所以 $\Psi$ 是对角矩阵。



### 3.3 两种模型的参数个数比较

#### ① 因子分析的参数个数：

$$
\begin{aligned}
O(\mu + \Lambda + \Psi) &= O(n + nd + n) \\
&\Rightarrow O(nd)
\end{aligned}
$$

#### ② 回顾 GMM 的参数个数：

$$
\phi_j \in \mathbb{R}\ (j = 1, \dots, k) \text{ 但概率值满足归一化}; \quad \mu_j \in \mathbb{R}^n \ (j = 1, \dots, k); \quad \Sigma_j \in \mathbb{R}^{n \times n} \ (j = 1, \dots, k) \text{ 且对称}
$$

所以 GMM 总参数个数为：

$$
\begin{aligned}
O(\phi + \mu + \Sigma) &= O((k-1) + nk + k \cdot \frac{n(n+1)}{2}) \\
&= O(kn^2)
\end{aligned}
$$

> 直观比较后**显然**：因子分析的总自由参数个数**比 GMM 少很多**。



### 3.4 一个例子

用一个简单例子展示因子分析模型中<u>"高维数据其实落在低维子空间附近"</u>这一核心思想。

假设：

$$
x \in \mathbb{R}^2 \ (n=2), \quad z \in \mathbb{R}^1 \ (d = 1), \quad m = 7
$$

由于 $z \sim \mathcal{N}(0, 1)$，先从数轴上的一维高斯分布中采样 7 个 $z$ 值：

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs229/Lec15_Guassian_sample-z.webp" alt="$z$'s sampled from a Gaussian" width="100%" loading="lazy" decoding="async" /></div>

不妨设：

$$
\Lambda = \begin{pmatrix} 2 \\ 1 \end{pmatrix}, \qquad \mu = \begin{pmatrix} 0 \\ 0 \end{pmatrix}
$$

现在线性函数就是：

$$
x = \mu + \Lambda z = \begin{pmatrix} 0 \\ 0 \end{pmatrix} + \begin{pmatrix} 2 \\ 1 \end{pmatrix} z
$$

**所以当没有噪声 $\epsilon$ 时，$x$ 都落在一条直线上**：

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs229/Lec15_x_without_noise.webp" alt="$x$'s without Gaussian noises" width="100%" loading="lazy" decoding="async" /></div>

再不妨设：

$$
\Psi = \begin{pmatrix} 1 & 0 \\ 0 & 2 \end{pmatrix}
$$

这意味着 $x_2$ 维度方向的噪声方差 比 $x_1$ 维度方向上的噪声方差更大。

此时我们对 $x$ **加上高斯噪声** $\epsilon$：

$$
x = \mu + \Lambda z + \epsilon
$$

相当于在每个 $x$ 上加一个高斯等高线（噪声就会让 $x$ 产生一定的偏移量）：

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs229/Lec15_actual_sample_x.webp" alt="actual sampled $x$'s" width="100%" loading="lazy" decoding="async" /></div>

从这些高斯里采样，得到红色叉号——这就是我们建模后所认为的数据集 $x$ 的生成方式。

> 这里 $n = 2, d = 1$，意味着虽然数据本身是二维的，但**大部分数据落在一维子空间附近**（带一点噪声）。

> 当我们手上有高维数据 + 很少的样本时，没法通过这些数据拟合很复杂的模型——所以此时把它们拟合在一个子空间里可能就是合理的。

---





## 4. 多元高斯分布的性质 (Properties of Multivariate Gaussians)

设（把 $x$ 拆成上下两个子向量的拼接）：

$$
x = \begin{pmatrix} x_1 \\ \hline x_2 \end{pmatrix} \in \mathbb{R}^{r+s}, \quad x_1 \in \mathbb{R}^r, \quad x_2 \in \mathbb{R}^s
$$

若 $x \sim \mathcal{N}(\mu, \Sigma)$，则：

$$
\mu = \begin{pmatrix} \mu_1 \\ \hline \mu_2 \end{pmatrix} \in \mathbb{R}^{r+s}, \quad
\Sigma = \begin{pmatrix} \Sigma_{11} & \Sigma_{12} \\ \Sigma_{21} & \Sigma_{22} \end{pmatrix}, \quad
\Sigma_{11} \in \mathbb{R}^{r \times r}, \quad \Sigma_{22} \in \mathbb{R}^{s \times s}
$$

### 4.1 边缘概率 $P(x_1) = ?$

$$
\begin{aligned}
P(x) &= P(x_1, x_2) \\
\Rightarrow P(x_1) &= \int_{x_2} P(x_1, x_2) \, dx_2 = \int_{x_2} P(x) \, dx_2
\end{aligned}
$$

代入高斯密度 $P(x)$：

$$
= \int_{x_2} \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}} \exp\!\left(-\frac{1}{2} \begin{pmatrix} x_1 - \mu_1 \\ x_2 - \mu_2 \end{pmatrix}^T \begin{pmatrix} \Sigma_{11} & \Sigma_{12} \\ \Sigma_{21} & \Sigma_{22} \end{pmatrix}^{-1} \begin{pmatrix} x_1 - \mu_1 \\ x_2 - \mu_2 \end{pmatrix}\right)
$$

对 $x_2$ 积分后得：

$$
x_1 \sim \mathcal{N}(\mu_1, \Sigma_{11})
$$

### 4.2 条件概率 $P(x_1 \mid x_2) = ?$

$$
P(x_1 \mid x_2) = \frac{P(x_1, x_2)}{P(x_2)} = \frac{P(x)}{P(x_2)}
$$

它也服从高斯分布：

$$
\Rightarrow x_1 \mid x_2 \sim \mathcal{N}\!\left(\mu_{x_1 \mid x_2}, \Sigma_{x_1 \mid x_2}\right)
$$

其中：

$$
\begin{aligned}
\mu_{x_1 \mid x_2} &= \mu_1 + \Sigma_{12} \Sigma_{22}^{-1} (x_2 - \mu_2) \\
\Sigma_{x_1 \mid x_2} &= \Sigma_{11} - \Sigma_{12} \Sigma_{22}^{-1} \Sigma_{21}
\end{aligned}
$$

---





## 5. Factor Analysis 上的 EM 推导

### 5.1 准备工作——求联合分布 $P(x, z)$

把多元高斯性质套到因子分析模型上，求**联合分布** $P(x, z)$：

$$
\begin{pmatrix} z \\ x \end{pmatrix} \sim \mathcal{N}(\mu_{x,z}, \Sigma_{x,z}), \quad
z \sim \mathcal{N}(0, I) \in \mathbb{R}^d, \quad
\epsilon \sim \mathcal{N}(0, \Psi) \in \mathbb{R}^n, \quad
x = \mu + \Lambda z + \epsilon \in \mathbb{R}^n
$$

可以推出：

$$
\mathbb{E}[x] = \mu \in \mathbb{R}^n \;\Rightarrow\; \mu_{x,z} = \begin{pmatrix} \vec{0} \\ \mu \end{pmatrix} \in \mathbb{R}^{d+n}
$$

类似地：

$$
\Sigma_{x,z} = \begin{pmatrix} \Sigma_{zz} & \Sigma_{zx} \\ \Sigma_{xz} & \Sigma_{xx} \end{pmatrix}
$$

其中：

$$
\begin{aligned}
\Sigma_{zz} &= \text{Cov}(z) = I \\
\Sigma_{zx} &= \mathbb{E}[(z - \mathbb{E}[z])(x - \mathbb{E}[x])^T] = \mathbb{E}[z (\Lambda z + \epsilon)^T] \\
&= \mathbb{E}[zz^T] \Lambda^T + \mathbb{E}[z\epsilon^T] = \Lambda^T \\
\Sigma_{xz} &= \Lambda \\
\Sigma_{xx} &= \mathbb{E}[(x - \mathbb{E}[x])(x - \mathbb{E}[x])^T] = \mathbb{E}[(\Lambda z + \epsilon)(\Lambda z + \epsilon)^T] \\
&= \mathbb{E}[\Lambda z z^T \Lambda^T + \epsilon z^T \Lambda^T + \Lambda z \epsilon^T + \epsilon \epsilon^T] \\
&= \Lambda \Lambda^T + \Psi
\end{aligned}
$$

所以：

$$
\Sigma_{x,z} = \begin{pmatrix} I & \Lambda^T \\ \Lambda & \Lambda \Lambda^T + \Psi \end{pmatrix}
$$

$$
\begin{pmatrix} z \\ x \end{pmatrix} \sim \mathcal{N}\!\left( \begin{pmatrix} \vec{0} \\ \mu \end{pmatrix}, \begin{pmatrix} I & \Lambda^T \\ \Lambda & \Lambda \Lambda^T + \Psi \end{pmatrix} \right)
$$

> 现在可以写出 $P(x)$（就是这个高斯密度），对它对参数求 log-likelihood 的导数会发现**没有闭式解**。

> 所以要用 EM 算法来拟合参数。



### 5.2 E-step

计算：

$$
Q_i(z^{(i)}) = P(z^{(i)} \mid x^{(i)}; \theta)
$$

> **重要**：拟合 GMM 时 $z^{(i)}$ 是**离散的**；但这里 $z^{(i)}$ 是**连续密度**。可以用条件概率密度来表示这个连续密度：

$$
z^{(i)} \mid x^{(i)} \sim \mathcal{N}\!\left(\mu_{z^{(i)} \mid x^{(i)}}, \Sigma_{z^{(i)} \mid x^{(i)}}\right)
$$

用前面推导的多元高斯性质：

$$
\begin{aligned}
\mu_{z^{(i)} \mid x^{(i)}} &= \mu_z + \Sigma_{zx} \Sigma_{xx}^{-1} (x^{(i)} - \mu_x) \\
&= \vec{0} + \Lambda^T (\Lambda \Lambda^T + \Psi)^{-1} (x^{(i)} - \mu) \\
\Sigma_{z^{(i)} \mid x^{(i)}} &= \Sigma_{zz} - \Sigma_{zx} \Sigma_{xx}^{-1} \Sigma_{xz} \\
&= I - \Lambda^T (\Lambda \Lambda^T + \Psi)^{-1} \Lambda
\end{aligned}
$$

> E-step 中算这些，把它们存储起来，$Q_i$ 用一个**高斯密度**表示。E-step 完成后，我们得到每个样本 $x^{(i)}$ 对应的**后验高斯分布**。



### 5.3 M-step

$$
\begin{aligned}
\theta &= \arg\max_\theta \sum_i \int_{z^{(i)}} Q_i(z^{(i)}) \log \frac{P(x^{(i)}, z^{(i)})}{Q_i(z^{(i)})} \, dz^{(i)} \\
\Rightarrow \theta &= \sum_i \mathbb{E}_{z^{(i)} \sim Q_i}\!\left[ \log \frac{P(x^{(i)}, z^{(i)})}{Q_i(z^{(i)})} \right]
\end{aligned}
$$

> 分子分母都代入高斯密度（一般有 $\log$ 在前面时，就代入高斯密度）。代入后是个**二次型**，对各参数求导令其为 0 后就能更新参数。



### 5.4 坐标上升法视角

> 与 GMM 类似，因子分析模型的 EM 同样是可以视为一种**坐标上升法**：固定 $\mathcal{J}(\theta, Q)$ 后，先 E-step 固定 $\theta$ 优化 $Q$，再 M-step 固定 $Q$ 选最优参数 $\theta$。

---





## 6. GMM vs 因子分析模型

| 对比项 | GMM | 因子分析 |
| --- | --- | --- |
| **模型框架** | 生成式模型骨架 | 生成式模型骨架 |
| **隐变量 $z$** | <span style="color:red">**discrete**</span>：$z \in \{1, \dots, k\}$ | <span style="color:red">**continuous**</span>：$z \in \mathbb{R}^d$ |
| **$z$ 含义** | 簇标签 | 低维潜在因子 |
| **先验 $P(z)$** | $P(z = j) = \phi_j$ | $z \sim \mathcal{N}(0, I)$ |
| **条件分布 $P(x \mid z)$** | $x \mid z = j \sim \mathcal{N}(\mu_j, \Sigma_j)$ | $x \mid z \sim \mathcal{N}(\mu + \Lambda z, \Psi)$ |
| **边缘分布 $P(x)$** | 混合高斯：$\sum_{j=1}^{k} \phi_j \cdot \mathcal{N}(\mu_j, \Sigma_j)$ | 单个高斯：$\mathcal{N}(\mu, \Lambda \Lambda^T + \Psi)$ |
| **EM 步骤** | ① 算后验概率 $w_j^{(i)}$；② 加权 MLE | ① 算出后验高斯分布；② 对高斯期望求导 |
| **建模目标** | 聚类、密度估计 | 降维、高维小样本密度估计 |

---




## 参考资料

- [CS229 Factor Analysis](https://cs229.stanford.edu/notes2021fall/cs229-notes9.pdf) — Lec 15 主讲义：Factor Analysis + EM 推导
- [CS229 2018](https://www.bilibili.com/video/BV1b4anzMEUv) — B 站上 CS229 Lec 15 讲课视频
- [Bishop, Pattern Recognition and Machine Learning, Ch.12](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) — Continuous Latent Variables（含 Factor Analysis 完整推导）
- [Deep Learning](https://www.deeplearningbook.org/) — Goodfellow, Bengio, Courville；Ch.13 Linear Factor Models（含 PCA / FA / ICA）