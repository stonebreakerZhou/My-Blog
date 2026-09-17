---
title: "CS229 : Lec 14 — K-Means + 高斯混合模型 (GMM) + EM 算法"
description: CS229 Lecture 14 学习笔记。进入无监督学习：① K-Means 聚类；② Gaussian Mixture Model (GMM)；③ EM（Expectation-maximum）算法。
pubDate: 2026-09-16
series: cs229
subSeries: personal-lecture-notes
order: 14
categories:
  - CS229
  - Unsupervised Learning
  - K-Means
  - Density Estimation
  - Gaussian Mixture Model
  - GMM
  - EM Algorithm
  - Jensen's Inequality
---


> **TL;DR**:
> - **K-Means**：初始化 K 个聚类中心 → 内循环两步骤——(a) 把每个样本分到相应最近的中心；(b) 把中心移到该簇样本质心处。
> - **密度估计 (Density Estimation)**：直接学 $p(x)$——给定一个新 $x$，若 $p(x) < \epsilon$ 则判为异常。
> - **GMM**：把 $p(x)$ 用 **K 个高斯的加权和**来建模：$p(x) = \sum_{j=1}^{k} \phi_j \mathcal{N}(x; \mu_j, \Sigma_j)$，其中 $\phi_j$ 是混合系数（先验权重）。。
> - **EM 算法**：迭代两步骤——
>   - **E-step**：用当前参数算 $z^{(i)}$ 的后验（即**软分类**概率 $w_j^{(i)}$）；
>   - **M-step**：把 $w_j^{(i)}$ 当成"伪标签"，重新用加权 MLE 更新 $\phi_j, \mu_j, \Sigma_j$
> - **Jensen 不等式视角**：EM 本质是**通过构造对数似然的下界、迭代地最大化下界**，对应 E、M两步骤，本质也是一种 MLE 




## 引子

前面 Lec 都是有监督学习（$x, y$ 都给）。Lec 14 进入**无监督学习**——只有 $x$，没有 $y$。三大主题：

- **K-Means Clustering**：最经典的聚类算法
- **Density Estimation**：直接建模 $p(x)$
- **Gaussian Mixture Model (GMM) + EM**：把 K-Means 的"硬分类"升级为"软分类" + 严格的最大似然估计

---





## 1. K-Means Clustering

给定一个**无标签**数据集，希望算法自动找出数据分成的若干簇：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec14_k-means_1.webp" alt="unlabeled dataset" width="50%" loading="lazy" decoding="async" /></div>

### 1.1 算法步骤

**第一步**：选两个叉号作为**聚类中心 (cluster centroids)**——根据每个样本到哪个中心更近来染色：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec14_k-means_2.webp" alt="initial centroids" width="50%" loading="lazy" decoding="async" /></div>

**第二步**：分别计算蓝色 / 红色点的均值，把中心**移到均值位置**：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec14_k-means_3.webp" alt="move centroids to means" width="50%" loading="lazy" decoding="async" /></div>

**迭代**：按到新中心的距离重新染色，再把中心移到新的均值位置：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec14_k-means_4.webp" alt="reassign and update" width="50%" loading="lazy" decoding="async" /></div>

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec14_k-means_5.webp" alt="converged" width="50%" loading="lazy" decoding="async" /></div>

**迭代到收敛**为止。


### 1.2 算法的数学描述

给定数据集 $x^{(1)}, \dots, x^{(m)}$（无标签）。

**① 初始化聚类中心**（通常**随机选** $k$ 个训练样本作为初始中心即可）：

$$
\mu_1, \dots, \mu_k \in \mathbb{R}^n
$$

**② Repeat until convergence**：

**(a)** 给每个样本分配簇：

$$
c^{(i)} = \arg\min_j \|x^{(i)} - \mu_j\|_2
$$

（用 $\ell_2$ 范数，开不开平方都行）

**(b)** 对 $j = 1, \dots, k$，把每个中心更新到该簇的样本均值：

$$
\mu_j := \frac{\sum_{i=1}^{m} \mathbb{1}\{c^{(i)} = j\} \, x^{(i)}}{\sum_{i=1}^{m} \mathbb{1}\{c^{(i)} = j\}}
$$


### 1.3 收敛性与代价函数

> K-Means 算法**一定会收敛**。如果写成 **cost function (也称distortion function)**：

$$
\mathcal{J}(c, \mu) = \sum_{i=1}^{m} \|x^{(i)} - \mu_{c^{(i)}}\|^2
$$

它是**分配 $c$** 和**中心 $\mu$** 的函数。**每一步迭代都会让 $\mathcal{J}$ 下降**——又因为 $\mathcal{J} \ge 0$——所以算法必收敛。

> **注意**：K-Means **可能陷入局部最优**（不同的初始化可能会得到不同的结果）。

---





## 2. 密度估计 (Density Estimation)

### 2.1 Motivating Example：飞机发动机异常检测

给定发动机的**振动值**和**热量值**组合，判断它是不是异常样本：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec14_GMM_1.webp" alt="engine anomaly detection" width="55%" loading="lazy" decoding="async" /></div>

一种实现是**直接建模** $p(x)$ 这个函数告诉我们"特征组合 $x$ 出现的密度有多高"。当 $p(x) < \epsilon$ 时，就认为是**异常**。

> 注意图中绿点的有趣之处：**单独看振动或热量都在正常范围内**——但两者**组合起来**就是异常。这就是密度估计需要看**联合分布**的原因。


### 2.2 复杂分布的处理

如果只看数据点，是 **"L" 形分布**，**很难用一个单一分布直接建模**。


### 2.3 Mixture of Gaussian

我们换个思路——把它看成**两个高斯分布的混合 (mixture of Gaussian)**。下图中两个椭圆分别是两个高斯分布的等高线：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec14_GMM_2.webp" alt="two Gaussian ellipses" width="55%" loading="lazy" decoding="async" /></div>

> 注意二维高斯分布的密度函数的等高线就是一个椭圆——两个椭圆的拼接就是最终的密度。

---





## 3. Gaussian Mixture Model (GMM)

### 3.1 一维例子

假设我们有这样一组**一维**数据点：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec14_GMM_3.webp" alt="1D GMM example" width="75%" loading="lazy" decoding="async" /></div>

我们假设这些数据来自**两个高斯分布**，但**不知道**每个点来自哪个分布（无标签）。

> **EM (Expectation-Maximum)** 算法能帮我们在这种不知道每个样本属于哪个分布的情况下拟合 GMM。


### 3.2 引入隐变量 $z^{(i)}$

我们引入一个变量 $z^{(i)}$ 表示数据点 $x^{(i)}$ 来自哪个高斯分布。设 $x^{(i)}, z^{(i)}$ 有联合分布：

$$
P(x^{(i)}, z^{(i)}) = P(x^{(i)} \mid z^{(i)}) \, P(z^{(i)})
$$

其中 $z^{(i)}$ 服从参数为 $\phi$ 的多项分布：

$$
\begin{aligned}
z &\sim \text{Multinomial}(\phi) \\
z^{(i)} &\in \{1, \dots, k\} \quad \text{（总共 } k \text{ 种取值）}\\
\phi_j &\ge 0, \quad \sum_{j=1}^{k} \phi_j = 1
\end{aligned}
$$

> 参数 $\phi_j$ 给出 $P(z^{(i)} = j)$，即第 $i$ 个样本属于第 $j$ 个高斯分布的概率值。



### 3.3 数据生成过程

在我们上面的例子中，假设一共有 $k$ 个簇，每个簇内部是一个高斯分布：

$$
x^{(i)} \mid z^{(i)} = j \;\sim\; \mathcal{N}(\mu_j, \Sigma_j)
$$

所以可以视为数据点的产生是这样的：
1. 先按概率 $\phi_j$ 抽到 $z^{(i)} = j$；
2. 再在第 $j$ 个簇的高斯分布里抽到 $x^{(i)}$。

**独立重复** $m$ 次得到数据集 $\{x^{(1)}, \dots, x^{(m)}\}$。



### 3.4 边缘分布 $P(x^{(i)})$

通过联合分布求 $z^{(i)}$ 的边缘：

$$
\begin{aligned}
P(x^{(i)}, z^{(i)} = j) &= P(x^{(i)} \mid z^{(i)}) \, P(z^{(i)} = j) \\
&= P(x^{(i)} \mid z^{(i)} = j) \, \phi_j \\
&= \phi_j \cdot \mathcal{N}(x^{(i)}; \mu_j, \Sigma_j)
\end{aligned}
$$

实际问题中我们只观测到 $x^{(i)}$，看不到 $z^{(i)}$。所以**对 $z^{(i)}$ 求和，得到 $x^{(i)}$ 的边缘分布**——也就是我们最终要对原始数据建模的概率分布：

$$
\begin{aligned}
P(x^{(i)}) &= \sum_{j=1}^{k} P(x^{(i)}, z^{(i)} = j) \\
\Leftrightarrow \quad P(x^{(i)}) &= \sum_{j=1}^{k} \phi_j \cdot \mathcal{N}(x^{(i)}; \mu_j, \Sigma_j)
\end{aligned}
$$

> <span style="color:red">**把 </span>$\textcolor{red}{P(x^{(i)})}$<span style="color:red"> 这个最终要建模的概率分布"参数化"为参数 </span>$\textcolor{red}{\phi_j, \mu_j, \Sigma_j}$<span style="color:red"> 来表示。**</span>


### 3.5 要估计的参数

只要估出参数最优值，这个建模概率分布就有了：

$$
\theta = \{\phi_j, \mu_j, \Sigma_j\}
$$


### 3.6 最大似然估计 (MLE)

给定数据点估计参数值，本能想法是**最大似然估计**：

$$
\begin{aligned}
\text{want to max:} \quad \mathcal{L}(\theta) &= \prod_{i=1}^{m} p(x^{(i)}; \theta) \\
\Leftrightarrow \quad \max \ell(\theta): \quad \log \mathcal{L}(\theta) &= \sum_{i=1}^{m} \log p(x^{(i)}; \theta)
\end{aligned}
$$

> 最大似然估计的本质：把 $P(x)$ 这个要最终建模的概率分布"参数化"为一个由 $\theta$ 控制的函数族 $P(x; \theta)$，最大化时的参数取值就是最后的估计值。

代入 GMM 的 $P(x^{(i)})$ 表达式：

$$
\begin{aligned}
\ell(\phi, \mu, \Sigma) &= \sum_{i=1}^{m} \log P(x^{(i)}; \phi, \mu, \Sigma) \\
&= \sum_{i=1}^{m} \log \left( \sum_{j=1}^{k} \phi_j \cdot \mathcal{N}(x^{(i)}; \mu_j, \Sigma_j) \right)
\end{aligned}
$$

> **直接对上面这个对数似然函数求最大化（令导数为 0）会发现<u>无法用闭式解</u>找到参数的最大似然估计。** 所以后面引入 EM 算法来解决这个问题。

---





## 4. $w^{(i)}_j$ 的引入

如果我们对上面的对数似然函数直接求最大化来解参数，发现无法用闭式解——**为什么？**

### 4.1 一个有趣的现象：已知 $z^{(i)}$ 时会发生什么

如果我们已知 $z^{(i)}$ 的值（也就是我们<span style="color:red">**不仅观测到 </span>$\textcolor{red}{x^{(i)}}$<span style="color:red">，而且还知道它属于哪个分量 </span>$\textcolor{red}{z^{(i)}}$<span style="color:red">**</span>），那么此时似然函数可以拆写为：

$$
\begin{aligned}
\ell(\phi, \mu, \Sigma) &= \sum_{i=1}^{m} \log \sum_{z^{(i)}=1}^{k} P(x^{(i)} \mid z^{(i)}; \mu, \Sigma) \, P(z^{(i)}; \phi) \\
&= \sum_{i=1}^{m} \big[ \log P(x^{(i)} \mid z^{(i)}; \mu, \Sigma) + \log P(z^{(i)}; \phi) \big]
\end{aligned}
$$

> 此时 $z^{(i)}$ 已经确定，不再对 $z^{(i)}$ 从 1 求和到 $k$。

这时再求 MLE——对 $\phi_j, \mu_j, \Sigma_j$ 求偏导并令其为 0——就能方便地解出：

$$
\begin{aligned}
\phi_j &= \frac{1}{m} \sum_{i=1}^{m} \mathbb{1}\{z^{(i)} = j\} \\
\mu_j &= \frac{\sum_{i=1}^{m} \mathbb{1}\{z^{(i)} = j\} \, x^{(i)}}{\sum_{i=1}^{m} \mathbb{1}\{z^{(i)} = j\}} \\
\Sigma_j &= \frac{\sum_{i=1}^{m} \mathbb{1}\{z^{(i)} = j\} \, (x^{(i)} - \mu_j)(x^{(i)} - \mu_j)^T}{\sum_{i=1}^{m} \mathbb{1}\{z^{(i)} = j\}}
\end{aligned}
$$


### 4.2 启发

> 既然已知 $z^{(i)}$ 后就能轻松求得 MLE，那能不能**先为每个样本预设一个类别值** $z^{(i)}$，再迭代地更新？

> **EM 的核心思想**：为每个样本当前设定一个 **"软" 类别值**（按一定概率分到不同类别）作为这一轮的"假设"，据此做优化；然后迭代地进入下一步，用新的"假设"继续更新——最后应该会收敛到真实的最优值。

---





## 5. EM 算法概览

### 5.1 E-step：猜 $z^{(i)}$ 的值

设：

$$
w_j^{(i)} = P(z^{(i)} = j \mid x^{(i)}; \phi, \mu, \Sigma)
$$

> 这是一种**"软分类"**——把每个样本点按**不同概率值**分到不同分布里（不像 K-Means 那样每一步每个样本点只能属于一个分布）。

由贝叶斯公式（与生成式学习算法类似）：

$$
w_j^{(i)} = \frac{\textcolor{red}{P(x^{(i)} \mid z^{(i)} = j)} \cdot \textcolor{blue}{P(z^{(i)} = j)}}{\sum_{l=1}^{k} P(x^{(i)} \mid z^{(i)} = l) \, P(z^{(i)} = l)}
$$

> - $\textcolor{red}{P(x^{(i)} \mid z^{(i)} = j)}$ 来自高斯密度
> - $\textcolor{blue}{P(z^{(i)} = j)}$ 来自 $z \sim \text{Multinomial}(\phi)$ 的假设
> - 分母项也来自高斯分布的密度以及 $\phi$

也可以写成：

$$
w_j^{(i)} = \frac{\textcolor{red}{\mathcal{N}(x^{(i)}; \mu_j, \Sigma_j)} \cdot \textcolor{blue}{\phi_j}}{\sum_{l=1}^{k} \mathcal{N}(x^{(i)}; \mu_l, \Sigma_l) \, \phi_l}
$$

> $w_j^{(i)}$ 依赖于（旧的）参数。

**E-step** 就是猜每个样本的 $z^{(i)}$，并把概率存在 $w_j^{(i)}$ 里——**$w^{(i)}$ 就是 $z^{(i)}$ 的后验分布**。

> $w_j^{(i)}$：$x^{(i)}$ **被分到第 $j$ 个高斯的程度**。


### 5.2 M-step：用 MLE 更新参数

用 $w_j^{(i)}$ 和 MLE 估计参数（注意 $\Sigma_j$ 要用**更新后**的 $\mu_j$ 来算）：

$$
\begin{aligned}
\phi_j &= \frac{1}{m} \sum_{i=1}^{m} w_j^{(i)} \\
\textcolor{blue}{\mu_j} &= \frac{\sum_{i=1}^{m} w_j^{(i)} \, x^{(i)}}{\sum_{i=1}^{m} w_j^{(i)}} \\
\Sigma_j &= \frac{\sum_{i=1}^{m} w_j^{(i)} \, (x^{(i)} - \textcolor{blue}{\mu_j})(x^{(i)} - \textcolor{blue}{\mu_j})^T}{\sum_{i=1}^{m} w_j^{(i)}}
\end{aligned}
$$

> 注：$w_j^{(i)} = \mathbb{E}[\mathbb{1}\{z^{(i)} = j\}]$。

**迭代**：重新算 $w_j^{(i)}$，再代入上面的步骤——**$w_j^{(i)}$ 依赖于上一轮的参数**。


### 5.3 K-Means vs GMM（软分配）

> GMM 的一个直观理解：它就像 K-Means 但用**软分配**。
> K-Means 中一旦更新中心，每个点就**硬**分到一个中心；而 EM 用**概率作为权重**把每个点分到各簇，再更新相应的均值。

---





## 6. EM 算法的严格数学推导

### 6.1 工具——Jensen 不等式

设 $f$ 是凸函数（例如 $f'' > 0$）；$X$ 是随机变量：

$$
f(\mathbb{E}[X]) \le \mathbb{E}[f(X)]
$$

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec14_Jensen-inequality.webp" alt="Jensen's inequality" width="50%" loading="lazy" decoding="async" /></div>

进一步，如果 $f'' > 0$（$f$ 是**严格凸**），则：

$$
f(\mathbb{E}[X]) = \mathbb{E}[f(X]) \iff X \text{ 是常数}
$$

> 当 $f$ 由凸变为凹时，所有不等号反向。


### 6.2 最优化目标

有 $P(x, z; \theta)$（$\theta$ 是参数），但只观测到 $x : \{x^{(1)}, \dots, x^{(m)}\}$：

$$
\begin{aligned}
\ell(\theta) &= \sum_{i=1}^{m} \log P(x^{(i)}; \theta) \\
&= \sum_{i=1}^{m} \log \left[ \sum_{z^{(i)}} P(x^{(i)}, z^{(i)}; \theta) \right]
\end{aligned}
$$

我们想求 $\max_\theta \ell(\theta)$，并推导一个迭代算法找 MLE 估计。（由于 log 里有求和，直接求导**没有闭式解**。因此我们想找一个**下界**，通过最大化下界来间接最大化 $\ell(\theta)$——**目标思想还是 MLE** 。）


### 6.3 EM 的几何直观

把整个迭代优化过程画出来：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec14_EM_1.webp" alt="log-likelihood curve" width="50%" loading="lazy" decoding="async" /></div>

**首先**，随机初始化 $\theta$：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec14_EM_2.webp" alt="initialize theta" width="50%" loading="lazy" decoding="async" /></div>

**1) E-step**：在当前 $\theta$ 处为对数似然曲线（绿线）**构造下界**，有两条性质——① 在下方；② 在 $\theta$ 处**与对数似然相切（相等）**：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec14_EM_3.webp" alt="E-step: lower bound" width="50%" loading="lazy" decoding="async" /></div>

**2) M-step**：找到使绿线下界最大化的新 $\theta$ 值，把 $\theta$ **更新过去**：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec14_EM_4.webp" alt="M-step: maximize bound" width="50%" loading="lazy" decoding="async" /></div>

**接着**在新的 $\theta$ 处再构造新的下界，迭代更新——最终收敛到**局部最优**：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec14_EM_5.webp" alt="convergence" width="50%" loading="lazy" decoding="async" /></div>


### 6.4 数学推导

把上面的几何过程写成数学：

$$
\begin{aligned}
\text{Goal:} \quad \max_\theta \sum_i \log P(x^{(i)}; \theta) &\Rightarrow \sum_i \log \sum_{z^{(i)}} P(x^{(i)}, z^{(i)}; \theta) \\
&\Rightarrow \sum_i \log \sum_{z^{(i)}} Q_i(z^{(i)}) \left[ \frac{P(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})} \right]
\end{aligned}
$$

其中 $Q_i(z^{(i)})$ 是一个**概率分布**，满足 $\sum_{z^{(i)}} Q_i(z^{(i)}) = 1$。

把方括号里的项看成 $z^{(i)}$ 的函数——也就是对 $z^{(i)}$ 求期望：

$$
\Rightarrow \sum_i \log \mathbb{E}_{z^{(i)} \sim Q_i}\!\left[ \frac{P(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})} \right]
$$

由 Jensen 不等式（**log 是凹函数**！）：

$$
\Rightarrow \ge \sum_i \mathbb{E}_{z^{(i)} \sim Q_i}\!\left[ \log \frac{P(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})} \right]
$$

展开期望：

$$
\Rightarrow \ge \sum_i \sum_{z^{(i)}} Q_i(z^{(i)}) \, \log \frac{P(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})}
$$

所以：

$$
\ell(\theta) \ge \sum_i \sum_{z^{(i)}} Q_i(z^{(i)}) \, \log \frac{P(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})}
$$

> **注意**：右边表达式是 $\theta$ 的函数！（$x^{(i)}$ 是数据，对 $z^{(i)}$ 求和消掉了。）它可以作为 $\ell(\theta)$ 的**下界**！



### 6.5 让 Jensen 不等式取等

我们希望这个下界**在当前 $\theta$ 处与 $\ell(\theta)$ 相等**——也就是让 Jensen 取等：

$$
\begin{aligned}
\sum_i \log \mathbb{E}_{z^{(i)} \sim Q_i}\!\left[ \frac{P(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})} \right] &= \sum_i \mathbb{E}_{z^{(i)} \sim Q_i}\!\left[ \log \frac{P(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})} \right] \\
\Leftrightarrow \log \mathbb{E}_{z^{(i)} \sim Q_i}\!\left[ \frac{P(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})} \right] &= \mathbb{E}_{z^{(i)} \sim Q_i}\!\left[ \log \frac{P(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})} \right]
\end{aligned}
$$

Jensen 取等的条件：

$$
\frac{P(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})} = \text{constant} \quad (\forall z^{(i)})
$$

所以可以设：

$$
Q_i(z^{(i)}) \propto P(x^{(i)}, z^{(i)}; \theta)
$$

由于 $Q_i$ 是 $z^{(i)}$ 的概率分布，可以归一化为：

$$
\begin{aligned}
Q_i(z^{(i)}) &= \frac{P(x^{(i)}, z^{(i)}; \theta)}{\sum_{z^{(i)}} P(x^{(i)}, z^{(i)}; \theta)} \\
&= P(z^{(i)} \mid x^{(i)}; \theta)
\end{aligned}
$$

> 注：$\sum_{z^{(i)}} P(x^{(i)}, z^{(i)}; \theta) = P(x^{(i)})$。这其实就是**后验概率 (posterior probability)**。

---





## 7. EM 算法小结

**E-step**：

$$
Q_i(z^{(i)}) = P(z^{(i)} \mid x^{(i)}; \theta)
$$

> 注：之前设的 $w_j^{(i)}$ 现在就放在 $Q_i$ 分布里。

**M-step**：

$$
\theta := \arg\max_\theta \sum_i \sum_{z^{(i)}} Q_i(z^{(i)}) \, \log \frac{P(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})}
$$

> 找到下界的最大点，把 $\theta$ 更新过去！

> <span style="color:red">**EM 算法本质上是"通过构造对数似然的下界、迭代地最大化下界"的 MLE 算法——它不放弃 MLE 的目标，而是把"无法直接最大化"的问题转化为"逐步构造紧贴的下界、最大化下界"。**</span>

---





## 参考资料

- [CS229 Mixture of Gaussians & EM](https://cs229.stanford.edu/notes2021fall/cs229-notes8.pdf) — Lec 14 主讲义：GMM + EM 算法
- [CS229 2018](https://www.bilibili.com/video/BV1b4anzMEUv) — B 站上 CS229 Lec 14 讲课视频
- [Bishop, Pattern Recognition and Machine Learning, Ch.9](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) — Mixture Models & EM 算法完整推导