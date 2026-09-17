---
title: "CS229 : Lec 14 补充证明 — K-Means 收敛性 + EM 推导细节"
description: CS229 Lec 14 的两篇补充证明。① K-Means 收敛性证明；② EM 步骤中的微积分细节：为什么 MLE 没有闭式解、以及驻点方程为什么与 EM M-step 的更新公式形式一致。
pubDate: 2026-09-17
series: cs229
subSeries: personal-lecture-notes
order: 14.5
categories:
  - CS229
  - K-Means
  - Convergence Proof
  - EM Algorithm
  - MLE Closed Form
  - Posterior Responsibility
---


> **TL;DR**:
> - **K-Means 收敛**：distortion function $\mathcal{J} \ge 0$ 有下界 + 内循环**精确单减** → 必收敛
> - **GMM MLE 无闭式解**：对观测对数似然求偏导得到的驻点方程是**隐式耦合**的，无法通过有限步解出。
> - **驻点方程 ≈ M-step 更新公式**：两者形式**完全相同**，但**含义不同**——驻点方程里 $w_j^{(i)}$ 依赖当前正在求的 $\theta$（耦合）；M-step 里 $w_j^{(i)}$ 由**上一轮的旧参数**算出后**当成常数**代入更新。
> - **本质统一**：观测对数似然的梯度 = **以 $P(z^{(i)} \mid x^{(i)}; \theta)$（即 $w_j^{(i)}$）为权重，对完全数据对数似然的梯度做加权平均**——这正是 EM 算法做的事。




## 引子

Lec 14 主笔记讲了 K-Means / GMM / EM 算法的整体思路。这篇是 Lec 14 的两篇**补充证明**：

1. **K-Means 收敛性证明**——为什么 K-Means 一定收敛？
2. **EM 步骤中的微积分细节**——为什么 GMM 的 MLE 没有闭式解？以及 EM 的 M-step 为什么和直接求导得到的驻点方程**形式完全相同**？

---





## 1. K-Means 收敛性证明 (Proof of K-Means' Convergence)

### 1.1 回顾 distortion function

之前在 K-Means 处定义的损失函数（distortion function）：

$$
\mathcal{J}(c, \mu) = \sum_{i=1}^{m} \|x^{(i)} - \mu_{c^{(i)}}\|^2
$$

> 它是**每个训练样本** $x^{(i)}$ **到它被分配的聚类中心** $\mu_{c^{(i)}}$ **的平方距离之和**。


### 1.2 收敛性论证

> 显然 $\mathcal{J}(c, \mu) \ge 0$——所以只要证明 K-Means 的**每一次迭代都让损失函数严格下降**即可：**单调递减** + **有下界（0）** $\Rightarrow$ 必收敛！

K-Means 中参数分为两部分：

- $c^{(i)}$：当前第 $i$ 个样本 $x^{(i)}$ 被分配给哪个簇
- $\mu_j$：第 $j$ 个簇的位置

K-Means 的目标就是**同时优化**这两组参数，让 distortion function 最小化。


### 1.3 Coordinate Descent 的视角

既然有两组参数需要优化，K-Means 采取的是**坐标优化法 (coordinate descent)**：<u>每次固定其他变量，只优化其中一个变量（或一组变量），使目标函数下降。</u>

所以在 K-Means 的内循环里：

- ① 固定 $\mu$，优化 $c$：对每个样本选取最合适的分类标签
- ② 固定 $c$，优化 $\mu$：由于此时各簇之间相互独立，因此可以分别对每个 $\mu_j$ 最小化——选各簇内的**质心**即可

> 由于内循环中**每一步都是精确的单减优化**，损失函数不会增加。


### 1.4 两个注意点

> ① 虽然 $\mathcal{J}$ 会收敛，但严格来说 $c$ 和 $\mu$ **不一定收敛到唯一值**——理论上可能出现：算法在几个不同的聚类结果之间来回跳，但这些结果的 $\mathcal{J}$ 完全相同。不过**实际中一般不会出现**这种情况。
>
> ② $\mathcal{J}$ **非凸**，所以收敛到的是**局部最优**，不保证全局最优。

---






## 2. EM 步骤中的微积分细节 (Detailed Calculus in EM Steps)

### 2.1 为什么 MLE 没有闭式解？

回顾 GMM 的对数似然：

$$
\begin{aligned}
\ell(\phi, \mu, \Sigma) &= \sum_{i=1}^{m} \log P(x^{(i)}; \phi, \mu, \Sigma) \\
&= \sum_{i=1}^{m} \log \left( \sum_{j=1}^{k} \phi_j \cdot \mathcal{N}(x^{(i)}; \mu_j, \Sigma_j) \right)
\end{aligned}
$$

#### ① 定义后验责任度

$$
w_j^{(i)} = P(z^{(i)} = j \mid x^{(i)}; \phi, \mu, \Sigma) = \frac{\phi_j \, \mathcal{N}(x^{(i)}; \mu_j, \Sigma_j)}{\sum_{l=1}^{k} \phi_l \, \mathcal{N}(x^{(i)}; \mu_l, \Sigma_l)}
$$

> 表示**在当前参数下，第 $i$ 个样本属于第 $j$ 个高斯分布的概率**。

#### ② 对 $\mu_j$ 求偏导

高斯密度对 $\mu_j$ 的导数：

$$
\frac{\partial}{\partial \mu_j} \log \mathcal{N}(x; \mu_j, \Sigma_j) = \Sigma_j^{-1} (x - \mu_j)
$$

所以：

$$
\Rightarrow \frac{\partial \ell}{\partial \mu_j} = \sum_{i=1}^{m} w_j^{(i)} \, \Sigma_j^{-1} (x^{(i)} - \mu_j) = 0
$$

因为 $\Sigma_j^{-1}$ 可逆，得到：

$$
\sum_{i=1}^{m} w_j^{(i)} (x^{(i)} - \mu_j) = 0
$$

$$
\Rightarrow \mu_j = \frac{\sum_{i=1}^{m} w_j^{(i)} \, x^{(i)}}{\sum_{i=1}^{m} w_j^{(i)}}
$$

#### ③ 对 $\Sigma_j$ 求导

高斯对数密度对 $\Sigma_j$ 的矩阵导数为：

$$
\frac{\partial}{\partial \Sigma_j} \log \mathcal{N}(x; \mu_j, \Sigma_j) = -\frac{1}{2} \Sigma_j^{-1} + \frac{1}{2} \Sigma_j^{-1} (x - \mu_j)(x - \mu_j)^T \Sigma_j^{-1}
$$

令导数为 0，得：

$$
\Sigma_j = \frac{\sum_{i=1}^{m} w_j^{(i)} \, (x^{(i)} - \mu_j)(x^{(i)} - \mu_j)^T}{\sum_{i=1}^{m} w_j^{(i)}}
$$

#### ④ 对 $\phi_j$ 求导

由于概率值变量自带**归一化约束** $\sum_{j=1}^{k} \phi_j = 1$，用拉格朗日乘子可得：

$$
\phi_j = \frac{1}{m} \sum_{i=1}^{m} w_j^{(i)}
$$

#### ⑤ 为什么得不到闭式解？

上面得到三个驻点方程：

$$
\mu_j = \frac{\sum_i w_j^{(i)} \, x^{(i)}}{\sum_i w_j^{(i)}}, \quad
\Sigma_j = \frac{\sum_i w_j^{(i)} \, (x^{(i)} - \mu_j)(x^{(i)} - \mu_j)^T}{\sum_i w_j^{(i)}}, \quad
\phi_j = \frac{1}{m} \sum_i w_j^{(i)}
$$

但关键问题是：

$$
w_j^{(i)} = \frac{\phi_j \, \mathcal{N}(x^{(i)}; \mu_j, \Sigma_j)}{\sum_{l=1}^{k} \phi_l \, \mathcal{N}(x^{(i)}; \mu_l, \Sigma_l)}
$$

> **$w_j^{(i)}$ 本身依赖于所有待估参数** $\phi, \mu, \Sigma$。

**所以**：

> 因此这些方程不是显式解，而是关于 $\phi, \mu, \Sigma$ 的**非线性隐式耦合方程组**。不能像完全数据情形那样，一步直接算出 $\mu_j, \Sigma_j, \phi_j$。

更准确地说：

> 不是数学上证明了**绝对不存在**任何闭式表达式，而是直接对观测似然求导得到的驻点方程是**隐式**的，<u>无法通过有限步初等运算显式解出</u>。因此通常说 GMM 的 MLE 没有闭式解——这也说明**为什么只能用迭代算法来做**。

---



### 2.2 为什么驻点方程 ≈ EM 的 M-step 更新公式？

直接对观测似然求偏导得到的驻点方程，和 EM 算法 M-step 里用的更新公式**长得完全一样**——为什么？

#### ① 形式一样，但含义不同

> **驻点方程**中 $w_j^{(i)}$ **依赖于正在被优化的参数** $\phi, \mu, \Sigma$——而这些参数里面又含 $w_j^{(i)}$，属于**隐式耦合关系**，无法解出。
>
> **M-step** 中 $w_j^{(i)}$ **先由旧参数算出**，然后**作为一个常量**去更新 $\phi, \mu, \Sigma$ 参数——这是**两个步骤**，所以可以用迭代方法不断更新收敛到最优值。

#### ② 事实上二者是统一的

由于：

$$
\ell(\theta) = \sum_{i=1}^{m} \log P(x^{(i)}; \theta)
$$

对 $\theta$ 求偏导，整理后可以写成：

$$
\begin{aligned}
\frac{\partial \ell}{\partial \theta} &= \sum_{i=1}^{m} \frac{1}{P(x^{(i)}; \theta)} \cdot \frac{\partial}{\partial \theta} P(x^{(i)}; \theta) \\
&= \sum_{i=1}^{m} \frac{1}{P(x^{(i)}; \theta)} \sum_{z^{(i)}} \frac{\partial}{\partial \theta} P(x^{(i)}, z^{(i)}; \theta) \\
&= \sum_{i=1}^{m} \frac{1}{P(x^{(i)}; \theta)} \sum_{z^{(i)}} P(z^{(i)} \mid x^{(i)}; \theta) \cdot \frac{1}{P(z^{(i)} \mid x^{(i)}; \theta)} \cdot \frac{\partial}{\partial \theta} P(x^{(i)}, z^{(i)}; \theta) \\
&= \sum_{i=1}^{m} \sum_{z^{(i)}} P(z^{(i)} \mid x^{(i)}; \theta) \cdot \frac{1}{P(x^{(i)}, z^{(i)}; \theta)} \cdot \frac{\partial P(x^{(i)}, z^{(i)}; \theta)}{\partial \theta} \\
&= \sum_{i=1}^{m} \sum_{z^{(i)}} P(z^{(i)} \mid x^{(i)}; \theta) \cdot \frac{\partial}{\partial \theta} \log P(x^{(i)}, z^{(i)}; \theta)
\end{aligned}
$$

这个式子就是在说：

> 观测数据对数似然的梯度 = **以 $P(z^{(i)} \mid x^{(i)}; \theta)$（也即是 $w_j^{(i)}$）为权重，对完全数据对数似然的梯度做加权平均**。

**这样我们就在**：
- **"无闭式解"**（直接对观测数据求偏导得到不可解的隐式耦合方程）

与
- **EM 算法**（用后验责任度 $w_j^{(i)}$ 对完全数据对数似然做加权，然后求导）

之间**建立了联系**——所以会发现最后的式子形式上**居然是一样的**！

---




## 参考资料

- [CS229 Mixture of Gaussians & EM](https://cs229.stanford.edu/notes2021fall/cs229-notes8.pdf) — Lec 14 主讲义：GMM + EM 算法
- [CS229 2018](https://www.bilibili.com/video/BV1b4anzMEUv) — B 站上 CS229 Lec 14 讲课视频
- [Bishop, Pattern Recognition and Machine Learning, Ch.9](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) — Mixture Models & EM 算法完整推导（含 Jensen 不等式、K-Means 收敛性）