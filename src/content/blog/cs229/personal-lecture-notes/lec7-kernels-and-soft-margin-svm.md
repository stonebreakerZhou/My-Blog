---
title: "CS229 : Lec 7 — 核方法（Kernels） 与  Soft-Margin SVM"
description: CS229 Lecture 7 笔记，从上一讲的 Optimal Margin Classifier 出发，推导 SVM 的对偶问题（Dual Optimization），然后引入 Kernel Trick 让 SVM 能学习非线性边界；接着讨论 Kernel 有效性判定的 Mercer Theorem；最后通过 ℓ₁ 范数 Soft-Margin SVM 放宽上一讲我们一直保留的数据集完全线性可分假设。
pubDate: 2026-08-30
series: cs229
subSeries: personal-lecture-notes
order: 7
categories:
  - CS229
  - SVM
  - Kernel Trick
  - Mercer Theorem
  - Soft Margin
---

> **TL;DR**:
> - **Representation Theorem**：假设 $w$ 可以写成训练样本的线性组合 $w = \sum_{i=1}^m \alpha_i y^{(i)} x^{(i)}$，这是推导 SVM 对偶形式的桥梁。这个假设可由 Logistic 回归梯度下降或 $w$ ⟂ 决策边界的几何论证支撑。
> - **Dual Optimization**：在 representation 假设下，最优间隔分类器的目标可以被改写成只关于 $\alpha$ 的最大化问题——目标函数与约束**仅通过内积 $\langle x^{(i)}, x^{(j)} \rangle$ 与训练样本接触**。
> - **Kernel Trick**：用映射 $\Phi: X \to$ 高维特征空间后，只要能高效地计算 $K(x, z) = \Phi(x)^T \Phi(z)$，就可以把算法里所有 $\langle x, z \rangle$ 替换成 $K(x, z)$，从而把 SVM 训练等价地"搬"到高维空间中去。
> - **Mercer Theorem**：$K$ 是一个有效核（即存在 $\Phi$ 使 $K(x, z) = \Phi(x)^T \Phi(z)$）**当且仅当** 对应的 Kernel Matrix 是半正定的。
> - **证明高斯核是有效核**：利用泰勒展开与多项式核，可以证明高斯核的 Kernel Matrix 满足半正定条件。
> - **$\ell_1$-norm Soft Margin SVM**：通过引入 slack variable $\xi_i \ge 0$ 与惩罚项 $C \sum_i \xi_i$，允许某些点的 functional margin $< 1$，让 SVM 对噪声和异常值更鲁棒

## 引子

Lec 6 我们把 SVM 的优化目标写成了

$$
\min_{w, b} \frac{1}{2}\|w\|^2 \qquad \text{s.t. } y^{(i)}(w^T x^{(i)} + b) \ge 1,\ i = 1, \dots, m
$$

这是一个凸二次规划。但它有一个**根本性限制**：解出来的决策边界是**线性**的，遇到非线性可分的数据就无能为力。

这节课要做两件事：

1. 借助 **representation theorem** 把上面这个原始（primal）问题改写成 **对偶（dual）** 形式——改写之后整个算法可以只通过**两个样本的内积**来表达，从而为引入 核方法（Kernel trick） 打开大门。
2. 通过 $\ell_1$-norm **soft margin** 放宽"训练集必须线性可分"的假设。

---

## 1. Recap & Representation Theorem

### 1.1 上节课得到的最终目标式

上一讲我们推导出，Optimal Margin Classifier 的目标是

$$
\min_{w, b} \frac{1}{2}\|w\|^2 \qquad \text{s.t. } y^{(i)}(w^T x^{(i)} + b) \ge 1,\ i = 1, \dots, m
$$

约束的含义是：**每个样本的 functional margin 都要 $\ge 1$**；目标 $\frac{1}{2}\|w\|^2$ 越小，几何间隔 $\gamma = \hat\gamma / \|w\|$ 就越大。

### 1.2 Representation Theorem

为了把 SVM 改写成对偶形式，需要先做一个额外的假设：

> **Assumption（Representation Theorem）**：$w$ 可以被表示成训练样本的线性组合
> $$ w = \sum_{i=1}^m \alpha_i x^{(i)} $$

（这个假设的完整证明比较长，这里只给两个直觉性论证。）

加入 $y^{(i)}$ 后（注意SVM分类标签 $y^{(i)} = \pm 1$，所以乘进去不改变线性组合的"形式"），可以写成更对称的形式：

$$
w = \sum_{i=1}^m \alpha_i y^{(i)} x^{(i)}
$$

#### 直觉一：来自 Logistic 回归的梯度下降

在 Logistic 回归中，参数 $\theta$ 按如下规则更新：

$$
\theta := \theta - \alpha \big(h_\theta(x^{(i)}) - y^{(i)}\big) x^{(i)}
$$

初始化 $\theta = \vec{0}$，那么经过若干步迭代后，$\theta$ 必然是训练样本 $\{x^{(i)}\}$ 的线性组合。Batch Gradient Descent 同样如此。

#### 直觉二：$w$ 与决策边界正交

可以证明：解出的 $w$ 永远**垂直于决策边界**——$w$ 决定边界的方向，$b$ 只决定边界的相对位置。

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec7_w_perpendicular_boundary.webp" alt="w is orthogonal to the boundary" width="40%" loading="lazy" decoding="async" /></div>

既然 $w$ 与边界垂直，而训练样本 $\{x^{(i)}\}$ 张成的空间里必然包含这个方向，$w$ 落在 $\text{span}(\{x^{(i)}\})$ 里也就很自然了。

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec7_w-boudary_eg1.webp" alt="orthogonality example 1" width="40%" loading="lazy" decoding="async" /></div>

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec7_w-boudary_eg2.webp" alt="orthogonality example 2" width="40%" loading="lazy" decoding="async" /></div>

---

## 2. 把目标函数改写成对偶形式

把 $w = \sum_{i=1}^m \alpha_i y^{(i)} x^{(i)}$ 代回原最终优化式。

### 2.1 目标项的改写

$$
\min_{w, b} \frac{1}{2}\|w\|^2
= \min_{\alpha, b} \frac{1}{2}
\Big( \sum_{i=1}^m \alpha_i y^{(i)} x^{(i)} \Big)^T
\Big( \sum_{j=1}^m \alpha_j y^{(j)} x^{(j)} \Big)
$$

注意 $w^T w = \|w\|^2$，展开后：

$$
= \min_{\alpha, b} \frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m
\alpha_i \alpha_j\, y^{(i)} y^{(j)} \color{blue}{\langle x^{(i)}, x^{(j)} \rangle}
$$

> 注：这里用 $\langle \cdot, \cdot \rangle$ 表示两个向量的内积。

### 2.2 约束的改写

$$
y^{(i)} (w^T x^{(i)} + b) \ge 1
\iff y^{(i)} \Big( \sum_{j=1}^m \alpha_j y^{(j)} x^{(j)} \Big)^T x^{(i)} + b \ge 1
\iff y^{(i)} \Big( \sum_{j=1}^m \alpha_j y^{(j)} \color{blue}{\langle x^{(j)}, x^{(i)} \rangle} + b \Big) \ge 1
$$

**关键观察**：无论目标项还是约束，特征向量 $x^{(i)}$ 出现的唯一形式都是**内积** $\langle x^{(i)}, x^{(j)} \rangle$。

所以，只要能找到方法高效地算出这个向量内积，我们可以处理高维甚至无穷维的特征向量。

---

## 3. Dual Optimization Problem

利用凸优化理论（或纯代数地消掉 $b$），上面的问题可以进一步简化为只关于 $\alpha$ 的**对偶优化问题**：

$$
\max_{\alpha} \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m y^{(i)} y^{(j)} \alpha_i \alpha_j \langle x^{(i)}, x^{(j)} \rangle
$$

$$
\text{s.t. } \alpha_i \ge 0,\quad \sum_{i=1}^m y^{(i)} \alpha_i = 0
$$

求解流程：

1. 解出最优的 $\alpha_i$；
2. 再回代得到 $w, b$，做预测时计算：

$$
h_{w, b}(x) = g(w^T x + b) = g\Big( \sum_{i=1}^m \alpha_i y^{(i)} \langle x^{(i)}, x \rangle + b \Big)
$$

---

## 4. Kernel Trick

上一节我们发现，整个 SVM 算法只用到了样本 $x^{(i)}, x^{(j)}$ 之间的内积。下面这一招就是大名鼎鼎的 **Kernel Trick**：

1. **将整个算法改写成内积形式**：把算法里所有出现的 $\langle x^{(i)}, x^{(j)} \rangle$ 显式地写出来（上面已经做到）。

2. **引入特征映射**：令映射 $\Phi: X \to$ 某个（可能非常高维甚至无穷维）的特征空间
   $$
   \begin{pmatrix} x_1 \\ x_2 \end{pmatrix} \to \Phi(x) = \begin{pmatrix} x_1 \\ x_2 \\ x_1 x_2 \\ x_1^2 x_2 \\ \vdots \end{pmatrix}
   $$
3. **定义 Kernel 函数**：
   $$
   K(x, z) = \Phi(x)^T \Phi(z)
   $$
   这个就是 kernel 函数。即使 $\Phi(x)$、$\Phi(z)$ 维度极高，也存在一些技巧可以高效地算出它们的内积。


4. **把算法里所有 $\langle x, z \rangle$ 替换为 $K(x, z)$**：因为只要能高效算出 $K(x, z)$，我们就相当于把整个学习算法"搬"到了高维特征空间里。

   > 直接在高维空间里跑算法计算量会非常大。Kernel Trick 的实质在于：算法已经写成纯内积形式，所以我们完全可以只算 kernel，无需显式构造 $\Phi(x)$。


Kernel Trick 的本质：

> 因为整个算法只用到了样本间的内积，所以我们只要会算 $K(x, z)$，就**完全不需要显式地构造 $\Phi(x)$**——这避免了高维特征空间带来的巨大计算开销。


最终的总结：

> **SVM = Optimal Margin Classifier + Kernel Trick**

极力推荐的可视化视频：[StatQuest — SVM and Kernels](https://www.youtube.com/watch?v=OdlNM96sHio)

> 也就是说，SVM 其实是在高维空间里找到一个**线性**最优分隔面（原数据集在高维空间里面就线性可分了）；但回到原始特征空间来看，这个分隔面就是**非线性**的（非线性决策边界）。Kernel就是用来帮助我们高效找到高维甚至无穷维空间中的线性最优分割边界！

---

## 5. Kernel 的一个简单例子

为了把 kernel trick 的"高效性"讲清楚，看一个具体例子。

### 5.1 平凡的 $\Phi$ 与显式计算

设

$$
x = \begin{pmatrix} x_1 \\ x_2 \\ x_3 \end{pmatrix} \in \mathbb{R}^n \xrightarrow{\Phi} \Phi(x) = \begin{pmatrix} x_1 x_1 \\ x_1 x_2 \\ x_1 x_3 \\ x_2 x_1 \\ x_2 x_2 \\ x_2 x_3 \\ x_3 x_1 \\ x_3 x_2 \\ x_3 x_3 \end{pmatrix} \in \mathbb{R}^{n^2}
$$

（这里 $\Phi(x)$ 实际是 $x x^T$ 拉直后的所有 $n^2$ 个二阶单项式。）

显式计算 $\Phi(x)^T \Phi(z)$ 需要 $O(n^2)$ 时间。

### 5.2 用 Kernel Trick

但如果我们令 $K(x, z) = (x^T z)^2$，有：

$$
K(x, z) = (x^T z)^2 = \Big( \sum_{i=1}^n x_i z_i \Big)^2
= \sum_{i=1}^n \sum_{j=1}^n x_i z_i x_j z_j = \sum_{i,j} (x_i x_j)(z_i z_j)
$$

逐项对照 §5.1 里 $\Phi(x)$ 第 $k$ 个分量和 $\Phi(z)$ 第 $k$ 个分量，正好**完全相等**：

$$
\Phi(x)^T \Phi(z) = (x^T z)^2
$$

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec7_Phi_dot-product.webp" alt="dot product illustration" width="40%" loading="lazy" decoding="async" /></div>

计算 $(x^T z)^2$ 只需要 $O(n)$ 时间（先算 $x^T z$ 这个标量，再平方），相比显式构造 $\Phi$ 省了一个量级。

### 5.3 推广到带常数项的 Polynomial Kernel

稍微改动一下 kernel：

$$
K(x, z) = (x^T z + c)^2, \quad c \in \mathbb{R}
$$

展开后对应的特征向量变为：

$$
\Phi(x) = \begin{pmatrix} x_1 x_1 \\ x_1 x_2 \\ x_1 x_3 \\ x_2 x_1 \\ x_2 x_2 \\ x_2 x_3 \\ x_3 x_1 \\ x_3 x_2 \\ x_3 x_3 \\ \sqrt{2c}\, x_1 \\ \sqrt{2c}\, x_2 \\ \sqrt{2c}\, x_3 \end{pmatrix}
$$

进一步推广：

$$
K(x, z) = (x^T z + c)^d, \quad c, d \in \mathbb{R}
$$

对应的 $\Phi(x)$ 包含**所有阶数 $\le d$ 的单项式**，特征空间维度为 $\dbinom{n + d}{d}$。

---

## 6. 如何判断一个 （设计出来的）Kernel 是否有效？

### 6.1 直觉性原则 (Guiding Principle)

> *"If $x, z$ are 'similar', $K(x, z) = \Phi(x)^T \Phi(z)$ is 'large'"* (the inner product of two similar vectors should be large). And the other way round.
>
> 有了这个指导性的直觉，我们相应的 kernel function 也应该满足当 $x, z$ 接近的时候值大，当不相近的时候值小。

这给设计 kernel 提供了直觉。

### 6.2 一个必要条件

> As the condition is that $K(x,z) = \Phi(x)^T \Phi(z)$. This puts some constraints on our kernel functions that we could choose:

由此 $K$ 必须满足的一些性质，例如直接可推得：

$$
K(x, x) = \Phi(x)^T \Phi(x) \ge 0
$$

### 6.3 Kernel Matrix 必须是半正定

给定任意 $d$ 个点 $\{x^{(1)}, \dots, x^{(d)}\}$，定义 **Kernel Matrix** $K \in \mathbb{R}^{d \times d}$，其中

$$
K_{ij} = K(x^{(i)}, x^{(j)}) = \Phi(x^{(i)})^T \Phi(x^{(j)})
$$

对任意向量 $z$，考察二次型：

$$
z^T K z
= \sum_i \sum_j z_i \Phi(x^{(i)})^T \Phi(x^{(j)})\, z_j
= \sum_k \Big( \sum_i z_i \Phi(x^{(i)})_k \Big)^2 \ge 0
$$

> 因此 $K$（Kernel Matrix）是**半正定的**。

这是一个**充分条件**——下面马上会看到它其实也是必要条件。

### 6.4 Mercer Theorem

> **Mercer's Theorem**：$K$ 是一个有效的 kernel（即存在 $\Phi$ 使得 $K(x, z) = \Phi(x)^T \Phi(z)$）**当且仅当** 对任意 $d$ 个点 $\{x^{(1)}, \dots, x^{(d)}\}$，对应的 Kernel Matrix $K$ 是**半正定的**。

换句话说：**Kernel Matrix 的半正定性就是判定 $K$ 是否能写成 $\Phi(x)^T \Phi(z)$ 的充要条件**。

### 6.5 常用的几个 Kernel

- **Linear Kernel**：$K(x, z) = x^T z$，$\Phi(x) = x$（不做高维映射）
- **Gaussian Kernel**：$K(x, z) = \exp\!\left(-\dfrac{\|x - z\|^2}{2\sigma^2}\right)$，$\Phi(x) \in \mathbb{R}^\infty$
- **Polynomial Kernel**：$K(x, z) = (x^T z)^d$，$\Phi(x) \in \mathbb{R}^{\binom{n+d}{d}}$

---

## 7. 高斯核是有效核的证明

现在我们来使用 Mercer Theorem 验证一下 Gaussian Kernel 确实有效。

> 首先验证直觉：$x, z$ 接近时 $K(x, z)$ 值较大，符合 §6.1 的直觉性原则。

下面证明 $\forall z,\ z^T K z \ge 0$（$K$ 是 Gaussian Kernel 对应的 Kernel Matrix）。

记 $K_{ij} = \exp\!\left(-\dfrac{\|x^{(i)} - x^{(j)}\|^2}{2\sigma^2}\right)$，于是

$$
z^T K z = \sum_i \sum_j z_i \exp\!\left(-\frac{\|x^{(i)} - x^{(j)}\|^2}{2\sigma^2}\right) z_j
$$

**方法：对 $K_{ij}$ 做 Taylor 展开。**

先把指数里的范数那部分展开：

$$
K_{ij}
= \exp\!\left(-\frac{\|x^{(i)}\|^2}{2\sigma^2}\right)
\cdot \exp\!\left(\frac{x^{(i)T} x^{(j)}}{\sigma^2}\right)
\cdot \exp\!\left(-\frac{\|x^{(j)}\|^2}{2\sigma^2}\right)
$$

令

$$
f(x^{(i)}) = \exp\!\left(-\frac{\|x^{(i)}\|^2}{2\sigma^2}\right)
$$

由于 $f(x^{(i)}) > 0$ 且只与 $i$ 有关，可以把两端合并到系数里，定义

$$
\tilde{z}_i = z_i\, f(x^{(i)}), \quad \tilde{z}_j = z_j\, f(x^{(j)})
$$

那么

$$
z^T K z = \sum_i \sum_j \tilde{z}_i \exp\!\left(\frac{x^{(i)T} x^{(j)}}{\sigma^2}\right) \tilde{z}_j
$$

对指数部分做 Taylor 展开（展成无穷级数）：

$$
z^T K z
= \sum_i \sum_j \tilde{z}_i \tilde{z}_j \sum_{k=0}^\infty \frac{1}{k!} \left(\frac{x^{(i)T} x^{(j)}}{\sigma^2}\right)^k
$$

由于所有项都是正数、绝对收敛，可以任意交换求和顺序：

$$
z^T K z
= \sum_{k=0}^\infty \frac{1}{k!\,\sigma^{2k}}
\sum_i \sum_j \tilde{z}_i \tilde{z}_j\, (x^{(i)T} x^{(j)})^k
$$

注意到 $(x^{(i)T} x^{(j)})^k$ 恰好是 $k$ 次多项式核！！！于是我们可以代入多项式核函数对应的 $\Phi_k$，有

$$
(x^{(i)T} x^{(j)})^k = \Phi_k(x^{(i)})^T \Phi_k(x^{(j)})
$$

所以

$$
z^T K z
= \sum_{k=0}^\infty \frac{1}{k!\,\sigma^{2k}}
\Big[ \sum_i \sum_j \tilde{z}_i \tilde{z}_j\, \Phi_k(x^{(i)})^T \Phi_k(x^{(j)}) \Big]
$$

把分别关于 $i, j$ 的求和合并：

$$
z^T K z
= \sum_{k=0}^\infty \frac{1}{k!\,\sigma^{2k}}
\Big[ \Big(\sum_i \tilde{z}_i \Phi_k(x^{(i)})\Big)^T \Big(\sum_j \tilde{z}_j \Phi_k(x^{(j)})\Big) \Big]
$$

注意 $\sum_i \tilde{z}_i \Phi_k(x^{(i)})$ 与 $\sum_j \tilde{z}_j \Phi_k(x^{(j)})$ **只是哑变量不同，本质上完全相同**。令

$$
v_k = \sum_i \tilde{z}_i \Phi_k(x^{(i)})
$$

则右侧就是 $\|v_k\|^2$：

$$
z^T K z = \sum_{k=0}^\infty \frac{1}{k!\,\sigma^{2k}}\, \|v_k\|^2 \ge 0
$$

由 Mercer Theorem 可知，高斯核是一个**有效的核**。

> 注：法二是把里面的高斯函数那一块写成无穷积分形式（此处略）。

> 注意：Kernel Trick 同样可以嫁接到许多其他学习算法上（例如 PCA），但其中最成功、应用最广的，还是 SVM。

---

## 8. 放宽数据集能完全线性可分的假设：使用 Soft Margin SVM

### 8.1 为什么需要放宽？

把数据映射到高维空间后，数据集会变得**更容易分开**。但如果数据本身有噪声，我们不应该追求把每个点都分对——那样会让决策边界变得过于复杂、过拟合。

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec7_not-linearly-separable-data.webp" alt="not linearly separable data boundary" width="40%" loading="lazy" decoding="async" /></div>

### 8.2 $\ell_1$-norm Soft Margin SVM

回顾硬间隔 SVM：

$$
\min \frac{1}{2}\|w\|^2 \qquad \text{s.t. } y^{(i)}(w^T x^{(i)} + b) \ge 1,\ i = 1, \dots, m
$$

约束其实是说："每个样本的 functional margin 必须 $\ge 1$"。

Soft Margin 通过引入 **slack variable** $\xi_i \ge 0$ 把约束放松：

$$
\min \frac{1}{2}\|w\|^2 + \color{red}{C \sum_{i=1}^m \xi_i}
$$

$$
\text{s.t. } y^{(i)}(w^T x^{(i)} + b) \ge 1 - \color{red}{\xi_i},\ i = 1, \dots, m
$$

$$
\color{red}{\xi_i \ge 0}
$$

直观解释：

- 只要 functional margin $\ge 0$，就算分类正确（不再强制要求 $\ge 1$）；
- $\xi_i$ 衡量的是"放松了多少"，出现在目标里作为惩罚项；
- 参数 $C$ 控制**间隔大小**与**分类错误容忍度**之间的权衡。

> 用 Soft Margin 的另一个理由：如果只有一个 outlier （异常值），硬间隔 SVM 会让决策边界被严重拉偏；所以 Soft Margin 更鲁棒。

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec7_soft_SVM_boudary.webp" alt="soft margin SVM boundary" width="40%" loading="lazy" decoding="async" /></div>

### 8.3 对偶形式

把 $w$ 重新表达成 $\alpha$ 的函数（与第 3 节同样的推导），soft margin SVM 的对偶问题为：

$$
\max_\alpha \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m y^{(i)} y^{(j)} \alpha_i \alpha_j \langle x^{(i)}, x^{(j)} \rangle
$$

$$
\text{s.t. } \sum_{i=1}^m y^{(i)} \alpha_i = 0
$$

$$
\color{red}{0 \le \alpha_i \le C},\quad i = 1, \dots, m
$$

与硬间隔 SVM 的对偶相比，soft margin **只多了一个上界约束 $\alpha_i \le C$**。

---

## 9. Kernel Trick 的一个应用：蛋白质序列分类器

### 9.1 问题设定 ($e.g$ : Protein sequence classifier)

蛋白质是氨基酸序列，如果将所有 21 种氨基酸各自编码，那么蛋白质就是一串编码序列。

这个时候我们应该思考，拿到一串编码序列作为输入后，如何将这个输入 $x$ 投影至 $\Phi(x)$？我们应该构造怎样的特征空间？

$$
\Phi(x) = ?
$$

一种构造特征向量的方法是，**列出所有 4 种氨基酸的组合**（不妨假设氨基酸一共 26 种使用 $A \sim Z$ 来编码表示）：

$$
\begin{bmatrix} A, A, A, A \\ A, A, A, B \\ A, A, A, C \\ \vdots \\ Z, Z, Z, Z \end{bmatrix}
$$

然后，根据这些序列在氨基酸中出现的次数来构建 $\Phi(x)$（**这里相当于是创新性构造出核函数**）：

$$
\begin{bmatrix} A, A, A, A \\ A, A, A, B \\ A, A, A, C \\ \vdots \\ Z, Z, Z, Z \end{bmatrix} \to \begin{bmatrix} \text{count of } AAAA \\ \text{count of } AAAB \\ \text{count of } AAAC \\ \vdots \\ \text{count of } ZZZZ \end{bmatrix} = \Phi(x)
$$

$$
\Phi(x) \in \mathbb{R}^{21^4}
$$

> Therefore, SVM allows us to invent kernel functions to measure the similarity.
>
> （这里可以回顾一下 kernel trick 的实质：如果我们手动去构建那个 $21^4$ 维的向量，然后计算这样两个向量 $x, z$ 之间的相似度（此处相似度就是求内积），那么计算量会多到爆炸！核函数其实也就是给我提供了一种不需要显式计算这种高维向量内积的捷径。）
>
> 在上述例子中，如果我们先手动去构造输入 $x, z$ 的对应 $21^4$ 维向量，再作点积，那么计算量过大。但如果换一种思路，我们直接去计数 $x, z$ 里面的"相同 4 元序列"出现的次数，$x, z$ 相同序列出现次数再取对应乘积和，那么问题就得到简化。

### 9.2 一个小例子：$\Sigma = \{A, B, C, D\}$，取 2-mer（就是每次只考虑两个连续氨基酸）

特征空间维度 $D = 4^2 = 16$，按字典序排列 16 维特征空间：

$$
\begin{bmatrix} AA \\ AB \\ AC \\ \vdots \\ DC \\ DD \end{bmatrix}
$$

**输入序列**：

$$
X = \{ABABD\}, \quad Y = \{ABACD\}
$$

**(i) 方法一：显式特征映射（笨办法）**

第一步：构建 16 维频次向量 $\Phi(x)$

$$
\Phi(X) = (0, 2, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0)^T
$$

$$
\Phi(Y) = (0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0)^T
$$

点积结果：

$$
\Phi(X)^T \Phi(Y) = 0\cdot 0 + 2\cdot 1 + 0\cdot 1 + 0\cdot 0 + 1\cdot 1 + 0 + 0 + 1\cdot 0 + \dots = 2 + 1 + 0 + 0 + 0 = \mathbf{3}
$$

**(2). 方法二：使用核技巧**

无需显式构造 16 维向量，直接计算交集片段的频次乘积和：

$$
\begin{aligned}
K(X, Y)
&= \sum_m \text{count}_X(m) \cdot \text{count}_Y(m) \\
&= \underbrace{(2 \cdot 1)}_{AB} + \underbrace{(1 \cdot 1)}_{BA} + \underbrace{(1 \cdot 0)}_{BD} + \underbrace{(0 \cdot 1)}_{AC} + \underbrace{(0 \cdot 1)}_{CD} \\
&= 2 + 1 + 0 + 0 + 0 = \mathbf{3}
\end{aligned}
$$

> **结论**：显式映射的内积 $\Phi(X)^T \Phi(Y) = 3$ 与 kernel $K(X, Y) = 3$ 在数学上完全等价。Kernel Trick 把复杂度从高维特征空间的 $O(|\Sigma|^k)$ 降到仅与实际出现的片段数相关的 $O(n)$。

---

## 参考资料

- [CS229 Lecture Notes 3](https://cs229.stanford.edu/notes2021fall/cs229-notes3.pdf) — SVM 的主体部分（Optimal Margin Classifier + Kernels）
- [CS229 2018](https://www.bilibili.com/video/BV1b4anzMEUv) — 看 B 站上 Andrew Ng 的讲课视频 CS229 Lec 7
- [Bishop, Pattern Recognition and Machine Learning, Ch.6](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) — Chapter 6 : Kernel Methods（Mercer Theorem 与 RKHS 的更深入展开）
- [StatQuest — SVM and Kernels](https://www.youtube.com/watch?v=OdlNM96sHio) — 强烈推荐的 SVM 可视化简短理解，把 Kernel Trick 可视化展示能直观快速深刻理解！