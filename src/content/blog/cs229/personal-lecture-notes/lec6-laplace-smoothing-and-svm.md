---
title: "CS229 : Lec 6 — Laplace 平滑与 SVM 入门"
description: CS229 Lecture 6 学习笔记，先补充 Naive Bayes 的 Laplace 平滑问题，并介绍多项式事件模型（Multinomial Event Model）；最后进入支持向量机（SVM），推导最优间隔分类器（Optimal Margin Classifier）。
pubDate: 2026-08-24
series: cs229
subSeries: personal-lecture-notes
order: 7
categories:
  - CS229
  - Naive Bayes
  - Laplace Smoothing
  - SVM
  - Optimal Margin Classifier
---

> **TL;DR**:
> - **Laplace 平滑**：解决 Naive Bayes 中由于训练集里从没出现过某词，导致MLE 直接给出概率 0，进而整个 Score 被乘成 0"的崩溃问题；做法是分子 +1、分母 +类别数 $k$（或词汇表大小 $|V|$），把零概率"软化"成一个很小的正数。
> - **Multinomial Event Model**：把邮件文本表示成"词索引序列"（不定长），参数是"词 $k$ 在位置 $j$ 出现的概率"。这种文本表示方式记录每个单词在邮件中出现的**次数**，而 Multivariate Bernoulli 只能记录这个单词"出现 / 没出现"。
> - **SVM 的核心思想**：从几何直观出发，找一个"离两边数据都最远"的最优分隔超平面；为此引入**函数间隔** $\hat\gamma$、**几何间隔** $\gamma$，最后把问题转化为一个**凸二次规划**。
> - **符号约定切换**：SVM 不再用 $y \in \{0,1\}$，而是 $y \in \{-1,+1\}$，并把 $h_\theta(x) = g(\theta^T x)$ 改写成 $h_{w,b}(x) = g(w^T x + b)$——后者更方便把截距 $b$ 和法向量 $w$ 分开讨论。
> - **Kernels 预告**：用核函数可以构造无穷维特征空间，从而让 SVM 在高维空间学到一个**非线性**的决策边界（细节将留给下节Lecture讲）。

## 引子

Lec 5 结尾我们留了一个问题：**Naive Bayes 在某个词从未在训练集某类邮件中出现过时，整个 Score 会被乘成 0**——本节先用一个简单技巧（**Laplace 平滑**）把它补完，并介绍一种能记录"词频"的新文本表示：**Multinomial Event Model**。

后半节我们离开生成式学习算法的范畴，进入一种新的判别式算法：**支持向量机（Support Vector Machine, SVM）**。它从**几何间隔最大化**出发，能学到比 Logistic 回归更强的分类边界。

---

## 1. Naive Bayes 续讲

### 1.1 Laplace 平滑

**问题回顾**：上节课 Naive Bayes 的 MLE 结论中，我们得到最后拟合好的参数值为：

$$
\phi_y = \frac{\sum_{i=1}^m \mathbb{1}\{y^{(i)} = 1\}}{m}, \qquad \phi_{j \mid y=1} = \frac{\sum_{i=1}^m \mathbb{1}\{x_j^{(i)} = 1,\ y^{(i)} = 1\}}{\sum_{i=1}^m \mathbb{1}\{y^{(i)} = 1\}}
$$

预测时则计算这个条件概率

$$
p(y = 1 \mid x) = \frac{p(x \mid y = 1)\, p(y = 1)}{p(x \mid y = 1)\, p(y = 1) + p(x \mid y = 0)\, p(y = 0)}
$$

设想一个特别问题：训练集里第 1000 个词 $x_{1000}$ 从未出现过——也就是说

$$
\phi_{1000 \mid y=1} = p(x_{1000} = 1 \mid y = 1) = 0, \qquad \phi_{1000 \mid y=0} = p(x_{1000} = 1 \mid y = 0) = 0
$$

代回到 Naive Bayes 公式里：

$$
p(y = 1 \mid x) = \frac{\color{red}{p(x \mid y = 1)}\, p(y = 1)}{\color{red}{p(x \mid y = 1)}\, p(y = 1) + \color{blue}{p(x \mid y = 0)}\, p(y = 0)}
$$

而条件概率在 Naive Bayes 假设下是连乘式：

$$
p(x \mid y = 1) = \prod_{i=1}^n p(x_i \mid y = 1)
$$

因为 $p(x_{1000} \mid y = 1) = 0$，**红色部分整体为 0**；同理**蓝色部分也整体为 0**。于是

$$
p(y = 1 \mid x) = \frac{0}{0 + 0}
$$

——分类器将直接崩溃，给不出任何有意义的结果。

> 而且，从统计意义上说，**仅因为训练集没见过就把概率估计为 0 也是非常粗暴的**——真实分布里这个词完全可能以小概率出现。

**Laplace 平滑** 的核心思想：在分子、分母上同时加一个常数，把"零概率"软化成一个很小的正数。

#### Laplace 平滑的一般形式

假设某离散随机变量取值于 $\{1, \dots, k\}$。MLE 给出的参数估计是：

$$
\widehat{p(x = j)}_{\text{MLE}} = \frac{\sum_{i=1}^m \mathbb{1}\{x^{(i)} = j\}}{m}
$$

加入 Laplace 平滑后参数估计变为：

$$
\widehat{p(x = j)}_{\text{Laplace}} = \frac{\sum_{i=1}^m \mathbb{1}\{x^{(i)} = j\} \color{red}{+1}}{m \color{red}{+k}}
$$

> 直观解释：想象我们在每一种取值上都**预先观察到了 1 次**，所以分子加 $1$，分母则加上类别数 $k$ 。

可以用所有情况概率总和为1来验证这仍然是一个合法的概率分布：

$$
\sum_{j=1}^k \widehat{p(x = j)}_{\text{Laplace}} = \frac{m + k}{m + k} = 1
$$

#### 将 Laplace Smoothing 用于 Naive Bayes 分类器

回到邮件分类问题（$y$ 与 $x_j$ 都是二值的，所以 $k = 2$），加入 Laplace 平滑后参数的估计形式如下：

$$
\phi_{j \mid y = 1} = \frac{\sum_{i=1}^m \mathbb{1}\{x_j^{(i)} = 1,\ y^{(i)} = 1\} \color{red}{+1}}{\sum_{i=1}^m \mathbb{1}\{y^{(i)} = 1\} \color{red}{+2}}
$$

$$
\phi_{j \mid y = 0} = \frac{\sum_{i=1}^m \mathbb{1}\{x_j^{(i)} = 1,\ y^{(i)} = 0\} \color{red}{+1}}{\sum_{i=1}^m \mathbb{1}\{y^{(i)} = 0\} \color{red}{+2}}
$$

> 也就是分子 $\mathbf{+1}$，分母 $\mathbf{+2}$（因为 $x_j \in \{0, 1\}$，所以分母上对应要加的常数是 2 而不是 1）。

这样既避免了概率为 0 导致的崩溃问题，又保证了概率的归一化。

### 1.2 Multivariate Bernoulli 表示（回顾）

Lec 5 中我们用**多重伯努利表示**（Multivariate Bernoulli representation）来描述一封邮件：把邮件表示成一个 $n$ 维二值向量（$n = |V|$ = 词汇表大小）：

$$
x = \begin{bmatrix} x_1 \\ \vdots \\ x_i \\ \vdots \\ x_n \end{bmatrix}, \quad x_i \in \{0, 1\}
$$

其中 $x_i = 1$ 表示第 $i$ 个词出现在邮件里，等于0则说明未出现在邮件中。

> 这个表示法有一个明显的缺点：**它把每个词都简化成"出现 / 没出现"**，丢失了"这个词到底出现了几次"的信息。

下面要介绍的 **Multinomial Event Model** 就能解决这个问题。

### 1.3 Multinomial Event Model（多项式事件模型）

**核心思路**：把第 $i$ 封邮件编码成一个**长度等于邮件长度的整数序列**——依次把邮件里出现的每个词替换成它在词汇表中的索引。

$$
x = \begin{bmatrix} 1200 \\ \vdots \\ 6300 \\ \vdots \\ 400 \end{bmatrix} \in \mathbb{R}^{n_i}, \quad x_j \in \{1, \dots, n\}
$$

其中 $n_i$ = 第 $i$ 封邮件的**单词数**，$n = |V|$ = 词汇表大小。

> **与 Naive Bayes 的对比**：上节课 Naive Bayes 中的 $n_i$ 是**词汇表大小**（一个固定维度），这里的 $n_i$ 是**邮件长度**（因邮件而异），而且 $x_j$ 是词索引（多值），不再是 0/1（伯努利）。

### 1.4 生成式建模与 MLE

我们仍然要建模 $p(x, y)$，使用同样的条件独立假设：

$$
p(x, y) = p(x \mid y)\, p(y) \overset{\text{assume}}{=} \prod_{j=1}^{n_i} p(x_j \mid y) \cdot p(y)
$$

参数与之前 Naive Bayes 中用到的参数一样：

$$
\phi_y = p(y = 1), \qquad \phi_{k \mid y = 0} = p(x_j = k \mid y = 0), \qquad \phi_{k \mid y = 1} = p(x_j = k \mid y = 1)
$$

> 第二个参数**只有 $k$，没有 $j$**：我们假设"词 $k$ 在邮件的任意位置出现的概率都相同"——也就是说，词的概率与它出现的位置 $j$ 无关。

给定训练集 $\{(x^{(i)}, y^{(i)})\}_{i=1}^m$，其中 $x^{(i)} = (x_1^{(i)}, x_2^{(i)}, \dots, x_{n_i}^{(i)})$（$n_i$ = 第 $i$ 个训练样本的单词数），那么这个训练集的联合似然函数如下所示：

$$
\begin{aligned}
\mathcal{L}(\phi_y, \phi_{k \mid y=0}, \phi_{k \mid y=1})
&= \prod_{i=1}^m p(x^{(i)}, y^{(i)}) \\
&= \prod_{i=1}^m \left( \prod_{j=1}^{n_i} p(x_j^{(i)} \mid y;\ \phi_{k \mid y=0}, \phi_{k \mid y=1}) \right) p(y^{(i)};\ \phi_y)
\end{aligned}
$$

> 双层连乘理解：
>
> - **外层 $\prod_{i=1}^m$**：遍历训练集中所有 $m$ 个邮件样本
> - **内层 $\prod_{j=1}^{n_i}$**：对当前第 $i$ 封邮件里的每个位置 $j$ 上的词做连乘

对联合似然函数做 MLE，得到最后的参数估计：

$$
\phi_{k \mid y = 1} = \frac{\sum_{i=1}^m \mathbb{1}\{y^{(i)} = 1\} \left( \sum_{j=1}^{n_i} \mathbb{1}\{x_j^{(i)} = k\} \right)}{\sum_{i=1}^m \mathbb{1}\{y^{(i)} = 1\} \cdot n_i}
$$

$$
\phi_{k \mid y = 0} = \frac{\sum_{i=1}^m \mathbb{1}\{y^{(i)} = 0\} \left( \sum_{j=1}^{n_i} \mathbb{1}\{x_j^{(i)} = k\} \right)}{\sum_{i=1}^m \mathbb{1}\{y^{(i)} = 0\} \cdot n_i}
$$

$$
\phi_y = \frac{\sum_{i=1}^m \mathbb{1}\{y^{(i)} = 1\}}{m}
$$

> 以 $\phi_{k \mid y = 0}$ 为例解读一下直观的计算过程：把所有 $y = 0$ 的邮件挑出来，查看里面所有的单词（不再区分是哪封邮件、哪个位置），统计词 $k$ 在其中的出现占比——得到的值就是"词 $k$ 出现在 $y = 0$ 邮件中任意位置"的概率大小估计。

### 1.5 多项式模型下的 Laplace 平滑

对上面得到的 $\phi_{k \mid y = 0}$ 和 $\phi_{k \mid y = 1}$ 再做 Laplace 平滑——分子 $\mathbf{+1}$，分母 $\mathbf{+|V|}$：

$$
\phi_{k \mid y = 1} = \frac{\sum_{i=1}^m \mathbb{1}\{y^{(i)} = 1\} \left( \sum_{j=1}^{n_i} \mathbb{1}\{x_j^{(i)} = k\} \right) \color{red}{+1}}{\sum_{i=1}^m \mathbb{1}\{y^{(i)} = 1\} \cdot n_i \color{red}{+|V|}}
$$

$$
\phi_{k \mid y = 0} = \frac{\sum_{i=1}^m \mathbb{1}\{y^{(i)} = 0\} \left( \sum_{j=1}^{n_i} \mathbb{1}\{x_j^{(i)} = k\} \right) \color{red}{+1}}{\sum_{i=1}^m \mathbb{1}\{y^{(i)} = 0\} \cdot n_i \color{red}{+|V|}}
$$

> 为什么分母是 $+ |V|$ 而不是 $+ 2$？因为这里 $x_j$ 取值范围是 $\{1, \dots, |V|\}$，一共有 $|V|$ 个不同的词。Laplace 平滑要求"每个取值都被预先观察到 1 次"，所以分母要补 $|V|$。
>
> 归一性验证：$\sum_{k=1}^{|V|} \phi_{k \mid y = 1} = 1$。

#### 补充：处理未登录词（OOV）

如果测试时碰到了**训练集词汇表里根本没有的词**怎么办？两种常见处理方式：

1. **直接丢弃**：忽略所有不在词汇表 $V$ 里的词，不参与概率计算。
2. **映射到 `UNK`**：把所有"低频 / 未登录"词都映射到一个特殊标记 `UNK`（unknown），统一处理。

#### Naive Bayes 的优势总结

最后再回顾一下 Naive Bayes（包括 Multinomial Event Model）相对其他方法的优势：

- **计算高效**：参数更新只是简单的统计计数，不需要梯度下降等迭代算法。
- **实现简单**：几行代码就能搭出一个 baseline 分类器。

---

## 2. 支持向量机 (Support Vector Machine, SVM)

### 2.1 引子：为什么需要非线性边界？

一个简单的例子：给出下面的二分类数据集，要求给出分类的决策边界：

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs229/Lec6_non-linear-classification.webp" alt="non-linear classification" width="250" loading="lazy" decoding="async" /></div>

但是普通的 Logistic 回归只能给出**线性**决策边界——因为它本质上是在拟合 $\theta^T x = 0$ 这一个超平面。

一个绕过这个限制的小技巧是**手工构造高维特征**。例如把特征向量从

$$
x = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
$$

升级到

$$
x = \begin{bmatrix} x_1 \\ x_2 \\ x_1^2 + x_2^2 \end{bmatrix}
$$

则决策边界变为

$$
\theta^T x = \theta_1 x_1 + \theta_2 x_2 + \theta_3 (x_1^2 + x_2^2) = 0
$$

——这是一个**非线性**决策边界（圆形）。

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs229/Lec6_non-linear-boudary.webp" alt="non-linear decision boundary" width="350" loading="lazy" decoding="async" /></div>

> **问题**：手工挑选高维的特征非常困难——我们不知道到底哪些特征组合能给出合适的边界。**SVM 的关键能力**就是它能自动从原始特征 $x_1, x_2, \dots$ 出发，**映射到高维特征空间**，并在这个高维空间中学到一个**线性的分类器**，等价于在原始空间中产生**非线性**的边界。这就是后面要讲的**核函数（Kernels）** 的核心思想（见下一个Lecture）。

### 2.2 函数间隔 vs 几何间隔 初步引入

#### 1）函数间隔 (Functional margin)

回顾一下之前的 Logistic 回归方法：

$$
h_\theta(x) = g(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}
$$
在预测时候：

- 若 $\theta^T x \geq 0$，预测 $y = 1$；
- 若 $\theta^T x < 0$，预测 $y = 0$。

分类的"信心"：

- 当 $y^{(i)} = 1$ （真实标签为1） 时，我们希望计算时的 $\theta^T x \gg 0$ ；
- 当 $y^{(i)} = 0$ （真实标签为0） 时，我们希望计算时的 $\theta^T x \ll 0$ 。

> **问题**：之前的 $y \in \{0, 1\}$ 标签不方便统一成一个优化目标——因为当 $y = 0$ 时 $y \cdot (\theta^T x) \equiv 0$，无法与我们在 $y=1$ 时的目标统一为一个函数的极值化 。这正是后面要切换 SVM 符号约定的根本原因。

#### 2）几何间隔 (Geometric margin)

假设数据集是**线性可分**的。（保证每一个数据点都能被正确线性分隔到对应类别）

SVM 要做的就是在低维空间中找一个**最优分隔线**（optimal margin classifier）——目标是让分隔线**离两边数据都尽可能远**。分隔线与最近数据点之间的距离，就叫**几何间隔** (geometric margin)。

例如对下面的数据集画出最优分割线时，虽然蓝线和绿线都能将各个数据点正确划分到对应类别，但是蓝线要更优一些：

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs229/Lec6_geometric-margin-comparison.webp" alt="maximizing geometric margin" width="360" loading="lazy" decoding="async" /></div>

现在如果有一条**线性分类器** $w^T x + b = 0$。考虑一个被正确分类的数据点 $(x^{(i)}, y^{(i)})$，我们把这个训练样本的**几何间隔**定义为**数据点到决策边界之间的欧氏距离**，如下图所示：

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs229/Lec6_geometric_margin.webp" alt="geometric margin illustration" width="360" loading="lazy" decoding="async" /></div>

### 2.3 SVM 的符号约定

由于上述函数间隔的不便，SVM 采用新的符号约定：

① **标签**：$y \in \{-1, +1\}$（不再是 $\{0, 1\}$）

② **输出**：$h \in \{-1, +1\}$（不再是概率，而是**硬决策**）

$$
g(z) = \begin{cases} +1, & z \geq 0 \\ -1, & \text{otherwise} \end{cases}
$$

> **注**：这是一个**硬阶跃**函数——这与"Optimal Margin Classifier"问题的特殊结构密切相关（后文会回扣这一点）。

### 2.4 参数形式

| Logistic 回归                        | SVM                                        |
| ---------------------------------- | ------------------------------------------ |
| 参数：$\theta$                        | 参数：$w, b$（$b$ 是单独的偏置）                      |
| $x \in \mathbb{R}^{n+1}$，$x_0 = 1$ | $x \in \mathbb{R}^n$（**不需要再约定 $x_0 = 1$**） |
| $h_\theta(x) = g(\theta^T x)$      | $h_{w,b}(x) = g(w^T x + b)$                |

直觉上两种写法是等价的——把 $\theta$ 拆成 $b$（对应 $\theta_0$）和 $w$（对应 $\theta_1, \dots, \theta_n$）：

$$
\theta = \begin{bmatrix} \theta_0 \\ \theta_1 \\ \vdots \\ \theta_n \end{bmatrix}, \quad b = \theta_0, \quad w = \begin{bmatrix} \theta_1 \\ \vdots \\ \theta_n \end{bmatrix}
$$

> **好处**：把偏置 $b$ 和法向量 $w$ 解耦后，公式推导更清爽，几何解释更直接。

### 2.5 对于 函数、几何间隔 的正式定义

#### 1）函数间隔

**① 单个样本**的函数间隔（对超平面 $(w, b)$ 关于样本 $(x^{(i)}, y^{(i)})$）：

$$
\hat\gamma^{(i)} = y^{(i)} (w^T x^{(i)} + b)
$$

直观理解：$w^T x + b$ 本身定义了"分隔正负例的超平面"。对单个样本：

- 若 $y^{(i)} = +1$，希望 $w^T x + b \gg 0$；
- 若 $y^{(i)} = -1$，希望 $w^T x + b \ll 0$。

两种情况合并后就是希望 $\hat\gamma^{(i)} \gg 0$。而且 **$\hat\gamma^{(i)} > 0$ 等价于分类正确**，这样我们两种类别的目标统一为一个目标函数的最大化问题。

**② 整个训练集**的函数间隔（也就是取最差的那个样本）：

$$
\hat\gamma = \min_{i = 1, \dots, m} \hat\gamma^{(i)}
$$

> 也就是在问："你对训练集中**最差**的那个样本有多确信？"

> **重要陷阱**：$\hat\gamma$ 可以被任意放大——只要把 $(w, b)$ 同时乘以一个常数 $k$，$\hat\gamma$ 就跟着乘以 $k$。一个常见的归一化方法是强制要求 $\|w\| = 1$（或者等价地把参数替换为 $(w/\|w\|, b/\|w\|)$）。

#### 2）几何间隔

**① 单个样本**的几何间隔（$(w, b)$ 关于 $(x^{(i)}, y^{(i)})$）：

$$
\gamma^{(i)} = \frac{y^{(i)} (w^T x^{(i)} + b)}{\|w\|}
$$

直观理解：在分类正确时，这就是数据点到决策边界的**欧氏距离**。


**② 整个训练集**的几何间隔：

$$
\gamma = \min_{i = 1, \dots, m} \gamma^{(i)}
$$


**几何间隔 vs 函数间隔**的关系：

$$
\text{geometric} = \frac{\text{functional}}{\|w\|}
$$

> 也就是说，函数间隔除以 $\|w\|$ 才得到真正的"几何距离"。

### 2.6 最优间隔分类器 (Optimal Margin Classifier)

**目标**：找 $w, b$ 使得几何间隔 $\gamma$ 最大：

$$
\begin{aligned}
\max_{\gamma, w, b}\ & \gamma \\
\text{s.t.}\ & \frac{y^{(i)} (w^T x^{(i)} + b)}{\|w\|} \geq \gamma, \quad i = 1, \dots, m
\end{aligned}
$$

> 最大化 $\gamma$，同时保证每个样本的几何间隔都至少是 $\gamma$ （下面那行是约束条件）。

> 不过这里有个关键问题：**这个原始问题其实是一个**非凸（non-convex）**优化问题**，直接用梯度下降之类的数值方法很难求解出最优的 $w, b$。所以我们需要做一步转化，把它变成一个凸问题。

#### 转化：用函数间隔替换

由 $\gamma = \hat\gamma / \|w\|$，目标等价于

$$
\max_{w, b}\ \frac{\hat\gamma}{\|w\|}
$$

> 其中 $\hat\gamma = \min_i \hat\gamma^{(i)}$ 是训练集上最小的函数间隔。

#### 关键观察：缩放不变性

我们注意到：如果把 $(w, b)$ 整体乘以常数 $k$：

- **决策超平面 $w^T x + b = 0$ 不变**（因为 $k \cdot 0 = 0$）；
- **几何间隔 $\gamma$ 不变**（分子分母同时缩放 $k$）；
- **函数间隔 $\hat\gamma$ 会同步缩放 $k$ 倍**。

既然缩放不会改变 $\gamma$，我们可以**人为固定** $\hat\gamma = 1$ （因为$\hat\gamma$ 可以随意缩放不影响$\gamma$大小）。这样目标函数就变成

$$
\gamma = \frac{1}{\|w\|}
$$

也就等价于要去最小化 $\|w\|$，即等价于去最小化 $\frac{1}{2} \|w\|^2$（加 $\frac{1}{2}$ 是为了求导方便，并且最优解的位置不会改变）。

#### 约束条件的推导

现在把目光转向**约束条件**。最开始的约束是

$$
\frac{y^{(i)} (w^T x^{(i)} + b)}{\|w\|} \geq \gamma, \quad i = 1, \dots, m
$$

两边同时乘以 $\|w\|$，简单变形得到

$$
y^{(i)} (w^T x^{(i)} + b) \geq \gamma\, \|w\|, \quad i = 1, \dots, m
$$

接下来关键的一步：因为我们已经令 $\hat\gamma = 1$，于是

$$
\gamma\, \|w\| = \frac{\hat\gamma}{\|w\|} \cdot \|w\| = \hat\gamma = 1
$$

代入上式右端，约束条件最终化为

$$
y^{(i)} (w^T x^{(i)} + b) \geq 1, \quad i = 1, \dots, m
$$

#### 最终版本

$$
\begin{aligned}
\min_{w, b}\ & \frac{1}{2} \|w\|^2 \\
\text{s.t.}\ & y^{(i)} (w^T x^{(i)} + b) \geq 1, \quad i = 1, \dots, m
\end{aligned}
$$

> **这是一个凸二次规划问题**（convex QP）——可以直接用现成的优化求解器高效地解出 $w, b$。

#### 前提基本假设

> **假设**：训练集是**线性可分**的——每一个样本都必须能被正确分类。最优间隔分类器就是 SVM 的基本构件（building block）。

### 2.7 核函数 (Kernels) 预告

> *本节详细讨论将留到下一讲*

SVM 的真正威力在于配合**核函数**之后，可以把特征 $x_1, x_2, \dots$ 映射到（甚至是无穷维的）高维特征空间，再在高维空间中学一个**线性分类器**——这等价于在原始空间中给出一个**高度非线性**的决策边界。

---

## 参考资料

- [CS229 Lecture Notes 2](https://cs229.stanford.edu/notes2021fall/cs229-notes2.pdf) — 看讲义最后的 Laplace 平滑与 Multinomial Event Model 部分
- [CS229 2018](https://www.bilibili.com/video/BV1b4anzMEUv) — 看B 站上 Andrew Ng 的讲课视频CS229 Lec6
- [CS229 Lecture Notes 3](https://cs229.stanford.edu/notes2021fall/cs229-notes3.pdf) — SVM + Kernels （本文为前面部分）
- [Bishop, Pattern Recognition and Machine Learning, Ch.6](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) — Chapter 6 : Kernel Methods
