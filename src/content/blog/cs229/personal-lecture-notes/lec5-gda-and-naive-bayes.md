---
title: "CS229 : Lec 5 — GDA 与朴素贝叶斯"
description: CS229 Lecture 5 学习笔记，主要介绍两种生成式学习算法：GDA（高斯判别分析）以及 Naive Bayes （朴素贝叶斯）
pubDate: 2026-08-09
series: cs229
subSeries: personal-lecture-notes
order: 5
categories:
  - CS229
  - GDA
  - Naive Bayes
  - 生成学习算法
---

> **TL;DR**:
> - **判别式 vs 生成式 学习算法**：之前Lec讲到的算法（线性回归、Logistic 回归）都是判别式，直接对 $p(y \mid x; \theta)$ 建模；生成式则对 $p(x \mid y)$ 和 $p(y)$ 分别建模，再用贝叶斯公式反推 $p(y \mid x)$
> - **GDA 假设**：$x \mid y \sim \mathcal{N}(\mu, \Sigma)$，$y \sim \text{Bernoulli}(\phi)$
> - **GDA 的 MLE 闭式解**：用统计方法直接算出 $\phi$、$\mu_0$、$\mu_1$、$\Sigma$ （不是用迭代更新方法！）
> - **GDA 的决策边界**：从 GDA 假设可以推出 $p(y=1 \mid x)$ 是 logistic 函数（sigmoid），因此 GDA 的边界是线性的
> - **GDA vs Logistic 回归**：GDA 假设更强，若假设正确则更高效；若假设错误则 Logistic 回归更鲁棒
> - **Naive Bayes**：对高维稀疏特征（如词袋），加入"$x_i$ 在给定 $y$ 时条件独立"假设，把指数级参数个数压成线性级个数；同样使用 MLE 拟合参数

## 引子

Lec 1–4 讲的算法（线性回归、Logistic 回归、GLM、Softmax）都属于**判别式学习算法**（discriminative learning algorithms）——它们都对 $p(y \mid x; \theta)$ 直接建模，给定输入 $x$ 就输出 $y$ 的分布，本节转入一类完全不同的思路：**生成式学习算法**（generative learning algorithms）。

我们先讲 GDA（高斯判别分析），它假设 $x \mid y$ 服从多元高斯分布；再讲 Naive Bayes，它把"高维稀疏二值特征下的指数级参数"用一个朴素的条件独立假设压成线性级个数。

## 1. 判别式 vs 生成式

到目前为止，我们讲过的学习算法都是判别式：

- **判别式（Discriminative）**：直接学 *$p(y \mid x)$*，即给定 $x$ 下 $y$ 的条件分布（以 $\theta$ 为参数）。hypothesis $h_\theta(x)$ 直接对应 $x \to y$ 的映射。

设想一个二分类问题：要根据动物的某个特征分辨它是大象（$y = 1$）还是小狗（$y = 0$）。用 Logistic 回归或感知机可以找到一条直线作为决策边界。给定一个新动物，程序检查它的特征落到了直线的哪一侧，就给出对应预测。这一路就是在做 MLE。

- **生成式（Generative）**：不直接学 $p(y \mid x)$，而是分别学 *$p(x \mid y)$*（给定类别 $y$，特征 $x$ 长什么样）和 *$p(y)$*（类别先验）。

> "生成式"这个名字来自于：一旦学到 $p(x \mid y)$，我们就**知道每个类别下数据是怎么生成的**，进而可以从这个分布中"生成"出新的样本。

测试时，把新样本分别喂给两个类别的模型 $p(x \mid y=0)$ 与 $p(x \mid y=1)$，看哪个匹配得更像，就把它判给哪一类。

两类用贝叶斯公式联系起来：

$$
p(y = 1 \mid x) = \frac{p(x \mid y = 1)\, p(y = 1)}{p(x)}
$$

$$
p(x) = p(x \mid y = 1)\, p(y = 1) + p(x \mid y = 0)\, p(y = 0)
$$

合并得到：

$$
p(y = 1 \mid x) = \frac{p(x \mid y = 1)\, p(y = 1)}{p(x \mid y = 1)\, p(y = 1) + p(x \mid y = 0)\, p(y = 0)}
$$

---

## 2. GDA 模型与假设

### 2.1 多元高斯分布

假设 $x \in \mathbb{R}^n$（连续值），并且我们不再约定 $x_0 = 1$（在 Lec 1 的线性模型里，$x_0 = 1$ 是为了把截距 $\theta_0$ 通过 $\theta_0 1$ 一起塞进 $\theta^T x$；这里 GDA 直接用不带截距的特征向量）。

**关键假设**：$p(x \mid y)$ 服从**高斯分布**。

由于 $x$ 是 $n$ 维向量，所以它应该服从**多元高斯分布**：

$$
x \sim \mathcal{N}(\mu, \Sigma)
$$

其中：

- $\mu$ 是均值向量（$\mu \in \mathbb{R}^n$）
- $\Sigma$ 是协方差矩阵（$\Sigma \in \mathbb{R}^{n \times n}$）（类似于一维数据的方差）

这两个参数共同决定了多元高斯分布的形状——它们控制了分布的中心位置和散布方式。

多元高斯分布的概率密度公式（PDF）：

$$
p(x; \mu, \Sigma) = \frac{1}{(2\pi)^{n/2} \, |\Sigma|^{1/2}} \exp\!\left( - \frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu) \right)
$$

下面几张图建立参数变化对于多元高斯分布影响的直观理解：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec5_multivariate_Guassian_bumps1.webp" alt="Multivariate Gaussian bumps1" width="350" loading="lazy" decoding="async" /></div>

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec5_multivariate_Guassian_bumps2.webp" alt="Multivariate Gaussian bumps2" width="300" loading="lazy" decoding="async" /></div>

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec5_multivariate_Guassian_contours.webp" alt="Multivariate Gaussian contours" width="450" loading="lazy" decoding="async" /></div>

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec5_multivariate_Guassian_bumps3.webp" alt="Multivariate Gaussian bumps3" width="300" loading="lazy" decoding="async" /></div>

### 2.2 GDA 模型设定

两类用各自的均值，但**共享同一个协方差矩阵** $\Sigma$（这是标准设定，少数情况下也允许 $\Sigma_0 \neq \Sigma_1$，但通常不这样做，这样事实上会导致最后的决策边界从线性变成非线性的（二次型））：

$$
p(x \mid y = 0) = \frac{1}{(2\pi)^{n/2} \, |\Sigma|^{1/2}} \exp\!\left( - \frac{1}{2} (x - \mu_0)^T \Sigma^{-1} (x - \mu_0) \right)
$$

$$
p(x \mid y = 1) = \frac{1}{(2\pi)^{n/2} \, |\Sigma|^{1/2}} \exp\!\left( - \frac{1}{2} (x - \mu_1)^T \Sigma^{-1} (x - \mu_1) \right)
$$

同时还要对 $y$ 建模（其实就是看成满足一个伯努利分布）：

$$
p(y) = \phi^{\,y} (1 - \phi)^{1-y}
$$

即 $p(y = 1) = \phi$。

**参数**：$\mu_0, \mu_1 \in \mathbb{R}^n$，$\Sigma \in \mathbb{R}^{n \times n}$，$\phi \in (0, 1)$——一共 4 个参数。

> 共享 $\Sigma$ 而不是用 $\Sigma_0, \Sigma_1$ 的原因见 §3 末尾的 Note。

只要拟合出这 4 个参数，就定义了 $p(x \mid y)$ 与 $p(y)$，再用贝叶斯公式就能算出 $p(y = 1 \mid x)$ 与 $p(y = 0 \mid x)$，最终做预测：

$$
p(y = 1 \mid x) = \frac{p(x \mid y = 1)\, p(y = 1)}{p(x \mid y = 1)\, p(y = 1) + p(x \mid y = 0)\, p(y = 0)}
$$

### 2.3 联合似然与 MLE

训练集为 $\{(x^{(i)}, y^{(i)})\}_{i=1}^m$。要拟合上面的参数，我们的目标是**最大化联合似然**（joint likelihood）：

$$
\begin{aligned}
\mathcal{L}(\phi, \mu_0, \mu_1, \Sigma)
&= \prod_{i=1}^m p(x^{(i)}, y^{(i)};\ \phi, \mu_0, \mu_1, \Sigma) \\
&= \prod_{i=1}^m p(x^{(i)} \mid y^{(i)})\, p(y^{(i)})
\end{aligned}
$$

也就是要最大化 $p(x^{(i)}, y^{(i)})$。

> 生成式学习算法最大化的是联合似然；而判别式学习算法（如 Logistic 回归）最大化的是**条件似然**：
>
> $$
> \mathcal{L}(\theta) = \prod_{i=1}^m p(y^{(i)} \mid x^{(i)}; \theta)
> $$
>
> 也就是选 $\theta$ 来最大化 $p(y^{(i)} \mid x^{(i)})$。

对联合似然取对数得到对数似然 $\ell$，然后对 4 个参数做最大化（具体推导略，参见 Lecture Notes 5）：

$$
\max_{\phi, \mu_0, \mu_1, \Sigma} \log \mathcal{L}(\phi, \mu_0, \mu_1, \Sigma) = \max_{\phi, \mu_0, \mu_1, \Sigma} \ell(\phi, \mu_0, \mu_1, \Sigma)
$$

得到闭式 MLE 解：

$$
\phi = \frac{\sum_{i=1}^m \mathbb{1}\{y^{(i)} = 1\}}{m} \quad \text{(label 1 的样本占比)}
$$

$$
\mu_0 = \frac{\sum_{i=1}^m \mathbb{1}\{y^{(i)} = 0\} \, x^{(i)}}{\sum_{i=1}^m \mathbb{1}\{y^{(i)} = 0\}} \quad \text{(所有 } y=0 \text{ 样本的特征均值)}
$$

$$
\mu_1 = \frac{\sum_{i=1}^m \mathbb{1}\{y^{(i)} = 1\} \, x^{(i)}}{\sum_{i=1}^m \mathbb{1}\{y^{(i)} = 1\}} \quad \text{(所有 } y=1 \text{ 样本的特征均值)}
$$

$$
\Sigma = \frac{1}{m} \sum_{i=1}^m \left( x^{(i)} - \mu_{y^{(i)}} \right) \left( x^{(i)} - \mu_{y^{(i)}} \right)^T
$$

> 注意：**指示函数**（indicator function）$\mathbb{1}\{\cdot\}$ 在条件为真时取值 1，为假时取值 0，后面会频繁用到这个函数的写法。

直观理解：所有解都是"对应类别下的统计量"——没有梯度下降，没有迭代法，只有简单的计数与求和。这与判别式的 Logistic 回归形成鲜明对比——后者只能用梯度上升 / Newton 法迭代求解。因此生成式学习算法的参数更新以及计算都更加高效简便。

### 2.4 决策边界

**预测规则**（比较后验概率的大小，概率大的就是最后要输出的 class）：

$$
\arg\max_y p(y \mid x) = \arg\max_y \frac{p(x \mid y)\, p(y)}{p(x)}
$$

> $\arg\max$ 返回的是使表达式最大的那个 $y$ 取值（这里 $y \in \{0, 1\}$）。

注意分母 $p(x)$ 与 $y$ 无关，所以上式等价于比较：

$$
\arg\max_y p(x \mid y)\, p(y)
$$

---

## 3. GDA vs Logistic 回归对比

### 3.1 几何直观与对比图

**Logistic 回归（判别式）**：直接拟合一条参数为 $\theta$ 的直线，输出 $\theta^T x$ 作为决策边界。

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec5_LW_figure.webp" alt="logistic regression iteration" width="400" loading="lazy" decoding="async" /></div>

**GDA（生成式）**：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec5_GDA_approach.webp" alt="GDA approach" width="450" loading="lazy" decoding="async" /></div>

GDA 的步骤：

① 分别为正负例拟合高斯分布。使用**相同的协方差矩阵** $\Sigma$，为每个类别定位高斯分布的位置，确定参数：

$$
\phi,\ \mu_0,\ \mu_1,\ \Sigma
$$

② 决策边界出现在"判别为正负例概率相同"的地方，即满足：

$$
p(y = 1 \mid x) = p(y = 0 \mid x)
$$

由贝叶斯公式得到：

$$
p(x \mid y = 1) \cdot p(y = 1) = p(x \mid y = 0) \cdot p(y = 0)
$$

把之前在拟合步骤得到的 $\mu_0, \mu_1, \Sigma$ 分别代入左右两侧的高斯 PDF：

**左侧（类别 1）**：

$$
\frac{1}{(2\pi)^{n/2} \, |\Sigma|^{1/2}} \exp\!\left( - \frac{1}{2} (x - \mu_1)^T \Sigma^{-1} (x - \mu_1) \right) \cdot \phi
$$

**右侧（类别 0）**：

$$
\frac{1}{(2\pi)^{n/2} \, |\Sigma|^{1/2}} \exp\!\left( - \frac{1}{2} (x - \mu_0)^T \Sigma^{-1} (x - \mu_0) \right) \cdot (1 - \phi)
$$

化简后决策边界是一根直线（线性边界！）：

$$
\theta^T x + \theta_0 = 0
$$

其中：

$$
\theta = \Sigma^{-1} (\mu_1 - \mu_0)
$$

> 表明决策边界直线一定**垂直**于 $\mu_0$ 与 $\mu_1$ 的连线方向（在 $\Sigma$ 定义的度量空间下）。

$$
\theta_0 = - \frac{1}{2} \mu_1^T \Sigma^{-1} \mu_1 + \frac{1}{2} \mu_0^T \Sigma^{-1} \mu_0 + \log \frac{\phi}{1 - \phi}
$$

> 则决定直线在空间中的具体位置（偏移量）。

从示意图可以看出：**同一个训练集，用 GDA 和 Logistic 回归得到的决策边界通常并不相同**。

> **Note**：为什么选同一个协方差矩阵 $\Sigma$？
>
> 答案：共享 $\Sigma$ 时决策边界是线性的；若让两类各自用 $\Sigma_0, \Sigma_1$，则会得到**非线性**的决策边界——这通常不太合理。

### 3.2 假设的强/弱 与 实际应用的取舍

给定一组固定参数 $\phi, \mu_0, \mu_1, \Sigma$，画出预测的后验概率：

$$
p(y = 1 \mid x) = \frac{p(x \mid y = 1)\, p(y = 1)}{p(x)}
$$

> 各部分的参数来源：
>
> - $p(y = 1 \mid x)$：参数化为 $\phi, \mu_0, \mu_1, \Sigma$
> - $p(x \mid y = 1)$：参数化为 $\mu_1, \Sigma$（label 1 的高斯 PDF）
> - $p(y = 1)$：仅参数化为 $\phi$（伯努利分布）
> - $p(x)$：参数化为 $\phi, \mu_0, \mu_1, \Sigma$（可用贝叶斯公式展开）

对每一个给定的 $x$ 都可以算出这个比值，得到"给定 $x$ 下 $y = 1$ 的概率"。

下面这张手绘图把这条流水线分步拆开来看：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec5_GDA_simple_example_figure1.webp" alt="GDA simple example figure 1" width="400" loading="lazy" decoding="async" /></div>

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec5_GDA_simple_example_figure2.webp" alt="GDA simple example figure 2" width="400" loading="lazy" decoding="async" /></div>

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec5_GDA_simple_example_figure3.webp" alt="GDA simple example figure 3" width="400" loading="lazy" decoding="async" /></div>

可以看出：**若训练集中 $\phi = 0.5$，则 $p(y = 1 \mid x)$ 就是一个标准的 sigmoid 函数**——事实上 $p(y = 0 \mid x)$ 也是一个标准的 sigmoid 函数（作业里会有严格证明）。

既然 GDA 和 Logistic 回归本质上都是用 sigmoid 来算最终预测比率 $p(y = 1 \mid x)$，**两种算法分别在什么情况下更优？**

#### 深入对比

> **GDA（生成式）假设**：
>
> $$
> \begin{aligned}
> x \mid y = 0 &\sim \mathcal{N}(\mu_0, \Sigma) \\
> x \mid y = 1 &\sim \mathcal{N}(\mu_1, \Sigma) \\
> y &\sim \text{Bernoulli}(\phi)
> \end{aligned}
> $$

> **Logistic 回归（判别式）假设**：
>
> $$
> p(y = 1 \mid x) = \frac{1}{1 + \theta^T x}, \quad \text{with detail like } x_0 = 1
> $$
>
> 也就是说，假设 $p(y = 1 \mid x)$ 是一个 logistic 函数。

从示意图可以看出（之后作业中会做严格证明）：**从 GDA 的假设出发，确实可以推出 $p(y = 1 \mid x)$ 是逻辑函数**。但反过来，从 logistic 函数这一条件**并不能推出 GDA 的假设**——也就是说，这是一种**充分不必要**的关系。GDA 的假设条件更强（stronger set of assumptions）。

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec5_GDA_logistic_comparison.webp" alt="GDA-logistic comparison" width="450" loading="lazy" decoding="async" /></div>

> **题外话**：对任何广义线性模型中的指数族分布而言，如果加上类似的"条件概率属于某个指数族"的假设，都会推出 $p(y = 1 \mid x)$ 是逻辑函数的结论。例如：
>
> $$
> \begin{aligned}
> x \mid y = 0 &\sim \text{Poisson}(\mu_0, \Sigma) \\
> x \mid y = 1 &\sim \text{Poisson}(\mu_1, \Sigma) \\
> y &\sim \text{Bernoulli}(\phi)
> \end{aligned}
> $$
>
> 也能得到类似的结论。

在模型中，当我们给模型提供更多正确信息之后，模型通常会表现得更好。因此：

- **若 GDA 的假设正确**：因为假设更强，GDA 效果更好，**且计算更高效**（闭式解）。
- **若假设错误**：GDA 会表现得很差。
- **若训练集很小**：做出更多假设的模型反而能表现更好（更稳定）；Logistic 回归假设更弱，对偏离的模型假设更鲁棒（robust）。
- **若训练集很大且数据确实非高斯**：Logistic 回归几乎总是比 GDA 更好。

因此，在实际中 Logistic 回归的使用频率**远高于** GDA。

---

## 4. Naive Bayes  朴素贝叶斯算法

Naive Bayes 是另一类生成式算法，专门针对**高维二值稀疏特征**（而不是连续实数值）——典型场景是**邮件好坏分类**（spam vs non-spam）。

### 4.1 邮件分类问题设定

① 把一封邮件表示成一个特征向量 $x$。一种最常用的方式是构造一个**one-hot 向量**（词汇表里面的词在邮件里面出现就在对应维度上标为 1，未出现则标为 0）：

$$
x \in \{0, 1\}^n \quad \text{(n 维二元向量，n = 词汇表大小)}
$$

$$
x_j = \mathbb{1}\{\text{word } j \text{ appears in the email}\}
$$

> $x_j = 1$ 表示第 $j$ 个词在邮件里出现过。

对于 Naive Bayes，我们的目的同样是要建模 $p(x \mid y)$ 与 $p(y)$。但 $x$ 是 $n$ 维二元向量——如果直接对 $p(x)$ 用多项分布建模，需要处理 $2^n$ 种情况（相当于要建模词与词之间两两搭配的所有条件概率），参数的个数**指数级爆炸**。

所以我们要引入额外的假设来解决参数过多的问题。

### 4.2 条件独立假设

> **Naive Bayes 假设**：
>
> $x_i$ 之间**在给定 $y$ 时条件独立**。

由概率链式法则（这一步是概率的基本性质）：

$$
p(x_1, \dots, x_n \mid y) = p(x_1 \mid y)\, p(x_2 \mid x_1, y) \cdots p(x_n \mid x_1, \dots, x_{n-1}, y)
$$

Naive Bayes 假设上式可以化简为：

$$
\begin{aligned}
&\overset{\text{assume}}{=} p(x_1 \mid y)\, p(x_2 \mid y) \cdots p(x_n \mid y) \\
&= \prod_{i=1}^n p(x_i \mid y)
\end{aligned}
$$

即"已知 $y$ 的标签后，$x_i$ 的存在与否不再影响 $x_j$ 的存在与否"。

> 这个假设在数学上**并不严格成立**——例如"machine" 和 "learning" 在邮件中倾向于一起出现。但它抓住了"每个词对类别贡献相对独立"这一关键直觉，因此有很好的实践效果。

从图模型的角度看，这是一个有向概率图模型，其中 $y$ 是父节点，所有 $x_i$ 是它的子节点（**星形结构**）：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec5_naive_Bayes_prob_graph.webp" alt="Naive Bayes prob graph" width="200" loading="lazy" decoding="async" /></div>

引入条件独立假设之后，我们只需要存储 $p(x_i \mid y)$ 这 $n$ 个独立的概率——参数个数从指数级降到线性级。

### 4.3 参数与 MLE

Naive Bayes 的参数：

$$
\begin{aligned}
\phi_{j \mid y = 1} &= p(x_j = 1 \mid y = 1) \\
\phi_{j \mid y = 0} &= p(x_j = 1 \mid y = 0) \\
\phi_y &= p(y = 1)
\end{aligned}
$$

由于 $x$ 是 $n$ 维二元向量，这里 $p(x_i \mid y)$ 实际上相当于满足 $\text{Bernoulli}$。

> 对比 GDA：
>
> - GDA：$p(x \mid y)$ 服从**高斯**分布，$p(y) = \phi$（$y \sim \text{Bernoulli}(\phi)$）
> - Naive Bayes：$p(x \mid y)$ 各分量互相**独立**，$y$ 同样服从 Bernoulli
>
> 两者都对 $y$ 用 Bernoulli，但对条件概率的假设完全不同。Naive Bayes 的条件独立假设换来一个直接的好处：MLE 的解非常简便。

**联合似然**（与 GDA 类似，都是要对这个值最大化，从而拟合出参数值）：

$$
\begin{aligned}
\mathcal{L}(\phi_y, \phi_{i \mid y})
&= \prod_{i=1}^m p(x^{(i)}, y^{(i)};\ \phi_y, \phi_{j \mid y}) \\
&= \prod_{i=1}^m p(x^{(i)} \mid y^{(i)})\, p(y^{(i)})
\end{aligned}
$$

> 其中 $m$ 为训练样本数量（邮件数量）。

**MLE 结论**：

$$
\phi_y = \frac{\sum_{i=1}^m \mathbb{1}\{y^{(i)} = 1\}}{m}
$$

$$
\phi_{j \mid y = 1} = \frac{\sum_{i=1}^m \mathbb{1}\{x_j^{(i)} = 1,\ y^{(i)} = 1\}}{\sum_{i=1}^m \mathbb{1}\{y^{(i)} = 1\}}
$$

> $\phi_y$：$y = 1$ 的样本比例。
> $\phi_{j \mid y = 1}$：在所有 $y = 1$ 的样本中，第 $j$ 个词出现的比例。

回顾 GDA 的 MLE 结论做对比：

$$
\begin{aligned}
\phi &= \frac{\sum_{i=1}^m \mathbb{1}\{y^{(i)} = 1\}}{m} \\
\mu_0 &= \frac{\sum_{i=1}^m \mathbb{1}\{y^{(i)} = 0\} \, x^{(i)}}{\sum_{i=1}^m \mathbb{1}\{y^{(i)} = 0\}} \\
\mu_1 &= \frac{\sum_{i=1}^m \mathbb{1}\{y^{(i)} = 1\} \, x^{(i)}}{\sum_{i=1}^m \mathbb{1}\{y^{(i)} = 1\}} \\
\Sigma &= \frac{1}{m} \sum_{i=1}^m \left( x^{(i)} - \mu_{y^{(i)}} \right) \left( x^{(i)} - \mu_{y^{(i)}} \right)^T
\end{aligned}
$$

可以看出，**Naive Bayes 与 GDA 同属生成式模型，两者的参数更新都是简单的统计计算**——不需要梯度下降等复杂的迭代法。

> **Biggest problems**：如果某些统计量等于 0 怎么办？（比如训练集里某个词从没在 $y = 1$ 的邮件里出现过）下一节要介绍的 **Laplace 平滑**（Laplace smoothing）可以解决这个问题。

### 4.4 预测规则与分数对比

拟合好参数（$\phi_y$ 与 $\phi_{j \mid y}$）之后，在实际应用中，就可以对于给定的数据做标签预测了。

给定 $x$，对二分类问题，比较两类后验概率的大小——$p(y = 1 \mid x)$ 与 $p(y = 0 \mid x)$。用贝叶斯公式代入，因为分母 $p(x)$ 相同，可以省略，所以实际上就是比较下面这两个式子的大小：

$$
p(x \mid y = 1)\, p(y = 1) \overset{?}{=} p(x \mid y = 0)\, p(y = 0)
$$

① **Score ($y = 1$)**：

$$
\begin{aligned}
\text{Score}(y = 1) &= p(x \mid y = 1)\, p(y = 1) \\
&= p(y = 1) \prod_{j=1}^n p(x_j \mid y = 1)
\end{aligned}
$$

② **Score ($y = 0$)**：

$$
\begin{aligned}
\text{Score}(y = 0) &= p(x \mid y = 0)\, p(y = 0) \\
&= p(y = 0) \prod_{j=1}^n p(x_j \mid y = 0)
\end{aligned}
$$

> 哪一个 Score 大，预测就输出对应的 label。

## 参考资料

- [CS229 Lecture Notes 2](https://cs229.stanford.edu/notes2021fall/cs229-notes2.pdf) — 原始讲义；"Generative Learning Algorithms" 是本文主要参考章节
- [CS229 Lecture 5](https://www.bilibili.com/video/BV1b4anzMEUv) — B 站上 Andrew Ng 的讲课视频 Lec5，对应本文覆盖的章节
- [Mitchell, Machine Learning (CMU), Ch.3](https://www.cs.cmu.edu/~tom/mlbook/NBayesLogReg.pdf) — Chapter 3 对 Naive Bayes 的概率视角做了更系统的展开