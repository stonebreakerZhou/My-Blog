---
title: "CS229 : Lec 9 — 学习理论 (Learning Theory)"
description: CS229 Lecture 9 学习笔记，先给出两个前提假设；再从 Bias / Variance 的角度理解泛化；接着引入ERM；再证明一致收敛；最后简单用 VC 维推广。
pubDate: 2026-09-08
series: cs229
subSeries: personal-lecture-notes
order: 9
categories:
  - CS229
  - Learning Theory
  - Bias-Variance
  - ERM
  - Uniform Convergence
  - VC Dimension
---

> **TL;DR**:
> - **基本假设**：存在一个数据分布 $\mathcal{D}$，所有样本 $(x, y) \sim \mathcal{D}$ **独立同分布 (i.i.d.)**；存在一个真实参数 $\theta^*$（或 $h^*$），它是个**未知的常数**。
> - **偏差与方差（Bias & Variance）**：算法在每一次送入的抽样数据集上得到的估计 $\hat\theta$ 形成一个分布；Bias 是分布的中心偏离真实参数 $\theta^*$ 的距离，Variance 则表示分布的离散程度，两者可以独立变化。
> - **泛化误差的分解**：估计出的参数 $\hat h$ 的泛化误差：$\varepsilon(\hat h) = \text{Bayes error} + \text{Approximation error} + \text{Estimation error}$，分别对应"不可消除 / 模型选择 / 数据有限"这三类。
> - **Empirical Risk Minimizer (ERM)**：$\hat h_{\text{ERM}} = \arg\min_{h \in \mathcal{H}} \hat\varepsilon_S(h)$，优化目标是最小化经验风险(empirical risk）。
> - **Uniform Convergence 一致收敛**：对**有限**假设类的大小 $|\mathcal{H}| = k$，$P(\forall h,\,|\hat\varepsilon_S(h) - \varepsilon(h)| \le \gamma) > 1 - 2k \exp(-2\gamma^2 m)$；并由此推出 $\varepsilon(\hat h) \le \varepsilon(h^*) + 2\gamma$。
> - **VC 维**：衡量**无限**假设类的"有效表达能力"，把有限类公式里的 $k$ 替换成 $\text{VC}(\mathcal{H})$ 后，泛化误差界仍成立。

## 引子

前面几讲我们搭建了一整套"建模工具箱"——线性回归、逻辑回归、GDA、Naive Bayes、SVM、Kernels。但一直有一个根本性问题没回答：

> **"为什么这些算法在新数据上也能 work？"**

也就是说，学习算法为什么具有**泛化能力**？Lec 8 给出了 Bias / Variance 的直觉刻画，这节课会**严格化**这种直觉——用概率论和统计学给出一个回答，下面是本节 Lecture 的 outline：

- **Setup / Assumptions**："数据从哪来"
- **Bias / Variance**：Learning Algorithm 的某种性质，且受数据量 $m$ 的影响
- **Approx / Estimate**：泛化误差 = 三部分之和
- **ERM**：一个朴素的学习算法
- **Uniform Convergence**：为什么 ERM 能泛化
- **VC dimension**：处理无穷假设类

---

## 1. 基本假设 (Setup & Assumptions)

要谈学习理论，必须先把**设定**写清楚：

1. 假设一 ： **存在数据分布** $\mathcal{D}$：训练集和测试集的所有 $(x, y)$ 都从这个分布采样：

   $$
   (x, y) \sim \mathcal{D}
   $$

   这条假设既适用于监督学习也适用于非监督学习——它要求**训练集和测试集同分布**。

2. 假设二： **样本独立同分布 (i.i.d.)**：所有样本 $(x^{(i)}, y^{(i)})$ 相互独立地从 $\mathcal{D}$ 采样。

3. **存在真实参数** $\theta^*$（或真实函数 $h^*$）：确定性函数把样本映射到一个常数。注意 $\theta^*$ 不是随机变量，只是**未知的固定值**。

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec9_learning_algo_procedure.webp" alt="learning procedure" width="80%" loading="lazy" decoding="async" /></div>

---

## 2. Bias & Variance 的参数视角

在 Lec 8 我们从模型复杂度的角度看 Bias / Variance，这一讲换一个视角——**把"参数估计"看作一个统计问题**。

### 2.1 数据视图 vs 参数视图

以线性回归为例：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec9_data_view.jpg" alt="data view" width="85%" loading="lazy" decoding="async" /></div>

上面这种"画散点 + 拟合直线"的视角是**数据视图**。

如果换到**参数视图**：我们关心的是真实参数 $\theta^*$ 的位置，而不是具体某一次拟合出来的直线长什么样。

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec9_parameter_view.jpg" alt="parameter view" width="75%" loading="lazy" decoding="async" /></div>

### 2.2 Bias / Variance 作为采样分布的性质

在参数视图里，如果我们多次实验，每次用不同的数据集 $\mathcal{D}$ 训练算法 $\mathcal{A}$，会得到多个估计 $\hat\theta$（图中的蓝色点）；真实参数 $\theta^*$ 是红色点。

> 蓝色点的集合本质上就是 $\hat\theta$ 的**采样分布**。

- **Bias**：采样分布的**中心**是否偏离 $\theta^*$（即 $E[\hat\theta] = \theta^*$ 是否成立）
- **Variance**：采样分布的**离散程度**（即 $\text{Var}[\hat\theta]$ 的大小）

所以 Bias 和 Variance 其实是**采样分布的一阶矩与二阶矩**。

### 2.3 Consistent Algorithm（一致性）

随着样本量 $m \to \infty$，Variance 会趋于 0：

$$
m \to \infty, \quad \text{Var}[\hat\theta] \to 0
$$

这个收敛到 0 的速度叫做 **statistical efficiency**——它衡量算法从数据里提取信息的能力。

如果一个算法满足

$$
\text{as } m \to \infty, \quad \hat\theta \;\xrightarrow{P}\; \theta^*
$$

即对所有 $m$ 都有

$$
E[\hat\theta] = \theta^*
$$

则称它是 **consistent（一致）** 的。

- **高 bias 算法**：无论喂多少数据，$\hat\theta$ 的分布永远以某个偏离 $\theta^*$ 的点为中心
- **高 variance 算法**：估计值容易被数据中的噪声带偏，也就是其离散程度比较高

Bias 和 Variance 是**互相独立**的，它们都是"在固定数据量 $m$ 下的算法自身的属性"。

---

## 3. Approximation / Estimation：泛化误差的三层分解

### 3.1 一些记号

为了将泛化形式化表示，我们先画出一个假设空间的图：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec9_hypothesis_space.jpg" alt="hypothesis space" width="50%" loading="lazy" decoding="async" /></div>

- $g$：**理论最优** hypothesis（在所有可能函数中最好的那一个）
- $\mathcal{H}$：一个**假设类**（比如所有线性分类器；所有SVM...）
- $h^* \in \mathcal{H}$：$\mathcal{H}$ 中**用无穷多数据学习得到的最优** hypothesis
- $\hat h \in \mathcal{H}$：只利用当前**有限数据**学习得到的 hypothesis
- $\varepsilon(h)$：**风险 / 泛化误差**（衡量模型 $h$ 在分布 $\mathcal{D}$ 上的错误率，因为 $\mathcal{D}$ 可以无穷采样所以这是一个"无穷过程"）

  $$
  \varepsilon(h) = E_{(x, y) \sim \mathcal{D}}\big[\,\mathbf{1}\{h(x) \ne y\}\,\big]
  $$

- $\hat\varepsilon_S(h)$：**经验风险**（衡量模型 $h$ 在当前 $m$ 个样本上的错误率，是一个"有限过程"）

  $$
  \hat\varepsilon_S(h) = \frac{1}{m} \sum_{i=1}^m \mathbf{1}\{h(x^{(i)}) \ne y^{(i)}\}
  $$

- $\varepsilon(g)$：**Bayes error / 不可约误差**：即使选了最优的 $g$，仍然会犯错的概率——这是任何模型都跨不过去的天花板，原因是数据自带噪声。

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec9_error_risk_plot.jpg" alt="error plot" width="70%" loading="lazy" decoding="async" /></div>

### 3.2 误差的分解

上面这套记号天然地把泛化误差分解成三层之和：

$$
\varepsilon(\hat h) = \underbrace{\varepsilon(\hat h) - \varepsilon(h^*)}_{\text{Estimation error}} + \underbrace{\varepsilon(h^*) - \varepsilon(g)}_{\text{Approximation error}} + \underbrace{\varepsilon(g)}_{\text{Bayes error}}
$$

直观解释这三层：
- **Bayes error**：数据本身带噪声，无论用什么算法、用多少数据都消不掉
- **Approximation error**：因为我们把模型限定在某个类 $\mathcal{H}$ 里（比如说所有的线性回归）（而不是所有可能的假设函数），那么这部分限制导致的代价是 $\varepsilon(h^*) - \varepsilon(g)$
- **Estimation error**：因为我们只有有限数据（这些数据相当于是从 $\cal D$ 中抽样得到的）来学习 ，所以学出来的 $\hat h$ 不等于 $\mathcal{H}$ 里最优的 $h^*$，这部分代价是 $\varepsilon(\hat h) - \varepsilon(h^*)$

进一步拆分后组合：Estimation error 拆成 (Estimation Variance + Estimation Bias)，于是

$$
\varepsilon(\hat h) = \text{Variance} + \text{Bias} + \text{Bayes error}
$$

> **注**：这里的 Bias 刻画的是 $\hat h$ 离理论最优 $g$ 有多远（可以看上面的图进行直观理解）——和"模型简单 vs 复杂"那种直觉的 Bias 略有差异。


### 3.3 对症下药

| 问题 | 应对手段 |
|---|---|
| 高 Variance | ① 增加样本数 $m$；② 加正则化（注意：会涨 Bias） |
| 高 Bias | ① 把假设类 $\mathcal{H}$ 做大（注意：会涨 Variance） |

> **注**：Regularization 由于在限制范数大小，实际上是在**收缩** $\mathcal{H}$——它用涨一点 Bias 来换取 Variance 的下降。（因为增大 $\cal H$ 能让假设空间类更有可能包含 $g$ / 离 $g$ 更近，Bias 更小）

---



## 4. Empirical Risk Minimizer (ERM)

最朴素的学习算法：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec9_ERM.jpg" alt="ERM" width="85%" loading="lazy" decoding="async" /></div>

$$
\hat h_{\text{ERM}} = \arg\min_{h \in \mathcal{H}} \frac{1}{m} \sum_{i=1}^m \mathbf{1}\{h(x^{(i)}) \ne y^{(i)}\}
$$

这个算法实际上就是"挑出在当前训练集上分类错误最少的 $h$"，就把最后得到的 $h$ 作为结果。ERM 本身是一个**算法模板**——在它之上可以构造各种具体的学习算法（比如 SVM、Logistic 回归等都可以看作 ERM 的一种实现）。

但要回答 **"为什么 ERM 泛化得好"**，需要做下面的理论分析。

---



## 5. Uniform Convergence

### 5.1 两个核心问题

① 既然 ERM 最小化的是**训练误差**（经验风险），那它对**泛化误差**有什么影响？也就是说 **$\varepsilon(\hat h)$ 和 $\hat\varepsilon(\hat h)$ 的关系是什么？**

② 我们拿 ERM 选出来的 $\hat h$，和**用无穷数据**学出来的最优 $h^*$ 比，泛化误差上界是多少？也就是 **$\varepsilon(\hat h)$ 和 $\varepsilon(h^*)$ 的关系是什么？**

### 5.2 两个工具

> 工具① **Union bound（联合界）**：如果有 $k$ 个事件 $A_1, \dots, A_k$（不必独立），则
>
> $$
 P(A_1 \cup A_2 \cup \dots \cup A_k) \le P(A_1) + \dots + P(A_k)
 $$

> 工具② **Hoeffding 不等式**：设 $Z_1, Z_2, \dots, Z_m \sim \text{Bernoulli}(\phi)$，$\hat\phi = \frac{1}{m} \sum_{i=1}^m Z_i$，$\gamma > 0$，则
>
> $$
 P(|\hat\phi - \phi| > \gamma) \le 2 \exp(-2 \gamma^2 m)
 $$

### 5.3 问题①：简单情况出发，固定一个 $h_i$，经验误差与泛化误差的关系

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec9_epsilon%28hat%28h%29%29_with_epsilon%28h%29.webp" alt="ε(ĥ) and ε(h)" width="80%" loading="lazy" decoding="async" /></div>

挑出某个具体的 $h_i$，定义

$$
Z_j = \mathbf{1}\{h_i(x^{(j)}) \ne y^{(j)}\}
$$

由于每个 $Z_j$ 是 Bernoulli 变量，其均值为

$$
\phi = E_{(x^{(j)}, y^{(j)}) \sim \mathcal{D}}\big[\mathbf{1}\{h_i(x^{(j)}) \ne y^{(j)}\}\big] = \varepsilon(h_i)
$$

我们套入 Hoeffding 不等式：

$$
P\!\left(\left|\frac{1}{m} \sum_{j=1}^m Z_j - \varepsilon(h_i)\right| > \gamma\right) \le 2 \exp(-2 \gamma^2 m)
$$

即

$$
P\!\left(|\hat\varepsilon(h_i) - \varepsilon(h_i)| > \gamma\right) \le 2 \exp(-2 \gamma^2 m)
$$

> **结论**：对**固定的** $h_i$，它的 $\hat\varepsilon(h_i)$ 会随 $m$ 增大而**集中在** $\varepsilon(h_i)$ 周围——所以用 ERM 最小化经验误差，确实可以（在直觉上）最小化泛化误差。

但这里有一个**逻辑漏洞**：上面的推导是"先固定 $h$，再去采样数据"；实际情况却是"先采样数据，再用 ERM 选出 $\hat h$"——$\hat h$ 和数据并不是独立的， $\hat h$ 是对于这个特定数据下最优的那个假设。


### 5.4 一致收敛 (Uniform Convergence)：把上面的结果推广到所有 $h$

要修复这个漏洞，我们需要把"对一个 $h$ 集中"推广到"**对 $\mathcal{H}$ 中所有 $h$ 同时集中**"——这就是 **uniform convergence**：

> **定义 (Uniform Convergence)**：$\mathcal{H}$ 中**所有** $h$ 的 $\hat\varepsilon_S(h)$ 都同时接近 $\varepsilon(h)$。


### 5.5 情形 1：有限假设类

**假设** $\mathcal{H}$ 的大小有限：$|\mathcal{H}| = k$。

对每个 $h \in \mathcal{H}$，Hoeffding 都给出一个界。套 Union bound 把 $k$ 个 bound 加起来：

$$
P\!\left(\exists h \in \mathcal{H},\ |\hat\varepsilon_S(h) - \varepsilon(h)| > \gamma\right) \le k \cdot 2 \exp(-2 \gamma^2 m)
$$

取反（所有 $h$ 都不出问题的概率）：

$$
P\!\left(\forall h \in \mathcal{H},\ |\hat\varepsilon_S(h) - \varepsilon(h)| \le \gamma\right) > 1 - k \cdot 2 \exp(-2 \gamma^2 m)
$$

三个变量：
- $\delta = k \cdot 2 \exp(-2 \gamma^2 m)$：**出错概率**（差距超过 $\gamma$ 的概率）
- $\gamma$：**误差容忍度**
- $m$：**样本量**

> **设计问题**：固定上面三个参数的其中两个，就可以求第三个（范围）。

例如：固定 $\delta, \gamma > 0$，反解 $m$：

$$
m \;\ge\; \frac{1}{2\gamma^2} \log\!\left(\frac{2k}{\delta}\right)
$$

这个界叫做 **sample complexity**——告诉我们：要达到 $\gamma$ 容忍度 + $\delta$ 失败率，至少需要多少样本。


### 5.6 问题②：$\varepsilon(\hat h)$ vs $\varepsilon(h^*)$

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec9_epsilon%28hat%28h%29%29_with_epsilon%28h%5Estar%29.webp" alt="ε(ĥ) and ε(h*)" width="85%" loading="lazy" decoding="async" /></div>

Hoeffding 不等式给出

$$
\varepsilon(\hat h) \le \hat\varepsilon(\hat h) + \gamma
$$

而 ERM 保证 $\hat\varepsilon(\hat h) \le \hat\varepsilon(h^*)$，于是

$$
\varepsilon(\hat h) \le \hat\varepsilon(\hat h) + \gamma \le \hat\varepsilon(h^*) + \gamma
$$

再由 **uniform convergence**：$\hat\varepsilon(h^*) \le \varepsilon(h^*) + \gamma$，代回去：

$$
\varepsilon(\hat h) \le \varepsilon(h^*) + 2\gamma
$$

> **结论**：以概率 $1 - \delta$、训练集大小为 $m$，$\varepsilon(\hat h)$ 和 $\varepsilon(h^*)$ 的差距不超过 $2\gamma$，即
>
> $$
> \varepsilon(\hat h) \le \varepsilon(h^*) + 2\sqrt{\frac{1}{2m} \log\!\left(\frac{2k}{\delta}\right)}
> $$


### 5.7 情形 2：无限假设类

把"有限类"的结果推广到"无限类"——这正是下一节 **VC 维**要做的事。

---



## 6. VC Dimension

现实中的模型（线性回归、神经网络…）权重 $\theta$ 可以取任意实数，假设空间 $\mathcal{H}$ 是**无穷大**的——直接代入 $|\mathcal{H}| = k = \infty$，所有 bound 都失效。

但统计学家发现：**虽然参数能取无穷多个实数，模型的"有效表达能力"却是有限的**。这个"有效大小"就是 **VC dimension**。

> **直观**：VC 维衡量的是 $\mathcal{H}$ 能"打散"（shatter）多少个点的能力。

把有限类公式里的 $k$ 替换成 $\text{VC}(\mathcal{H})$，就可以得到一个不发散的泛化界：

$$
\varepsilon(\hat h) \;\le\; \varepsilon(h^*) \;+\; O\!\left(\sqrt{\frac{\text{VC}(\mathcal{H})}{m} \log\!\left(\frac{m}{\text{VC}(\mathcal{H})}\right) + \frac{1}{m} \log\!\left(\frac{1}{\delta}\right)}\right)
$$

这告诉我们：**只要训练样本 $m$ 远大于 $\text{VC}(\mathcal{H})$，泛化误差就会很小**——这正是学习算法能泛化的根本原因。

---


## 参考资料

- [CS229 Bias&Variance PPT](https://cs229.stanford.edu/notes2021fall/lecture10-bias-variance.pdf) — Bias / Variance 以及这一讲的简单
- [CS229 Lecture Notes 4](https://cs229.stanford.edu/notes2021fall/cs229-notes4.pdf) — 涵盖这一讲主要内容！
- [CS229 2018](https://www.bilibili.com/video/BV1b4anzMEUv) — B站上Andrew Ng的讲课视频
- [Understanding Machine Learning: From Theory to Algorithms](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/understanding-machine-learning-theory-algorithms.pdf) — Shalev-Shwartz & Ben-David，第 6 章对 VC 维、Uniform Convergence 有更严格的证明