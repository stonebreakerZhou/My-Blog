---
title: "CS229 : Lec 8 — 学习算法的实践应用"
description: CS229 Lecture 8 学习笔记，讲学习算法应用中的相关知识：从 Bias / Variance 理解过拟合与欠拟合；再使用正则化方法；通过 Train / Dev / Test 划分与 K折交叉验证 选出泛化能力强的模型；最后简单介绍特征选择。
pubDate: 2026-09-01
series: cs229
subSeries: personal-lecture-notes
order: 9
categories:
  - CS229
  - Bias-Variance
  - Regularization
  - Train-Dev-Test Split
  - Cross-Validation
  - Feature Selection
---

> **TL;DR**:
> - **Bias & Variance**：欠拟合 = 高 bias，过拟合 = 高 variance
> - **正则化（Regularization）**：在线性/逻辑回归目标里加 $\lambda \|\theta\|^2$（或减去）防止参数过大；过大 $\lambda$ 会反过来欠拟合。MAP 视角下，这等价于对 $\theta \sim \mathcal{N}(0, \tau^2 I)$ 的高斯先验做最大后验估计。
> - **数据集划分的三个子集：Train / Dev / Test**：Train 用来拟合参数、Dev 用来选模型、Test 只能用来报告无偏结果。
> - **K折交叉验证（K-fold Cross-Validation）**：当数据量较小时，把训练集切成 $k$ 份轮流做验证，更充分利用数据；极端情况是 留一法交叉验证（Leave-One-Out CV）。
> - **特征选择 (Forward Search 法)**：贪心地逐步加入对 Dev 集性能提升最大的特征。

## 引子

前面几讲我们搭好了监督学习的一整套工具箱：线性回归、逻辑回归、GDA、Naive Bayes、SVM、Kernels。这些模型在新数据上的**泛化能力**到底如何？怎么判断自己的模型是"学得太死"还是"学得太浅"？这节课就围绕"**把学习算法用到实践中**"这一主题展开：

- **Bias / Variance**：从直觉上区分欠拟合与过拟合
- **Regularization**：缓解高方差
- **Train / Dev / Test splits**：合理划分数据
- **Model Selection & Cross-Validation**：小数据集上更稳健地选模型
- **Feature Selection (Forward Search)**：特征工程的一个简单算法

---

## 1. Bias & Variance

给定一个数据集，我们希望拟合得"刚刚好"——既不是欠拟合也不是过拟合：

$$
\text{underfit (high } \underline{bias}\text{)} \;\longrightarrow\; \text{“just right”} \;\longrightarrow\; \text{overfit (high } \underline{variance}\text{)}
$$

- **High bias**：算法对数据有很强的先验（perconception），认为数据可以用（线性）函数拟合；当真实关系是非线性时就会欠拟合。
- **High variance**：如果我们再做一次实验（换一份数据），可能得到完全不同的预测——模型对训练数据中的噪声过度敏感。

---

## 2. Regularization

正则化是缓解高方差（过拟合）的常用手段。

### 2.1 线性回归的正则化

我们以前做线性回归的时候的优化目标是

$$
\min_\theta \frac{1}{2} \sum_{i=1}^m \left\| y^{(i)} - \theta^T x^{(i)} \right\|^2
$$

加上**正则化项**：

$$
\min_\theta \frac{1}{2} \sum_{i=1}^m \left\| y^{(i)} - \theta^T x^{(i)} \right\|^2 \color{red}{+ \lambda \|\theta\|^2}
$$

直观地看，以一个五阶多项式拟合数据集为例：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec8_regularization_dataset.webp" alt="dataset" width="60%" loading="lazy" decoding="async" /></div>

- ① $\lambda = 0$：相当于直接拟合一个五阶多项式，会**过拟合**：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec8_regularization_overfit.webp" alt="overfit" width="60%" loading="lazy" decoding="async" /></div>

- ② $\lambda$ 太大：相当于把 $\theta$ 强行压向 0，当 $\lambda$ 充分大时 $h_\theta(x) \approx 0$，会**欠拟合**：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec8_regularization_underfit.webp" alt="underfit" width="60%" loading="lazy" decoding="async" /></div>

- ③ 选一个合适的 $\lambda$，既不让参数太大，又允许模型刻画数据：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec8_regularization_just-right.webp" alt="just right" width="60%" loading="lazy" decoding="async" /></div>

### 2.2 逻辑回归的正则化

逻辑回归里正则化是**减号**（因为目标是最大化）：

$$
\arg\max_\theta \sum_{i=1}^n \log p(y^{(i)} \mid x^{(i)}; \theta) \color{red}{- \lambda \|\theta\|^2}
$$

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec8_regularization_logistic.webp" alt="logistic regression regularization" width="60%" loading="lazy" decoding="async" /></div>

> **Rule of thumb for logistic regression （经验法则）**：如果不加正则化，至少需要训练样本数与待拟合参数数量**同一量级**，否则容易过拟合。

> **Q：为什么 SVM 在无穷维空间里也不容易过拟合？**
>
> **A**：因为 SVM 的优化目标是 $\min \|w\|^2$——这跟 $\min \lambda \|\theta\|^2$ 有类似的正则化效果。强制参数小让 SVM 难以把训练集拟合得过死。

### 2.3 文本分类等高维场景

如果训练样本只有 100 个，但特征向量维度 $x \in \mathbb{R}^{10000}$，模型几乎一定会过拟合。加上正则化的逻辑回归通常是一个稳健的选择（但需要用梯度下降求解局部最优参数）。

> **Q：为什么我们不针对每个参数分别用不同的 $\lambda_j$，写成 $\sum_j \lambda_j (\theta_j)^2$？**
>
> **A**：那样的话我们最后会得到一堆需要调的 $\lambda_j$。而我们并没有很好的"先验"来给每个维度选权重。统一用一个 $\lambda$ 更简单也更常用。

> **注**：实际使用中我们还会做**特征预处理**——把每一维特征缩放到相同的区间（如 $(-1, +1)$ 或 $(0, 1)$），即**归一化**。这会让梯度下降跑得更快。

### 2.4 正则化的数学由来：MAP 视角

下面给正则化一个概率解释。

给定训练集 $S = \{(x^{(i)}, y^{(i)})\}_{i=1}^m$，我们想找最可能的 $\theta$：

$$
P(\theta \mid S) = \frac{P(S \mid \theta)\, P(\theta)}{P(S)}
$$

由于 $P(S)$ 与 $\theta$ 无关，求 $\arg\max_\theta P(\theta \mid S)$ 等价于求

$$
\arg\max_\theta P(S \mid \theta)\, P(\theta)
$$

**i.i.d. 假设（样本之间相互独立） + 条件概率公式展开**：

$$
\begin{aligned}
P(S \mid \theta)
&= \prod_{i=1}^m P(x^{(i)}, y^{(i)} \mid \theta) \\
&= \prod_{i=1}^m P(y^{(i)} \mid x^{(i)}, \theta) \cdot P(x^{(i)} \mid \theta)
\end{aligned}
$$

> 对于判别式模型（如逻辑回归等所有 GLM 广义线性模型），我们关心的是 $P(y \mid x, \theta)$，而 $P(x^{(i)} \mid \theta)$ **不依赖于**模型参数 $\theta$，所以 $p(x^{(i)} \mid \theta) = p(x^{(i)})$ 是常数，可以从优化目标中略去：
>
> $$
> P(S \mid \theta) \propto \prod_{i=1}^m P(y^{(i)} \mid x^{(i)}, \theta)
> $$

所以最终的优化目标是

$$
\arg\max_\theta \prod_{i=1}^m P(y^{(i)} \mid x^{(i)}, \theta)\, P(\theta)
$$

**假设 $\theta$ 满足多元高斯分布**：

$$
\text{assume}: \theta \sim \mathcal{N}(0, \tau^2 I)
$$

（注意 $\tau^2 I$ 这个协方差矩阵说明 $\theta$ 各维度独立、方差相等。）

写出概率密度：

$$
\begin{aligned}
P(\theta)
&= \frac{1}{(2\pi)^{n/2}\, |\tau^2 I|^{1/2}} \exp\!\left(- \frac{1}{2}\, \theta^T (\tau^2 I)^{-1} \theta\right) \\
&= \frac{1}{(2\pi)^{n/2}\, |\tau^2 I|^{1/2}} \exp\!\left(- \frac{1}{2\tau^2}\, \theta^T \theta\right)
\end{aligned}
$$

代回原目标，取对数后要最大化的是

$$
\arg\max_\theta \log\!\left(\prod_{i=1}^m P(y^{(i)} \mid x^{(i)}, \theta)\, P(\theta)\right)
= \arg\max_\theta \left[ \sum_{i=1}^m \log P(y^{(i)} \mid x^{(i)}, \theta) + \log P(\theta) \right]
$$

记作 $\ell(\theta)$。

代入 $\log P(\theta)$：

$$
\log P(\theta) \propto - \frac{1}{2\tau^2}\, \theta^T \theta
$$

所以最终的优化目标是

$$
\arg\max_\theta \left[\, \color{red}{\sum_{i=1}^m \log P(y^{(i)} \mid x^{(i)}, \theta)} \color{blue}{- \frac{1}{2\tau^2}\, \|\theta\|^2} \,\right]
$$

> 注意红色那一块
>
> $$
> \arg\max_\theta \sum_{i=1}^m \log P(y^{(i)} \mid x^{(i)}, \theta)
> $$
>
> 其实就是**最大似然估计（MLE）**——也就是没有正则化时我们做的优化。
>
> 后面蓝色那一块
>
> $$
> \arg\max_\theta \left(- \frac{1}{2\tau^2}\, \|\theta\|^2\right)
> $$
>
> 就是我们额外加上的正则化项。

**小结**：上面"加 $\lambda \|\theta\|^2$ 正则化"这种做法的数学本质，其实就是**对参数 $\theta$ 引入一个多元高斯分布的先验**。这种做法称为 **MAP（Maximum A Posteriori，最大后验估计）**——它在做最大似然拟合数据的同时，还要求 $\theta$ 符合预设的规律（这里即高斯先验，因此 $\theta$ 不会太大）。而 **MLE（最大似然估计）** 则认为 $\theta$ 是一个未知但固定的值，只让模型在训练集上充分拟合。

> **MLE vs MAP**：这两个估计方法实际上涉及统计学两大学派的根本区别：
>
> ① **Frequentist（频率学派）**：认为存在一个真实的 $\theta$ 使当前数据最可能，我们要去估计这个值——所以用 MLE：
>
> $$ P(S \mid \theta) \;\xrightarrow{\text{MLE}}\; \theta $$
>
> ② **Bayesian（贝叶斯学派）**：认为 $\theta$ 是未知的，但在看到任何数据之前，我们已经对数据生成机制有先验信念，这些先验信念被编码在概率分布中：
>
> $$ P(\theta) \;\to\; \text{prior distribution} $$
>
> 回顾一下，在上个例子中，我们用 **Gaussian prior** 是合理的——一方面世界上大多数事物的分布近似高斯，另一方面在我们对 $\theta$ 一无所知时，把它的均值设为 0 也很自然。
>
> $$ \max_\theta P(\theta \mid S) \;\xrightarrow{\text{MAP}}\; \theta $$

---

## 3. 数据集分割：Train / Dev / Test splits

### 3.1 误差曲线

一般地，我们可以画出训练误差与泛化误差随模型复杂度变化的曲线：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec8_error-curve.webp" alt="error curve" width="75%" loading="lazy" decoding="async" /></div>

训练误差会随模型复杂度增加而单调下降（模型越复杂拟合训练集越紧）；但泛化误差会先下降后上升——它的最低点就是我们要找的"balanced point"。

下面给出一种**机械化算法**帮我们自动找到这个平衡点。

### 3.2 完整流程

给定一个数据集，我们把它分成若干子集：

$$
S \to S_{\text{train}},\; S_{\text{dev}},\; S_{\text{test}}
$$

完整流程：

1. **Train**：在 $S_{\text{train}}$ 上训练一连串模型（不同超参，例如多项式的阶数），得到若干 hypothesis $h_i$；
2. **Dev**：在 $S_{\text{dev}}$ 上测每个 $h_i$ 的误差，**选误差最小**的那个；
3. **Test**：用独立的 $S_{\text{test}}$ 对最终选出的模型评估一次，给出**无偏**的发布结果。

> **关键约束**：测试集 $S_{\text{test}}$ 不能用来做模型选择决策（比如说判断是否要回滚模型等等）——否则得到的报告就不是无偏的。测试集只能用来**最终报告或追踪**性能。

### 3.3 数据集的划分比例

**经验法则**（数据量**不大**时）：

$$
S \to \begin{cases} \text{Train} : 70\% \\ \text{Test} : 30\% \end{cases}
$$

或者

$$
S \to \begin{cases} \text{Train} : 60\% \\ \text{Dev} : 20\% \\ \text{Test} : 20\% \end{cases}
$$

**Massive dataset**（数据量很大时）：分给 Dev / Test 的比例可以显著缩小——因为我们只需要 Dev/Test 用来"测量模型之间的小差异"就够了。如果只是想比较不同算法的整体效果，更不需要很大的 Dev/Test。

> **Hold-out cross validation**：对 Train / Dev 集合做这样的"留出"操作，业界也叫 **hold-out cross validation**。"development set" 也称为 "cross validation set"。

### 3.4 数据量很小时：K-fold Cross-Validation （K 折交叉验证）

如果数据集很小，但又不想浪费任何数据用于训练（因为之前我们预留给 Dev 的不能拿来 Train），可以用 **K-fold Cross-Validation**：

例如：

$$
S = \{(x^{(i)}, y^{(i)})\}_{i=1}^{100}
$$

取 $k = 5$（典型是 $k = 10$），把数据集切成 5 份，每份 20 个样本。

**算法**：

```
For i = 1, ..., k:
    Train: 在 k-1 份上拟合参数
    Test:  在剩下那 1 份上测试
Average(k 次得到的误差)
```

如果想同时选模型的超参（例如多项式阶数），就在外层再加一个循环：

$$
\text{For } d = 1, \dots, s \quad \text{(degree of polynomial)}
$$

对每个候选的 $d$，重复上面的 $k$-fold 流程，最后比较每个 $d$ 的**平均误差**，取最小平均误差对应的那个 d 值就是我们要找的最优的多项式的次数。

**可选的最后一步**（在选完超参之后）：用 **100% 的数据**重新拟合一次模型。

相比简单留出法，K-fold 把每次留出的数据比例降到 $\frac{1}{k}$，**数据利用率更高**；但代价是**计算量翻 $k$ 倍**，所以只适用于小数据集。

**极端版本：Leave-one-out cross-validation**：每次只留一个样本做测试。适用于 $m \le 100$ 这种极小数据集。

> **Q：K-fold CV 的最后平均步骤里，我们要估计这 $k$ 个误差估计的方差吗？**
>
> **A**：其实不太需要——这 $k$ 个估计**高度相关**，因为每次训练集都共享了 $(k-2)/(k-1)$ 的样本，方差不会告诉我们什么新信息。

---

## 4. Feature Selection

最后介绍一个简单的**特征选择**算法：**Forward Search**。

```
初始化：F = ∅  (当前选中的特征集合为空)

Repeat:
    1) 对每个候选特征 i，试着把 i 加入 F，评估在 Dev 集上的性能
    2) 选出那个能让 Dev 性能提升最多的特征 i*
    3) 把 i* 加入 F

Until: 再加入新特征反而让 Dev 集性能下降
```

本质上是**贪心地逐步加入特征**，直到性能不再提升为止。

---

## 参考资料

- [CS229 Lecture Notes 5](https://cs229.stanford.edu/notes2021fall/cs229-notes5.pdf) 以及[CS229 Lecture Note 4](https://cs229.stanford.edu/notes2021fall/cs229-notes5.pdf) — Bias / Variance、Regularization、Learning Theory
- [CS229 2018](https://www.bilibili.com/video/BV1b4anzMEUv) — 看 B 站上 Andrew Ng 的讲课视频 CS229 Lec 8
- [Bishop, Pattern Recognition and Machine Learning, Ch.3](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) — Chapter 3 : Linear Models for Regression（Regularization 的概率视角）
- [Bishop PRML, Ch.3.3](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) — Bayesian Linear Regression（MAP / MLE 对比）
