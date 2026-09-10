---
title: "CS229 : Lec 10 — 决策树与集成方法 (Decision Tree & Ensemble Methods)"
description: CS229 Lecture 10 学习笔记。第一个非线性模型——决策树 (Decision Tree)：贪心分裂、损失函数：交叉熵 / Gini 、正则化、回归树；再讲集成方法：Bagging 降低方差、Random Forest、Boosting 降低偏差。
pubDate: 2026-09-10
series: cs229
subSeries: personal-lecture-notes
order: 10
categories:
  - CS229
  - Decision Tree
  - Ensemble Methods
  - Bagging
  - Random Forest
  - Boosting
  - Adaboost
---

> **TL;DR**:
> - **决策树 (Decision Tree)**：第一个**非线性**模型。对特征空间做"贪心、自顶向下、递归"的二分裂。
> - **分裂准则：损失函数 (Loss)**：用 **交叉熵** 或 **Gini**——它们都是**严格凹**的，所以无论子区域样本量是否均匀，split 后总能保证 loss 下降；**misclassification loss** 不严格凹，可能 split 后 loss 不变甚至上升。
> - **决策树的优劣**：优点——可解释、能直接处理类别特征、快；缺点——**高方差**、不适合加性可分的边界、单棵树预测精度一般。
> - **集成 (Ensemble)**：决策树高方差低偏差 → 天然适合集成。要降低集成预测的方差 $\text{Var}[\overline X] = \rho \sigma^2 + \frac{1-\rho}{n} \sigma^2$——就要想办法同时降 $\rho$（让基模型去相关）和升 $n$。
> - **Bagging (Bootstrap Aggregation)**：从训练集 $S$ **有放回**抽样得 $B$ 个 bootstrap 子集，各训练一棵决策树，预测取平均。靠"不同子集训练"降低各棵树之间的相关性 $\rho$。
> - **Random Forest = Bagging + DT + 额外随机化**：在每个 split 时只考虑**部分特征**——防止"所有树都用同一个强特征进行分裂"，从而进一步降 $\rho$。
> - **Boosting (Adaboost)**：串行训练，每轮**提升被错分样本的权重**；最终预测是各基分类器的加权和 $G(x) = \sum_m \alpha_m G_m(x)$


## 引子

前面几讲学的都是**线性模型**（线性回归、逻辑回归、SVM 等）。它们的决策边界都是线性的，遇到下图中这种数据就束手无策（当然也可以比如说用 SVM 把数据投影到高维空间，但是比较麻烦）：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec10_DT_dataset_1.webp" alt="dataset 1" width="80%" loading="lazy" decoding="async" /></div>

这一讲进入**非线性模型**的第一个代表——**决策树 (Decision Tree)**。然后会讲它的"加强版"——**集成方法 (Ensemble Methods)**：Bagging / Random Forest / Boosting。

---


## 1. Decision Tree

### 1.1 基本思想：贪心、自顶向下、递归分裂

决策树的核心思想非常朴素：**对特征空间贪心、自顶向下、递归地划分**，每个叶节点区域最后给出一个预测。

对于上面那张图，决策树可能会这样分：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec10_DT_eg1.webp" alt="decision tree on dataset 1" width="55%" loading="lazy" decoding="async" /></div>

最终得到的区域划分像这样：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec10_DT_dataset_1_split.webp" alt="dataset 1 split" width="80%" loading="lazy" decoding="async" /></div>

### 1.2 分裂函数

对父区域 $R_p$，我们找一个分裂函数

$$
S_p(j, t)
$$

其中 $j$ 是**特征编号**，$t$ 是**阈值**。输出是父区域切成的两个子区域：

$$
S_p(j, t) = \big(\{X \mid X_j < t,\, X \in R_p\},\ \{X \mid X_j > t,\, X \in R_p\}\big)
$$

记两个子区域为 $R_1, R_2$。



### 1.3 如何选分裂？——损失函数

#### ① 第一种思路：Misclassification Loss

设总共 $C$ 个类，$\hat P_c$ 是区域 $R$ 中属于类 $c$ 的样本占比。**误分类损失**定义为

$$
L_{\text{misclass}} = 1 - \max_c \hat P_c
$$

> 直觉：选叶节点预测时按多数表决，$1 - \max_c \hat P_c$ 就是分类错误的比例。

我们想选一个 split，让"父区域损失 - 子区域加权损失" 这个”损失增益“值最大：

$$
\max_{j, t}\; L(R_p) - \left(\frac{|R_1|}{|R_p|} L(R_1) + \frac{|R_2|}{|R_p|} L(R_2)\right)
$$

由于 $L(R_p)$ 已经定下来了，等价于最小化**子区域加权损失**：

$$
\min_{j, t}\; \frac{|R_1|}{|R_p|} L(R_1) + \frac{|R_2|}{|R_p|} L(R_2)
$$

#### ② Misclassification Loss 不够用 ！

来看一个例子：900 个正例 + 100 个负例。

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec10_misclassification_simple-eg1_dataset.webp" alt="dataset 2" width="50%" loading="lazy" decoding="async" /></div>

两种 split：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec10_misclassification_simple-eg1_split1.webp" alt="dataset 2 split 1" width="80%" loading="lazy" decoding="async" /></div>

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec10_misclassification_simple-eg1_split2.webp" alt="dataset 2 split 2" width="80%" loading="lazy" decoding="async" /></div>

直觉上 split ② 更好（多分离出了一些正例）。但 misclassification loss 算出来两者**完全一样**：

$$
\frac{|R_1|}{|R_p|} L(R_1) + \frac{|R_2|}{|R_p|} L(R_2) = \frac{800}{1000} \cdot \frac{100}{800} + 0 = 0.1
$$

$$
\frac{|R_1'|}{|R_p|} L(R_1') + \frac{|R_2'|}{|R_p|} L(R_2') = \frac{500}{1000} \cdot \frac{100}{500} = 0.1
$$

**misclassification loss 没法分辨这两个 split**——它的曲线不是严格凹的，某些 split 之后 loss 不下降。


#### ③ Cross-Entropy Loss（推荐用）

$$
L_{\text{cross}} = -\sum_c \hat P_c \log_2 \hat P_c
$$

> 直观：来自信息论——"告诉一个样本它属于哪个类"所需的信息 bit 数。

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec10_CE-loss_curve1.webp" alt="cross-entropy loss curve" width="80%" loading="lazy" decoding="async" /></div>

这是一条**严格凹**曲线。如果两个子区域 $R_1, R_2$ **样本数相同**，那 loss 变化量就是

$$
L(R_p) - \frac{L(R_1) + L(R_2)}{2} > 0
$$

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec10_CE-loss_curve2.webp" alt="CE curve with two regions" width="80%" loading="lazy" decoding="async" /></div>

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec10_CE-loss_curve3.webp" alt="CE curve midpoint" width="80%" loading="lazy" decoding="async" /></div>

由于样本数相同，所以两个子区域的 $\hat P_c$ 平均起来就是父区域的 $\hat P_c$：

$$
\hat P_c(R_p) = \frac{\hat P_c(R_1) + \hat P_c(R_2)}{2}
$$

> **关键**：因为 cross-entropy 是**严格凹**的，无论 $R_1, R_2$ 样本量是否均匀，每次 split 都能保证 loss 下降——这就是我们要的。

用上面的 900 / 100 例子验证：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec10_CE-loss_curve_midpoint_eg.webp" alt="CE loss midpoint verification" width="70%" loading="lazy" decoding="async" /></div>

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec10_loss-change_on_CE-loss_curve.webp" alt="loss change on CE curve" width="100%" loading="lazy" decoding="async" /></div>

#### ④ Gini Loss

$$
L_{\text{Gini}} = \sum_c \hat p_c\, (1 - \hat p_c)
$$

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec10_Gini-loss_curve.webp" alt="Gini loss curve" width="80%" loading="lazy" decoding="async" /></div>

也是**严格凹**的（实际上严格凸也是——这里不严格区分凸凹的关键在于"midpoint 处的值低于两端均值"），所以行为和 cross-entropy 一样。

> **对比**：misclassification 曲线是平的——这意味着某些 split 不带来任何 loss 下降。

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec10_misclassification-loss_curve.webp" alt="misclassification loss curve" width="70%" loading="lazy" decoding="async" /></div>



### 1.4 回归树 (Regression Tree)

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec10_regression-tree_dataset.webp" alt="regression tree dataset" width="85%" loading="lazy" decoding="async" /></div>

回归任务下，叶节点的预测输出不再是"多数类"，而是**所有落到这个区域里的样本目标值的均值**：

$$
\hat y_m = \frac{\sum_{i \in R_m} y_i}{|R_m|}
$$

相应的损失函数用**平方误差**：

$$
L_{\text{squared}} = \frac{\sum_{i \in R_m} (y_i - \hat y_m)^2}{|R_m|}
$$

> 在平方误差意义下，区域内的**最优常数预测就是均值**——所以叶节点直接取均值是合理的。



### 1.5 类别变量 (Categorical Variables)

回归树同样能处理类别变量：把一个 $Q$ 类的集合分成 2 个子集，可能的 split 数有 $(2^{Q-1} - 1)$ 种。

> **实践中的结论（技巧）**：对于分类任务（用 Gini 或 cross-entropy），最优的二分方式一定可以通过**按正例比例（或回归任务中的均值）排序后，在相邻位置之间切分**得到。所以只需要先排序，再做一次线性扫描就能找到最优二分，不需要考虑数学上所有可能的 split 方案。


### 1.6 正则化

决策树如果不加任何限制一直往下分，每个样本都会落到唯一的叶节点——这就是**严重过拟合**。所以需要正则化：

**启发式的正则化方法**：
1. **设定一个最小叶节点大小**：叶节点样本数低于阈值就停止分裂
2. **设定决策树的最大深度**：限制树的高度
3. **设定决策树的最大节点数**：限制节点总数
4. **设定一个最小 loss 下降值**：loss 下降不够多就放弃 split（通常不太好用）
5. **后剪枝 (Post-pruning)**：先让决策树完全生长，再用验证集评估每个内部节点"剪掉子树变成叶节点"是否更好——**自底向上考虑剪枝**，对每个候选剪枝节点用验证集评估，剪后验证误差不升反降就剪

> **为什么有时鼓励"先长满再剪枝"**：强制早停可能丢掉"几个分裂组合起来"才能产生的提升。后剪枝能避免这个问题。


### 1.7 运行时间

- **测试时**：$O(d)$（从根到叶，深度 $d$）
- **训练时**：$O(n f d)$（每个样本经过 $d$ 层，每层扫 $f$ 个特征）

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec10_DT_depth.webp" alt="decision tree depth" width="85%" loading="lazy" decoding="async" /></div>

> 最优情况是平衡二叉树，$d < \log_2 n$；但实践中很少严格平衡。


### 1.8 缺点：不适合加性可分

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec10_DT_dataset3.webp" alt="dataset 3" width="80%" loading="lazy" decoding="async" /></div>

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec10_DT_no-additive.webp" alt="decision tree on dataset 3" width="80%" loading="lazy" decoding="async" /></div>

左图这种**斜对角可分**的数据，线性 / 逻辑回归一条黑虚线就能搞定；但决策树只能沿着坐标轴画阶梯状边界，要用很多次分裂才能近似——边界很粗糙。

> **根本原因**：决策树的每次分裂边界**只能平行于某个坐标轴**，无法直接画出一条像 $\theta^T x$ 那样带斜率的线。


### 1.9 Recap

**优点**：
1. 可解释
2. 非黑箱
3. 能直接处理类别变量（无需独热编码）
4. 快

**缺点**：
1. **高方差**（对数据噪声敏感、容易过拟合）
2. 不擅长加性可分
3. 前两个缺点导致**单棵树预测精度一般**

> **但！** 通过**集成方法**可以大幅提升决策树的性能。

---




## 2. Ensemble Methods （集成方法）

### 2.1 数学基础：均值估计的方差

先回顾两个有用的公式。

**情况 ①：$X_1, X_2, \dots$ i.i.d.**，$\text{Var}[X_i] = \sigma^2$，则

$$
\text{Var}\!\left[\overline X\right] = \text{Var}\!\left[\frac{1}{n} \sum_i X_i\right] = \frac{\sigma^2}{n}
$$

**情况 ②：$X_i$ 之间有相关性**，相关系数 $\rho$，则

$$
\text{Var}\!\left[\overline X\right] = \rho \sigma^2 + \frac{1-\rho}{n} \sigma^2
$$

> 两个极端：$\rho = 0$ 时退化成 i.i.d. 情况；$\rho = 1$ 时 $\text{Var}[\overline X] = \sigma^2$，**取平均也没用**——所有 $X_i$ 完全一样。


### 2.2 决策树为什么要做集成？

回顾 Lec 9 的误差分解：

$$
\text{总误差} = \text{Bias} + \text{Variance} + \text{Inreducible error}
$$

决策树是**高方差、低偏差**的模型 → 集成的目标主要是**降方差**。

把刚才数学基础 §2.1 的公式翻译成决策树的语言：

| 符号 | 含义 |
|---|---|
| $X_i$ | 第 $i$ 个模型在某个测试点上的预测 |
| $n$ | 集成中模型的数量 |
| $\overline X$ | 集成后的预测（$n$ 个模型预测的平均） |
| $\rho$ | 模型之间预测的相关系数 |
| $\text{Var}[\overline X]$ | 集成后预测的方差 |

根据公式 $\text{Var}[\overline X] = \rho\sigma^2 + \frac{1-\rho}{n}\sigma^2$，降低集成方差有**两条路**：
1. **$n$ 变大**：训练更多模型
2. **$\rho$ 变小**：让模型之间去相关


### 2.3 集成方法的几条路径

1. **用不同算法**：神经网络 + 随机森林 + SVM 等取平均——不实际，开销太大
2. **用不同训练集**：收集新数据 → 成本太高
3. **Bagging（Bootstrap Aggregation）**：本讲重点
4. **Boosting（Adaboost / XGBoost）**：本讲重点

---




## 3. Bagging —— Bootstrap Aggregation

### 3.1 Bootstrap

**Bootstrap** 是一种统计学里估计量不确定性的方法。

**思路**：设真实分布是 $P$，训练集 $S \sim P$。理想情况下，我们想画很多个不同的 $S_1, S_2, \dots$ 训练很多模型——但这要无穷多数据。

Bootstrap 的近似做法：**假设 $S = P$**，从 $S$ 里**有放回**地抽 $N$ 次得到"Bootstrap 样本"：

$$
Z \sim \hat P_S
$$

其中 $\hat P_S$ 是由 $S$ 定义的**经验分布**。

然后在每个 $Z_m$ 上分别训练一个模型 $G_m$，最后看这些模型预测的**变异性**——这就是对不确定性的一种度量。


### 3.2 Bagging 流程

$$
G_{\text{bag}}(x) = \frac{1}{M} \sum_{m=1}^M G_m(x)
$$

流程：**取 bootstrap 样本 → 分别训练 → 把所有模型预测聚合**。


### 3.3 为什么 Bagging 有用？——Bias-Variance 分析

$$
\text{Var}[\overline X] = \rho \sigma^2 + \frac{1-\rho}{n} \sigma^2 \quad (n = M)
$$

$$
= \rho \sigma^2 + \frac{1-\rho}{M} \sigma^2
$$

- **Bootstrap 抽样让每个模型看到不同的训练集** → 模型之间**去相关**，$\rho$ 下降
- 增大 $M$ → $\frac{1-\rho}{M}\sigma^2$ 项进一步下降

**好处**：增大 $M$ 只降方差，**不会带来过拟合**。

**代价**：bootstrap 抽样会导致每个模型看到的训练集变小一些 → 可能略增 bias。但相对收益来说 bias 的影响小很多。

---



## 4. Random Forest

决策树高方差低偏差 → **天然适合 Bagging**。

Random forest = Bagging + Decision Tree + **额外随机化**

$$
\text{Random forests} = \text{Bagging} + \text{Decision Tree} + \text{额外的随机化}
$$

**两个维度的随机化**：
1. **样本随机化**（Bagging 自带）：每个树看不同 bootstrap 子集
2. **特征随机化**（Random Forest 特有）：每次 split 时只在一个**小的特征子集**里挑最优特征

> 特征随机化的目的也是降 $\rho$。如果数据里有一个非常强的特征，所有树可能都在很多层用这个特征分裂，导致树之间结构相似、$\rho$ 仍很大。强制每次只看部分特征就能打破这种同步。

---



## 5. Boosting

### 5.1 Bagging vs Boosting

| | Bagging | Boosting |
|---|---|---|
| 训练方式 | **并行**训练多个独立模型 | **串行**训练，每个新模型修正前一个的错误 |
| 主要降 | Bias（顺便也降一些 Variance） | **Variance** |
| 最终预测 | 各模型平均 | 各模型**加权和** |

Boosting 的**加性 (additive) 本质**：最终的模型是之前多个基模型的**加权和**。


### 5.2 Adaboost 例子

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec10_boosting_dataset.webp" alt="dataset 4" width="70%" loading="lazy" decoding="async" /></div>

用**深度为 1 的决策树 (decision stump)** 做基分类器——通过限制深度来**降低单个模型的方差、允许较高的偏差**，这正好适合 boosting。

第 1 轮：训出一个决策边界：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec10_boosting_decision-boundary1.webp" alt="boosting decision boundary 1" width="70%" loading="lazy" decoding="async" /></div>

标出错分的样本：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec10_boosting_decision-boundary1_mistakes.webp" alt="mistakes identified" width="75%" loading="lazy" decoding="async" /></div>

下一轮：**提高错分样本的权重**，在这个修改后的训练集上训下一个 stump：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec10_boosting_decision-boundary2.webp" alt="boosting decision boundary 2" width="75%" loading="lazy" decoding="async" /></div>

递归重复这个过程。


### 5.3 Adaboost 的权重

第 $m$ 个分类器 $G_m$ 的权重：

$$
\alpha_m = \log \frac{1 - \text{err}_m}{\text{err}_m}
$$

> 错得越少（$\text{err}_m$ 越小），$\alpha_m$ 越大。

最终的总分类器：

$$
G(x) = \sum_m \alpha_m G_m(x)
$$

每个 $G_m$ 在重新加权的训练集上训练。

> **直观**：每个基分类器就像一个**阶跃函数**；多个阶跃函数的线性组合最终能拼成**非线性的、阶梯状的决策边界**。

---



## 参考资料

- [CS229 Boosting](https://cs229.stanford.edu/notes2021fall/lecture11-boosting.pdf) — Boosting 课件 pdf
- [CS229 Decision Trees](https://cs229.stanford.edu/notes2021fall/lecture11-decision-trees.pdf) — Decision Trees 课件 pdf
- [CS229 Ensemble Techniques](https://cs229.stanford.edu/notes2021fall/section8notes-ensembling-techniques.pdf) — Ensemble Techniques 课件 pdf
- [CS229 2018](https://www.bilibili.com/video/BV1b4anzMEUv) — B 站上 Andrew Ng 的讲课视频 CS229 Lec 10
- [Bishop, Pattern Recognition and Machine Learning, Ch.14](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) — 涵盖 Tree-based Models, Boosting, 