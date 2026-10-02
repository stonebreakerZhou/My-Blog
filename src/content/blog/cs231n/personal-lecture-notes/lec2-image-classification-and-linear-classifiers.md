---
title: "CS231n : Lec 2 — 图像分类与线性分类器"
description: CS231n Lecture 2 学习笔记，覆盖图像分类任务、KNN（K-Nearest Neighbor）与线性分类器（L1/L2 距离度量、超参数设置），以及 Softmax 和 Hinge Loss。
pubDate: 2026-09-30
series: cs231n
subSeries: personal-lecture-notes
order: 2
categories:
  - CS231n
  - KNN
  - Linear Classifier
  - Softmax
  - Hinge Loss
---

> **TL;DR**:
> - **图像分类任务**：给定一张图片预测其类别，受视角、光照、遮挡、形变等多重因素影响
> - **数据驱动方法**：KNN（基于距离的非参数方法）+ 线性分类器（基于参数的判别方法）
> - **KNN**：用 $L_1$（Manhattan）或 $L_2$（Euclidean）距离找最近的 $k$ 个邻居做投票
> - **超参数设置**：3 种思路（训练集调 / 测试集调 / 验证集调）+ $k$-fold Cross-Validation
> - **线性分类器**：$f(x, W) = Wx + b$
> - **损失函数**：Softmax + Cross-Entropy（概率解释）vs Hinge Loss（SVM，间隔解释）


## 引子

本节切入计算机视觉第一个具体任务：**图像分类（Image Classification）**

我们从最朴素的两类分类器入手：**K-Nearest Neighbor**（非参数）和**线性分类器**（参数），并由此引出整个 Lec 2 的核心问题——**怎么定义"分类得好不好"**（损失函数），以及**怎么在训练集上学到好参数**（优化）。后者会留到 Lec 3，本节重点放在损失函数上。

---



## 1. 图像分类任务

**任务定义**：给定一张输入图片，预测它属于 $K$ 个预定义类别中的哪一类。

### 1.1 七大挑战

一张图片从"看起来像猫"到"算法判定是猫"，中间要跨越七重障碍：

| # | 挑战 | 说明 |
|---|---|---|
| ① | Viewpoint variation（视角变化）| 同一个物体从不同角度看像素完全不同 |
| ② | Illumination（光照）| 光线强弱改变像素值 |
| ③ | Background Clutter（背景干扰）| 物体和背景颜色相近 |
| ④ | Occlusion（遮挡）| 部分被挡住 |
| ⑤ | Deformation（形变）| 物体本身份子变形 |
| ⑥ | Intraclass variation（类内变化）| 同类物体长相差很大 |
| ⑦ | Context（上下文）| 物体依赖周围场景判断 |

### 1.2 数据驱动方法

不像规则式编程可以"硬编码"识别猫的规则，**机器学习**采用数据驱动：

```
① Collect a dataset of images and labels
② Use Machine Learning algorithms to train a classifier
③ Evaluate the classifier on new images
```

---






## 2. First classifier: Nearest Neighbor

**核心思想**：计算查询数据与训练数据（带标签）的距离，把最近邻的标签作为预测。

### 2.1 距离度量

**$L_1$ distance（Manhattan）**：

$$
d_1(I_1, I_2) = \sum_p \left| I_1^p - I_2^p \right|
$$

即所有像素差值的**绝对值**之和。

**$L_2$ distance（Euclidean）**：

$$
d_2(I_1, I_2) = \sqrt{\sum_p \left( I_1^p - I_2^p \right)^2 }
$$

即像素差值的**平方和**的平方根。

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs231n/Lec2_L1_Manhattan_distance.webp" alt="L1 Manhattan distance" width="45%" loading="lazy" decoding="async" /></div>

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs231n/Lec2_L2_Euclidean_distance.webp" alt="L2 Euclidean distance" width="45%" loading="lazy" decoding="async" /></div>

**两种度量的差异**：

|       | $L_1$               | $L_2$         |
| ----- | ------------------- | ------------- |
| 边界形状  | 方格线                 | 圆弧            |
| 对轴的依赖 | **依赖于坐标轴**（旋转会改变距离） | 不依赖（任意旋转距离不变） |
| 适用场景  | 特征维度"有特定意义"时        | 特征任意时         |

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs231n/Lec2_k_nearest_neighbors_with_L1-metric_comparison.webp" alt="kNN L1 vs L2 决策边界对比" width="100%" loading="lazy" decoding="async" /></div>

> **直觉**：$L_2$ 的等距线是圆，$k$-NN 拼接出的决策边界更光滑；$L_1$ 的等距线是方格，决策边界呈"齿轮状"。


### 2.2 K-Nearest Neighbor

**朴素 1-NN 的问题**：

- 预测时遍历 $N$ 个训练样本 → **$O(N)$** 时间
- 对噪声/异常点非常敏感（中间黄色孤立点会把整片区域划错类）

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs231n/Lec2_1_nearest_neighbor.webp" alt="1-nearest neighbor" width="60%" loading="lazy" decoding="async" /></div>


**$k$-NN 改进**：取**最近的 $k$ 个**邻居，**多数投票**决定标签。这让决策边界更平滑、对异常点更鲁棒。

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs231n/Lec2_k_nearest_neighbors.webp" alt="k-nearest neighbor" width="100%" loading="lazy" decoding="async" /></div>

> 图中**白色区域**是"无法判断"的区域——多个类别票数相等。


### 2.3 超参数设置

超参数（$k$、距离度量）怎么选？有 4 种思路：

| 思路 | 做法 | 问题 |
|---|---|---|
| **Idea 1** | 选在训练集上表现最好的 | ❌ $k=1$ 永远在训练集上完美（过拟合） |
| **Idea 2** | 选在测试集上表现最好的 | ❌ 无法泛化 |
| **Idea 3** | 划分 train / val / test，在 val 调 | ⚠️ 常用，但牺牲了一部分训练数据 |
| **Idea 4** | **$k$-fold Cross-Validation** | ✅ 小数据集尤其有用，但深度学习不常用 |

> **关键原则**：**测试集只能最后用一次**，否则就是在"偷看测试集"，所有调参都基于 val。


### 2.4 实践结论

> **在实践中，KNN with pixel distance 几乎从不用于真实图像分类**——像素级距离提供的信息量太低（比如一张图片平移所有像素完全错位）。Lec 2 这里把它作为"最简单的分类器"引出概念，真正的图像分类要用 CNN。

---







## 3. Linear Classifier

第二个基础分类器——**参数化方法**。

### 3.1 模型形式

$$
f(x, W) = W x + b \;\longrightarrow\; 输出 10 \text{ 个类别分数}
$$

其中：
- $x \in \mathbb{R}^D$ 是输入（展平的图像，$D = 32 \times 32 \times 3 = 3072$）
- $W \in \mathbb{R}^{K \times D}$ 是权重矩阵（$K$ = 类别数）
- $b \in \mathbb{R}^K$ 是偏置向量

训练目标：**学习 $W, b$ 使得 $f(x, W)$ 在训练集上预测得准**。


### 3.2 三个等价视角

**① 代数视角（algebraic viewpoint）**

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs231n/Lec2_linear_classifier_illustration.webp" alt="线性分类器代数视角" width="80%" loading="lazy" decoding="async" /></div>

矩阵乘法 $Wx$——每一行对应一个类别的权重。

**② 视觉视角（visual viewpoint）**

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs231n/Lec2_Linear_classifier_visual_viewpoint.webp" alt="线性分类器视觉视角" width="60%" loading="lazy" decoding="async" /></div>

把 $W$ 的每一行**重塑**为图像——可以直观看到"模型在找什么模式"。

**③ 几何视角（geometric viewpoint）**

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs231n/Lec2_Linear_classifier_geometric_viewpoint.webp" alt="线性分类器几何视角" width="80%" loading="lazy" decoding="async" /></div>

输入空间里的**超平面**把不同类别的样本分开。

---







## 4. 损失函数：怎么定义"分类得好不好"

有了 $f(x, W) = Wx + b$ 输出原始分数 $s$，接下来要回答：

> 怎么把"分数"变成"预测"？怎么衡量"预测得不准"？

Lec 2 介绍两个经典答案：**Softmax Classifier**（概率解释）vs **SVM Classifier**（间隔解释）。


### 4.1 Softmax Classifier（Cross-Entropy Loss）

我们希望把分类器的原始输出分数 $\mathbf{s}$ 解释为概率。

给定输入 $\mathbf{x} \in \mathbb{R}^D$，线性分类器产生原始分数：

$$
\mathbf{s} = f(\mathbf{x}; \mathbf{W}, \mathbf{b}) = \mathbf{W}\mathbf{x} + \mathbf{b} \quad \in \mathbb{R}^K
$$

对原始分数向量 $\mathbf{s}$ 应用 softmax 函数，得到概率向量 $\mathbf{p}$：

$$
p(Y = k \mid x) = \frac{e^{s_k}}{\sum_{j=1}^K e^{s_j}}, \quad k = 1, \dots, K
$$

现在我们思考怎样定义目标函数：


**法一：**

输出的概率向量可以看作是我们得到的预测概率分布，我们可以计算真实概率分布（对应向量应该是独热的）与这个预测概率分布之间的 KL 散度 $D_{KL}(P \| Q)$ 值：

$$
\begin{aligned}
D_{KL}(P \| Q) &= \sum_k P_k \log \frac{P_k}{Q_k} \\
&= \sum_k P_k \log P_k - \sum_k P_k \log Q_k
\end{aligned}
$$

其中：

$$
H(P) = -\sum_k P_k \log P_k
$$

是真实分布 $P$ 的熵。

$$
H(P, Q) = -\sum_k P_k \log Q_k
$$

是交叉熵。

所以：

$$
D_{KL}(P \parallel Q) = H(P, Q) - H(P)
$$

由于此时 $P$ 是真实标签分布，它不依赖于模型参数。所以 $H(P)$ 是一个常数：

$$
\arg\min_Q D_{KL}(P \parallel Q) = \arg\min_Q H(P, Q)
$$

因此我们实际上就是使用的交叉熵损失函数公式。



**法二：**

我们用最大似然估计（**MLE**）：

整个数据集出现的似然是每个样本正确类别概率的连乘：

$$
L(W, b) = \prod_{i=1}^N p(y_i \mid x_i; W, b)
$$

MLE 的目标就是：

$$
\max_{W,b} L(W, b) = \max_{W,b} \prod_{i=1}^N p(y_i \mid x_i; W, b)
$$

取对数后取负为：

$$
-\ell(W, b) = -\log L(W, b) = -\sum_{i=1}^N \log p(y_i \mid x_i; W, b)
$$

则最终的目标就是：

$$
\min_{W,b} -\ell(W, b) = \min_{W,b} -\sum_{i=1}^N \log p(y_i \mid x_i; W, b)
$$

之前对单个样本，真实分布 $P_i$ 是 one-hot，模型预测分布是 $Q_i$，其交叉熵为：

$$
\begin{aligned}
H(P_i, Q_i) &= -\sum_{k=1}^K P_{i,k} \log Q_{i,k} \\
&= -\log p(y_i \mid x_i; W, b)
\end{aligned}
$$

所以对于这个分类任务，最大似然估计与交叉熵会推导出相同的目标函数进行优化。



### 4.2 Hinge Loss（SVM Classifier）

前面讲了 Softmax 分类器：它把线性分数 $s = W x + b$ 通过 Softmax 变成概率，再用交叉熵损失衡量预测分布和真实 one-hot 分布的差距。

现在换一个思路：我们不一定非要输出概率，也可以只要求正确类别的分数比错误类别的分数高出至少一个间隔。

这就是 SVM（Support Vector Machine）所使用的 合页损失（hinge loss）。

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs231n/Lec3_SVM_vs_Softmax.webp" alt="Softmax vs SVM" width="100%" loading="lazy" decoding="async" /></div>

**Hinge Loss**

对于第 $i$ 个样本，正确类别是 $y_i$，线性分类器给出的分数是 $s$，SVM 损失定义为：

$$
L_i = \sum_{j \neq y_i} \max(0, s_j - s_{y_i} + \Delta)
$$

通常取间隔 $\Delta = 1$，所以：

$$
L_i = \sum_{j \neq y_i} \max(0, s_j - s_{y_i} + 1)
$$

含义：
① 如果正确类别的分数 $s_{y_i}$ 比某个错误类别 $s_j$ 高出至少 1，那么这一项为 0，没有损失；
② 否则，损失等于"还差多少才满足间隔"。

整个训练集的损失是：

$$
L = \frac{1}{N} \sum_{i=1}^N L_i
$$

但是我们选用 hinge loss 会出现一个问题，即满足损失值相同的权重值 $W$ 并非唯一！有些时候 $W$ 与 $2 W$ 都会得到相同的损失值！

e.g：一个损失为 0 的例子

假设某个样本的正确类别是 frog，分数如下：

| 类别 | 分数 |
|---|---|
| cat | 1.3 |
| frog | 4.9 |
| car | 2.0 |

正确类别 frog 的分数是 4.9，对 cat 和 car 分别计算：

$$
\max(0, 1.3 - 4.9 + 1) = \max(0, -2.6) = 0
$$

$$
\max(0, 2.0 - 4.9 + 1) = \max(0, -1.9) = 0
$$

所以这个样本的 SVM 损失为 0。正确类别已经比错误类别高出超过 1，满足间隔要求。

现在把权重 $W$ 变成 $2W$，所有分数也翻倍：

| 类别 | 原分数 | $2W$ 下的分数 |
|---|---|---|
| cat | 1.3 | 2.6 |
| frog | 4.9 | 9.8 |
| car | 2.0 | 4.0 |

再算 SVM 损失：

$$
\max(0, 2.6 - 9.8 + 1) = \max(0, -6.2) = 0
$$

$$
\max(0, 4.0 - 9.8 + 1) = \max(0, -4.8) = 0
$$

我们会发现 $W$ 与 $2 W$ 下损失值均为 0！

但 $W$, $2 W$ 对应两种截然不同的模型：2W 的权重更大，对输入变化更敏感；可能更容易过拟合；数值上也可能更不稳定。

所以最简单的方式就是加入 L2 正则化！！！这样模型会选取 W 而非 2W！

---





## 参考资料

- [CS231n Lecture 2 — Image Classification](https://cs231n.stanford.edu/slides/2026/lecture_2.pdf) — 2026 slide PDF
- [CS231n 2025 spring](https://www.bilibili.com/video/BV1YJ3PzLEiW) — CS231n Spring 2025 视频
- [K-Nearest Neighbors (Wikipedia)](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) — $k$-NN 算法的更详细介绍
