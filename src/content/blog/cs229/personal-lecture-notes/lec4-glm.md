---
title: "CS229 : Lec 4 — 广义线性模型"
description: CS229 Lecture 4 学习笔记，从感知机学习算法起步，覆盖指数族分布，再到广义线性模型（GLM）的三大假设与推导，从广泛视角看待前两节课的内容，并引出 Softmax 回归作为多分类推广。
pubDate: 2026-08-06
series: cs229
subSeries: personal-lecture-notes
order: 5
categories:
  - CS229
  - 广义线性模型
  - Softmax 回归
---

> **TL;DR**:
> - **感知机**：把 logistic regression 的 sigmoid 替换成阈值（阶跃）函数，更新规则形式上与 LMS 相同，但**本质上不属于概率模型**，不可用极大似然推导
> - **指数族分布**：$p(y; \eta) = b(y) \exp(\eta^T T(y) - a(\eta))$；Bernoulli、Gaussian、Poisson、Gamma、Beta 等都属此族
> - **GLM 三大假设**：① $y \mid x; \theta \sim \text{ExponentialFamily}(\eta)$；② $\eta = \theta^T x$；③ $h(x) = E[y \mid x]$（即输出的预测 = 条件期望）
> - **均值-参数恒等式**：$h(x) = E[y \mid x] = \frac{\partial a(\eta)}{\partial \eta}$，把线性模型部分的输出 $\eta$ 和最终预测用一个公式直接连接
> - **统一更新规则**：$\theta_j := \theta_j + \alpha (y^{(i)} - h_\theta(x^{(i)})) x_j^{(i)}$，对所有 GLM 都成立
> - **Softmax 回归**：多分类问题的推广，损失函数用交叉熵（cross entropy）

## 引子

CS229 Lecture 4 主要话内容是**广义线性模型**（Generalized Linear Models, GLM）。我们会先复习一个看起来像 logistic regression、但其实完全不是概率模型的算法——**感知机**（perceptron）；然后介绍一个相当基础的工具——**指数族分布**（exponential family），用它统一 Gaussian、Bernoulli、Poisson 等常见分布；最后用三个假设推导出 GLM 的整体框架，并把它应用于最小二乘、logistic 回归，以及多分类的 **Softmax 回归**。


---

## 1. 感知机学习算法

Logistic regression 使用的 sigmoid 函数 $g(z) = 1 / (1 + e^{-z})$ 把整个实数轴 $(- \infty, + \infty)$ 压缩到 $(0, 1)$。设想把 logistic regression 稍微改一下，让它**只输出 0 或 1**——一个自然的做法是改用**阈值函数**（unit step function）：

$$
g(z) = \begin{cases}
1 & \text{if } z \ge 0 \\
0 & \text{if } z < 0
\end{cases}
$$

<div style="text-align: center;"><img src="/My-Blog/blog-images/unit_step_func.webp" alt="unit step function" width="280" loading="lazy" decoding="async" /></div>

然后仍然定义 hypothesis $h_\theta(x) = g(\theta^T x)$，但这里的 $g$ 是上面的阈值函数，再使用下面的更新规则（形式上和 logistic regression 一模一样）：

$$
\theta_j := \theta_j + \alpha \left( y^{(i)} - h_\theta(x^{(i)}) \right) x_j^{(i)}
$$

就得到了 **感知机学习算法**（perceptron learning algorithm）。

仔细看这条更新规则：如果预测正确，$y^{(i)} - h_\theta(x^{(i)}) = 0$，参数不更新；否则 $y^{(i)} - h_\theta(x^{(i)}) = \pm 1$。具体来看：

① 若 $y^{(i)} = 0$，$h_\theta(x^{(i)}) = 1$，即把负例错判成了正例——此时执行 $\theta_j := \theta_j - \alpha x_j$，相当于给 $\vec{\theta}$ 加上 $-\alpha \vec{x}$，让 $\vec{\theta}$ 与 $\vec{x}$ 之后尽量呈**负的点积**（即方向相反）。

② 类似地，若把正例错判成了负例，则给 $\vec{\theta}$ 加上 $\alpha \vec{x}$，让 $\vec{\theta}$ 与 $\vec{x}$ 之后尽量呈**正的点积**（即方向相同）。

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec4_perceptron.webp" alt="perceptron update rule" width="400" loading="lazy" decoding="async" /></div>

> **题外话**：20 世纪 60 年代，感知机被认为是"单个神经元工作方式"的粗糙模型。因为它足够简单，它也成为后面学习理论讨论的起点。但要强调：**感知机和 logistic regression、最小二乘线性回归**只是表面上相似，**本质上完全不同**。特别是：很难给感知机的预测附加一个有意义的概率解释；感知机学习算法也不能被推导为某个极大似然估计。


---

## 2. 指数族分布

我们先定义一下**指数族分布**（exponential family distributions）。如果一个分布能写成下面的形式，我们就说这类分布属于指数族：

$$
p(y; \eta) = b(y) \exp\!\left( \eta^T T(y) - a(\eta) \right)
$$

等价地：

$$
p(y; \eta) = \frac{b(y) \, e^{\eta^T T(y)}}{e^{a(\eta)}}
$$

其中：

- $y$——数据
- $\eta$——此分布的**自然参数**（natural parameter / canonical parameter）
- $T(y)$——**充分统计量**（sufficient statistic），通常就是 $y$ 本身
- $\eta$ 与 $T(y)$ 的维度应当匹配（要做向量点积）
- $b(y)$——basic measure（标量）
- $a(\eta)$——**对数分割函数**（log partition function），只与 $\eta$ 有关！！！（这一点很重要！）

$\exp(-a(\eta))$ 这个量本质上扮演**归一化常数**（normalization constant）的角色，确保 $p(y; \eta)$ 的总和或积分等于 $1$ （即：当参数 $\eta$ 固定，对于 $y$ 从负无穷到正无穷的积分之和为1），因此我们可以看出分母这个函数显然只与参数 $\eta$ 有关。

给定 $T$、$a$ 和 $b$，就定义了一个用 $\eta$ 参数化的分布族（family）；通过改变 $\eta$，就能得到这个族里的不同分布（也就是说这是一个单参数的分布，给定 $\eta$ 就能拿到分布的全部概率密度公式，进而推出更新公式）。

### 2.1 Bernoulli 分布属于指数族

回顾 Bernoulli 分布的密度函数：

$$
p(y; \phi) = \phi^y (1 - \phi)^{1-y}
$$

改写为指数族形式（技巧：凑成 log(exp) 的形式）：

$$
\begin{aligned}
p(y; \phi)
&= \exp\!\left( y \log \phi + (1 - y) \log(1 - \phi) \right) \\
&= \exp\!\left( \left( \log \frac{\phi}{1 - \phi} \right) y + \log(1 - \phi) \right)
\end{aligned}
$$

对比通用形式：

$$
p(y; \eta) = \frac{b(y) \, e^{\eta^T T(y)}}{e^{a(\eta)}}
$$

Bernoulli 对应的各个量为：

$$
\begin{aligned}
b(y) &= 1 \\
\eta &= \log \frac{\phi}{1 - \phi} \\
T(y) &= y \\
a(\eta) &= -\log(1 - \phi) = \log(1 + e^\eta)
\end{aligned}
$$

### 2.2 Gaussian 分布也属于指数族

假设 $\sigma^2 = 1$：

$$
\begin{aligned}
p(y; \mu)
&= \frac{1}{\sqrt{2\pi}} \exp\!\left( - \frac{(y - \mu)^2}{2} \right) \\
&= \frac{1}{\sqrt{2\pi}} \exp\!\left( - \frac{1}{2} y^2 \right) \cdot \exp\!\left( \mu y - \frac{1}{2} \mu^2 \right)
\end{aligned}
$$

于是：

$$
\begin{aligned}
b(y) &= \frac{1}{\sqrt{2\pi}} \exp\!\left( - \frac{y^2}{2} \right) \\
\eta &= \mu \\
T(y) &= y \\
a(\eta) &= \frac{\eta^2}{2} = \frac{\mu^2}{2}
\end{aligned}
$$

注意事项：在这里我们简化假设为 $\sigma^2 = 1$ , 实际上即使 $\sigma^2$ 是一个别的常数值，这个常数值也会顺带被之后线性模型的参数 $\theta$ 给学习到，所以简便起见我们就说成 $\sigma^2 = 1$

### 2.3 指数族的几个性质

- **MLE 关于 $\eta$ 是凹函数** $\Longleftrightarrow$ 负对数似然（NLL）是凸函数
- **均值**：

$$
E[y; \eta] = \frac{\partial a(\eta)}{\partial \eta}
$$

- **方差**：

$$
\mathrm{Var}[y; \eta] = \frac{\partial^2 a(\eta)}{\partial \eta^2}
$$

> 通常求分布的均值与方差要做积分，但这里只要做求导，更易操作。

针对不同类型的数据，我们可以采用指数族中的不同分布进行建模：

| 数据类型                  | 选用的分布                          |
| --------------------- | ------------------------------ |
| ① 实值（real number）     | Gaussian                       |
| ② 二分类（binary）         | Bernoulli                      |
| ③ 计数（count，1/2/3…）    | Poisson                        |
| ④ 正实数（$\mathbb{R}^+$） | Gamma, Exponential             |
| ⑤ 概率分布之上的概率分布         | Beta, Dirichlet（贝叶斯机器学习、统计里常见） |


---

## 3. 广义线性模型（GLMs）

设想一个分类或回归问题：要预测随机变量 $y$ 的值，作为 $x$ 的函数。要导出对应的**广义线性模型**，要对我们的模型——给定 $x$ 下 $y$ 的条件分布——做下面三个假设。本节我们会依次立下这三个假设，推出 GLM 的整体框架，最后看看如何从这个框架落到一个统一的学习算法上。

**假设 1**：

$$
y \mid x; \theta \sim \text{ExponentialFamily}(\eta)
$$

即给定 $x$ 和 $\theta$，$y$ 的分布属于指数族，是一个参数为 $\eta$ 的（单参数）指数分布。

**假设 2**：

$$
\eta = \theta^T x, \quad \theta \in \mathbb{R}^n, x \in \mathbb{R}^n
$$

自然参数 $\eta$ 与输入 $x$ 线性相关：$\eta = \theta^T x$。如果 $\eta$ 是向量，则 $\eta_i = \theta_i^T x$。

> 三个假设里，**假设 2 看起来最不像"假设"**，因此更合适把它视为我们在设计广义线性模型时的一个**设计选择**（design choice）。自然界或许并不真的遵循 $\eta = \theta^T x$，但我们把它当成一个设计准则——线性是最简单的起点；如果效果不够好，可以在 $x$ 上加非线性特征（比如 $\log x$、$x^2$）来强行变好。

**假设 3**：

$$
h(x) = E[y \mid x]
$$

给定 $x$，目标是预测对应这个 $x$ 的 $T(y) (=y)$ 的**期望值**，也就是 hypothesis $h(x)$ 要满足 $h(x) = E[y \mid x]$ ，即最后的输出应当是一个条件期望值。

例如在 logistic regression 中：

$$
\begin{aligned}
h_\theta(x)
&= \left[ p(y = 1 \mid x; \theta) \right] \\
&= \left[ 0 \cdot p(y = 0 \mid x; \theta) + 1 \cdot p(y = 1 \mid x; \theta) \right] \\
&= E[y \mid x; \theta]
\end{aligned}
$$

这三个假设/设计选择共同推导出了一个学习算法类别——**广义线性模型**（GLMs），它有一系列友好的理想性质：易于学习、对不同类型的 $y$ 分布建模都很高效。

GLM 的整体思路可以分成两部分：**模型 + 分布**。先看一张 mental map 把整个流水线串起来：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec4_GLM_mental_map.webp" alt="GLM mental map" width="540" loading="lazy" decoding="async" /></div>

流水线是这样的：拿到输入 $x$，由于我们假设模型是线性模型，于是模型通过可学习参数 $\theta$ 输出 $\theta^T x$ 作为参数 $\eta$（即自然参数），$\eta$ 传递给 Exponential Family 作为核心参数（分布类型的选择取决于任务：预测实值就选 Gaussian，预测 $\in \{0, 1\}$ 就选 Bernoulli……再相应地选择 $b(y)$、$a(\eta)$、$T(y)$），最终在所选分布上得到预测值 $h(x)$，定义为给定 $x$ 下的条件期望 $E[y \mid x]$。

而这个期望值恰好等于指数族分布的对数分割函数 $a(\eta)$ 的一阶导数：

$$
h(x) = E[y \mid x] = \frac{\partial a(\eta)}{\partial \eta}
$$

> 这个数学性质（指数族的**均值-参数恒等式**）是连接线性部分 $\eta$ 和最终预测值的重要桥梁！！！也就是后面要提到的 **canonical response function**。

剩下的就是**学习**（学习参数 $\theta$ ）：GLM 的训练目标是对样本做 **maximum likelihood**：

$$
\max_\theta \log P(y^{(i)}; \theta^T x^{(i)})
$$

注意优化的是线性模型里的参数 $\theta$，而不是分布里的 $\mu$、$\sigma^2$、$\eta$。

### 3.1 统一的更新规则

有趣的是，无论在做哪种 GLM 或选哪种分布，更新规则都是一样的：

$$
\theta_j := \theta_j + \alpha \left( y^{(i)} - h_\theta(x^{(i)}) \right) x_j^{(i)}
$$

> 这条公式总是成立的统一更新规则，本质上来自我们使用 MLE 推导出来的结果。

### 3.2 术语小结

$$
\eta \rightarrow \text{natural parameter}
$$

$$
\mu = E[y; \eta] = g(\eta) = \frac{\partial}{\partial \eta} a(\eta) \rightarrow g(): \text{canonical response function}
$$

$$
\eta = g^{-1}(\mu) \rightarrow g^{-1}(): \text{canonical link function}
$$

### 3.3 三层参数化

GLM 在学习时涉及到三层不同的"参数"，注意区分：

- ① **模型参数（model parameter）**：$\theta$
- ② **自然参数（natural parameter）**：$\eta$
- ③ **典范参数（canonical parameter）**：

$$
\begin{cases}
\phi & \text{— Bernoulli} \\
\mu, \sigma^2 & \text{— Gaussian} \\
\lambda & \text{— Poisson}
\end{cases}
$$

只要学习 GLM，我们学到的是 $\theta$（在线性模型中）。它们之间的关系是：

- $\theta^T x = \eta$（这是我们的设计选择）
- $g(\eta) =$ canonical parameter
- $g^{-1}(\text{canonical param}) = \eta$

### 3.4 用 GLM 看 logistic regression 与 线性回归

先看线性回归：输入 $x$，得到 $\theta^T x = \eta$。这里我们用 Gaussian 作为分布，所以 $\eta = \mu$。在 Gaussian 假设下，我们假设对每个 $x$，对应的 $y$ 服从方差为 1（注意：方差不为 1 时方差的大小可以被学习在 $\theta$ 中，所以简化起见我们通常令方差为 1）、均值为 $\theta^T x$ 的高斯分布：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec4_GLM_Guassian.webp" alt="GLM linear regression(Gaussian)" width="540" loading="lazy" decoding="async" /></div>

> 所以相当于**先假设存在上述数据分布规律**，而我们实际拿到的数据是这个规律下产生的样本，实际做的是一个**从右向左的倒推过程**，最终要找到合适的 $\theta$。

同理对 logistic 回归任务，我们也要做这样的倒推：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec4_GLM_logistic.webp" alt="GLM logistic regression(Bernoulli)" width="540" loading="lazy" decoding="async" /></div>

回顾 logistic regression：

$$
h_\theta(x) = E[y \mid x; \theta] = \phi
$$

（我们选了 Bernoulli 分布，所以这里的典范参数是 $\phi$）。

另外

> 为什么这里的 hypothesis 必须长成 sigmoid 的样子？
> 其实答案就藏在第 2.1 节里——
> 
> ① 首先，我们知道最后的输出是一个条件期望！对于伯努利分布而言，这个期望值就是 $\phi$ , 所以我们就是要找 $\phi$ 关于输入 $x$ 的函数关系！！！
> ② 对 Bernoulli 分布我们推出过 $\eta = \log(\phi / (1 - \phi))$。把这条公式两边对 $\phi$ 反解，立刻就得到下面的 $\phi = 1 / (1 + e^{-\eta})$ ， 再代入 $\eta = \theta^T x$ 就得到了 $\phi$ 关于 $x$ 的函数关系，而这就是 sigmoid函数
> 
> 换句话说，**sigmoid 不是我们"挑"出来的，而是 Bernoulli + GLM 假设的数学后果**：只要你选了 Bernoulli（也就是二分类场景下的标准选择），把自然参数 $\eta = \theta^T x$ 代进去，hypothesis 就只能长成 sigmoid 的样子：

$$
\phi = \frac{1}{1 + e^{-\eta}} = \frac{1}{1 + e^{-\theta^T x}}
$$

> 至此我们知道，在做二分类任务时，logistic （sigmoid） 函数就是假设函数形式的一个自然选择！


---

## 4. Softmax 回归

在第 3 节我们看到，logistic regression 用 sigmoid 把线性输出压到 $(0,1)$，解决二分类。本节把这条思路推广到 **$k$ 类**——也就是 **Softmax 回归**。Softmax 可以理解为 GLM 家族中的一员（把多项分布 multinomial 视作指数族），但更常用**交叉熵**（cross entropy）的方式直接推导，所以这里采用非 GLM 的方法，并把整节按"几何直观 → 参数表示 → 前向计算 → 损失函数"的顺序展开。

### 4.1 问题设定

考虑一个**多分类（$k$ 类）问题**：

$$
x^{(i)} \in \mathbb{R}^n, \quad \text{label } y \in \left[ \{0, 1\}^k \right]
$$

其中 $y$ 是一个 **one-hot 向量**（表示每个输入只属于其中一个类别）。

### 4.2 几何直观

先看一张图建立直观：数据被分成 $k$ 类，每一类都有一个"专属的分割线"——也就是下面的 $\theta_{\text{class}}$。

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec4_softmax.webp" alt="Multi-class classification" width="330" loading="lazy" decoding="async" /></div>

所以我们实际上就是要找这些组参数 $\theta$ , 并画出图中那些最合适的分割线

每个类别 (class) 都有自己的专属参数：$\theta_{\text{class}} \in \mathbb{R}^n$，共有 $k$ 套，$\text{class} \in \{1, \dots, k\}$（one-hot 编码）。

所有类别的参数也可以堆成一个矩阵：

$$
\underbrace{\begin{bmatrix}
\theta_1^T \\
\theta_2^T \\
\vdots \\
\theta_k^T
\end{bmatrix}}_{\text{共 $k$ 行}}
$$

### 4.3 Softmax：把线性输出变成概率分布

给定一个 $x$，$\theta_{\text{class}}^T x \in (-\infty, +\infty)$ 是个实数。我们的目标是得到一个关于**所有类别**的概率分布——这意味着每个类别的输出都必须是**非负的**且加和为 1。一个标准做法是：先对每个 $\theta_i^T x$ 取指数 $e^{\theta_i^T x}$（使其变为正数），然后归一化（使其之和为1）：

$$
p(y = i \mid x; \theta) = \frac{e^{\theta_i^T x}}{\sum_{j=1}^k e^{\theta_j^T x}}
$$

> 给定一个 $x$，跑完这套流程，我们就能得到一个**所有类别上的概率分布**——直接告诉我们该样本最可能属于哪一类。这一步就是 **softmax**，名字里的 "max" 指取最大的概率，"soft" 指不是硬挑一个而是输出软概率。

### 4.4 损失函数：交叉熵

**现在的问题是**：怎么训练以找到最优的 $\theta$？注意 softmax 的输出 $p(y \mid x; \theta)$ **本身就是一个概率分布**——所以我们自然可以用"预测分布 vs 真实分布之间的距离"作为损失。

真实标签 $y$ 也可以看成一个概率分布：在真实类别上概率为 1，其他类别上概率为 0（即 one-hot 向量）。于是**最小化两个概率分布之间的距离**就转化为一个标准度量——**交叉熵**（cross entropy）：

$$
\mathrm{Cross\,Entropy}(p, \hat{p}) = - \sum_{y \in \text{classes}} p(y) \log \hat{p}(y)
$$

对一个样本而言，因为真实分布 $p$ 是 one-hot（只有真实类别 $y_0$ 处概率为 1），上式立刻化简为：

$$
\begin{aligned}
&= - \log \hat{p}(y_0) \\
&= - \log \frac{e^{\theta_{y_0}^T x}}{\sum_{c \in \text{classes}} e^{\theta_c^T x}}
\end{aligned}
$$

之后对参数做梯度下降即可。

值得指出的是：softmax 的梯度更新**同样满足** GLM 节的统一更新规则 $\theta_j := \theta_j + \alpha (y^{(i)} - h_\theta(x^{(i)})) x_j^{(i)}$——这正是它能被纳入 GLM 家族的具体体现。


---

## 参考资料

- [CS229 Lecture Notes 1](https://cs229.stanford.edu/notes2021fall/cs229-notes1.pdf) — 原始讲义，靠后的部分覆盖感知机、指数族、GLM 等内容
- [CS229 Lecture 4 (Autumn 2018) — Lecture 4](https://www.bilibili.com/video/BV1b4anzMEUv) — B 站上 Andrew Ng 的讲课视频（但是是2018年的(●'◡'●)）
- [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) — 矩阵恒等式、矩阵微积分速查表
- [Cross-entropy](https://en.wikipedia.org/wiki/Cross-entropy)——交叉熵的数学知识，自行查阅