---
title: "CS229 : Lec 12 — 反向传播算法推导 + 神经网络优化技巧"
description: CS229 Lecture 12 学习笔记。上半场：用链式法则手动推导反向传播的每一步梯度，证明前向 + 反向传播的"复用性"为什么高效；下半场：讲神经网络训练的各种"经验技巧"——激活函数 / 输入归一化 / 权重初始化 / Mini-batch / Momentum。
pubDate: 2026-09-14
series: cs229
subSeries: personal-lecture-notes
order: 12
categories:
  - CS229
  - Backpropagation
  - Chain Rule
  - Activation Function
  - Input Normalization
  - Weight Initialization
  - Mini-batch Gradient Descent
  - Momentum
---


> **TL;DR**:
> - **反向传播 (Backpropagation) 的本质**：用链式法则求 $\partial \mathcal{J} / \partial w^{[\ell]}$；从输出层往回算梯度，**复用上一层算好的中间梯度**——省去大量重复计算。
> - **激活函数**：sigmoid / ReLU / tanh 三大选择；**如果没有激活函数** → 无论多少层都等价于一个简单的线性变换 $Wx + B$。
> - **输入归一化 (Input Normalization)**：把 $x$ 零均值化、标准化，避免落在 sigmoid 的饱和区。
> - **权重初始化**：要避免 **vanishing / exploding gradients**；对于sigmoid，ReLU， tanh 用不同的优化方案。
> - **Mini-batch GD**：把训练集切成 $T$ 个小批，每批算一次梯度，单步噪声大但是总体比较高效。
> - **Momentum 动量更新法**：参数 $w$ 想象成一个小球的位置，小球有**惯性**。它不会因为当前坡度突然改变就立刻掉头，而是保留一部分之前的速度。



## 引子

Lec 11 讲了神经网络的基本结构（前向传播），但**梯度到底怎么算？** Lec 12 上半场就手动把反向传播的每一步梯度**从链式法则**一行一行推出来。

下半场则讲"训练神经网络的**经验技巧**"——光有算法不够，现实训练中还会遇到：
- 激活函数选什么？
- 输入 / 权重怎么初始化？
- 训练要不要用 mini-batch？
- 怎么让梯度下降"方向更对、不用来回震荡"？

---




## 1. 反向传播 (Backpropagation) 算法推导

### 1.1 Cost function & update rule

回顾神经网络要最小化的代价函数（针对整体全部样本）：

$$
\mathcal{J}(\hat y, y) = \frac{1}{m} \sum_{i=1}^{m} \mathcal{L}^{(i)}(\hat y, y)
$$

其中单个样本的 loss（二分类）：

$$
\mathcal{L}^{(i)} = - \big[\, y^{(i)} \log \hat y^{(i)} + (1 - y^{(i)}) \log (1 - \hat y^{(i)}) \,\big]
$$

每一层 $\ell$ 都有自己的参数 $W^{[\ell]}, b^{[\ell]}$，其更新规则为：

$$
W^{[\ell]} := W^{[\ell]} - \alpha \, \frac{\partial \mathcal{J}}{\partial W^{[\ell]}}
$$

下面以这个简单三层网络为例子：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec12_simple-nn_eg.webp" alt="simple neural network" width="100%" loading="lazy" decoding="async" /></div>

> **约定**：$\mathcal{J}$ 对 $W^{[\ell]}$ 求导时，先利用求和的线性性 → 只算 $\partial \mathcal{L} / \partial W^{[\ell]}$，最后对 $m$ 个样本求和取平均即可。下面从**最靠近 cost 的 $W^{[3]}$** 开始往回倒推。

---


### 1.2 求 $\partial \mathcal{L} / \partial W^{[3]}$

回忆 $\hat y = a^{[3]} = \sigma(W^{[3]} a^{[2]} + b^{[3]})$，所以：

$$
\frac{\partial \mathcal{L}}{\partial W^{[3]}} = - \Big[\, y^{(i)} \frac{\partial}{\partial W^{[3]}} \log\big(\sigma(W^{[3]} a^{[2]} + b^{[3]})\big) + (1 - y^{(i)}) \frac{\partial}{\partial W^{[3]}} \log\big(1 - \sigma(W^{[3]} a^{[2]} + b^{[3]})\big) \,\Big]
$$

此处求导涉及的两个性质：

$$
\frac{\partial \log(\sigma(f))}{\partial w} = \frac{1}{\sigma(f)} \cdot \frac{\partial \sigma(f)}{\partial w}
$$

$$
\sigma'(x) = \sigma(x)\,(1 - \sigma(x))
$$

逐项展开：

$$
\begin{aligned}
\frac{\partial}{\partial W^{[3]}} \log(\sigma(\cdots)) &= \frac{1}{\sigma(\cdots)} \cdot \sigma(\cdots)\,(1 - \sigma(\cdots)) \cdot a^{[2]T} \\
&= (1 - \sigma(W^{[3]} a^{[2]} + b^{[3]})) \, a^{[2]T}
\end{aligned}
$$

$$
\begin{aligned}
\frac{\partial}{\partial W^{[3]}} \log(1 - \sigma(\cdots)) &= \frac{1}{1 - \sigma(\cdots)} \cdot \big(- \sigma(\cdots) (1 - \sigma(\cdots))\big) \cdot a^{[2]T} \\
&= - \sigma(W^{[3]} a^{[2]} + b^{[3]}) \, a^{[2]T}
\end{aligned}
$$

两项合到一起：

$$
\frac{\partial \mathcal{L}}{\partial W^{[3]}} = - \big[\, y^{(i)} (1 - \sigma(\cdots)) a^{[2]T} - (1 - y^{(i)}) \sigma(\cdots) a^{[2]T} \,\big]
$$

$$
= - \big(y^{(i)} - \sigma(W^{[3]} a^{[2]} + b^{[3]})\big) \, a^{[2]T}
$$

$$
= - \big(y^{(i)} - a^{[3]}\big) \, a^{[2]T}
$$

> **得到对第三层参数的偏导**：$\partial \mathcal{L} / \partial W^{[3]} = -(y^{(i)} - a^{[3]}) \, a^{[2]T}$——这是 sigmoid + 交叉熵这套组合的特殊便利之处。
>
> 对 $b^{[3]}$ 的求导类似，只是最后少了 $a^{[2]T}$ 这一项。

对 $m$ 个样本求平均：

$$
\frac{\partial \mathcal{J}}{\partial W^{[3]}} = \frac{1}{m} \sum_{i=1}^{m} \big(- (y^{(i)} - a^{[3]})\, a^{[2]T}\big)
$$

---


### 1.3 求 $\partial \mathcal{L} / \partial W^{[2]}$

现在往回推到第二层求 $W^{[2]}$ 的梯度——链式路径**比 $W^{[3]}$ 更长**：

$$
\frac{\partial \mathcal{L}}{\partial W^{[2]}} = \textcolor{red}{\frac{\partial \mathcal{L}}{\partial a^{[3]}} \cdot \frac{\partial a^{[3]}}{\partial Z^{[3]}}} \cdot \textcolor{blue}{\frac{\partial Z^{[3]}}{\partial a^{[2]}}} \cdot \textcolor{green}{\frac{\partial a^{[2]}}{\partial Z^{[2]}}} \cdot \textcolor{purple}{\frac{\partial Z^{[2]}}{\partial W^{[2]}}}
$$

(we need variables that directly connect to each other to pass down the chain rule)

注意到 $\partial \mathcal{L} / \partial W^{[3]}$ 对应 $\partial \mathcal{L} / \partial W^{[2]}$ 里的前两项（红色部分）。所以直接代入得：

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial W^{[3]}} &= \textcolor{red}{\frac{\partial \mathcal{L}}{\partial a^{[3]}} \cdot \frac{\partial a^{[3]}}{\partial Z^{[3]}}} \cdot \frac{\partial Z^{[3]}}{\partial W^{[3]}} \\
&= \textcolor{red}{\frac{\partial \mathcal{L}}{\partial a^{[3]}} \cdot \frac{\partial a^{[3]}}{\partial Z^{[3]}}} \cdot a^{[2]T}
\end{aligned}
$$

又因为我们已知：

$$
\frac{\partial \mathcal{L}}{\partial W^{[3]}} = - (y^{(i)} - a^{[3]}) \, a^{[2]T}
$$

所以红色部分就等于 $- (y^{(i)} - a^{[3]})$。

代入 $\partial \mathcal{L} / \partial W^{[2]}$ 后得：

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial W^{[2]}} &= - (y^{(i)} - a^{[3]}) \cdot \textcolor{blue}{\frac{\partial Z^{[3]}}{\partial a^{[2]}}} \cdot \textcolor{green}{\frac{\partial a^{[2]}}{\partial Z^{[2]}}} \cdot \textcolor{purple}{\frac{\partial Z^{[2]}}{\partial W^{[2]}}} \\
&= (a^{[3]} - y^{(i)}) \cdot \textcolor{blue}{W^{[3]T}} \cdot \textcolor{green}{a^{[2]} (1 - a^{[2]})} \cdot \textcolor{purple}{a^{[1]T}}
\end{aligned}
$$


### 数据形状分析

各项的形状（输入 $x \in \mathbb{R}^{3 \times 1}$，隐藏层 2 神经元，输出层 1 神经元）：

$$
\begin{aligned}
(a^{[3]} - y^{(i)}) &\in \mathbb{R}^{1 \times 1} \quad \text{(scalar)} \\
W^{[3]T} &\in \mathbb{R}^{2 \times 1} \\
a^{[2]} (1 - a^{[2]}) &\in \mathbb{R}^{2 \times 1} \quad \text{(element-wise product)} \\
a^{[1]T} &\in \mathbb{R}^{1 \times 3}
\end{aligned}
$$

严格按顺序相乘：

$$
\frac{\partial \mathcal{L}}{\partial W^{[2]}} = W^{[3]T} \, * \, a^{[2]} (1 - a^{[2]}) \cdot (a^{[3]} - y^{(i)}) \cdot a^{[1]T}
$$

$$
= \mathbb{R}^{2 \times 1} \, * \, \mathbb{R}^{2 \times 1} \cdot \mathbb{R}^{1 \times 1} \cdot \mathbb{R}^{1 \times 3} = \mathbb{R}^{2 \times 3}
$$

形状正好 = $W^{[2]}$ 本身的形状  ✓  （注意 $*$ 表示 Hadamard product）

> **关键观察**：算 $W^{[2]}$ 的时候**复用了算 $W^{[3]}$ 时已经得到的红色部分**——这就是"反向传播"省时间的本质：每往回推一层，只是再多乘一个**本地梯度**（蓝色 $W^{[3]T}$ + 绿色 $\sigma'(Z^{[2]})$ + 紫色 $a^{[1]T}$），不需要重新从 $\mathcal{L}$ 一路链式乘到这一层。

### Cache 缓存的作用

> 前向传播时，要保存**几乎所有算出来的中间值**——反向传播时直接取用，不必重算。

---





## 2. 改进神经网络 (Improving NNs)

光有反向传播算法还不够——下面这些**训练技巧**在实践中不可或缺。

### 2.1 激活函数 (Activation Function)

#### ① Sigmoid

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec12_sigmoid_plot.webp" alt="sigmoid plot" width="60%" loading="lazy" decoding="async" /></div>

$$
\sigma(z) = \frac{1}{1 + e^{-z}}, \quad \sigma'(z) = \sigma(z)\,(1 - \sigma(z))
$$

- **优点**：能把 $(-\infty, +\infty)$ 压到 $(0, 1)$，输出可解释为概率。
- **缺点**：如果 $z$ 太大或太小，梯度**几乎为 0** → 反向传播时参数几乎不更新（**梯度消失**）。


#### ② ReLU

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec12_ReLU_plot.webp" alt="ReLU plot" width="60%" loading="lazy" decoding="async" /></div>

$$
\text{ReLU}(z) = \begin{cases} 0 & \text{if } z \le 0 \\ z & \text{if } z > 0 \end{cases}, \quad \text{ReLU}'(z) = \mathbb{1}\{z > 0\}
$$

> ReLU 在正区间梯度恒为 1，**不存在梯度消失问题**——这是它在深度网络中如此流行的根本原因。


#### ③ Tanh

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec12_tanh_plot.webp" alt="tanh plot" width="60%" loading="lazy" decoding="async" /></div>

$$
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}, \quad \tanh'(z) = 1 - \tanh^2(z)
$$

> 和 sigmoid 类似——只是把输出从 $(0, 1)$ 平移到 $(-1, 1)$，使得均值为 0（实践中往往比 sigmoid 更易收敛）。


#### 为什么必须有激活函数？

假设没有激活函数（即 $a^{[\ell]} = Z^{[\ell]}$）：

$$
\begin{aligned}
\hat y = a^{[3]} = Z^{[3]} &= W^{[3]} a^{[2]} + b^{[3]} = W^{[3]} Z^{[2]} + b^{[3]} \\
&= W^{[3]} (W^{[2]} Z^{[1]} + b^{[2]}) + b^{[3]} \\
&= W^{[3]} (W^{[2]} (W^{[1]} x + b^{[1]}) + b^{[2]}) + b^{[3]} \\
&= W x + B
\end{aligned}
$$

> 没有激活函数，无论堆多少层都**等价于单层线性回归** $Wx + B$——非线性能力完全丧失。激活函数是神经网络能拟合任意复杂函数的根源。

---


### 2.2 输入归一化 (Input Normalization)

#### 为什么？

假设输入 $x = \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$ 的原始分布是这样：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec12_raw_input.webp" alt="raw input distribution" width="60%" loading="lazy" decoding="async" /></div>

问题在于：做 $W^{[1]} x + b^{[1]}$ 算 $Z^{[1]}$ 时，如果 $x$ 太大，那 $Z^{[1]}$ 也很容易进入激活函数的**饱和区**（sigmoid 的两端）→ 梯度消失 → 学不动。

#### 做法

对训练集算均值和方差：

$$
\begin{pmatrix} \mu_1 \\ \mu_2 \end{pmatrix} = \mu = \frac{1}{m} \sum_{i=1}^{m} x^{(i)}, \qquad \sigma^2 = \frac{1}{m} \sum_{i=1}^{m} (x^{(i)} - \mu)^2
$$

然后归一化：

$$
x := \frac{x - \mu}{\sigma}
$$

归一化后的分布：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec12_normalized_input.webp" alt="normalized input distribution" width="60%" loading="lazy" decoding="async" /></div>

> **注意**：**测试**时，必须用**训练集上算出来的 $\mu, \sigma$** 来归一化测试集；**不能**在测试集上重新算 $\mu, \sigma$。


#### 直观效果

loss 曲面对比：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec12_normalized_loss-plot_compare.webp" alt="normalized loss plot comparison" width="100%" loading="lazy" decoding="async" /></div>

梯度下降路径对比：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec12_normalized_loss-plot-route_compare.webp" alt="normalized optimizing route comparison" width="100%" loading="lazy" decoding="async" /></div>

> 归一化后（右图），loss 曲面更"圆"——梯度下降不会在陡峭方向上反复震荡，**收敛效率显著提升**。

---



### 2.3 权重初始化 (Weights Initialization)

#### ① 问题：梯度消失 / 爆炸 (Vanishing / Exploding Gradients)

考虑一个**没有激活、没有 bias** 的神经网络（设激活 = 恒等，$b = 0$）：

$$
\hat y = W^{[\ell]} W^{[\ell-1]} \cdots W^{[1]} x
$$

如果 $W = \begin{pmatrix} s & 0 \\ 0 & s \end{pmatrix}$：
- 若 $s < 1$：连续乘法让 $\hat y$ **指数级衰减** → 梯度消失。
- 若 $s > 1$：连续乘法让 $\hat y$ **指数级爆炸** → 梯度爆炸。

> 一个直接的解法：把 $W$ 初始化到合适的范围（如上面的 $s \approx 1$）



#### ② 单神经元推导：方差约束

考虑只有一个神经元、多个输入输出的情况：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec12_one_neuron_eg.webp" alt="single neuron" width="45%" loading="lazy" decoding="async" /></div>

例如 $a = \sigma(Z)$，$Z = w_1 x_1 + \cdots + w_n x_n$。

假设：输入 $x_i$ 独立同分布（均值为 0，方差为 $\text{Var}(x)$）；权重 $w_i$ 同样独立同分布（均值为 0，方差为 $\text{Var}(w)$）。

> 为了避免 $Z$ 进入激活函数的饱和区，我们希望 $Z$ 的方差在**每一层传播时保持不变**：$\text{Var}(z) \approx \text{Var}(x)$。

$$
\begin{aligned}
\text{Var}(Z) &= \text{Var}\!\left(\sum_{i=1}^{n} w_i x_i\right) = \sum_{i=1}^{n} \text{Var}(w_i x_i) \\
&= n \cdot \text{Var}(w) \cdot \text{Var}(x) \\
&\approx \text{Var}(x)
\end{aligned}
$$

> 这就要求 $\text{Var}(w) \approx 1/n$——其中 $n$ 是该层的输入数。



#### ③ 各种初始化方案

**① Sigmoid 作为激活函数时**：

```py
W^(l) = np.random.randn(shape) * np.sqrt(1 / n^(l-1))
```

> 对应上面的推导——初始化权重方差 = $\dfrac{1}{n^{[\ell-1]}}$（$n^{[\ell-1]}$ 是第 $\ell$ 层的输入个数）。


**② ReLU 作为激活函数时**：

```py
W^(l) = np.random.randn(shape) * np.sqrt(2 / n^(l-1))
```

> ReLU 会把一半的神经元（$z \le 0$ 的那部分）置零——这意味着实际**有效输入**只有一半。
> 为了补偿这种"减半"效应，要把方差放大一倍——所以只用把分子从 $1$ 改成 $2$。


**③ Xavier 初始化**：

Xavier 同时考虑**前向传播**和**反向传播**两边的方差稳定性：
- 前向要求：$\text{Var}(w^{[\ell]}) \approx 1 / n^{[\ell-1]}$
- 反向要求：$\text{Var}(w^{[\ell]}) \approx 1 / n^{[\ell]}$

两者折中，取调和平均：

$$
\text{Var}(w^{[\ell]}) = \frac{2}{n^{[\ell-1]} + n^{[\ell]}}
$$

即

$$
w^{[\ell]} \sim \mathcal{N}\!\left(0, \frac{2}{n^{[\ell-1]} + n^{[\ell]}}\right)
$$

> 这种初始化适用于 **tanh / sigmoid** 作激活函数的时候。


> **为什么一定要"随机"初始化？** 如果没有随机性，所有神经元会**对称地初始化为相同值**——然后每次更新也得到完全相同的梯度，最终学到**完全相同**的特征（symmetry problem）。随机初始化就是为了打破这种对称，让每个神经元学不同的东西。

---



### 2.4 Mini-batch 梯度下降

#### 流程

假设训练集：

$$
X = (x^{(1)}, \dots, x^{(m)}), \quad Y = (y^{(1)}, \dots, y^{(m)})
$$

把它切成 $T$ 个 mini-batch：

$$
X = (x^{\{1\}}, \dots, x^{\{T\}}), \quad Y = (y^{\{1\}}, \dots, y^{\{T\}})
$$

Mini-batch GD 算法：

```
For iteration t = 1, ... :
    Select a batch of data (x^{t}, y^{t}) :
        Forward propagate this batch
        Backpropagate this batch
        Update all w^l, b^l for all layers
```

> **Forward propagate**：把当前 batch 的所有样本送进网络，算整个 batch 的 cost。

#### 与 Batch GD 的对比

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec12_cost_function_compare.webp" alt="cost function comparison" width="80%" loading="lazy" decoding="async" /></div>

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec12_mini-batch_GD_plot.webp" alt="mini-batch GD route" width="80%" loading="lazy" decoding="async" /></div>

| | Batch GD（左图）| Mini-batch GD（右图）|
|---|---|---|
| 单步梯度 | 用**全部** $m$ 个样本算出的**真实**梯度 | 用**一个 batch** 算出的**近似**梯度 |
| Cost 曲线 | **平滑**单调下降 | 趋势下降，但**带噪声** |
| 单步代价 | 大（要算 $m$ 个样本）| 小（只算 batch size 个）|
| 总迭代次数 | 少 | 多 |

> **虽然** mini-batch 需要**更多次**迭代，但每次迭代的**算量小得多**——总体上比 batch GD **高效**得多，尤其当 $m$ 很大的时候。

---


### 2.5 Momentum 动量算法

#### 动机

来看一个典型的 loss 等高线图（横长纵窄的山谷形）：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec12_momentum_loss-plot1.webp" alt="loss contour plot" width="80%" loading="lazy" decoding="async" /></div>

普通梯度下降的方向**总是垂直于等高线**——结果就是：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec12_momentum_loss-plot2.webp" alt="ordinary GD route" width="80%" loading="lazy" decoding="async" /></div>

> 在窄长山谷里走"之字形"——水平方向进展很慢，垂直方向反复震荡。

#### 直觉

我们想要的效果：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec12_momentum_loss-plot3.webp" alt="desired optimization route" width="100%" loading="lazy" decoding="async" /></div>

**水平方向**一直同方向累积得多 → 大步向前；**垂直方向**上下震荡有抵消 → 下一次垂直走向变小。

#### 做法：Momentum

让参数更新方向变成**过去梯度的指数加权平均**：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec12_momentum_loss-plot4.webp" alt="momentum route" width="90%" loading="lazy" decoding="async" /></div>

> 初始化 $v = 0$，每一步：

$$
v := \beta v + (1 - \beta) \frac{\partial \mathcal{L}}{\partial w}, \qquad w := w - \alpha v
$$

> "$w$ 的更新直接使用 $v$"——这样 $v$ 同时携带了**当前梯度**和**历史方向**，行为像物理里的**惯性**："momentum" 就是这个意思：让更新有"惯性"，不会轻易被噪声带偏方向。

> **关键属性**：因为是"带权"的——所以方向的改变**不会太剧烈**（哪怕某一步的梯度方向突然跳一下，$v$ 不会一下子跟着跳）。

---




## 参考资料

- [CS229 DeepLearning Notes](https://cs229.stanford.edu/notes2021fall/deep_learning_notes.pdf) — 包含 Lec 12 内容
- [CS229 2018](https://www.bilibili.com/video/BV1b4anzMEUv) — B 站上 CS229 Lec 12 视频
- [Bishop, Pattern Recognition and Machine Learning, Ch.5](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) — Neural Networks 章节：BP 算法详细推导
- [Deep Learning](https://www.deeplearningbook.org/) — Goodfellow, Bengio, Courville；Ch.6 Feedforward Networks、Ch.8 Optimization for Training（含 SGD / Mini-batch / Backprop）