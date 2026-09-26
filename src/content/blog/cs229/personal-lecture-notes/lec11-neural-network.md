---
title: "CS229 : Lec 11 — 神经网络"
description: CS229 Lecture 11 笔记。从Logistic Regression 出发，把它看作一个最简单的"神经元"；再堆叠出层和引入非线性激活函数，便得到多层前馈神经网络 (Neural Network)；最后初步介绍前向传播、反向传播。
pubDate: 2026-09-12
series: cs229
subSeries: personal-lecture-notes
order: 12
categories:
  - CS229
  - Neural Networks
  - Logistic Regression
  - Softmax
  - Activation Function
  - Forward Propagation
  - Backpropagation
---


> **TL;DR**:
> - **神经元 (Neuron) = 线性部分 + 激活函数**：**Logistic Regression** 本质上就是单个神经元：先算 $Z = w^Tx + b$，再过激活 $a = \sigma(Z)$，可以做二分类问题。
> - 解决 K-分类时，输出层放 K 个神经元 + Softmax，把输出归一化成概率分布。
> - ① *神经元 = 线性 + 激活*；② *模型 = 网络结构 + 参数* 
> - **神经网络**：把多个神经元按"层 (Layer)" 组织起来——同一层内的神经元互不通信、相邻层之间全连接。深度 = 隐藏层层数。
> - **前向传播 (Forward Propagation)**：从输入 $x$ 一层层算到输出 $\hat y$。每一层都做 $Z^{[\ell]} = W^{[\ell]} a^{[\ell-1]} + b^{[\ell]}$ 和 $a^{[\ell]} = \sigma(Z^{[\ell]})$
> - **反向传播 (Backpropagation)**：从输出层往回链式求 $\partial \mathcal{J} / \partial W^{[\ell]}$，并进行迭代代值，是一个**自顶向下的递归过程**。


## 引子

前面几讲学的都是**线性 / 浅层模型**（线性回归、逻辑回归、SVM）。这一讲是深度学习的第一讲，简单认识**神经网络**

> **深度学习为什么很成功？**
> - 新的计算方法（GPU、TPU）
> - **海量数据**（互联网 + 移动设备 + 物联网爆发）
> - 算法的进步（更好的激活函数、更深的网络、正则化技巧、…）

这一讲从 **Logistic Regression** 出发，把它看作"只有一个神经元的网络"；然后堆叠层，得到真正的多层神经网络；最后讲**前向 + 反向传播**。

---




## 1. Logistic Regression：一个神经元的视角

### 1.1 二分类第一个例子 (e.g.¹)：找图里有没有猫

**任务**：给定一张图，判断"图里有没有猫"。

$$
y = \begin{cases} 0 & \text{absence} \\ 1 & \text{presence} \end{cases}
$$

**做法**：把一个像素图按 RGB 三个通道展开成一个长向量作为输入 $x$，然后过 $w x + b$，最后套一个 **sigmoid 函数**：

$$
\hat y = \sigma(w^T x + b), \quad \sigma(z) = \frac{1}{1 + e^{-z}}
$$

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec11_LR_eg1.webp" alt="logistic regression as a single neuron" width="100%" loading="lazy" decoding="async" /></div>

> **关键观察**：**Logistic Regression 本质上就是一个"神经元"**——线性部分 + sigmoid 激活。

对于 $64 \times 64 \times 3$ 的像素图：

$$
x \in \mathbb{R}^{12288 \times 1}, \quad w \in \mathbb{R}^{1 \times 12288}
$$

**训练的三个步骤**：
1. 初始化参数 $w, b$
2. 定义 loss（此处是来自 MLE 方法得到的）：

$$
\mathcal{L}(\hat y, y) = - \big[ y \log \hat y + (1 - y) \log (1 - \hat y) \big]
$$

3. 用梯度下降找最优 $w, b$。

> 二分类任务的标签必须是 $\{0, 1\}$。

---


### 1.2 多分类 (e.g.²)：猫 / 狮子 / 鬣蜥

**任务**：给定一张图，判断"图里有哪种动物"——3 类（猫、狮子、鬣蜥）。

最简单粗暴的做法：**输出层放 3 个独立神经元**，各管各的——这样会得到 $\hat y_1, \hat y_2, \hat y_3$ 三维输出：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec11_LR_eg2.webp" alt="multi-class logistic regression as 3 neurons" width="100%" loading="lazy" decoding="async" /></div>

> **记号**：`[1]` 表示 **层 (layer)** 的编号，同一层内的神经元彼此**互不通信**；下标 $1, 2, 3$ 是层内的神经元索引。

对应的标签用**独热 (one-hot) 编码**：

$$
\begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix}, \quad \begin{pmatrix} 0 \\ 1 \\ 0 \end{pmatrix}, \quad \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix}
$$

> **鲁棒性 (Robustness)**：因为这一层三个神经元之间互不通信，所以它们可以**独立训练**。如果你把数据标注里"代表狮子的那一维"从第二维挪到第三维，相应的神经元也会自动跟着迁移——第三个神经元会接手识别狮子的任务。

---


### 1.3 Softmax 多分类 (e.g.³)：互斥的多类识别

**任务**：图里**只有一种动物**，判断是哪一种。

这是一个**互斥**的多分类问题——三个输出不能各自独立（因为它们加起来必须 = 1）。所以引入记号 $Z_1^{[1]}, Z_2^{[1]}, Z_3^{[1]}$ 表示第一层三个神经元的**线性部分**（还没过激活）。

去掉激活只算 $Z$：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec11_softmax_eg.webp" alt="softmax classification network" width="100%" loading="lazy" decoding="async" /></div>

Softmax 把 $Z$ 归一化成概率分布（和 = 1），三个输出之间因此**相互依赖**——不能像 e.g.² 那样独立训练。

> **注意**：这里不能像 e.g.² 那样继续用二元的损失函数 $\mathcal{L} = -y \log \hat y + (1-y)\log(1 - \hat y)$。因为 Softmax 强制三个输出之和为 1，但若对每个输出**独立**使用二元交叉熵，损失函数会鼓励每个输出独立地接近目标，但归一化约束使它们无法同时独立满足，导致**梯度冲突**——所以我们应该换一个损失函数。

**如果我们尝试使用这种 loss function**：

$$
\mathcal{L}_{3N} = -\sum_{k=1}^{3} \big[\, y_k \log \hat y_k + (1 - y_k) \log(1 - \hat y_k) \,\big]
$$

> 假如我们现在对第二个神经元的权重 $w^{[2]}$ 求导，结果会非常复杂——因为每个 $\hat y_k$ 都依赖于所有 $Z_j$，而每个 $Z_j$ 又对应自己的一套参数。所以对 $w^{[2]}$ 求导时，不仅第二项有贡献，第一项和第三项也会通过分母中的 $e^{Z_2}$ 产生影响。链式法则展开会牵扯到所有参数，导致**梯度表达式极其冗长**。

所以最终我们用的是 **Softmax cross-entropy loss function**——也叫 **Softmax 交叉熵损失**：

$$
\mathcal{L}_{\text{softmax}} = -\sum_{k=1}^{K} y_k \log \hat y_k
$$

> 其中 $K$ 是类别数（这里是 3）。


#### 回归任务的改造

如果想把网络改成"预测猫的年龄"，只需要改动激活函数以及损失函数即可，即可把最后一层激活函数从 sigmoid 换成：

- **线性激活**（$a = Z$）——直接输出数值
- **ReLU (Rectified Linear Unit)**——输出非负值（年龄 / 价格这类天然非负的目标很合适）

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec11_ReLU.webp" alt="ReLU activation" width="50%" loading="lazy" decoding="async" /></div>

$$
\text{ReLU}(z) = \max(0, z)
$$

相应的 loss 也要换成**平方误差** $\lVert y - \hat y \rVert^2$。

> 回归任务的 loss 一般比分类的 softmax loss 更好优化。

---


### 1.4 两个关键公式

整个 Lec 11 反复会用到两个公式：

**① 神经元 = 线性部分 + 激活函数**

$$
\text{neuron} = \underbrace{w x + b}_{\text{linear}} + \underbrace{\text{activation}}_{\sigma / \tanh / \text{ReLU}}
$$

> 例如：Logistic Regression = 一个神经元 = 线性 + sigmoid。

**② 模型 = 网络结构 + 参数**

$$
\text{model} = \text{architecture} + \text{parameters}
$$

- **网络结构 (architecture)**：多少层、每层多少神经元、用什么激活
- **参数 (parameters)**：各层的 $W^{[\ell]}, b^{[\ell]}$

---





## 2. 神经网络：把神经元堆成层

### 2.1 一个简单的两层神经网络

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec11_neural_network_architecture_eg.webp" alt="simple 2-layer neural network" width="100%" loading="lazy" decoding="async" /></div>

这个网络有 **1 个隐藏层**（hidden layer）+ 1 个输出层（output layer），输入是 3 个特征 $x_1, x_2, x_3$。

> **输出数量 = 任务类型**：
> - 1 个输出 → 回归
> - K 个输出 → K-分类（K ≥ 2）


### 2.2 参数数量怎么算

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec11_neural_network_parameters.webp" alt="parameters in the neural network" width="100%" loading="lazy" decoding="async" /></div>

每一条边 = 一个权重 $w$。所以参数总数 = 边的数量 + 偏置数量。


### 2.3 三层神经网络的术语

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec11_neural_network_3layers.webp" alt="three layers of the neural network" width="100%" loading="lazy" decoding="async" /></div>

| 层                              | 角色                                                         |
| ------------------------------ | ---------------------------------------------------------- |
| **Layer 1：输入层 (Input Layer)**  | 装原始特征 $x_1, x_2, x_3$——不算真正的"层"，因为没有参数也没有激活                |
| **Layer 2：隐藏层 (Hidden Layer)** | 名字叫"隐藏"是因为**输入和输出都不与其直接相邻**——它看到的永远只是抽象层面的数据，并且我们也不知道它在算什么 |
| **Layer 3：输出层 (Output Layer)** | 输出最终的预测 $\hat y$                                           |

> **经验**（以图像任务为例）：前几层倾向于**检测边缘**；中间几层把边缘组合成**部件**（耳朵、嘴）；最后几层把部件组合成**整体**（猫脸）并做出分类判断。每一层都比上一层**更抽象**。


### 2.4 房价预测例子：从知识驱动构建网络到端到端学习（全连接层）

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec11_house_price_neural_network_with-knowledge.webp" alt="neural network with prior knowledge" width="100%" loading="lazy" decoding="async" /></div>

如果我们**用人类先验知识**手工设计网络——比如先算"家庭规模"，再算"家庭收入"，再算"家庭购买力"，最后推"房价"——结构很清晰，但**靠人脑设计**。


**但更常见的做法**：相邻两层之间**全连接**，让人工神经网络**端到端 (end-to-end)** 自己学：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec11_house_price_neural_network_fully-connected.webp" alt="fully connected neural network" width="100%" loading="lazy" decoding="async" /></div>

> 这是一个**黑箱模型**——我们不约束中间在算什么，让网络自己决定。这就是 **end-to-end learning**。

---





## 3. 前向传播 (Forward Propagation)

### 3.1 单样本：3 层网络

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec11_propagation_architecture.webp" alt="3-layer network architecture" width="100%" loading="lazy" decoding="async" /></div>

把上面的图翻译成数学公式：

$$
Z^{[1]} = W^{[1]} x + b^{[1]}, \quad a^{[1]} = \sigma(Z^{[1]}) \\
Z^{[2]} = W^{[2]} a^{[1]} + b^{[2]}, \quad a^{[2]} = \sigma(Z^{[2]}) \\
Z^{[3]} = W^{[3]} a^{[2]} + b^{[3]}, \quad a^{[3]} = \sigma(Z^{[3]}) = \hat y
$$

数据的形状对应关系（输入 $x \in \mathbb{R}^{n \times 1}$，三层神经元数依次 3, 2, 1）：

$$
\begin{aligned}
Z^{[1]} \in \mathbb{R}^{3 \times 1} &\longleftarrow W^{[1]} \in \mathbb{R}^{3 \times n},\; b^{[1]} \in \mathbb{R}^{3 \times 1} &\longrightarrow a^{[1]} \in \mathbb{R}^{3 \times 1} \\
Z^{[2]} \in \mathbb{R}^{2 \times 1} &\longleftarrow W^{[2]} \in \mathbb{R}^{2 \times 3},\; b^{[2]} \in \mathbb{R}^{2 \times 1} &\longrightarrow a^{[2]} \in \mathbb{R}^{2 \times 1} \\
Z^{[3]} \in \mathbb{R}^{1 \times 1} &\longleftarrow W^{[3]} \in \mathbb{R}^{1 \times 2},\; b^{[3]} \in \mathbb{R}^{1 \times 1} &\longrightarrow a^{[3]} \in \mathbb{R}^{1 \times 1}
\end{aligned}
$$

> **直观规律**：
> - $Z^{[\ell]}$ 的维度 = 第 $\ell$ 层的**神经元数**
> - $W^{[\ell]}$ 的大小 = **连接两个相邻层的边数**

---


### 3.2 批量化 (Mini-batch)：一次算 $m$ 个样本

实际训练中我们不会一次只跑一个样本——而是把 $m$ 个样本**横向拼起来**成一个输入矩阵 $X$：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec11_batched_input.webp" alt="batched input X" width="70%" loading="lazy" decoding="async" /></div>

$$
X = \big[\, x^{(1)} \mid x^{(2)} \mid \cdots \mid x^{(m)} \,\big] \in \mathbb{R}^{n_x \times m}
$$

批量化后，前向传播变成纯矩阵运算：

$$
Z^{[\ell]} = W^{[\ell]} A^{[\ell-1]} + b^{[\ell]}, \quad A^{[\ell]} = \sigma(Z^{[\ell]})
$$

此时线性部分（中间变量）的形状变化：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec11_batched_linear_part.webp" alt="shape of batched Z" width="80%" loading="lazy" decoding="async" /></div>

$$
Z^{[1]} \in \mathbb{R}^{3 \times m}, \quad Z^{[2]} \in \mathbb{R}^{2 \times m}, \quad Z^{[3]} \in \mathbb{R}^{1 \times m}
$$

> **关键**：$W^{[\ell]}, b^{[\ell]}$ 的形状**完全不变**！其中 $b$ 靠的是广播机制（**broadcasting**）——$b^{[\ell]} \in \mathbb{R}^{3 \times 1}$ 自动沿列复制 $m$ 次，得到 $\mathbb{R}^{3 \times m}$ 的矩阵才能与前面的矩阵做加法。

---





## 4. 优化 (Optimizing)

### 4.1 损失/代价函数（Loss vs Cost）

> **Loss $\mathcal{L}(\hat y^{(i)}, y^{(i)})$**：只看**一个样本**的损失。
> **Cost $\mathcal{J}(W, b)$**：在**整个 batch 上**取平均。

$$
\boxed{\; \mathcal{J}(W, b) = \frac{1}{m} \sum_{i=1}^{m} \mathcal{L}\big(\hat y^{(i)}, y^{(i)}\big) \;}
$$

> 之所以除以 $m$，是因为接下来要做 **批量梯度下降 (batch gradient descent)**——整个 batch 一起算平均梯度，再更新参数。

对二分类任务，loss 仍然用 logistic regression 使用的 loss：

$$
\mathcal{L}^{(i)} = - \big[\, y^{(i)} \log \hat y^{(i)} + (1 - y^{(i)}) \log (1 - \hat y^{(i)}) \,\big]
$$

---





## 5. 反向传播 (Backpropagation)

### 5.1 为什么是"反向"？

参数更新公式（对每一层 $\ell$）：

$$
W^{[\ell]} := W^{[\ell]} - \alpha \frac{\partial \mathcal{J}}{\partial W^{[\ell]}}, \qquad b^{[\ell]} := b^{[\ell]} - \alpha \frac{\partial \mathcal{J}}{\partial b^{[\ell]}}
$$

我们从**离 loss 最近的层**（输出层）开始往回算梯度。

> **为什么倒着来？** 因为 $\mathcal{J}$ 依赖 $\hat y$，$\hat y$ 依赖 $Z^{[3]}$，$Z^{[3]}$ 依赖 $W^{[3]}$ 和 $a^{[2]}$——直接对每一层 $W^{[\ell]}$ 都从 $\mathcal{J}$ 一路链式乘下来，**重复计算**了大量中间量。倒着来，每一层只需要从"上一层已经算好的梯度"继续乘局部梯度——**省事又省时间**。


### 5.2 输出层梯度计算的链式法则

对最后一层 $W^{[3]}$：

$$
\frac{\partial \mathcal{J}}{\partial W^{[3]}} = \frac{\partial \mathcal{J}}{\partial a^{[3]}} \cdot \frac{\partial a^{[3]}}{\partial Z^{[3]}} \cdot \frac{\partial Z^{[3]}}{\partial W^{[3]}}
$$

这是一个标准的链式展开：$\mathcal{J} \to a^{[3]} \to Z^{[3]} \to W^{[3]}$。


### 5.3 中间层的链式法则

对中间层 $W^{[2]}$，完整的链式路径更长：

$$
\frac{\partial \mathcal{J}}{\partial W^{[2]}} = \underbrace{\frac{\partial \mathcal{J}}{\partial Z^{[3]}}}_{\text{已知}} \cdot \frac{\partial Z^{[3]}}{\partial a^{[2]}} \cdot \frac{\partial a^{[2]}}{\partial Z^{[2]}} \cdot \frac{\partial Z^{[2]}}{\partial W^{[2]}}
$$

> **技巧**：$\dfrac{\partial \mathcal{J}}{\partial Z^{[3]}}$ 在算 $W^{[3]}$ 梯度时**已经算出来了**——直接复用，乘上 $W^{[2]}$ 对应的局部梯度。这就是反向传播加速的本质。


### 5.4 反向传播的"路径选择"

> ⚠️ **路径选择很关键**：链式法则要求乘的是**当前变量到 loss 的合法路径**，不是任意的偏导都能乘。
>
> 例如：要求 $\dfrac{\partial \mathcal{J}}{\partial W^{[2]}}$，就不能去找 $\dfrac{\partial W^{[2]}}{\partial a^{[1]}}$——$W^{[2]}$ 不直接依赖 $a^{[1]}$，中间还隔着 $Z^{[2]}$。

正确的链式路径**必须穿过**当前变量真实依赖的中间变量。

---





## 参考资料

- [CS229 Deep Learning Note](https://cs229.stanford.edu/notes2021fall/deep_learning_notes.pdf) — Lec 11 主讲义：Neural Networks
- [CS229 2018](https://www.bilibili.com/video/BV1b4anzMEUv) — B 站上 CS229 Lec 11 讲课视频
- [Bishop, Pattern Recognition and Machine Learning, Ch.5](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) — Neural Networks 章节：多层感知机、BP 算法推导、Softmax / Cross-Entropy
- [Deep Learning](https://www.deeplearningbook.org/) — Goodfellow, Bengio, Courville