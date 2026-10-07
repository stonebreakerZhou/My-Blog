---
title: "CS231n : Lec 7 — 循环神经网络（RNN）"
description: CS231n Lecture 7 学习笔记，覆盖 RNN 基本原理（隐态更新、计算图、BPTT）、LSTM 与现代 RNN（状态空间模型 SSM）。
pubDate: 2026-10-06
series: cs231n
subSeries: personal-lecture-notes
order: 7
categories:
  - CS231n
  - RNN
  - LSTM
  - Sequence Modeling
  - SSM
---

## 引子

Lec 6 解决了"怎么训练 CNN"。Lec 7 转向另一类数据：**序列（sequences）**。

outline：

① Recurrent Neural Networks（RNNs）
② Sequence modeling（assumed fixed-length inputs so far）
③ Simple models commonly used before the era of transformers
④ RNNs and some variants
⑤ Relation to modern state-space models（e.g. Mamba）

---





## 1. 循环神经网络（Recurrent Neural Network, RNN）

序列任务有很多种形式，但都可以用 RNN 统一处理：

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec7_sequence_tasks.webp" alt="sequence tasks" width="100%" loading="lazy" decoding="async" />
  <figcaption>sequence tasks</figcaption>
</figure>


### 什么是 RNN ？

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec7_RNN_simple_figure.webp" alt="RNN simple figure" width="30%" loading="lazy" decoding="async" />
  <figcaption>RNN simple figure</figcaption>
</figure>

核心思想是：我们有一个输入序列 $x$ 和一个输出序列 $y$。RNN 具有这种循环结构（图中箭头）：**RNNs have an "internal state" (hidden state) that is updated as a sequence is processed**——也就是说，RNN 维护一个"内部状态"（hidden state），随着序列推进而更新。

我们可以画出 *unrolled RNN diagram*：

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec7_unrolled_RNN_diagram.webp" alt="unrolled RNN diagram" width="100%" loading="lazy" decoding="async" />
  <figcaption>unrolled RNN diagram</figcaption>
</figure>



### 隐状态更新规则（Hidden state update rule）

我们可以对向量序列 $x$ 在每一个时间步应用 *recurrence* 公式来逐步处理：

$$
h_t = f_W(h_{t-1}, x_t)
$$

其中 $h_t$ 是新状态，$f_W$ 是带参数 $W$ 的某个函数，$h_{t-1}$ 是旧状态，$x_t$ 是某个时间步的输入向量。（用隐状态来产生输出）

> **注意**：*同一个* 函数 $f$ 和 *同一套* 参数 $W$ 在每个时间步都会被复用。



### 输出（Output generation）

为了得到实际输出：

$$
y_t = f_{W_{hy}}(h_t)
$$

其中 $y_t$ 是输出，$f_{W_{hy}}$ 是另一组带参数 $W_{hy}$ 的函数，$h_t$ 是新状态。




### Vanilla (Elman) RNN

我们采用：

$$
\begin{aligned}
h_t &= \tanh(W_{hh} h_{t-1} + W_{xh} x_t) \\
y_t &= W_{hy} h_t
\end{aligned}
$$



### RNN Concrete Simple Example

下面我们手动构造一个 RNN 来检测输入序列中重复出现的 1。这是一个 "many to many" 序列建模任务。

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec7_RNN_concrete_simple_eg.webp" alt="input-output pair example" width="30%" loading="lazy" decoding="async" />
  <figcaption>input-output pair example</figcaption>
</figure>


为简便起见，我们可以这样设计：

$$
\begin{aligned}
h_t &= \text{ReLU}(W_{hh} h_{t-1} + W_{xh} x_t) \\
h_t &= \begin{pmatrix} \text{current} \\ \text{Previous} \\ 1 \end{pmatrix} \\
y_t &= \text{ReLU}(W_{hy} h_t)
\end{aligned}
$$


① $W_{xh} x_t$：此时 $x_t$ 就是当前的数值。由于 hidden state 是一个三维向量，故 $W_{xh}$ 也就是一个三维向量。两者的乘积的第一维应当代表当前所看到的数值，故设置：

$$
W_{xh} = \begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix}
$$

这样当前位置 $x = 0$ 时 $\to W_{xh} x = \begin{pmatrix} 0 \\ 0 \\ 0 \end{pmatrix}$；当前位置 $x = 1$ 时 $\to W_{xh} x = \begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix}$。


② 接着我们来设置：

$$
W_{hh} = \begin{pmatrix} 0, 0, 0 \\ 1, 0, 0 \\ 0, 0, 1 \end{pmatrix}
$$

第一行全设为 0 因为当前状态完全由 $W_{xh} x$ 决定而与前一个隐态无关；第二行 $(1, 0, 0)$ 相当于是将上一个隐态 $h_{t-1}$ 的 current 值拷贝到当前隐态的 previous 值处；第三行则是保持第三维的 1 不变。


③ 我们最后设置：

$$
W_{hy} = \begin{pmatrix} 1 \\ 1 \\ -1 \end{pmatrix}
$$

这样得到的值经过 ReLU 后能够得到正确的 $y_t$ 输出值！

---







### RNN 的计算图（Computational Graph）

注意：我们在每个时间步都复用同一套权重矩阵。

#### Many to Many

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec7_RNN_computational_graph_many-to-many.webp" alt="many to many" width="100%" loading="lazy" decoding="async" />
  <figcaption>many to many</figcaption>
</figure>

每一步都算损失，最后求和或是求平均。


#### Many to One

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec7_RNN_computational_graph_many-to-one.webp" alt="many to one" width="100%" loading="lazy" decoding="async" />
  <figcaption>many to one</figcaption>
</figure>

只在最后一个时间步有一个输出并得到一个损失值。


#### One to Many

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec7_RNN_computational_graph_one-to-many_figure1.webp" alt="one to many : figure1" width="100%" loading="lazy" decoding="async" />
  <figcaption>one to many : figure1</figcaption>
</figure>

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec7_RNN_computational_graph_one-to-many_figure2.webp" alt="one to many : figure2" width="100%" loading="lazy" decoding="async" />
  <figcaption>one to many : figure2</figcaption>
</figure>

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec7_RNN_eg_character_LM.webp" alt="RNN e.g. character-level LM" width="80%" loading="lazy" decoding="async" />
  <figcaption>RNN e.g. character-level LM</figcaption>
</figure>

这种情况下每一个时间步都会有一个输出以及对应的损失，最后总损失会对所有时间步的损失进行求和（或求平均）。

---





### Backpropagation through time (BPTT)

> 前向传播遍历整个序列以计算损失，然后反向传播遍历整个序列以计算梯度。


RNN 的参数 $W_{hh}, W_{xh}, W_{hy}$ 在所有时间步共享。

- 前向传播时，同一个 $W_{hh}$ 被用了 $T$ 次。
- 反向传播时，每个时间步都会算出"我对 $W_{hh}$ 的梯度贡献"。

最终 $W_{hh}$ 的梯度是每个时间步所有梯度的贡献总和：

$$
\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^T \frac{\partial L}{\partial W_{hh}}\bigg|_t
$$

然后只用这个总梯度更新一次：

$$
W_{hh} \leftarrow W_{hh} - \eta \frac{\partial L}{\partial W_{hh}}
$$

> 所以：每个时间步都有梯度贡献，但只更新一次参数。



many-to-one 情况下损失值由最后一个时间步计算；而 many-to-many 情况下每个时间步都有输出与损失，最终损失值一般取成所有时间步的损失值之和：

$$
L = \sum_{t=1}^T L_t \quad (\text{最终损失值})
$$

对 $W_{hh}$ 求导，拆分为每个时间步对 $W_{hh}$ 的梯度贡献：

$$
\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^T \frac{\partial L}{\partial h_t} \cdot \frac{\partial h_t}{\partial W_{hh}}\bigg|_{\text{直接}}
$$

但注意：现在 $\frac{\partial L}{\partial h_t}$ 有两部分：

$$
\frac{\partial L}{\partial h_t} = \underbrace{\frac{\partial L_t}{\partial h_t}}_{\text{当前输出}} + \underbrace{\frac{\partial L}{\partial h_{t+1}} \cdot \frac{\partial h_{t+1}}{\partial h_t}}_{\text{后续时间步}}
$$

① 第一项：当前时间步有输出，直接贡献梯度；
② 第二项：后续时间步的损失通过递推传回来。


所以 many-to-many 中，每个时间步的梯度贡献更直接，因为每一步都有输出损失直接贡献梯度。


无论是 many-to-one 还是 many-to-many 类型，最终梯度值都与 $(W_{hh}^T)^n$（以及激活函数导数的连乘项）有关。这可能会导致梯度爆炸 / 消失。（后面在gradient flow 那一节会详细严谨推导）



于是我们可以使用 *Truncated Backpropagation through time (TBPTT)*：

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec7_truncated_backpropagation_through_time.webp" alt="truncated backprop through time" width="100%" loading="lazy" decoding="async" />
  <figcaption>truncated backprop through time</figcaption>
</figure>

标准的 BPTT 前向时把整个序列跑完，计算总损失；反向时则从最后一步开始，一直回传到第一步；梯度需要穿过整个序列。

**TBPTT 则把长序列切成若干段（chunks），每段分别做前向和反向。** 注意所有段都是共享同一套参数，所以每一段的反向传播都会更新一次这一套参数。当前这一段更新完后，把输出的隐层继续给下一段，然后继续做前向 + 反向。

---





### Interpretable Cells

在 CNN 中，我们可以可视化第一层卷积核，看到它们学到了边缘、颜色等特征。但 RNN 的隐藏状态是一个高维向量，随时间步不断变化，我们很难直接理解每个维度在做什么。


于是作者提出一个问题：RNN 的隐藏状态中，是否也存在一些"可解释的神经元"？即：有没有某个神经元，在特定输入模式出现时激活，其他时候不激活？如果有，我们就能理解它在追踪什么。

（补充：注意"神经元"的概念：一个神经元的输出是一个标量，在 RNN 中，因为隐层状态 $h_t$ 是一个高维向量，我们就可以说这个高维向量每一维就是一个神经元。）

该论文最终在字符级语言模型中，观察到一些 LSTM 单元（神经元）会自发地学习去追踪输入序列中特定的、可解释的模式。

具体例子：这些神经元会分别负责追踪代码行的长度、引号的开闭、括号的嵌套深度，或是检测 `if` 语句等。


**意义**：这说明 RNN 并非一个完全无法理解的"黑盒"。它的隐藏状态中，确实存在有明确分工、负责处理特定信息的单元。

---




### RNN Tradeoffs

**RNN Advantages：**

① 可以处理任意长度的输入（没有 context length 限制）
② 第 $t$ 步的计算理论上可以使用许多步之前的信息
③ 输入越长，模型大小不会增大
④ 每个时间步都复用同一套权重，因此输入的处理方式具有对称性。

**RNN Disadvantages：**

① 循环计算较慢
② 实际上难以访问许多步之前的信息（因为我们把所有信息都塞进隐层 $h_t$ 中，这会导致随着序列变长我们总会损失一些信息）

---





### RNN 在 CV 上的应用

① **图像描述生成（Image Captioning）**
② **视觉问答（Visual Question Answering, VQA）**
③ **视觉对话（Visual Dialog）**：Conversations about images——围绕图像展开对话。



#### 应用的模型架构

**CNN + RNN（LSTM）**

用 CNN 对图像进行编码（coding the image），并把得到的特征作为 RNN 的初始隐层状态。

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec7_CNN+RNN.webp" alt="CNN + RNN" width="100%" loading="lazy" decoding="async" />
  <figcaption>CNN + RNN</figcaption>
</figure>

具体做法是：去掉原 CNN 最后的几个（FC）层，得到一个特征向量（after pooling）$v$。

$v$ 有两种使用方法：

**法一**：$v$ 只用来初始化 RNN 的隐层 $h_0$：

$$
h_0 = W_v v
$$

**法二**：$v$ 每一步都参与：

$$
h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + W_{ih} v)
$$

这样相当于每一个时间步都可以"看到" $v$，$v$ 是每一步都可见的上下文信息，是贯穿整个序列的条件信息。

---





### Multilayer RNNs

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec7_multilayer_RNN.webp" alt="multilayer RNN figure" width="80%" loading="lazy" decoding="async" />
  <figcaption>multilayer RNN figure</figcaption>
</figure>

其实就是把多层 RNN 堆叠在一起，核心思想：第一层的输出序列，作为第二层的输入序列；第二层的输出，作为第三层的输入；以此类推……

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec7_multilayer_RNN_structure.webp" alt="multilayer RNN structure" width="100%" loading="lazy" decoding="async" />
  <figcaption>multilayer RNN structure</figcaption>
</figure>

每一层都有一套独立的共享权重。

---







#### Vanilla RNN Gradient Flow

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec7_single_RNN_gradient_flow.webp" alt="single RNN gradient flow" width="70%" loading="lazy" decoding="async" />
  <figcaption>single RNN gradient flow</figcaption>
</figure>

更新规则为：

$$
\begin{aligned}
h_t &= \tanh(W_{hh} h_{t-1} + W_{xh} x_t) \\
    &= \tanh\begin{pmatrix} W_{hh} & W_{hx} \end{pmatrix} \begin{pmatrix} h_{t-1} \\ x_t \end{pmatrix} \\
    &= \tanh W \begin{pmatrix} h_{t-1} \\ x_t \end{pmatrix}
\end{aligned}
$$

由此可以得到梯度为：

$$
\frac{\partial h_t}{\partial h_{t-1}} = \tanh'(W_{hh} h_{t-1} + W_{xh} x_t) W_{hh}
$$

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec7_vanilla_RNN_gradient_flow.webp" alt="RNN gradient flow in backprop" width="100%" loading="lazy" decoding="async" />
  <figcaption>RNN gradient flow in backprop</figcaption>
</figure>

现在我们来看反向传播：

$$
\frac{\partial L}{\partial W} = \sum_{t=1}^T \frac{\partial L_t}{\partial W}
$$

where

$$
\begin{aligned}
\frac{\partial L_T}{\partial W} &= \frac{\partial L_T}{\partial h_T} \frac{\partial h_t}{\partial h_{T-1}} \cdots \frac{\partial h_1}{\partial W} \\
&= \frac{\partial L_T}{\partial h_T} \left( \prod_{t=2}^T \frac{\partial h_t}{\partial h_{t-1}} \right) \frac{\partial h_1}{\partial W}
\end{aligned}
$$

代入 $\frac{\partial h_t}{\partial h_{t-1}}$，得到：

$$
\frac{\partial L_T}{\partial W} = \frac{\partial L_T}{\partial h_T} \left( \prod_{t=2}^T \tanh'(W_{hh} h_{t-1} + W_{xh} x_t) \right) W_{hh}^{T-1} \frac{\partial h_1}{\partial W}
$$

注意蓝色项几乎总是 $<1$，于是我们就得到了 *梯度消失*！

> 即便我们假设没有非线性，或者选择不会产生这种问题的激活函数，只要看 $W_{hh}^{T-1}$ 仍然会有问题：如果它的最大奇异值 $> 1$，那就会出现梯度爆炸；反之若最大奇异值 $< 1$，则会出现梯度消失。

在这种情况下，如果遇到梯度爆炸，可以做 *梯度裁剪（gradient clipping）*：当梯度的范数过大时对梯度进行缩放；但如果遇到梯度消失，那就只能改 RNN 的架构了！

---







### LSTM（a historical note）

首先，我们有 *四个门（four gates）*：

- $i$：Input gate, whether to write to cell
- $f$：Forget gate, Whether to erase cell
- $o$：Output gate, How much to reveal cell
- $g$：Gate gate (?), How much to write to cell

$$
\begin{pmatrix} i \\ f \\ o \\ g \end{pmatrix} = \begin{pmatrix} \sigma \\ \sigma \\ \sigma \\ \tanh \end{pmatrix} W \begin{pmatrix} h_{t-1} \\ x_t \end{pmatrix}
$$

上面的计算可以直观地这样理解：

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec7_gate_intuitive_computation.webp" alt="LSTM gate intuitive computation" width="100%" loading="lazy" decoding="async" />
  <figcaption>LSTM gate intuitive computation</figcaption>
</figure>

然后我们定义 *细胞状态与隐状态（cell state and hidden state）*：

$$
\begin{aligned}
c_t &= f \odot c_{t-1} + i \odot g \\
h_t &= o \odot \tanh(c_t)
\end{aligned}
$$

现在我们来仔细看一下 LSTM 的 *梯度流*：

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec7_single_LSTM_gradient_flow_figure.webp" alt="LSTM gradient flow" width="100%" loading="lazy" decoding="async" />
  <figcaption>LSTM gradient flow</figcaption>
</figure>

看 $c_t$ 到 $c_{t-1}$ 的梯度：

$$
\frac{\partial c_t}{\partial c_{t-1}} = f_t
$$

这里只有逐元素乘法，没有矩阵乘法，没有 $\tanh$ 导数。如果 $f_t \approx 1$，那么梯度近乎无损传回。整条传导路径为：

$$
c_0 \leftarrow c_1 \leftarrow c_2 \leftarrow \cdots
$$

这是一条无中断的梯度流（uninterrupted gradient flow）！

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec7_LSTM_uninterrupted_gradient_flow.webp" alt="uninterrupted gradient flow" width="100%" loading="lazy" decoding="async" />
  <figcaption>uninterrupted gradient flow</figcaption>
</figure>

对比普通 RNN：

$$
h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t)
$$

每一步都经过 $\tanh$，信息被压缩；每一步都乘 $W_{hh}$，梯度容易衰减；隐藏状态被完全覆盖，旧信息无法直接保留。

而 LSTM 的 cell state 是门控简单加法（不是完全覆盖！），信息、梯度更易保留。

---


#### Similarity to ResNet

直接对输出做加法、跳过激活函数这些思路，与残差连接（residual connection）的思想是类似的！

它们的区别在于：LSTM 处理的是长时间序列，而 ResNet 处理的是很深的层数。

---


#### Do LSTMs solve the vanishing gradient problem ?

LSTM 的架构让 RNN 在许多时间步上保留信息变得更加容易。

例如，如果 $f = 1$ 且 $i = 0$，那么该细胞中的信息就会被无限期地保留下来。

相比之下，普通 RNN 要学到一个能保留隐状态中信息的循环权重矩阵 $W_h$ 是更困难的。

LSTM 并不保证不会发生梯度消失 / 爆炸，但 ***它确实为模型学习长距离依赖提供了一条更轻松的路径***。

---






### Modern RNNs

现代 RNN 有时被称为 *状态空间模型（State Space Models, SSM）*，因为现代 RNN 本质上是在维护一个隐藏状态 $h_t$，并让它随输入逐步演化。这种"状态随输入演化"的视角，就是状态空间模型。

所有 RNN 的共同点：用一个隐藏状态 $h_t$ 压缩过去的所有信息。

**主要优势：**

**① Unlimited context length（上下文长度不受限）**

传统 Transformer 的注意力机制：每个 token 都要和之前所有 token 计算注意力；上下文长度受限于显存和计算量；通常有最大窗口限制，比如 512、2048、8192。

现代 RNN / SSM：每步只依赖前一步的隐藏状态；不需要保存所有历史 token；理论上可以处理任意长度的序列；适合流式数据、超长文本、音频、视频。

**② Compute scales linearly with sequence length（计算量随序列长度线性增长）**

Transformer 随着序列增长计算量成平方次增长；而 RNN / SSM 随着序列增长计算量为线性增长！

---







## 参考资料

- [CS231n Lecture 7 — Recurrent Neural Networks](https://cs231n.stanford.edu/slides/2026/lecture_7.pdf) — 2026 slide PDF
- [CS231n 2025 spring](https://www.bilibili.com/video/BV1YJ3PzLEiW) — CS231n Spring 2025 视频
- [CS231n notes — Recurrent Neural Networks](https://cs231n.github.io/rnn/) — 网站上的笔记：RNN 架构
- [Understanding LSTM Networks (Olah, 2015)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — LSTM 可视化经典博客
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Goel, 2023)](https://arxiv.org/abs/2312.00752) — 现代 SSM 代表论文
- [The Unreasonable Effectiveness of Recurrent Neural Networks (Karpathy, 2015)](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) — 字符级 RNN 的早期经典