---
title: "CS231n : Lec 8 — 注意力机制（Attention）+ Transformer"
description: CS231n Lecture 8 学习笔记，覆盖 Seq2Seq 与注意力机制、Cross-Attention、Self-Attention、Multi-Head Attention、Transformer Block、ViT、Pre-Norm / QK-Norm / SwiGLU / MoE 等现代变体。
pubDate: 2026-10-07
series: cs231n
subSeries: personal-lecture-notes
order: 8
categories:
  - CS231n
  - Attention
  - Transformer
  - ViT
  - MoE
---

## 引子

Lec 7 讲了 RNN / LSTM 处理序列的思路。Lec 8 转向另一类对序列建模更强大的工具：**注意力机制（Attention）+ Transformer**。

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec8_outline.webp" alt="Lec8 outline" width="100%" loading="lazy" decoding="async" />
  <figcaption>Lec8 outline</figcaption>
</figure>
---






## 1. 动机：基于 RNN 的 Seq2Seq 任务

输入：序列 $x_1, \ldots, x_T$
输出：序列 $y_1, \ldots, y_{T'}$


### Encoder

首先，我们用一个 RNN 作为 encoder：

$$
h_t = f_W(x_t, h_{t-1})
$$

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec8_motivate_eg_RNN_encoder.webp" alt="RNN encoder" width="100%" loading="lazy" decoding="async" />
  <figcaption>RNN encoder</figcaption>
</figure>

这个 encoder 会输出一个"摘要向量" $c$（通常就是最后那个隐状态 $h_T$），用来概括输入序列中的全部信息。



### Decoder

现在我们用第二个 RNN（$g_U$）作为 *decoder*（用来"翻译"输入序列）。它的架构通常和 encoder 一样，但用的是另一套学习到的参数。

$$
s_t = g_U(y_{t-1}, s_{t-1}, c)
$$

encoder 在每一个时间步接收 3 个输入：

- $y_{t-1}$：上一个时间步的输出 token（previous output token）；
- $s_{t-1}$：输出序列的上一个隐状态（previous hidden state in the output sequence）；
- $c$：上下文向量，概括了整个输入序列（context vector, summarising the entire input sequence）。

（注意：每一个时间步最后的输出是由当前时间步的隐态 $s_t$ 加上一个线性层 + Softmax 输出概率得到的：$\hat{y}_t = \text{Softmax}(W_{hy} s_t) + b_y$）

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec8_motivate_eg_RNN_encoder_decoder.webp" alt="RNN encoder-decoder architecture" width="100%" loading="lazy" decoding="async" />
  <figcaption>RNN encoder-decoder architecture</figcaption>
</figure>

注意：我们会用 $s_0$ 来初始化 decoder 的隐状态。

> **问题（瓶颈）**：encoder 与 decoder 通信的唯一方式就是通过上下文向量 $c$。这导致 *输入序列的信息全部挤在固定大小的 $c$ 中*！（固定长度的向量没办法合理地概括整个输入序列）
>
> **解决方案**：换一种思路：*在生成输出序列的每一步，模型都可以回头"看"输入序列*。（在输出的每一步都可以回头看）

---




### Seq2Seq with RNN & attention

首先，encoder 与 decoder 初始状态 $s_0$ 保持不变。

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec8_RNN+attention_figure1.webp" alt="the same encoder + s₀" width="100%" loading="lazy" decoding="async" />
  <figcaption>the same encoder + s₀（decoder 初始状态）</figcaption>
</figure>

然后我们 *计算 alignment scores*（标量）：

$$
e_{t,i} = f_{\text{att}}(s_{t-1}, h_i)
$$

$f_{\text{attn}}$ 就是一个线性层（把 $s_{t-1}, h_i$ 拼接起来，再做一个线性变换）。

这些分数本质上回答的问题是：当前输入 token 和上一个时间步的 decoder 状态之间有多相似。

之后，我们 *把 alignment scores 归一化，得到 attention weights*（用 softmax 即可）：

$$
0 < a_{t,i} < 1, \quad \sum_i a_{t,i} = 1
$$

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec8_RNN+attention_figure2.webp" alt="the same encoder + s₀" width="100%" loading="lazy" decoding="async" />
  <figcaption>the same encoder + s₀（decoder 初始状态）</figcaption>
</figure>

我们会用"输入序列中所有隐状态的 *加权和*"来计算当前时间步的上下文向量：（这样 decoder 在每一步都能完整看到 encoder 的所有隐态！）

$$
c_t = \sum_i a_{t,i} h_i
$$

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec8_RNN+attention_figure3.webp" alt="compute the context vector" width="100%" loading="lazy" decoding="async" />
  <figcaption>compute the context vector</figcaption>
</figure>

再把这个上下文向量送进 decoder：

$$
s_t = g_U(y_{t-1}, s_{t-1}, c_t)
$$

其中 $g_U$ 是一个 RNN 单元（例如 LSTM、GRU）。

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec8_RNN+attention_figure4.webp" alt="use the context vector" width="100%" loading="lazy" decoding="async" />
  <figcaption>use the context vector</figcaption>
</figure>


**直觉**：上下文向量会让 decoder 关注输入序列中"相关"的部分。

> "vediamo" = "we see"
>
> 那么也许学习到的权重值就是：
>
> $a_{11} = a_{12} = 0.45, \quad a_{13} = a_{14} = 0.05$


注意这张大计算图是 *全可微* 的！Attention weights 不需要监督，我们只要做端到端的学习，对整张图反向传播就行。

不断重复这个过程：拿到 $s_1$ 后去算新的上下文向量 $c_2$，再算新的 alignment scores $e_{2, i}$ 与 attention weights $a_{2, i}$ ……

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec8_RNN+attention_figure5.webp" alt="repeat the former process" width="100%" loading="lazy" decoding="async" />
  <figcaption>repeat the former process</figcaption>
</figure>

通过观察每个时间步 $t$ 对每个词 $i$ 的 attention weights $a_{t, i}$，我们就可以"窥探"网络到底学到了什么。

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec8_attention_weights_introspect.webp" alt="attention weights introspection" width="70%" loading="lazy" decoding="async" />
  <figcaption>attention weights introspection</figcaption>
</figure>

（对角线式的 attention 表示词与词之间是一一对应的）

---





## 2. 推广形式（Generalized Case）

现在我们把 RNN 部分剥掉，把 attention 机制分离出来，并把它推广为神经网络的一个核心原语（primitive）：

① **Data vectors（数据向量）**：$X \in \mathbb{R}^{N_X \times D_X}$，也就是我们想去做摘要的数据，作用就相当于上面 encoder 各个时间步的 RNN 状态。
② **Query vectors（查询向量）**：$q \in \mathbb{R}^{D_Q}$，作用就相当于 decoder 的 RNN 状态。
③ **Output vectors（输出向量）**：context states。

> **每个 query 都会对所有 data vectors 做 attention，并产生一个输出向量。**


实际中，我们用 *缩放点积（scaled dot product）* 作为 $f_{\text{att}}$ 去算 data vectors 与 query vectors 之间的相似度分数（标量），方法简单但够用。

④ **similarities**：$e \in \mathbb{R}^{N_X}$

$$
e_i = \frac{q \cdot X_i}{\sqrt{D_Q}}
$$

> **注意：不能直接用点积而是用缩放点积！**
>
> 原因：当维度 $D_Q$ 很大时，未缩放的点积数值会非常大（维度越大，方差越大），导致 softmax 进入饱和区，梯度几乎为零，训练困难。


⑤ **attention weights**：$a \in \mathbb{R}^{N_X}$

$$
a = \text{softmax}(e)
$$

⑥ **output vector**：

$$
y = \sum_i a_i X_i \quad \in \mathbb{R}^{N_X}
$$




### 进一步推广：*一组* query vector


#### version 1（直接复用 data vectors 两次）

**Inputs：**

① Query vector：$Q \quad [N_Q \times D_X]$
② Data vectors：$X \quad [N_X \times D_X]$


**Computation：**

③ Similarities：（注意：这里要计算两组向量之间所有的点积值，恰好矩阵乘法可以做到）

$$
\begin{aligned}
E &= Q X^\top / \sqrt{D_X} \quad [N_Q \times N_X] \\
E_{ij} &= Q_i \cdot X_j / \sqrt{D_X}
\end{aligned}
$$

④ Attention weights：

$$
A = \text{softmax}(E, \text{dim}=1) \quad [N_Q \times N_X]
$$

⑤ Output vector：（注意：这里是要用一组向量的各维度值作为权重来取另一组向量的线性组合，恰好矩阵乘法也是有这样的性质！）

$$
\begin{aligned}
Y &= A X \quad [N_Q \times D_X] \\
Y_i &= \sum_j A_{ij} X_j
\end{aligned}
$$



#### version 2（key / value 投影）

由于 data vectors 会在计算 similarities 以及线性组合得到 output vectors 时候重复使用两次。所以我们想到每个 data vector 可以分别被当作 key 与 value 各使用一次，这也是为了区分开 data vectors 的两次使用，所以我们使用下面的方法。

*我们把每个 data vector 投影成两个向量：一个是 key 向量，另一个是 value 向量。*

（额外添加两个可学习矩阵：key 矩阵 $W_K$ + value 矩阵 $W_V$）

**Inputs：**

① Query vector：$Q \quad [N_Q \times D_Q]$
② Data vectors：$X \quad [N_X \times D_X]$
③ Key matrix：$W_K \quad [D_X \times D_Q]$
④ Value matrix：$W_V \quad [D_X \times D_V]$


**Computation：**

Keys：

$$
K = X W_K \quad [N_X \times D_Q]
$$

Values：

$$
V = X W_V \quad [N_X \times D_V]
$$

Similarities：

$$
\begin{aligned}
E &= Q K^\top / \sqrt{D_Q} \quad [N_Q \times N_X] \\
E_{ij} &= Q_i \cdot K_j / \sqrt{D_Q}
\end{aligned}
$$

Attention weights：

$$
A = \text{softmax}(E, \text{dim}=1) \quad [N_Q \times N_X]
$$

Output vector：

$$
\begin{aligned}
Y &= A V \quad [N_Q \times D_V] \\
Y_i &= \sum_j A_{ij} V_j
\end{aligned}
$$

#### visualization：（Cross-Attention Layer）

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec8_key-value_projection_visualization.webp" alt="data → key, value" width="70%" loading="lazy" decoding="async" />
  <figcaption>data 经投影得到 key, value</figcaption>
</figure>

（注意：softmax 是对 $E$（scores）的每一列分别做的。）

我们可以认为 key, value matrix 是两个 filter，在原始数据向量中我们通过这两个 filter 来过滤出 key, value 两方面的信息。


以上的这个可视化已经就是一个独立的神经网络，我们可以将其独立地插入到其他地方！（叫做 *cross attention layer*，因为它有两组输入：data vectors 与 query vectors，这两组输入可能来自不同来源）

---






## 3. 自注意力层（Self-Attention Layer）

我们只有一组输入向量，因此再额外加一个可学习的 *Query 矩阵*，把输入向量映射成 query 向量。

**Inputs：**

① Input vectors：$X \quad [N \times D_{\text{in}}]$
② Key matrix：$W_K \quad [D_{\text{in}} \times D_{\text{out}}]$
③ Value matrix：$W_V \quad [D_{\text{in}} \times D_{\text{out}}]$
④ Query matrix：$W_Q \quad [D_{\text{in}} \times D_{\text{out}}]$


**Computation：**

⑤ Queries：

$$
Q = X W_Q \quad [N \times D_{\text{out}}]
$$

⑥ Keys：

$$
K = X W_K \quad [N \times D_{\text{out}}]
$$

⑦ Values：

$$
V = X W_V \quad [N \times D_{\text{out}}]
$$

对单个输入来说，这三步通常会被 *合并成一次 matmul*：

$$
\begin{aligned}
[Q \quad K \quad V] &= X [W_Q \quad W_K \quad W_V] \\
[N \times 3 D_{\text{out}}] &= [N \times D_{\text{in}}] [D_{\text{in}} \times 3 D_{\text{out}}]
\end{aligned}
$$

⑧ Similarities：

$$
\begin{aligned}
E &= Q K^\top / \sqrt{D_{\text{out}}} \quad [N \times N] \\
E_{ij} &= Q_i \cdot K_j / \sqrt{D_{\text{out}}}
\end{aligned}
$$

⑨ Attention weights：

$$
A = \text{softmax}(E, \text{dim}=1) \quad [N \times N]
$$

⑩ Output vector：

$$
\begin{aligned}
Y &= A V \quad [N \times D_{\text{out}}] \\
Y_i &= \sum_j A_{ij} V_j
\end{aligned}
$$

**visualization：**

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec8_self_attention_visualization.webp" alt="self attention layer" width="70%" loading="lazy" decoding="async" />
  <figcaption>self attention layer</figcaption>
</figure>

每个输入都产生一个输出，这个输出是来自所有输入信息的混合。

---



### 置换等变性（Permutation Equivariance）

我们考虑把输入做一次置换：

此时 queries、keys、values、similarities、attention weights 都会保持原样，只是顺序被重排。

在这种情况下，我们看到 *输出也是原样，只是顺序被重排*！

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec8_self_attention_permutation.webp" alt="self attention permuted" width="70%" loading="lazy" decoding="async" />
  <figcaption>self attention permuted</figcaption>
</figure>

这意味着 self-attention 其实并不关心输入的顺序。把输入打乱，输出也会按相同的方式被打乱。

所以我们可以认为 self-attention 处理的对象并不是一个"序列"，而是 *一组向量*（a set of vectors），向量之间的位置无关紧要。


> **问题**：Self-attention 不知道序列的顺序。
>
> **解决方案**：给每个输入加上 *位置编码（positional encoding）*，这是一个与位置下标相关的固定函数得到的向量（例如 RoPE）。

---




### Masked Self-Attention Layer

如果我们不想让某些向量"看到"序列中靠后的部分，就把它们对应的 similarities 强制置为 $-\infty$；这样就可以控制每个向量能看到哪些输入。

（语言模型预测下一个词时就会用这个机制）

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec8_masked_self-attention_layer.webp" alt="masked self-attention layer" width="70%" loading="lazy" decoding="async" />
  <figcaption>masked self-attention layer</figcaption>
</figure>

---




### 多头自注意力层（Multiheaded Self-Attention Layer）

单头 self-attention 有一个问题：每个输入向量只能产生一组 attention weights，也就是只能关注一种模式。

但自然语言中，一个词可能需要同时关注多种关系：语法、语义……

解决思路：并行运行 $H$ 个独立的 self-attention，每个 head 有自己的 $W_Q, W_K, W_V$，分别学习不同的关注模式。

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec8_multiheaded_self-attention_layer.webp" alt="multiheaded self-attention layer" width="70%" loading="lazy" decoding="async" />
  <figcaption>multiheaded self-attention layer</figcaption>
</figure>

这里 $H = 3$，也就是有 3 个相互独立的 self-attention 层（称为 heads），每个 head 都有自己的一套权重。

把每个输入 $X$ 对应的 $H$ 个独立输出拼起来。

最后再用一个输出投影把各个 head 的信息融合到一起。


**Inputs：**

① Input vectors：$X \quad [N \times D]$
② Key matrix：$W_K \quad [D \times H D_H]$
③ Value matrix：$W_V \quad [D \times H D_H]$
④ Query matrix：$W_Q \quad [D \times H D_H]$
⑤ Output matrix：$W_O \quad [H D_H \times D]$

（注：一般取 $D_H = D / H$，那么输出维度就是 $H D_H = D$）



**Computation terms：**

*1 ) 投影 ：*

(注：reshape 成多头形式：把 $H D_H$ 维的输出，切成 $H$ 份，每份 $D_H$ 维，分给 $H$ 个 head)

Queries：

$$
Q = X W_Q \quad [N \times H D_H] \xrightarrow{\text{reshape}} [H \times N \times D_H]
$$

Keys：

$$
K = X W_K \quad [N \times H D_H] \xrightarrow{\text{reshape}} [H \times N \times D_H]
$$

Values：

$$
V = X W_V \quad [N \times H D_H] \xrightarrow{\text{reshape}} [H \times N \times D_H]
$$

*2 ) 对第 $h$ 个 head ：*

① Similarity：

$$
E^{(h)} = \frac{Q^{(h)} (K^{(h)})^\top}{\sqrt{D_H}} \in \mathbb{R}^{N \times N}
$$

注意：缩放因子是 $\sqrt{D_H}$，不是 $\sqrt{D}$。

② Attention weights：

$$
A^{(h)} = \text{softmax}(E^{(h)}) \in \mathbb{R}^{N \times N}
$$

③ Outputs：

$$
Y^{(h)} = A^{(h)} V^{(h)} \in \mathbb{R}^{N \times D_H}
$$

*3 ) 在第 2) 步基础上将所有 head 结果合为一体（相当于直接一起算）*

$$
E = \frac{Q K^\top}{\sqrt{D_H}} \in \mathbb{R}^{H \times N \times N}
$$

$$
A = \text{softmax}(E, \text{dim}=2) \in \mathbb{R}^{H \times N \times N}
$$

$$
\begin{aligned}
Y &= A V \in \mathbb{R}^{H \times N \times D_H} \\
  &\xrightarrow{\text{reshape}} \mathbb{R}^{N \times H \times D_H}
\end{aligned}
$$

再最后投影回 $D$ 维空间：

$$
O = Y W_O \in \mathbb{R}^{N \times D}
$$

---


**Computation steps conclusion：**

+ **QKV Projection**

$[N \times D] [D \times 3 H D_H] \rightarrow [N \times 3 H D_H]$

拆分并 reshape 得到 $Q$, $K$, $V$，每一项的形状都是 $[H \times N \times D_H]$。

+ **QK Similarity**

$[H \times N \times D_H] [H \times N \times D_H] \rightarrow [H \times N \times N]$

+ **V-Weighting**

$[H \times N \times N] [H \times N \times D_H] \rightarrow [H \times N \times D_H]$

Reshape 回 $[N \times H D_H]$。

+ **Output Projection**

$[N \times H D_H] [H D_H \times D] \rightarrow [N \times D]$

这 $H$ 个并行层每一层用的 qkv 维度都是 $D_H = \text{head dim}$。通常取 $D_H = D / H$，这样输入和输出的维度保持一致。

实际中，所有 $H$ 个 head 通常用 batched matmul 一次性并行算完。

---







## 处理序列的三种方式（Three Ways of Processing Sequences）

### 1) RNN

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec8_sequence_process_way1_RNN.webp" alt="RNN" width="70%" loading="lazy" decoding="async" />
  <figcaption>RNN</figcaption>
</figure>

**优点**：理论上擅长处理长序列：长度为 $N$ 的序列，计算和内存都是 $O(N)$。（计算复杂度、内存复杂度都较低）

**缺点**：*不能并行*。必须串行地算各个隐状态。


### 2) Convolution

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec8_sequence_process_way2_convolution.webp" alt="Convolution" width="70%" loading="lazy" decoding="async" />
  <figcaption>Convolution</figcaption>
</figure>

**缺点**：对长序列不友好：需要堆很多层才能建立足够大的感受野。

**优点**：*可并行*，输出可以同时算出来。


### 3) Self-Attention

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec8_sequence_process_way3_self-attention.webp" alt="Self-Attention" width="70%" loading="lazy" decoding="async" />
  <figcaption>Self-Attention</figcaption>
</figure>

**优点**：对长序列很友好；每个输出都直接依赖所有输入（一层就能建立任意两个位置之间的依赖）；*高度并行*，总共就 4 个 matmul。

**缺点**：代价大：长度为 $N$ 的序列，计算 $O(N^2)$、内存 $O(N)$。

---





## Transformer Block

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec8_transformer_block.webp" alt="transformer block" width="100%" loading="lazy" decoding="async" />
  <figcaption>transformer block</figcaption>
</figure>

输入：一组向量 $x$
输出：一组向量 $y$

- Self-Attention 是向量之间唯一的"互动"环节。
- LayerNorm 和 MLP 对每个向量都独立地工作（注意：各个 MLP 独立地处理不同时间步的输出；LayerNorm 对每个向量独立地做归一化）。
- Residual connecting（残差连接）：让梯度可以绕过子层直接传播，缓解梯度消失，即使子层的梯度很小，梯度也能传回去。
- 高度可扩展、并行友好，绝大部分计算就是 6 个 matmul：4 个来自 Self-Attention + 2 个来自 MLP。

> 一个 Transformer 就是把若干个相同的 Transformer Block 堆叠起来！








### Transformer 应用（Transformer Applications）


#### 用于语言模型（For Language Modeling, LLM）

在模型开头学一个 *embedding 矩阵*，把词变成向量。词表大小为 $V$、模型维度为 $D$ 时，它就是一个 $[V \times D]$ 的查找表。

在每个 Transformer block 内部用 *masked attention*，让每个 token 只能看到它前面的 token。

在模型最后学一个 $[D \times V]$ 的 *投影矩阵*，把每个 $D$ 维向量投影成 $V$ 维分数向量，对应词表中每个 token 的得分。

训练时用 *softmax + cross-entropy loss* 预测下一个 token。

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec8_transformer_for_LM_figure.webp" alt="transformer for LM (LLM)" width="50%" loading="lazy" decoding="async" />
  <figcaption>transformer for LM (LLM)</figcaption>
</figure>


#### Vision Transformers (ViT)

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec8_ViT_figure.webp" alt="ViT" width="100%" loading="lazy" decoding="async" />
  <figcaption>ViT</figcaption>
</figure>

把图像切成小块（patches），每个 patch 当成一个 token；每个 patch 展平成一个向量，然后通过一个线性层投影到模型维度 $D$；因为 Transformer 不知道顺序，需要加上位置编码；最后送入 Transformer 即可。

Transformer 输出经过全局平均池化后得到的汇总向量送入线性层最后得到分类。（当然，在经典ViT论文中使用额外的 CLS token ，对于 transformer 输出的最后一个向量进行线性层处理后能得到最后的分类（因为 attention使得最后一个向量输出时具有全局的注意力））

---








## 经典Transformer的微调（Tweaking Transformers）


### Pre-Norm Transformer

之前 layer normalization 放在残差连接之外。这种做法有点怪，因为模型其实没办法学到恒等函数。

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec8_transformer_block.webp" alt="previous transformer block" width="100%" loading="lazy" decoding="async" />
  <figcaption>previous transformer block</figcaption>
</figure>

**解决方案**：*把 layer normalization 移到 self-attention 和 MLP 之前、放在残差连接内部。* 训练更稳定。

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec8_pre-norm_transformer.webp" alt="pre-norm transformer" width="70%" loading="lazy" decoding="async" />
  <figcaption>pre-norm transformer</figcaption>
</figure>

注：post-norm 为：

$$
x' = \text{LayerNorm}(x + \text{SubLayer}(x))
$$

假设子层什么不做（为 0），那么输出结果为 $x$ 被层归一化后的结果，$x' \neq x$，无法学到恒等函数！

而 pre-norm 的残差连接在子层操作之外：

$$
x' = x + \text{SubLayer}(\text{LayerNorm}(x))
$$

若子层什么不做输出 0，那么最后 $x' = x$ 能成功学到恒等函数！

---



### QK-Norm

在注意力机制分数计算中，如果 $Q$ 和 $K$ 的数值很大，点积会很大，softmax 会饱和，梯度会消失。所以我们的改进方案是：Normalize queries and keys before computing attention similarities.

Queries：

$$
Q = \text{normalize}(X W_Q) \quad [H \times N \times D_H]
$$

Keys：

$$
K = \text{normalize}(X W_K) \quad [H \times N \times D_H]
$$

Values：

$$
V = X W_V \quad [H \times N \times D_H]
$$

Similarities：

$$
E = \frac{Q K^\top}{\sqrt{D_Q}} \quad [H \times N \times N]
$$

这能避免梯度尖峰（gradient spikes），从而让训练更稳定。

e.g：Normalize with RMSNorm：

$$
y_i = \frac{x_i}{\text{RMS}(x)} \cdot \gamma_i
$$

$$
\text{RMS}(x) = \sqrt{\epsilon + \frac{1}{N} \sum_{i=1}^N x_i^2}
$$

> 注意：只对 $Q, K$ 做归一化，$V$ 不做归一化。因为 QK-Norm 的目的是控制注意力分数的数值范围，防止 softmax 饱和，而 $V$ 不参与相似度计算，所以对它做归一化既没必要，还会损害信息。

---



### SwiGLU MLP

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec8_SwiGLU_figure.webp" alt="standard MLP vs. SwiGLU" width="100%" loading="lazy" decoding="async" />
  <figcaption>standard MLP vs. SwiGLU</figcaption>
</figure>

**① Classic MLP**

单通路结构，先做线性变换，过激活函数，再做第二次线性变换。

- *Input*：$X : \quad [N \times D]$
- *Weights*：

$$
W_1 : \quad [D \times 4D] \qquad W_2 : \quad [4D \times D]
$$

- *Output*：

$$
Y = \sigma(X W_1) W_2 : \quad [N \times D]
$$

**② SwiGLU MLP**

- *Branch 1（主通路/激活分支）*：输入经过线性变换 $W$ 后，通过 "Swish"（即 "SiLU"）激活函数。
- *Branch 2（门控分支）*：输入经过另一个独立的线性变换 $V$。
- *门控融合（$\odot$）*：两条分支的结果进行逐元素相乘（Hadamard Product）。
- *输出投影*：融合后的特征最后通过线性变换 $W_2$ 输出。

$$
\text{SwiGLU}(X) = (\text{Swish}(X W) \odot X V) W_2
$$

- *Input*：$X \quad [N \times D]$
- *Weights*：

$$
W_1, W_2 : \quad [D \times H] \qquad W_3 : \quad [H \times D]
$$

- *Output*：

$$
Y = (\sigma(X W_1) \odot X W_2) W_3
$$

> SwiGLU 就是把 GLU 中的激活函数 $\sigma$ 换成 swish。

> 取 $H = (8D)/3$ 时可以保持与原模型相同的总参数量。

---



### Mixture of Experts (MoE)

大模型需要大量参数来提升容量，但参数越多，计算量越大，训练和推理成本越高。

MoE 的核心思想：把一个大 MLP 拆成多个"专家"MLP，每个 token 只激活其中少数几个专家。这样总参数量可以很大，但每个 token 的计算量只和激活的专家数有关。

**基本结构**：在 Transformer 的每个 Block 中，把原来的 MLP 替换为：$E$ 个专家 MLP，每个专家有自己的权重；一个路由网络（Router），决定每个 token 去哪些专家。

> 在每个 block 中学 $E$ 套独立的 MLP 权重；每个 MLP 就是一个专家。

$$
W_1 : [D \times 4D] \Rightarrow [E \times D \times 4D]
$$

$$
W_2 : [4D \times D] \Rightarrow [E \times 4D \times D]
$$

> 每个 token 会被路由到 $A < E$ 个专家上，这些就是被激活的专家（active experts）。


**路由机制**：

对于每个 token $x$：

① 路由网络计算它到每个专家的得分：

$$
s = \text{softmax}(x W_r)
$$

其中 $W_r \in \mathbb{R}^{D \times E}$。

② 选择得分最高的 $A$ 个专家（通常 $A \ll E$），只有这 $A$ 个专家参与计算，输出加权求和：

$$
y = \sum_{i \in \text{Top-}A} s_i \cdot \text{Expert}_i(x)
$$

> 这样参数量增加了 $E$ 倍，但计算量只增加 $A$ 倍。


**Example**：Gemma4 26B-A4B (4/2/2026)

- 1 个"共享专家（shared expert）"处理所有 token
- 128 个"路由专家（routed experts）"；每个 token 选取 8/128 来处理它
- 总参数量 26B，但每个 token 只"激活"4B 参数

---







## 参考资料

- [CS231n Lecture 8 — Attention and Transformers](https://cs231n.stanford.edu/slides/2026/lecture_8.pdf) — 2026 slide PDF
- [CS231n 2025 spring](https://www.bilibili.com/video/BV1YJ3PzLEiW) — CS231n Spring 2025 视频
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) — Transformer 原始论文
- [An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929) — ViT 原始论文
- [RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)](https://arxiv.org/abs/2104.09864) — RoPE 位置编码
- [Gemma 4: Open Weights from Google](https://blog.google/technology/google-deepmind/gemma/) — Gemma 系列模型
- [The Illustrated Transformer (Alammar, 2018)](http://jalammar.github.io/illustrated-transformer/) — Transformer 图解经典博客