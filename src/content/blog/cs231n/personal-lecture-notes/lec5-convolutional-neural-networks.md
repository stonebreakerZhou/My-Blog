---
title: "CS231n : Lec 5 — 卷积神经网络（Convolutional Neural Networks, CNN）"
description: CS231n Lecture 5 学习笔记，覆盖卷积层（filter / activation map / padding / 感受野 / stride）、池化层（max pooling）以及平移等变性。
pubDate: 2026-10-04
series: cs231n
subSeries: personal-lecture-notes
order: 5
categories:
  - CS231n
  - CNN
  - Convolution
  - Pooling
  - Receptive Field
---

## 引子

Lec 4 我们讲了神经网络的层级化计算 + 反向传播，Lec 5 转向一个具体问题：**图像分类（Image Classification）**。

*卷积神经网络（Convolutional Neural Networks, CNN）* 用来解决图像分类问题。

> **Timeline**：
> 2012 – 2020 : ConvNets dominate all vision tasks
> 2021 – present : Transformers (ViT) have taken over

---




## 引子：为什么线性分类器不够用？

**Problem**：线性分类器不够强大（Linear classifiers are not powerful !）


### 原因 ① ：视觉视角（Visual viewpoint）

事实上，图像分类的训练过程中线性分类器对于每个类别只学一个 *模板（template）*。

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec5_problem_with_linear_classifier_viewpoint1.webp" alt="visualized templates" width="80%" loading="lazy" decoding="async" />
  <figcaption>visualized templates</figcaption>
</figure>

我们可以把线性分类器理解为：把学习到的权重矩阵 $W$ 看作一张 *图像（image）*，每一行就是对应类别的一张模板。

也就是说，线性分类器必须把它对每一个类别的全部知识都压缩进一张模板里。



### 原因 ② ：几何视角（Geometric viewpoint）

线性分类器只能在特征空间里做 *线性划分（linear separation）*。

因此，我们堆叠多个线性层并在中间插入非线性激活，就得到一个对图像分类更强大的机制。

---






## 1. 早期特征工程（Feature extraction back in the day）

现在我们直接把图像的 raw pixels 输入神经网络；但在更早的时候，人们会先从图像中 *抽取高层特征（high-level features）* 再喂给模型。

### Example 1 ：颜色直方图（Color histogram）

只看颜色信息，不看空间结构。

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec5_color_histogram.webp" alt="color histogram" width="100%" loading="lazy" decoding="async" />
  <figcaption>color histogram</figcaption>
</figure>



### Example 2 ：方向梯度直方图（Histogram of Oriented Gradients, HoG）

丢掉颜色信息，只看结构信息。

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec5_HoG.webp" alt="HoG" width="100%" loading="lazy" decoding="async" />
  <figcaption>HoG</figcaption>
</figure>



### 把多种特征拼起来

因此，过去的人们会用不同的特征抽取器得到不同表示，再把它们拼起来作为输入：

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec5_feature_extract_and_concatenate.webp" alt="feature extract and concatenate" width="100%" loading="lazy" decoding="async" />
  <figcaption>feature extract and concatenate</figcaption>
</figure>

不过现在我们直接用神经网络做 *端到端学习（end-to-end learning）*。唯一的区别是：特征抽取现在由梯度下降调出来、从训练数据中自动学到，而不是人手工设计。


### 为什么要强调"保留图像的空间结构"？

之前我们讨论过的简单 2-layer 神经网络，会把图像 raw pixels 展平成一个长向量再输入。

但这样做会 *破坏图像的空间结构（spatial structure）*。**当我们处理图像时，应当尊重图像的二维结构！**

---








## 2. 卷积神经网络（Convolutional Neural Networks）

这就引出了 *卷积神经网络（Convolutional Neural Networks）*：

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec5_CNN.webp" alt="CNN architecture design" width="100%" loading="lazy" decoding="async" />
  <figcaption>CNN architecture design</figcaption>
</figure>

**整个网络以端到端方式用反向传播 + 梯度下降训练。**

---


### 回顾 ：全连接层（Recap : Fully-connected layer）

假设我们有一张 $32 \times 32 \times 3$ 的图像，并把它拉成 $3072 \times 1$ 的向量：

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec5_fully_connected_layer_intuition.webp" alt="fully connected layer intuition" width="100%" loading="lazy" decoding="async" />
  <figcaption>fully connected layer intuition</figcaption>
</figure>

我们可以把 $W$ 的一行与输入向量的点积理解为 *模板匹配（template match）*！所以输出值就是 *模板匹配分数（template matching score）*，它告诉我们输入与哪个模板最匹配。

---



### 1 ) 卷积层（Convolutional layer）

现在我们有一张 $32 \times 32 \times 3$ 的图像，我们保留原始空间结构，所以它是一个 3 维张量（3 个通道）。

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec5_convolutional_layer_intuition1.webp" alt="convolutional layer intuition" width="100%" loading="lazy" decoding="async" />
  <figcaption>convolutional layer intuition</figcaption>
</figure>

我们的 filter 需要和输入张量具有相同的通道数。**然后我们把 filter 在图像上滑动（slide），并计算点积。**

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec5_convolutional_layer_filter_slide.webp" alt="filter slide over the image" width="45%" loading="lazy" decoding="async" />
  <figcaption>filter slide over the image</figcaption>
</figure>

我们可以把这个 filter 看作一个 *子模板（subtemplate）*，我们实际上是在 *匹配（matching）* 这个 subtemplate 与图像的子块！

计算完所有点积激活后，我们把它们收集起来，就得到一张 *二维激活图（2-D activation map）*。

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec5_convolutional_layer_activation_map.webp" alt="activation plane of a filter" width="100%" loading="lazy" decoding="async" />
  <figcaption>activation plane of a filter</figcaption>
</figure>

假设我们一共有 6 个这样的 filter，最终就得到 6 张激活图。把它们堆叠起来：

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec5_convolutional_layer_activation_maps_stacking.webp" alt="activation maps stacking" width="100%" loading="lazy" decoding="async" />
  <figcaption>activation maps stacking</figcaption>
</figure>

注意：卷积层把一个 3 维输入图像和 4 维（$6 \times 3 \times 5 \times 5$）的 filter 张量作为输入，给出 6 张响应（激活）平面。

收集所有响应平面并堆叠为一个 3 维张量后，在这个卷积神经网络里我们用一个 6 维偏置向量。偏置向量的每一维只对对应的那个 filter 生效（通过 *广播（broadcasting）*）。


更一般地，使用 *批输入（batched input）* 时，结构如下图：

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec5_convolutional_layer_batched_input.webp" alt="batched input" width="100%" loading="lazy" decoding="async" />
  <figcaption>batched input</figcaption>
</figure>

> ***ConvNet = Conv layers + activations***
>
> 即：卷积神经网络 = 卷积层 + 激活函数

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec5_ConvNet=Convlayers+activations.webp" alt="ConvNet = Conv layers + activations" width="100%" loading="lazy" decoding="async" />
  <figcaption>ConvNet = Conv layers + activations</figcaption>
</figure>

注意这里激活函数非常关键，因为点积是一个 *线性算子（linear operator）*，而 *线性算子的复合仍然是线性算子*。

---


#### 滤波器（filters）学到了什么？

之前，线性分类器（MLP）每个类别学一个模板，这些模板构成 *整图模板库（a bank of whole-image templates）*。

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec5_MLP_template_bank_learned.webp" alt="first layer learned" width="50%" loading="lazy" decoding="async" />
  <figcaption>first layer learned</figcaption>
</figure>

而现在，因为每个 filter 只是图像的一个子块，第一层卷积学到的就是 *局部图像模板（local image templates）*（通常是有向边、对立颜色等）。更深的卷积层倾向于学 *更大的结构（larger structures）*。

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec5_first_convlayer_learned.webp" alt="first layer learned" width="100%" loading="lazy" decoding="async" />
  <figcaption>first layer learned</figcaption>
</figure>

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec5_deeper_convlayers_learned.webp" alt="deeper layers learned" width="100%" loading="lazy" decoding="async" />
  <figcaption>deeper layers learned</figcaption>
</figure>

---



#### 填充（Padding）

输入：$W \times W$
filter：$K \times K$
输出：$W - K + 1$

**问题**：特征图每过一层都会缩小（Feature maps shrink with each layer）！
**解决方案**：在滑动 filter 之前先在输入周围加 *填充（padding）*。

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec5_padding.webp" alt="padding" width="50%" loading="lazy" decoding="async" />
  <figcaption>padding</figcaption>
</figure>

填充：$P$
输出：$W - K + 1 + 2P$

**常见设置**：$P = (K-1)/2$，这样输出就和输入保持同样大小尺寸。

---


#### 感受野（Receptive Fields）

> **感受野**：输出特征图上的一个点，能"看到"原始输入图像的多大区域，就是它的感受野。感受野决定了：一个神经元能利用多大范围的输入信息来做判断。

对于 kernel size 为 $K$ 的卷积，输出中的每一个元素都依赖于输入中一个 $K \times K$ 的感受野（注意 ：感受野 = 边长 = $K$ ）。

每多一层卷积，感受野就增加 $K - 1$。有 $L$ 层的话，感受野大小为：

$$
1 + L \times (K - 1)
$$

**简单推导**：

假设每一层都是 $K \times K$ 卷积，stride $= 1$，无 padding。

- 第一层：感受野 $= K \times K$。
- 第二层：每一个输出元素，来自第一层的一个 $K \times K$ 区域。而第一层每个元素又对应输入中一个 $K \times K$ 区域。所以第二层输出元素对应的输入区域是：

$$
(K + K - 1) \times (K + K - 1)
$$

感受野大小（注意是感受面积的边长！）为：

$$
1 + 2(K - 1)
$$

进而可得这个情况下第 $n$ 层感受野为：

$$
1 + n(K - 1)
$$

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec5_receptive_fields.webp" alt="receptive fields" width="100%" loading="lazy" decoding="async" />
  <figcaption>receptive fields</figcaption>
</figure>

进一步还有"**有效感受野**（effective receptive fields）"的概念，有效感受野包含于上述理论感受野当中。

**在 ConvNet 中，effective receptive fields 实际上随卷积层深度 *线性* 增长。**

> **问题**：对于大图像，我们需要很多层才能让每个输出"看"到整张图。
>
> **解决**：我们希望以一种更高效的方式更快地扩大感受野。**在网络内部做下采样**。
>
> （下采样：降低特征图的空间分辨率（高和宽），让特征图变小，会让后续层的感受野增长更快。）

---



#### 跨步卷积（Strided convolution）

输入：$W \times W$
filter：$K \times K$
填充：$P$
步长：$S$
输出：$(W - K + 2P) / S + 1$

当我们做跨步卷积时，实际上是在网络内部对图像做下采样。

**现在 effective receptive field 可以随着层数增加而以指数速度增长！这样用更少的层堆叠就可以让 effective receptive field 大到足以覆盖整张原图。**


> 关于 strided convolution 感受野大小的推导：


**1. 先定义"第 $l$ 层的步长" $R_l$**

设 $R_l$ 表示：第 $l$ 层输出移动 1 个位置，对应到原始输入中移动了多少个位置。

① 第 1 层直接作用在输入上，stride $= S$。
所以第 1 层输出移动 1 格，输入移动 $S$ 格：

$$
R_1 = S
$$

② 第 2 层作用在第 1 层输出上，stride $= S$。第 2 层输出移动 1 格 → 第 1 层移动 $S$ 格 → 输入移动 $S \cdot S = S^2$ 格。

$$
R_2 = S^2
$$

进而可以推出：

$$
R_l = S^l
$$


**2. 第 $l$ 层的感受野从哪里来？**

第 $l$ 层的某个输出元素，是由第 $l-1$ 层中连续 $K$ 个位置算出来的。

设这 $K$ 个位置为：

$$
p, p+1, p+2, \dots, p+K-1
$$

其中 $p$ 是起点。这 $K$ 个位置中，第一个和最后一个的输入覆盖范围：

- 第 $l-1$ 层位置 $p$ 的感受野，覆盖输入中某个区间，起点为 $a$；
- 第 $l-1$ 层位置 $p+K-1$ 的感受野，覆盖输入中某个区间，起点为：

$$
a + (K-1) \cdot R_{l-1}
$$

这里是因为第 $l-1$ 层输出每移动 1 格，对应到输入中移动 $R_{l-1}$ 格。所以从位置 $p$ 到位置 $p+K-1$，中间移动了 $K-1$ 格，对应输入中移动了：

$$
(K-1) \cdot R_{l-1}
$$


**3. 第 $l$ 层感受野的增量**

**第 $l$ 层的感受野，是这 $K$ 个第 $l-1$ 层位置的感受野的并集。**

第 1 个位置贡献了最左端；第 $K$ 个位置贡献了最右端；中间的位置都被包含在内。

所以第 $l$ 层感受野的边长，比第 $l-1$ 层感受野的边长多出：

$$
\begin{aligned}
\Delta_l &= (K-1) \cdot R_{l-1} \\
         &= (K-1) \cdot S^{l-1}
\end{aligned}
$$

有了这个增量式，进一步地，我们可以推出步长 stride $= S$ 下的感受野大小：

$$
\begin{aligned}
r_1 &= K \\
\Delta_l &= (K - 1) \cdot S^{l-1} \\
r_l &= r_{l-1} + \Delta_l \\
\Rightarrow r_l &= K + (K-1) \cdot \frac{S^l - S}{S - 1} \quad (S \neq 1)
\end{aligned}
$$

所以此时 $r_l$ 随着层数 $l$ 增大成指数形式增大！

---

#### 卷积层小结（Convolution Summary）

> **Input:** $C_{\text{in}} \times H \times W$
>
> **Hyperparameters:**
>
> - Kernel size: $K_H \times K_W$
> - Number filters: $C_{\text{out}}$（注意 filter 个数与输出 channels 数相等！）
> - Padding: $P$
> - Stride: $S$
>
> **Weight matrix:** $C_{\text{out}} \times C_{\text{in}} \times K_H \times K_W$
> giving $C_{\text{out}}$ filters of size $C_{\text{in}} \times K_H \times K_W$
>
> **Bias vector:** $C_{\text{out}}$
>
> **Output size:** $C_{\text{out}} \times H' \times W'$ where:
>
> - $H' = (H - K + 2P) / S + 1$
> - $W' = (W - K + 2P) / S + 1$
>
> **Common settings:**
>
> - $K_H = K_W$（small square filters）
> - $P = (K - 1) / 2$（"same" padding）
> - $C_{\text{in}}, C_{\text{out}} = 32, 64, 128, 256$（powers of 2）
> - $K = 3, P = 1, S = 1$（$3 \times 3$ conv）
> - $K = 5, P = 2, S = 1$（$5 \times 5$ conv）
> - $K = 1, P = 0, S = 1$（$1 \times 1$ conv）
> - $K = 3, P = 1, S = 2$（downsample by 2）

---

#### 1-D 与 3-D 卷积（1-D and 3-D convolution）

此外，卷积还可以从 2-D 推广到 *1-D* 和 *3-D*。

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec5_1D_convolution.webp" alt="1-D convolution" width="60%" loading="lazy" decoding="async" />
  <figcaption>1-D convolution</figcaption>
</figure>

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec5_3D_convolution.webp" alt="3-D convolution" width="60%" loading="lazy" decoding="async" />
  <figcaption>3-D convolution</figcaption>
</figure>

---





### 2 ) 池化层（Pooling Layer）

Pooling 是网络内部另一种 *下采样（downsample）* 方式。

Pooling 做下采样很便宜，几乎不增加计算量，大部分计算其实都发生在卷积层。

给定一个 $C \times H \times W$ 的输入，Pooling 分别对每个 $1 \times H \times W$ 的平面做下采样，输出 channels 数不变、空间尺寸变小：

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec5_pooling_figure.webp" alt="pooling" width="80%" loading="lazy" decoding="async" />
  <figcaption>pooling</figcaption>
</figure>

#### 最大池化（Maxpooling）

Pooling 使用的一种常见下采样方式是 *最大池化（max pooling）*。

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec5_maxpooling_figure.webp" alt="maxpooling" width="100%" loading="lazy" decoding="async" />
  <figcaption>maxpooling</figcaption>
</figure>

它实际上只是在空间上做压缩。

注意：Pooling 层通常不使用 padding。

> Max pooling 本身是非线性操作（取最大值），所以它引入了非线性；Average pooling 是线性操作，所以通常在 average pooling 前需要 ReLU 等非线性激活；实际网络中 max pooling 后通常仍会有 ReLU。

---

#### 池化层小结（Pooling Summary）

> **Input:** $C \times H \times W$
>
> **Hyperparameters:**
>
> - Kernel size: $K$
> - Stride: $S$
> - Pooling function: max, avg
>
> **Output size:** $C \times H' \times W'$ where:
>
> - $H' = (H - K) / S + 1$
> - $W' = (W - K) / S + 1$
>
> **No learnable parameters.**
>
> **Common setting:**
>
> - max, $K = 2, S = 2$ ⇒ $2 \times$ downsampling

---


### 卷积与池化 ：平移等变性（Translation Equivariance）

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec5_translation_equivariance.webp" alt="translation equivariance" width="100%" loading="lazy" decoding="async" />
  <figcaption>translation equivariance</figcaption>
</figure>

"**输入平移多少，输出也平移多少**"：先平移再卷积 $=$ 先卷积再平移。

e.g. 假设图像中有一只猫在左上角，卷积后某个位置会有一个强响应。现在把猫移到右下角，卷积后右下角对应的位置也会有同样的强响应。猫移动了，响应也移动了相同距离。

> **直觉**：图像的特征不依赖于它们在图像中的位置。

---





## 参考资料

- [CS231n Lecture 5 — Convolutional Neural Networks](https://cs231n.stanford.edu/slides/2026/lecture_5.pdf) — 2026 slide PDF
- [CS231n 2025 spring](https://www.bilibili.com/video/BV1YJ3PzLEiW) — CS231n Spring 2025 视频
- [CS231n Lecture 5 notes (Fei-Fei Li et al.)](https://cs231n.github.io/convolutional-networks/) — 配套讲义：卷积神经网络
- [Convolutional Neural Networks (Wikipedia)](https://en.wikipedia.org/wiki/Convolutional_neural_network) — CNN 的背景与历史
- [A guide to receptive fields in arithmetic (Araujo et al., 2019)](https://arxiv.org/abs/1901.07421) — effective receptive fields 的进一步讨论