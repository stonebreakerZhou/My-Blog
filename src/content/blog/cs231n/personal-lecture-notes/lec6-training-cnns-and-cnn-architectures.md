---
title: "CS231n : Lec 6 — CNN 架构与训练"
description: CS231n Lecture 6 学习笔记，覆盖归一化层（BN/LN/IN/GN）、Dropout、激活函数、VGGNet、ResNet、权重初始化、数据预处理与增强、迁移学习、超参数选择。
pubDate: 2026-10-05
series: cs231n
subSeries: personal-lecture-notes
order: 6
categories:
  - CS231n
  - CNN
  - ResNet
  - Normalization
---

## 引子

Lec 5 我们讲了 CNN 的核心构件（卷积层、池化层）。Lec 6 转向一个更实际的问题：**怎么训练、怎么搭一个能 work 的 CNN？**

这一讲分成两个大问题：

- **How to build CNNs?**
  - Layers in CNNs
  - CNN Architectures（CNN 架构）
  - Activation Functions
  - Weight Initialization

- **How to train CNNs?**
  - Data Preprocessing
  - Data Augmentation
  - Transfer Learning
  - Hyperparameter Selection

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec6_CNNs_components.webp" alt="CNNs components" width="100%" loading="lazy" decoding="async" />
  <figcaption>CNNs components</figcaption>
</figure>
---





## 1. CNN 中的层

### 归一化层（Normalizing Layers）

首先进行归一化：

$$
\hat{x}_{i,j} = \frac{x_{i,j} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}
$$

其中：

- $x_{i,j}$：第 $i$ 个样本、第 $j$ 个特征（或通道）的值；
- $\mu_j$：第 $j$ 个特征的均值；
- $\sigma_j^2$：第 $j$ 个特征的方差；
- $\epsilon$：防止除零的小常数。

**核心问题**：均值和方差是在哪些维度上计算的？这就决定了下面四种 normalization 的核心差异！

> **High-level Idea**：Learn parameters that let us *scale / shift* the input data
>
> ① Normalize input data
> ② Scale / shift using learned parameters

因为强行把每层都归一化到均值为 0、方差为 1，可能会限制网络的表达能力。加入可学习的缩放和平移，可以让网络自己决定：是否需要保留归一化后的分布，还是恢复一部分原始分布。

所以完整的归一化层是：

$$
y = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$


现在假设特征图张量形状为：

$$
(N, C, H, W)
$$

其中：$N$：batch size；$C$：通道数；$H$：高度；$W$：宽度。


下面来看四种 normalization 方法及其差异：

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec6_normalization_layers_intuition.webp" alt="normalization layers" width="100%" loading="lazy" decoding="async" />
  <figcaption>normalization layers</figcaption>
</figure>

#### ① Batch Normalization (BN)

对每个通道，在 $(N, H, W)$ 上计算均值和方差。

也就是跨样本、跨空间位置，对同一个通道做归一化。

- **适用场景**：CNN，batch size 较大时。
- **缺点**：batch size 小时，统计量不稳定。

#### ② Layer Normalization (LN)

对每个样本，在 $(C, H, W)$ 上计算均值和方差。

**含义**：对每个样本的所有通道和空间位置一起归一化。

- **适用场景**：RNN、Transformer。
- **优点**：不依赖 batch size。

#### ③ Instance Normalization (IN)

对每个样本的每个通道，在 $(H, W)$ 上计算均值和方差。

**含义**：对每张图的每个通道单独归一化。

- **适用场景**：风格迁移（style transfer）。
- **特点**：不依赖 batch，也不混合通道。

#### ④ Group Normalization（GN）

把通道分成若干组，对每个样本的每组通道，在 $(H, W)$ 和组内通道上计算均值和方差。

**含义**：介于 Layer Norm 和 Instance Norm 之间。

- 当 $G = 1$：退化为 Layer Norm；
- 当 $G = C$：退化为 Instance Norm。

- **适用场景**：batch size 小的情况，如目标检测、分割。
- **优点**：不依赖 batch size，效果稳定。

---




### Dropout

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec6_dropout_figure.webp" alt="dropout" width="100%" loading="lazy" decoding="async" />
  <figcaption>dropout</figcaption>
</figure>

In each forward pass, randomly set some neurons to zero. Probability of dropping is a hyperparameter; 0.5, 0.25 are common.

This is actually forcing the network to have a redundant representation, and prevents co-adaptation of features.（注：训练时随机丢弃一些决策特征能让模型不会过度依赖这些特征来做决策！）

> **另一种解读**：Dropout is training a large *ensemble* of models (that share parameters). Each binary mask is one model.
>
> 因为在 *test time* 我们会移除 dropout 并按丢弃概率对输出激活做缩放，所以把它理解为一种集成学习是合理的。

---







## 2. 激活函数（Activation Functions）

### Sigmoid

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec6_sigmoid_figure.webp" alt="sigmoid" width="50%" loading="lazy" decoding="async" />
  <figcaption>sigmoid</figcaption>
</figure>

历史上很流行，因为它可以被解读为神经元的"放电率"

> **Key problem**：大的正/负值会让梯度"饱和到零"。多层 sigmoid 堆叠起来，实际中梯度会一层比一层小。


### ReLU（Recified Linear Unit）

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec6_ReLU.webp" alt="ReLU" width="50%" loading="lazy" decoding="async" />
  <figcaption>ReLU</figcaption>
</figure>

**Advantages：**

① 不会饱和（在正区间），正区间上梯度恒为 1
② 计算非常高效
③ 实际中比 sigmoid 收敛快得多，例如 6×

**Problems：**

① 输出不是以零为中心的
② 一个烦人的问题：当 $x < 0$ 时就会出现 Dead ReLUs！


### GELU（Gaussian Error Linear Unit）

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec6_GELU_figure.webp" alt="GELU" width="70%" loading="lazy" decoding="async" />
  <figcaption>GELU</figcaption>
</figure>

**Advantages：**

① 在零点附近有非常平滑的行为
② 平滑性让训练更顺畅

**Problems：**

① 计算成本比 ReLU 高
② 当输入为大负值时梯度仍可能趋于 0

它是今天 Transformer 中主要使用的激活函数。

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec6_activation_func_zoo.webp" alt="activation function zoo" width="100%" loading="lazy" decoding="async" />
  <figcaption>activation function zoo</figcaption>
</figure>

GELU 的梯度值近似于累积高斯分布概率值（从0上升到1）

> CNN 中的激活函数通常放在 *线性算子之后*（linear operators）—— 比如前馈 / 全连接层、卷积层等。

---








## 3. CNN 架构（CNN Architectures）

### Case study ：VGGNet（2014）

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec6_VGGNet_architecture.webp" alt="VGG architecture" width="100%" loading="lazy" decoding="async" />
  <figcaption>VGG architecture</figcaption>
</figure>

如果把 AlexNet 和 VGGNet 比较，变化就是 **"Small filters, Deeper networks"**。

**Why *smaller filters* ($3 \times 3$ Conv) ?**

三层 $3 \times 3$ conv (stride 1) 堆起来，和一个 $7 \times 7$ conv 层有相同的 effective receptive field。（注意 $3 \times 3$ conv 的感受野每加深一层增加 2。）

但更深的堆叠允许 *更多的非线性*。并且 *参数更少* ！：$3 \times (3^2 C^2) < 7^2 C^2$（其中 $C$ 为每层通道数）。

---



### Case Study ：ResNet

如果我们在"plain"的卷积神经网络上继续堆叠更深的层，训练误差和测试误差都会上升！（这并不是 overfit 导致的！）

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec6_deeper_layer_higher_error.webp" alt="deeper layer with higher error" width="100%" loading="lazy" decoding="async" />
  <figcaption>deeper layer with higher error</figcaption>
</figure>

这让人困惑，因为更深的网络应该比浅层网络有更强的表征能力，进而性能应该是碾压才对。

如果做一个思想实验：当把深层网络比浅层网络多余的层设成 identity function 时，深层网络就变成了浅层网络。**所以深层网络至少应该表现得和浅层网络一样好。**

所以我们认为上面出现的那种深层网络被浅层网络"碾压"的现象应当是一个 **optimization problem**（深层模型优化更难！）

（注意：此时的优化问题是复杂模型在训练中被卡住了，通常是因为优化困难，可能是因为梯度消失、病态曲率或鞍点，并且这个训练瓶颈并不是因为训练时间不够而导致的！）

正是受到这个思想的启发，我们的直觉是：深层模型至少应当学习得与浅层模型一样好，那么在构造上一个简单的思想实验就是：copying the learned layers from the shallower model and setting additional layers to identity mapping.

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec6_copy_from_shallow_model.webp" alt="copy from shallow model + identity" width="30%" loading="lazy" decoding="async" />
  <figcaption>copy from shallow model + identity</figcaption>
</figure>

进一步地，ResNet 的思路就是：**Use network layers to fit a residual mapping instead of directly trying to fit a desired underlying mapping**.

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec6_ResNet_figure1.webp" alt="ResNet" width="100%" loading="lazy" decoding="async" />
  <figcaption>ResNet</figcaption>
</figure>

（也就是说，我们不再让网络直接去拟合深层网络的复杂映射 $H(x)$，而是让这些层去拟合一个残差映射 $F(x) = H(x) - x$。这个 $F(x)$ 正是浅层网络与深层网络映射之间的"差量" $\delta$！）


> **前向视角**：如果恒等映射就是最优解的话，那么让网络学习一个零映射比学习一个恒等映射容易得多！这个残差映射 $F(x) = H(x) - x$ 也更容易学习！

> **反向视角**：残差连接提供了一条恒等路径（残差流）。反向传播求导时：

$$
\frac{\partial x_{l+1}}{\partial x_l} = \frac{\partial F(x_l)}{\partial x_l} + 1
$$

这个 $+1$ 保证了即使残差分支 $F$ 的梯度很小，梯度仍然可以沿着恒等路径无损回传，不会指数衰减消失。

> Residual blocks help us use more data to build more complex model!
>
> Very deep networks are using residual connections.

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec6_ResNet_figure2.webp" alt="ResNet full architecture" width="100%" loading="lazy" decoding="async" />
  <figcaption>ResNet full architecture</figcaption>
</figure>

---






## 4. 权重初始化（Weight Initialization）

### Analysis

如果初始化的权重值太小，对于更深的网络层，所有激活值都会趋向于 0。

如果初始化的权重值太大，激活值会迅速爆炸（均值和方差都会爆炸）。

所以初始化的目的就是：每一层的输出（激活值）方差大致保持一致，不缩不放（输入与输出方差一样，不变）。


### Kaiming / MSRA Initialization

这是一个专门为 ReLU 网络设置的初始化方法。

以前的 Xavier 初始化做法：我们初始化权重参数分布为：

$$
\sigma_w = \frac{1}{\sqrt{D_{\text{in}}}}
$$

或者

$$
\sigma_w = \sqrt{\frac{2}{D_{\text{in}} + D_{\text{out}}}}
$$

但是注意 Xavier 初始化假设激活函数是线性的，**而 ReLU 会把一半神经元置零，导致方差减半**。所以我们改用这样的初始化：

$$
\sigma_w = \sqrt{\frac{2}{D_{\text{in}}}}
$$

---





## 5. 数据预处理（Data Preprocessing）

对于图像数据，现代 CNN 几乎都采用同一种预处理方法：

> **对每个通道，减去该通道的均值，再除以该通道的标准差。**

**Advantages：**

① **加速收敛**：归一化后各通道尺度一致，损失曲面更接近圆形，梯度下降更顺畅。
② **避免通道主导**：如果某个通道数值范围大，它的梯度也会大，可能主导参数更新。归一化后所有通道公平参与。
③ **数值稳定性**：将输入预处理为在 0 附近且方差为 1，激活值和梯度不会过大或过小，训练更稳定。

---





## 6. 数据增强（Data Augmentation）

首先我们回顾一下正则化（regularization）：

① Training: Add some kind of randomness
② Testing: Average out randomness (sometimes approximate)

正则化实际上使用的是集成学习（Ensemble）的思想：

每次训练时加入不同的随机性，相当于在训练很多个略有不同的模型；测试时把这些模型的效果平均起来；平均后的模型比单个模型更鲁棒，不容易过拟合。

之前我们讲过 dropout，现在来讲 **Data augmentation**，这是数据层面的一种正则化手段。


### Horizontal flips（水平翻转）

把图像左右翻转。对于大多数分类任务（猫、狗、汽车），翻转不改变类别。


### Random Crops and Scales（随机裁剪和缩放）

**1 ) ResNet 训练策略**

随机选一个 $L \in [256, 480]$；把训练图像的短边缩放到 $L$；随机裁剪一个 $224 \times 224$ 的 patch。

这样每次看到的都是图像的不同部分和不同尺度。

测试时：不做随机裁剪，而是用固定的一组裁剪，取平均。

**2 ) ResNet 测试策略（Test Time Augmentation）**

把图像缩放到 5 个尺度：$\{224, 256, 384, 480, 640\}$；每个尺度取 10 个 $224 \times 224$ 裁剪：4 个角 + 中心，加上水平翻转；总共 $5 \times 10 = 50$ 个裁剪，分别预测，取平均。

这是 Test Time Augmentation（TTA）。


### Color Jitter（颜色抖动）

随机改变图像的对比度和亮度。

**简单做法**：随机调整亮度；随机调整对比度；随机调整饱和度；随机调整色调。

这样模型不会过度依赖颜色信息。


### Cutout

训练时，随机把图像中的某些矩形区域置零（变成黑色或灰色）。测试时使用完整图像。

**效果**：迫使模型不能只依赖图像的某一块区域；学会从多个区域综合判断；对小数据集（如 CIFAR）效果很好。

> 进行了 data augmentation 操作之后我们相当于也是很好地利用了数据，让模型能够从多个角度学习已有数据，直观上就是"对输入图加噪点防止过拟合"这一种正则化思想。

---





## 7. 迁移学习（Transfer Learning）

如果我们手头数据不多，依然可以训练 CNN。

**1）步骤 1**：获取一个已经在大数据集（如 ImageNet）上预训练好的模型。

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec6_transfer_learning_figure1.webp" alt="step 1: train on a large dataset" width="20%" loading="lazy" decoding="async" />
  <figcaption>step 1: train on a large dataset</figcaption>
</figure>

**2）步骤 2**：冻结所有深层参数，只替换最后的线性层，在我们给定的小数据集上训练。

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec6_transfer_learning_figure2.webp" alt="step 2: change the last linear layer and train" width="60%" loading="lazy" decoding="async" />
  <figcaption>step 2: change the last linear layer and train</figcaption>
</figure>

**3）步骤 3**：如果给定的数据集足够大，就可以用这个数据集对预训练模型做 *finetune*（微调）。

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec6_transfer_learning_figure3.webp" alt="step 3: finetune the pretrained model" width="60%" loading="lazy" decoding="async" />
  <figcaption>step 3: finetune the pretrained model</figcaption>
</figure>

### 经验方法表

|                | **与预训练数据集非常相似**                                  | **与预训练数据集差异很大**           |
| -------------- | ------------------------------------------------ | ------------------------- |
| **当前任务的数据集极小** | 只在最后一层训练一个 *线性分类器*                               | 尝试换一个预训练模型，或者去收集更多数据      |
| **当前任务的数据集较大** | 对整个模型的各层都做 *finetune*（Finetune all model layers） | 对所有层做 finetune，或者从头训练一个模型 |

---





## 8. 超参数选择（Hyperparameter Selection）

一个 7 步流程：

1. **检查初始损失**：用随机初始化权重，跑一次前向传播；看初始损失值是否合理。这一步目的是验证网络结构和初始化是否正确。

2. **过拟合一个小样本**：取一小部分训练数据（比如 $5 \sim 10$ 张图）；关掉正则化（或设得很小）；训练几百步；看训练准确率能否达到 100%。这一步目的是验证网络有能力学习。

3. **找到能让损失下降的学习率**：使用上一步的网络结构；使用全部训练数据；打开很小的 weight decay；尝试不同的学习率，看哪个能在 100 次迭代内让损失显著下降。这一步目的是确定学习率的合理量级。

4. **粗网格搜索，训练 $1 \sim 5$ 个 epoch**：选定几个超参数：学习率、正则化强度、网络大小等；每个组合训练 $1 \sim 5$ 个 epoch；记录验证集准确率。这一步目的是在合理范围内粗筛超参数组合。

5. **细化网格，训练更久**：取粗筛中表现最好的几个组合；在它们附近细化网格；训练更久（比如 $10 \sim 20$ 个 epoch）；继续比较验证集表现。

6. **看损失和准确率曲线（3 种曲线情况）**：

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec6_accuracy_case1.webp" alt="case 1: need more training" width="70%" loading="lazy" decoding="async" />
  <figcaption>case 1: need more training</figcaption>
</figure>

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec6_accuracy_case2.webp" alt="case 2: overfit" width="80%" loading="lazy" decoding="async" />
  <figcaption>case 2: overfit</figcaption>
</figure>

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec6_accuracy_case3.webp" alt="case 3: underfit" width="80%" loading="lazy" decoding="async" />
  <figcaption>case 3: underfit</figcaption>
</figure>

7. **回到第 5 步，循环迭代**（调参是一个迭代过程）。

---


### 2 种搜索策略

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec6_hyperparameter_2_search_method.webp" alt="2 search methods" width="100%" loading="lazy" decoding="async" />
  <figcaption>2 search methods</figcaption>
</figure>

**Grid Search（网格搜索）**：在每个超参数上取固定的一组值；组合成网格，逐个尝试。

> **问题**：维度灾难：超参数越多，组合数指数增长；计算量巨大；很多组合其实效果差不多，浪费计算。

**Random Search（随机搜索）**：在每个超参数的范围里随机采样；不需要遍历所有组合。

> **优势**：研究表明，随机搜索通常比网格搜索更高效；因为通常只有少数几个超参数真正重要；随机搜索能更密集地覆盖重要参数，而网格搜索会在不重要的参数上浪费大量试验。

**结论**：优先使用随机搜索，而不是网格搜索。

---






## 参考资料

- [CS231n Lecture 6 — CNN Architectures and Training](https://cs231n.stanford.edu/slides/2026/lecture_6.pdf) — 2026 slide PDF
- [CS231n 2025 spring](https://www.bilibili.com/video/BV1YJ3PzLEiW) — CS231n Spring 2025 视频
- [Batch Normalization (Ioffe & Szegedy, 2015)](https://arxiv.org/abs/1502.03167) — BN 原始论文
- [Group Normalization (Wu & He, 2018)](https://arxiv.org/abs/1803.08494) — GN 原始论文
- [Deep Residual Learning for Image Recognition (He et al., 2015)](https://arxiv.org/abs/1512.03385) — ResNet 原始论文
- [Delving Deep into Rectifiers (He et al., 2015)](https://arxiv.org/abs/1502.01852) — Kaiming/MSRA 初始化原始论文