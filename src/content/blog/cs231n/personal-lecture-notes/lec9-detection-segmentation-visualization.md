---
title: "CS231n : Lec 9 — 语义分割、目标检测、实例分割、模型可视化"
description: CS231n Lecture 9 学习笔记，覆盖语义分割、目标检测（R-CNN / Fast R-CNN / YOLO / DETR）、实例分割（Mask R-CNN）以及模型可视化（Filters / Saliency / CAM / Grad-CAM）。
pubDate: 2026-10-08
series: cs231n
subSeries: personal-lecture-notes
order: 9
categories:
  - CS231n
  - Semantic Segmentation
  - Object Detection
  - Instance Segmentation
  - R-CNN
  - YOLO
  - DETR
  - Mask R-CNN
  - CAM
  - Grad-CAM
---

## 引子

Lec 8 讲了注意力机制（Attention）/ Transformer。Lec 9 回到计算机视觉（CV）的几大核心任务：**检测、分割、可视化**。

这一讲分成两大块：

- **计算机视觉任务（Computer Vision Tasks）**
  - 语义分割（Semantic Segmentation）
  - 目标检测（Object Detection）
  - 实例分割（Instance Segmentation）
- **可视化（Visualization）**
  - 模型层可视化（Model Layers Visualization）
  - 显著性图（Saliency Maps）
  - CAM 与 Grad-CAM

---






## 1. 语义分割（Semantic Segmentation）

**任务**：给图像中每个像素都打上一个类别标签。**不区分具体实例**，只关心像素本身的类别。

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec9_semantic_segmentation_eg1.webp" alt="semantic segmentation e.g.1" width="100%" loading="lazy" decoding="async" />
  <figcaption>semantic segmentation e.g.1</figcaption>
</figure>

要对每个像素都打标签，只看一个像素的局部信息分类是远远不够的。那怎么引入上下文信息呢？



### Idea 1：滑动窗口（Sliding Window）

我们在整张图上滑动一个窗口，裁出小图块（patches），用 CNN 分类每个图块的中心像素。

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec9_semantic_segmentation_idea1_sliding_window.webp" alt="sliding window" width="100%" loading="lazy" decoding="async" />
  <figcaption>sliding window</figcaption>
</figure>

**问题**：这种方式*非常低效* ！重叠的 patches 之间没有共享特征计算。



### Idea 2：纯卷积（Pure Convolution）

设计一个网络，全部由卷积层组成、不使用下采样算子，一次性输出所有像素的预测！

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec9_semantic_segmentation_idea2_pure_convolution.webp" alt="pure convolution" width="100%" loading="lazy" decoding="async" />
  <figcaption>pure convolution</figcaption>
</figure>

如果我们在卷积层不加入池化，不做 stride > 1 的卷积，不做下采样，每一步都通过 padding 来使得卷积结果尺寸与输入尺寸完全相同，那么经过多层卷积后，输出特征图的空间尺寸就仍然是 $H \times W$。

**问题**：

① 在原图分辨率上做卷积，计算量会非常大。
② 感受野太小，学不到高层语义。所以我们必须堆很多层。



### Idea 3：卷积中加入下、上采样（Convolution with Down + Up Sampling）

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec9_semantic_segmentation_idea3_convolution_with_down+up_sampling.webp" alt="convolution with down+up-sampling" width="100%" loading="lazy" decoding="async" />
  <figcaption>convolution with down+up-sampling</figcaption>
</figure>

把网络设计成一堆卷积层，并在网络内部加入 *下采样（downsampling）* 和 *上采样（upsampling）*！

我们之前已经学过 *下采样* 的方法：*池化（pooling）、跨步卷积（strided convolution）* ……

那么现在如何实现 *上采样* 呢？

---





#### 上采样（UpSampling）


##### 网络内上采样：Unpooling

**① Nearest Neighbor：**

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec9_upsampling_nearest_neighbor.webp" alt="nearest neighbor" width="80%" loading="lazy" decoding="async" />
  <figcaption>nearest neighbor</figcaption>
</figure>

**② "Bed of Nails"：**

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec9_upsampling_bed_of_nails.webp" alt="bed of nails" width="80%" loading="lazy" decoding="async" />
  <figcaption>bed of nails</figcaption>
</figure>


##### 网络内上采样：最大反池化（Max Unpooling）

Max Pooling 时记录最大值的位置索引，Max Unpooling 时把值放回原来的位置，其他位置填 0。

不需要学习参数，并且能保留 Max Pooling 时选中的位置信息。

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec9_upsampling_max-unpooling_figure1.webp" alt="max unpooling" width="100%" loading="lazy" decoding="async" />
  <figcaption>max unpooling</figcaption>
</figure>

> **Note**：Corresponding pairs of downsampling and upsampling layers！

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec9_upsampling_max-unpooling_figure2.webp" alt="max unpooling correspondence relation" width="80%" loading="lazy" decoding="async" />
  <figcaption>max unpooling correspondence relation</figcaption>
</figure>



##### 可学习上采样：跨步转置卷积（Strided Transposed Convolution）

**回顾**：普通的 $3 \times 3$ 卷积，stride = 1，padding = 1。

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec9_normal_3by3_conv_with_s1p1.webp" alt="normal convolution" width="100%" loading="lazy" decoding="async" />
  <figcaption>normal convolution</figcaption>
</figure>

普通卷积可以写成矩阵乘法形式！下面用一个简单的例子可视化：

$$
I = \begin{bmatrix} 1, 2, 3, 4 \\ 5, 6, 7, 8 \\ \vdots, \vdots, \vdots, \vdots \end{bmatrix} \in [4 \times 4] \qquad \text{kernel} = \begin{bmatrix} 1/2, 1/2 \\ 1/2, 1/2 \end{bmatrix} \in [2 \times 2]
$$

易知卷积结果为：

$$
O = \begin{bmatrix} 7, 9, 11 \\ \vdots, \vdots, \vdots \end{bmatrix} \in [3 \times 3]
$$

现在把 $I, O$ flatten 成列向量（按行优先）：

$$
I.\text{flatten} = \begin{bmatrix} 1 \\ 2 \\ 3 \\ \vdots \end{bmatrix} \in \mathbb{R}^{16} \qquad O.\text{flatten} = \begin{bmatrix} 7 \\ 9 \\ 11 \\ \vdots \end{bmatrix} \in \mathbb{R}^{9}
$$

现在我们就是要把这一整套操作改写为：

$$
O.\text{flatten} = R \cdot I.\text{flatten} \qquad R \in [9 \times 16]
$$

如果要直接构造写出矩阵 $R$ 比较麻烦因为很多地方需要填充为 0，由于 $R$ 的每一个行向量与 $I.\text{flatten}$ 的点积就是在一个位置进行一次卷积，所以我们分行向量来分析 $R$：

比如 kernel 一行一行移动，移动到第二个位置，此时感受野卷积为：

$$
W_2 = \begin{bmatrix} 0, 1/2, 1/2, 0 \\ 0, 1/2, 1/2, 0 \\ \vdots, \vdots, \vdots, \vdots \end{bmatrix}
$$

可以发现直接把 $W_2$ flatten 之后就能得到 $R$ 的第二行行向量！

最后可以顺利将卷积操作改写成矩阵乘法：

$$
O.\text{flatten} = R \cdot I.\text{flatten}
$$

这是 down sampling 的矩阵表示，那么进一步，假如我们要进行 up sampling，那么：

$$
\text{recovered}.\text{flatten} = R^\top \cdot O.\text{flatten} \qquad R^\top \in [16 \times 9]
$$

注意 $R^\top$ 的形状，所以转置卷积就是要学习这样一个矩阵实现"逆卷积"的操作（注意 $R^\top$ 不一定就是正向卷积的操作矩阵 $R$！）

---




### 具体架构实现：U-Net

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec9_U-Net.webp" alt="U-Net" width="100%" loading="lazy" decoding="async" />
  <figcaption>U-Net</figcaption>
</figure>

可以看出 U-Net 左侧在做 down sampling，提取高层语义特征，右侧则进行 up sampling。

中间是一个 skip connection：因为左侧 Encoder 在下采样降分辨率的过程中，不可避免地丢失了像素级的精细位置和边缘信息。所以将 Encoder 浅层提取到的高分辨率空间特征图，直接跨层复制并与 Decoder 对应层的特征图进行通道拼接，这样融合了 Encoder 的"精准位置/边缘"和 Decoder 的"高层语义"。

---







## 2. 目标检测（Object Detection）


### 单目标检测（Single Object Detection）

**目标（Goal）**：分类 + 定位

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec9_single_object_detection.webp" alt="single object detection" width="100%" loading="lazy" decoding="async" />
  <figcaption>single object detection</figcaption>
</figure>

这里我们使用 CNN，把定位当作一个回归问题来处理，总的多任务损失（multitask loss）由两个损失相加得到。




### 多目标检测（Multiple Objects Detection）

由于一幅图里面需要检测的物体的个数不定，所以对于不同图片最后的输出个数也是不一样的！

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec9_multiple_objects_detection_uncertain_outputs_number.webp" alt="multiple object detection outputs numbers" width="100%" loading="lazy" decoding="async" />
  <figcaption>multiple object detection outputs numbers</figcaption>
</figure>

但是一个神经网络的输出通常是固定大小的张量！

**A Naive Solution**：对图像的不同裁剪区域应用 CNN，CNN 把每个裁剪区域分类为物体或背景。

**Problem**：Need to apply CNN to huge number of locations, scales, and aspect ratios, very computationally expensive！（穷举所有窗口进行计算太浪费！）





### 候选区域（Region Proposal）：Selective Search

上面讲到穷举所有的 crops 不可行，那么候选区域法提出：不要穷举所有窗口，而是先找出可能包含物体的候选区域，再对这些区域分类。

生成候选区域的常用方法：**Selective Search**：基于颜色、纹理、大小等相似度，把图像分割成很多小区域；逐步合并相似区域，生成约 2000 个候选框；这些候选框覆盖了图像中可能物体的位置。




#### 候选区域网络（Region Proposal Network, RPN）

**① 输入 + 特征提取：**

输入图像后经过 CNN 得到特征图（RPN 会在这个特征图上操作而不是原始图像上！）

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec9_RPN_feature_extraction.webp" alt="feature extraction" width="100%" loading="lazy" decoding="async" />
  <figcaption>feature extraction</figcaption>
</figure>

**② Anchor Box（锚框）**

想象在特征图的每个点上，都有一个固定大小的锚框（此时我们先假设大小为 $20 \times 15$），一个锚框是一个矩形框，由 $(x, y, w, h)$ 定义，它是对"这个位置可能存在的物体"的初始猜测。

对每个锚框，用一个卷积层输出一个分数；分数表示"这个锚框包含物体"的概率；输出形状：每个位置一个分数。在每个点上，预测对应的锚框是否包含物体（二分类）。

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec9_RPN_anchor_boxes_scoring.webp" alt="anchor boxes binary classification" width="100%" loading="lazy" decoding="async" />
  <figcaption>anchor boxes binary classification</figcaption>
</figure>

对于正样本锚框，还要预测从锚框到真实框的修正量（每个位置回归 4 个数字），微调锚框的位置和大小，让它更接近真实框。

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec9_RPN_anchor_boxes_correction.webp" alt="anchor boxes correction" width="100%" loading="lazy" decoding="async" />
  <figcaption>anchor boxes correction</figcaption>
</figure>

实际中，在每个点上使用 $K$ 个不同大小/尺度的锚框：

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec9_RPN_K-different_anchor_boxes.webp" alt="K different anchor boxes" width="100%" loading="lazy" decoding="async" />
  <figcaption>$K$ different anchor boxes</figcaption>
</figure>

把所有 $K \times 20 \times 15$ 个锚框按 objectness 分数排序，取前约 300 个作为候选框。

---




### R-CNN（Region-based CNN）

**① "slow" R-CNN：**

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec9_slow_R-CNN.webp" alt="slow R-CNN" width="100%" loading="lazy" decoding="async" />
  <figcaption>"slow" R-CNN</figcaption>
</figure>

**问题（Problem）**：非常慢！每张图都要做大约 2k 次独立的前向传播！（因为有约 2k 个候选框 crops）

**思路（Idea）**：*先让整张图过一遍 convnet，再去裁剪特征图*（Crop the conv feature instead）！


**② "fast" R-CNN：**

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec9_fast_R-CNN.webp" alt="fast R-CNN" width="100%" loading="lazy" decoding="async" />
  <figcaption>"fast" R-CNN</figcaption>
</figure>

注：此处模型中 Regions of Interest 是用我们上述的 Region Proposal 方法选取得到的！





### 单阶段目标检测器：YOLO / SSD / RetinaNet

**YOLO**：（You Only Look Once，你只看一次）—— 一种实时目标检测器

R-CNN 问题：要经历 RPN + 检测两个阶段，计算量较大，检测速度慢！

YOLO 核心思想：把目标检测变成一个单次回归问题：一张图只跑一次网络，直接输出所有框和类别。

**step ①**：把图像分割成 $s \times s$ 的网格

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec9_YOLO_step1_grid.webp" alt="grid" width="50%" loading="lazy" decoding="async" />
  <figcaption>grid</figcaption>
</figure>

**step ② ：每个格子独立预测**

每个格子预测 $B$ 个边界框，每个边界框包含 5 个值：

$$
(x, y, w, h, \text{confidence})
$$

训练时：

$$
\text{confidence}(\text{目标值}) = P(\text{object}) \cdot (\text{IoU})_{\text{pred}}^{\text{truth}}
$$

分别表示框的位置、大小，以及置信度（这个框里有多大概率包含物体，并且框得有多准）。

每个格子还预测 $C$ 个类别概率 $P(\text{class}_i \mid \text{object})$，表示"如果这个格子里有物体，它属于第 $i$ 类的概率"。

**step ③ ：输出**

网络最后输出一个张量：$s \times s \times (5B + C)$。

**step ④ ：损失值**

YOLO 的损失由三部分组成：（定位 + 置信度 + 分类损失）

$$
L = \lambda_{\text{coord}} L_{\text{coord}} + L_{\text{conf}} + L_{\text{class}}
$$

**step ⑤ ：推理**

得到网络输出的张量后，对每个框计算最终分数：

$$
\text{score} = \text{confidence} \cdot P(\text{class}_i \mid \text{object})
$$

按阈值过滤低分框；对每个类别做 NMS，去除重叠框；得到最终检测结果。

> 注意在训练时，不是每个格子都有真实标签。如果一个物体的中心点落在某个格子内，那么这个格子负责检测这个物体。（其他格子的框的 confidence 都被设置为 0）






### 基于 Transformer 的目标检测：DETR

**DETR 核心思想**：

把目标检测变成一个集合预测问题：输入图像，直接输出一个固定大小的集合，包含所有物体。

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec9_DETR_basic_architecture.webp" alt="DETR basic architecture" width="100%" loading="lazy" decoding="async" />
  <figcaption>DETR basic architecture</figcaption>
</figure>

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec9_DETR_detailed_architecture.webp" alt="DETR detailed architecture" width="100%" loading="lazy" decoding="async" />
  <figcaption>DETR detailed architecture</figcaption>
</figure>

最后输出有两个头：分类头（物体类别） + 回归头（框位置）。

把物体检测任务看成一个集合预测问题：输入图像，输出一个集合（集合大小可变），每个元素是"类别 + 框"。

Transformer 天生就是处理集合的！输入序列，输出也是序列（可以是集合）。

Transformer 的自注意力可以全局建模，自动找到物体位置；不需要显式生成候选框。

最后可以进行端到端学习！

---








## 3. 实例分割（Instance Segmentation）

**回顾（Recall）**：fast R-CNN：

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec9_recall_fast-R-CNN.webp" alt="fast R-CNN" width="100%" loading="lazy" decoding="async" />
  <figcaption>fast R-CNN</figcaption>
</figure>

在 Fast R-CNN 的基础上，并行添加一个 mask 分支，为每个 RoI 预测一个 $28 \times 28$ 的二值掩码：

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec9_mask_R-CNN_architecture.webp" alt="mask R-CNN" width="100%" loading="lazy" decoding="async" />
  <figcaption>mask R-CNN</figcaption>
</figure>

RPN 生成 RoI（region of interest）之后，经过 pooling（注：mask R-CNN 用的是 RoI align）变成固定大小的特征。（e.g. 每一块都是 $14 \times 14 \times 256$ 的特征）


**① 分类头**：对每个 RoI 的特征，送入全连接层；输出 $C + 1$ 个类别的概率（$C$ 个物体类别 + 1 个背景）；用 softmax 得到概率分布。

**② 回归头**：对每个 RoI 的特征，送入另一个全连接层；输出 4 个修正量 $(t_x, t_y, t_w, t_h)$；用来微调 RoI 的位置和大小（精确边界框）。

**③ Mask 分支**：将 RoI 处理后得到的特征输入到 4 层 $3 \times 3$ Conv + Transposed Conv $2 \times 2$, stride 2 + $1 \times 1$ Conv，最后输出 $28 \times 28 \times K$，即映射到 $K$ 个类别上。


由于我们在分类头已经得到每个 RoI 的类别（假设此时是"猫"），那么从 Mask 分支输出中取出对应的那一层特征，对这 $28 \times 28$ 个 logits 逐像素做 sigmoid，得到每个像素属于"猫"的概率。设定一个概率阈值后可以取出那些属于"猫"的像素。（其实也是得到一个二值掩码）

最后把这个 $28 \times 28$ 特征图上采样还原至 RoI 大小（e.g. $120 \times 80$），我们就得到原始 RoI 图中哪些像素属于"猫"，相当于就是把"猫"的二值掩码贴到原图对应的位置，得到猫这个实例在原图中的像素级掩码！


> 注：使用 Mask 分支的意义：之前的 fast R-CNN 只是用方框框出每个类别物体的位置，但是现在是要得到每个物体的像素级轮廓。

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec9_mask_R-CNN_illustration.webp" alt="mask R-CNN illustration" width="100%" loading="lazy" decoding="async" />
  <figcaption>mask R-CNN illustration</figcaption>
</figure>

---








## 4. 模型可视化（Model Layers Visualization）


### 线性分类器的第一层：滤波器（Filters）

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec9_filters_visualization.webp" alt="filters visualization" width="100%" loading="lazy" decoding="async" />
  <figcaption>filters visualization</figcaption>
</figure>



### 通过反向传播做显著性图（Saliency Map）

> **把损失对输入像素求梯度，梯度大的像素说明它对分类结果影响大。**

e.g.：固定网络权重不变；把输入图像当作变量；求"猫"这个类别的分数对每个像素的偏导；偏导大的像素，就是重要像素。

最后得到一个和图像同尺寸的矩阵：

$$
\text{Saliency} \in \mathbb{R}^{H \times W}
$$

然后对绝对值归一化到 $[0, 1]$，在 RGB 三通道上取最大值，进行可视化：亮的地方：重要像素；暗的地方：不重要像素。

（这个想法有点类似之前我看过的模型的"激活空间"的概念与得来！！！）




### 类别激活图（Class Activation Mapping, CAM）

核心思想：用全局平均池化（GAP）+ 全连接层权重，把最后一层卷积特征图加权求和，得到类别激活图。

<figure style="text-align: center;">
  <img src="/My-Blog/blog-images/cs231n/Lec9_CAM_figure.webp" alt="CAM" width="100%" loading="lazy" decoding="async" />
  <figcaption>CAM</figcaption>
</figure>

**① 全局平均池化（GAP）**得到最后的一维向量 $F$ 的操作：

$$
F_k = \frac{1}{HW} \sum_{h, w} f_{h, w, k}
$$

**② 最后各类别分数计算的公式**：（其实就是一个线性层，我们暂且忽略偏置值）

$$
\begin{aligned}
S_c &= \sum_k w_{k, c} F_k \\
    &= \sum_k w_{k, c} \cdot \frac{1}{HW} \sum_{h, w} f_{h, w, k} \\
    &= \frac{1}{HW} \sum_{h, w} \sum_k w_{k, c} f_{h, w, k}
\end{aligned}
$$

现在我们定义 *class activation maps*：$M \in \mathbb{R}^{C, H, W}$：

$$
M_{c, h, w} = \sum_k w_{k, c} f_{h, w, k}
$$


直观理解：把最后一层卷积特征图 $f$ 想象成 $K$ 张"特征地图"：

- 第 1 张地图：检测某种纹理；
- 第 2 张地图：检测某种形状；
- $\cdots$
- 第 $K$ 张地图：检测另一种模式。


> **全连接层的权重 $w_{k, c}$ 告诉我们：对于类别 $c$，第 $k$ 张特征地图有多重要。**


如果 $w_{k, c}$ 很大，说明第 $k$ 个通道对类别 $c$ 的分数贡献很大。

所以：

$$
S_c = \sum_k w_{k, c} \cdot F_k
$$

也就是：把每个通道的全局平均的值，按重要性加权求和，就得到类别分数。

那么 CAM 就沿用 $w_{k, c}$ 作为"通道重要性"的直观含义，用其作为权重对最后一层卷积特征图加权求和，就得到每个空间位置对最后分类为类别 $c$ 的贡献。



> 注：或许疑惑为什么 CNN 这里最后一层采用的是全局平均池化（GAP）而不是 VGG 采用的将最后一层特征展平为向量后送入线性层呢？
>
> 因为这是 NIN, 2014 中提出的设计，使参数量大幅减少，不容易过拟合，且保留了通道语义，每个通道对应一个类别概念，CAM 采用了这种做法。




### 梯度加权类别激活图（Gradient-Weighted CAM, Grad-CAM）

上述所讲的 CAM 有一个问题：只有最后一层卷积能直接对应到全连接层权重；中间层没有这样的权重；所以 CAM 只能用于最后一个卷积层。

现在我们想推广到任意一层。Grad-CAM 思路：**既然全连接权重不好用，那就用梯度来衡量每个通道的重要性。**


**① 选任意一层，拿到它的 feature map** $A \in \mathbb{R}^{H \times W \times K}$（即该层的激活值）。

**② 对类别分数 $S_c$ 关于 $A$ 求梯度**：

$$
\frac{\partial S_c}{\partial A} \in \mathbb{R}^{H \times W \times K}
$$

**③ 对梯度做全局平均池化，得到权重 $\alpha \in \mathbb{R}^K$**（每个通道的权重 $\alpha_k$）：

$$
\alpha_k = \frac{1}{HW} \sum_{h, w} \frac{\partial S_c}{\partial A_{h, w, k}}
$$

**④ 计算激活图 $M_c \in \mathbb{R}^{H \times W}$**：

$$
M_c^{(h, w)} = \text{ReLU} \left( \sum_k \alpha_k A_{h, w, k} \right)
$$

---







## 参考资料

- [CS231n Lecture 9 — Detection, Segmentation, Visualization](https://cs231n.stanford.edu/slides/2026/lecture_9.pdf) — 2026 slide PDF
- [CS231n 2025 spring](https://www.bilibili.com/video/BV1YJ3PzLEiW) — CS231n Spring 2025 视频
- [U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015)](https://arxiv.org/abs/1505.04597) — U-Net 原始论文
- [Faster R-CNN: Towards Real-Time Object Detection (Ren et al., 2015)](https://arxiv.org/abs/1506.01497) — Faster R-CNN 原始论文
- [You Only Look Once: Unified, Real-Time Object Detection (Redmon et al., 2016)](https://arxiv.org/abs/1506.02640) — YOLO 原始论文
- [End-to-End Object Detection with Transformers (Carion et al., 2020)](https://arxiv.org/abs/2005.12872) — DETR 原始论文
- [Mask R-CNN (He et al., 2017)](https://arxiv.org/abs/1703.06870) — Mask R-CNN 原始论文
- [Grad-CAM: Visual Explanations from Deep Networks (Selvaraju et al., 2017)](https://arxiv.org/abs/1610.02391) — Grad-CAM 原始论文