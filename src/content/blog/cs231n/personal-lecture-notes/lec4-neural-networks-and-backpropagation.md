---
title: "CS231n : Lec 4 — 神经网络（Neural Networks）+ 反向传播（Backpropagation）"
description: CS231n Lecture 4 学习笔记，覆盖神经网络层级化计算、计算图与反向传播。
pubDate: 2026-10-03
series: cs231n
subSeries: personal-lecture-notes
order: 4
categories:
  - CS231n
  - Neural Networks
  - Backpropagation
  - Jacobian
---

## 引子

Lec 3 把"如何学习权重"（优化 + 正则化）讲完了，Lec 4 把"更复杂的网络能不能算"接上：

- **神经网络（Neural Networks）**：在线性分类器 $f=Wx$ 的基础上，叠一个非线性激活（如 $\max(0, \cdot)$），堆出多层结构；
- **反向传播（Backpropagation）**：把整张计算图跑完前向 + 反向一次，得到每个参数的梯度

---



## 1. Neural Networks

**Linear score function** : $f = W x$

**2-layer Neural Network** : $f = W_2 \max(0, W_1 x)$

**3-layer Neural Network** : $f = W_3 \max(0, W_2 \max(0, W_1 x))$

$$
x \in \mathbb{R}^D, \quad W_1 \in \mathbb{R}^{H_1 \times D}, \quad W_2 \in \mathbb{R}^{H_2 \times H_1}, \quad W_3 \in \mathbb{R}^{C \times H_2}
$$

**Why do we want non-linearity ?**
**如果数据不是线性可分（linearly separable）的**，我们就需要做 *feature transform*，把数据点映射到一个线性可分的空间中。（这个映射应当是一个非线性操作）

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs231n/Lec4_why_non-linearity.webp" alt="non-linear mapping" width="100%" loading="lazy" decoding="async" /></div>

**2-layer 神经网络层级化计算 :**

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs231n/Lec4_2-layer_neural_network.webp" alt="2-layer neural network" width="80%" loading="lazy" decoding="async" /></div>


**激活函数 :**
ReLU, Sigmoid, Leaky ReLU, Tanh, ELU, GELU, SiLU...

**ReLU 对大多数问题都是一个不错的选择.**




**另外 ： 设置层数，层大小，正则化强度 ：**

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs231n/Lec4_neural_layer_size.webp" alt="layer size" width="100%" loading="lazy" decoding="async" /></div>

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs231n/Lec4_lambda_value.webp" alt="regularization strength" width="100%" loading="lazy" decoding="async" /></div>

---







## 2. 计算图与反向传播（Computational graphs and Backpropagation）

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs231n/Lec4_backpropagation_figure.webp" alt="backpropagation" width="100%" loading="lazy" decoding="async" /></div>


**gradient backpropagation :**

> $$\texttt{"downstream"} = \texttt{"local"} \times \texttt{"upstream"}$$

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs231n/Lec4_gradient_backpropagation.webp" alt="gradient backpropagation" width="100%" loading="lazy" decoding="async" /></div>




### Patterns in gradient flow

**① 加法门（add gate）：梯度分发器（gradient distributor）**

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs231n/Lec4_add_gate.webp" alt="add gate" width="50%" loading="lazy" decoding="async" /></div>

**从上游传过来的 upstream gradient 会原封不动分发给下游每一个输入**


**② 乘法门（mul gate）："互换乘法器"（"swap multiplier"）**

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs231n/Lec4_mul_gate.webp" alt="mul gate" width="50%" loading="lazy" decoding="async" /></div>

**上游梯度会乘以另一个输入的值，然后加到该输入的梯度累积中**



**③ 复制门（copy gate）：梯度累加器（gradient adder）**

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs231n/Lec4_copy_gate.webp" alt="copy gate" width="50%" loading="lazy" decoding="async" /></div>

**复制门在反向传播时执行梯度累加**，多个分支的梯度又"累加"回同一个变量。



**④ 最大门（max gate）：梯度路由器（gradient router）**

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs231n/Lec4_max_gate.webp" alt="max gate" width="50%" loading="lazy" decoding="async" /></div>

**把上游梯度只路由到那个最大的输入**，其他输入得到零梯度。





### **导数知识回顾与补充：**

**① 输入、输出全为标量 :**

$$
x, y \in \mathbb{R}
$$

$$
\frac{\mathrm{d} y}{\mathrm{d} x}
$$

**② 输入向量，输出标量 :**

$$
x \in \mathbb{R}^N, \quad y \in \mathbb{R}
$$

$$
\nabla_x y = \begin{pmatrix} \frac{\partial y}{\partial x_1} \\ \vdots \\ \frac{\partial y}{\partial x_n} \end{pmatrix} \in \mathbb{R}^{N \times 1}
$$

$$
(\nabla_x y)_i = \frac{\partial y}{\partial x_i}
$$

**③ 输入输出都是向量 : 此时导数是 Jacobian**（约定以下雅可比矩阵采用分母布局）

$$
x \in \mathbb{R}^N, \quad y \in \mathbb{R}^M
$$

$$
J_{ij} = \frac{\partial y_j}{\partial x_i}
$$

$$
J = \begin{pmatrix}
\frac{\partial y_1}{\partial x_1} & \cdots & \frac{\partial y_M}{\partial x_1} \\
\vdots & \ddots & \vdots \\
\frac{\partial y_1}{\partial x_N} & \cdots & \frac{\partial y_M}{\partial x_N}
\end{pmatrix} \in \mathbb{R}^{N \times M}
$$





**同理可以推广到关于矩阵参数的反向传播，也就是 Tensor 类型的数据。下面先用一个简单图示说明怎样确定各梯度的形状 ：**

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs231n/Lec4_backprop_with_matrices_illustration.webp" alt="backprop with matrices — illustration e.g." width="100%" loading="lazy" decoding="async" /></div>


已知 ： 输入 $x \in \mathbb{R}^{D_x \times M_x}$，$y \in \mathbb{R}^{D_y \times M_y}$，输出为 $z \in \mathbb{R}^{D_z \times M_z}$

**① 先求 upstream gradient 形状 :**

由于最终的损失值 $L \in \mathbb{R}$，故有：

$$
\frac{\mathrm{d} L}{\mathrm{d} z} \in \mathbb{R}^{D_z \times M_z}
$$

（因为 $L$ 为标量，故偏导形状与 $z$ 形状一样）

**② 再求 local gradients 的形状 :**

**注意如果我们认为 $z, x, y$ 都是矩阵的话那么偏导求出来应该是一个四维张量**（因为有 4 个独立的索引）；**但是如果我们把 $z, x, y$ 展平为向量，并在分母布局下求偏导，就得到雅可比矩阵 (Jacobian matrices) ：**

$$
z \to \mathbb{R}^{D_z M_z \times 1} \\
x \to \mathbb{R}^{D_x M_x \times 1} \\
y \to \mathbb{R}^{D_y M_y \times 1} \\
$$

$$
\frac{\partial z}{\partial x} \in \mathbb{R}^{(D_x M_x) \times (D_z M_z)} \\
\frac{\partial z}{\partial y} \in \mathbb{R}^{(D_y M_y) \times (D_z M_z)}
$$

**③ 最后得到 downstream gradients 的形状 ：**

$$
\begin{aligned}
\frac{\partial L}{\partial x} & = \frac{\partial z}{\partial x} \cdot \frac{\partial L}{\partial z} \\
                              & \to \mathbb{R}^{(D_x M_x) \times (D_z M_z)} \cdot \mathbb{R}^{D_z \times M_z} \\
                              & \to \mathbb{R}^{(D_x M_x) \times (D_z M_z)} \cdot \mathbb{R}^{(D_z M_z) \times 1} \\
                              & \in \mathbb{R}^{D_x M_x}
\end{aligned}
$$

$$
\begin{aligned}
\frac{\partial L}{\partial y} & = \frac{\partial z}{\partial y} \cdot \frac{\partial L}{\partial z} \\
                              & \to \mathbb{R}^{(D_y M_y) \times (D_z M_z)} \cdot \mathbb{R}^{D_z \times M_z} \\
                              & \to \mathbb{R}^{(D_y M_y) \times (D_z M_z)} \cdot \mathbb{R}^{(D_z M_z) \times 1} \\
                              & \in \mathbb{R}^{D_y M_y}
\end{aligned}
$$

**最终 downstream gradients 也可以被 reshape 还原为 $\mathbb{R}^{D_x \times M_x}$ 与 $\mathbb{R}^{D_y \times M_y}$**




**In practice, the Jacobians would be too large to store so that we should do the process implicitly.**





### Simple example

设定 :

$$
x \in \mathbb{R}^{N \times D}, \quad w \in \mathbb{R}^{D \times M}, \quad y = x w \in \mathbb{R}^{N \times M}
$$

具体数值：

$$
x = \begin{pmatrix} 2 & 1 & -3 \\ -3 & 4 & 2 \end{pmatrix} \quad (N = 2, D = 3)
$$

$$
w = \begin{pmatrix} 3 & 2 & 1 & -1 \\ 2 & 1 & 3 & 2 \\ 3 & 2 & 1 & -2 \end{pmatrix} \quad (D = 3, M = 4)
$$

$$
y = x w = \begin{pmatrix} 1 & 3 & 9 & -2 \\ -6 & 5 & 2 & 17 \end{pmatrix} \quad (N = 2, M = 4)
$$

upstream gradient :

$$
\frac{\partial L}{\partial y} \in \mathbb{R}^{N \times M} = \begin{pmatrix} 2 & 3 & -3 & 9 \\ -8 & 1 & 4 & 6 \end{pmatrix}
$$

**如果我们直接计算 local gradient 的 Jacobian 的话 ：**

$$
\frac{\partial y}{\partial x} \in \mathbb{R}^{(N D) \times (N M)} \\
\frac{\partial y}{\partial w} \in \mathbb{R}^{(D M) \times (N M)}
$$


**对神经网络来说，假设 $N=64, D=M=4096$，那么每张 Jacobian 就要占约 256 GB 显存！所以我们必须隐式地完成这一过程，而不能显式存储 Jacobian。**


我们考虑下能否不借助 Jacobian 直接推出偏导数求解式 ？


**① Q1 : $x$ 的一个元素影响 $y$ 的哪些部分？**
**A1 : 由于 $y = x w$ 为矩阵乘法，故 $y$ 的 $i, j$ 元素由 $x$ 的第 $i$ 行向量与 $w$ 的第 $j$ 列向量内积得到，则 $x$ 第 $i$ 行的任意一个元素都会影响 $y$ 第 $i$ 行的所有元素值！**

**② Q2 : $x_{nd}$ 对 $y_{nm}$ 的影响有多大？**

**A2 : 由于 $y_{nm} = \sum_{k=1}^{D} x_{nk} w_{km}$，里面有关 $x_{nd}$ 的项就只有 $x_{nd} w_{md}$，所以 ：**

$$
\frac{\partial y_{nm}}{\partial x_{nd}} = w_{md}
$$

**综上 ① ②**，我们可以在不显式存储 Jacobian 的情况下直接推导出损失 $L$ 关于输入每一项 $x_{nd}$ 的偏导数：

$$
\begin{aligned}
\frac{\partial L}{\partial x_{nd}} & = \sum_{m=1}^{M} \frac{\partial L}{\partial y_{nm}} \cdot \frac{\partial y_{nm}}{\partial x_{nd}} \\
                                  & = \sum_{m=1}^{M} \frac{\partial L}{\partial y_{nm}} \cdot w_{md}
\end{aligned}
$$

**进一步还可以写成矩阵乘法形式：**

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} w^{\top}
$$

**形状验证：**

$$
\mathbb{R}^{N \times D} \equiv \mathbb{R}^{N \times M} \cdot \mathbb{R}^{M \times D}
$$

成立！

**同理还可得：**

$$
\frac{\partial L}{\partial w} = x^{\top} \frac{\partial L}{\partial y}
$$



**所以上述就直接推导出了损失关于输入 $x$ 和参数 $w$ 的偏导数计算方式，反向传播可以完全在矩阵层面完成，而不需要去显式存储庞大的 Jacobian ！**

---




## 参考资料

- [CS231n Lecture 4 — Neural Networks and Backpropagation](https://cs231n.stanford.edu/slides/2026/lecture_4.pdf) — 2026 slide PDF
- [CS231n 2025 spring](https://www.bilibili.com/video/BV1YJ3PzLEiW) — CS231n Spring 2025 视频
- [Backpropagation (Wikipedia)](https://en.wikipedia.org/wiki/Backpropagation) — 反向传播算法的数学背景
- [CS231n Lecture 4 notes (Fei-Fei Li et al.)](https://cs231n.github.io/neural-networks-1/) — 配套讲义：神经网络与激活函数
- [Matrix calculus (Wikipedia)](https://en.wikipedia.org/wiki/Matrix_calculus) — 矩阵形式 Jacobian / 布局约定的补充