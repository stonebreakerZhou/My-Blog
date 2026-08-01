---
title: "CS229 review : 线性代数回顾"
description: CS229 Lecture Notes 1 数学基础笔记第一篇，回顾线性代数核心概念：矩阵运算、范数、投影、特征值、二次型与最小二乘。
pubDate: 2026-08-01
series: cs229
subSeries: math-foundations
order: 1
categories:
  - CS229
  - 线性代数
  - 数学基础
---

> **TL;DR**:
> - 矩阵 = 线性变换；同一矩阵可以从 **4 种等价观点** 理解（行向量、列向量、变换、空间）
> - 范数 = 长度度量；ML 里 $\ell_2$（欧氏）范数最常用，矩阵用 Frobenius 范数
> - 投影 = 在子空间上找最近点；公式 $\operatorname{Proj}(y; A) = A(A^\top A)^{-1} A^\top y$
> - 特征值 = 变换的不变方向；对称矩阵可对角化为 $A = U \Lambda U^\top$
> - 二次型 = 对称矩阵的特征值决定符号（PD / PSD / ND / NSD / 不定）

## 引子

在正式学习CS229所有课程之前，我们先进行 linear algebra 必备基础知识的一些回顾

适合读者：学过线代但一段时间没用、想快速 rebuild mental model

## 分节知识小点


### 1. 矩阵

矩阵 $A \in \mathbb{R}^{m \times n}$ 的3种视角：

**行向量视角**
$$
A = \begin{bmatrix}
-a_1^\top- \\
-a_2^\top- \\
\vdots \\
-a_m^\top-
\end{bmatrix}
$$
**列向量视角**
$$
A = \begin{bmatrix}
\mid & \mid & & \mid \\
a_1 & a_2 & \cdots & a_n \\
\mid & \mid & & \mid
\end{bmatrix}
$$

**变换视角**——$x \mapsto Ax$ 把 $\mathbb{R}^n$ 映射到 $\mathbb{R}^m$。

### 2. 矩阵-向量乘法

$$
y = Ax = \underbrace{\begin{bmatrix}
-a_1^\top- \\
-a_2^\top- \\
\vdots \\
-a_m^\top-
\end{bmatrix}}_{\text{行视角}} x = \begin{bmatrix}
a_1^\top x \\
a_2^\top x \\
\vdots \\
a_m^\top x
\end{bmatrix}
$$

$$
y = Ax = \underbrace{\begin{bmatrix}
\mid & \mid & & \mid \\
a_1 & a_2 & \cdots & a_n \\
\mid & \mid & & \mid
\end{bmatrix}}_{\text{列视角}} \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} = \begin{bmatrix} \mid \\ a_1 \\ \mid \end{bmatrix} x_1 + \begin{bmatrix} \mid \\ a_2 \\ \mid \end{bmatrix} x_2 + \cdots + \begin{bmatrix} \mid \\ a_n \\ \mid \end{bmatrix} x_n
$$

> **$y$ 是 $A$ 各列的线性组合，系数由 $x$ 的分量给出。**

对称地，

$$
y^\top = x^\top A = x^\top \begin{bmatrix} \mid & \mid & & \mid \\ a_1 & a_2 & \cdots & a_n \\ \mid & \mid & & \mid \end{bmatrix} = \begin{bmatrix} x^\top a_1 & x^\top a_2 & \cdots & x^\top a_n \end{bmatrix}
$$

> **$y^\top$ 是 $A$ 各行的线性组合，系数由 $x$ 的分量给出。**

### 3. 对称矩阵 (Symmetric Matrices)

任意方阵 $A \in \mathbb{R}^{n \times n}$ 都可以分解为对称矩阵 + 反对称矩阵：

$$
A = \tfrac{1}{2}(A + A^\top) + \tfrac{1}{2}(A - A^\top)
$$

把所有 $n \times n$ 对称矩阵的集合记为 $\mathcal{S}^n$。

### 4. 矩阵的迹 (The Trace)

方阵 $A \in \mathbb{R}^{n \times n}$ 的迹 $\operatorname{tr}(A)$ 是对角元素之和：

$$
\operatorname{tr} A = \sum_{i=1}^n A_{ii}
$$

迹的几个性质（注意下面的矩阵 $A, B$ 都是 $n$ 阶方阵）：

- $\operatorname{tr} A = \operatorname{tr} A^\top$
- $\operatorname{tr}(A + B) = \operatorname{tr} A + \operatorname{tr} B$
- $\operatorname{tr}(tA) = t \operatorname{tr} A$
- $\operatorname{tr}(AB) = \operatorname{tr}(BA)$
- $\operatorname{tr}(ABC) = \operatorname{tr}(BCA) = \operatorname{tr}(CAB)$，更多矩阵同理

### 5. 范数 (Norms)

范数是满足 4 条公理的"长度"函数 $f: \mathbb{R}^n \to \mathbb{R}$：

1. $f(x) \geq 0$（非负）
2. $f(x) = 0 \iff x = 0$（正定）
3. $f(tx) = |t| f(x)$（齐次）
4. $f(x + y) \leq f(x) + f(y)$（三角不等式）

**欧氏范数 / $\ell_2$ 范数**：

$$
\|x\|_2 = \sqrt{\sum_{i=1}^n x_i^2}, \quad \|x\|_2^2 = x^\top x
$$

**$\ell_p$ 范数族**：

$$
\|x\|_p = \left( \sum_{i=1}^n |x_i|^p \right)^{1/p}
$$

**Frobenius 范数 （F-范数）**（给矩阵用）：

$$
\|A\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n A_{ij}^2} = \sqrt{\operatorname{tr}(A^\top A)}
$$

### 6. 线性无关与秩 (Linear Independence and Rank)

这一点较为简单略过

### 7. 矩阵的逆 (The Inverse)

$A \in \mathbb{R}^{n \times n}$ **非奇异**（可逆）

逆矩阵公式（**只用于理论理解**，实际不用这个算）：

$$
A^{-1} = \frac{1}{|A|} \operatorname{adj}(A)
$$

### 8. 正交矩阵 (Orthogonal Matrices)

向量正交：$x^\top y = 0$。向量归一化：$\|x\|_2 = 1$。

方阵 $U \in \mathbb{R}^{n \times n}$ 是**正交矩阵** $\iff$ 它的列两两正交且都归一化：

$$
UU^\top = I = U^\top U \implies U^\top = U^{-1}
$$

**关键性质**：正交矩阵乘以向量**不会改变向量的欧氏范数**（既不拉伸也不降维）：

$$
\|Ux\|_2 = \|x\|_2
$$

(p.s: 正交矩阵对应的线性变换作用在一个向量上对应对该向量"旋转 + 翻转"操作，不涉及任何长度变化或维度变化。这就是为什么向量的欧氏范数（长度）在变换后保持不变)

注意：如果 $U \in \mathbb{R}^{m \times n}$（$n < m$）但列仍然正交归一化，那么 $U^\top U = I$ 但 $UU^\top \neq I$。我们**只用「正交」形容方阵**。

### 9. 值域与零空间 (Range and Nullspace)

**投影**：给定向量 $y \in \mathbb{R}^m$ 和子空间 $\mathcal{S} = \operatorname{span}\{x_1, \dots, x_n\}$（每个 $x_i \in \mathbb{R}^m$），投影 $v$ 是子空间里**离 $y$ 最近**的点（注意：长度的度量是用欧式范数，即l_2范数）：

$$
\operatorname{Proj}(y; \{x_1, \dots, x_n\}) = \operatorname{arg\,min}_{v \in \mathcal{S}} \|v - y\|_2
$$

(注意：$n$ 与 $m$ 的大小关系不重要，关键是 $y$ 与 $x_i$ 维度一致)

**值域** (Range / Column space)：$A$ 的列张成的子空间

$$
\mathcal{R}(A) = \{v \in \mathbb{R}^m : v = Ax, x \in \mathbb{R}^n\}
$$

($A$ 乘以 $x$ 本质上就是对 $A$ 的每一列做线性组合，所以 $\mathcal{R}(A)$ 就是所有 $Ax$ 的集合)

**投影到 $A$ 的值域**（要求 $A$ 满秩且 $n < m$）：

误差平方：
$$
f(x) = \|y - Ax\|^2 = (y - Ax)^\top (y - Ax)
$$

对 $x$ 求导并令为 0 （要使得取极小值），得到**正规方程** (Normal Equation)：
$$
A^\top A x = A^\top y
$$

因为 $A$ 满秩且 $n < m$，$A^\top A$ 是正定对称矩阵，一定可逆， 两边左乘其逆矩阵：

$$
x = (A^\top A)^{-1} A^\top y
$$

投影点 $v = Ax$：
$$
v = A \left[(A^\top A)^{-1} A^\top y\right] = A (A^\top A)^{-1} A^\top y
$$

因此 y 在 A 域上的投影点对应的向量 v 为：

$$
\boxed{\operatorname{Proj}(y; A) = \operatorname{arg\,min}_{v \in \mathcal{R}(A)} \|v - y\|_2 = A(A^\top A)^{-1} A^\top y}
$$

当 $A$ 就是一个列向量 $a \in \mathbb{R}^m$，即投影到一条直线：

$$
\operatorname{Proj}(y; a) = \frac{a a^\top}{a^\top a} y
$$

**零空间** (Nullspace)：$A$ 乘出来等于 0 的所有向量：

$$
\mathcal{N}(A) = \{x \in \mathbb{R}^n : Ax = 0\}
$$

**正交补**：$\mathcal{R}(A^\top)$ 和 $\mathcal{N}(A)$ 互补正交：

$$
\{w : w = u + v, u \in \mathcal{R}(A^\top), v \in \mathcal{N}(A)\} = \mathbb{R}^n \quad \text{and} \quad \mathcal{R}(A^\top) \cap \mathcal{N}(A) = \{0\}
$$

记作 $\mathcal{R}(A^\top) = \mathcal{N}(A)^\perp$（正交且互补）。

### 10. 行列式 (The Determinant)

$$
S = \left\{ v \in \mathbb{R}^n : v = \sum_{i=1}^n \alpha_i a_i \ \text{where}\ 0 \leq \alpha_i \leq 1,\ i = 1, \dots, n \right\}
$$

$|A|$ 的绝对值是集合 $S$ 的"体积"度量。（可以看3b1b视频直观理解下高阶矩阵行列式大小的几何意义，在空间中的意义：链接）

**伴随矩阵** $A$：

$$
\operatorname{adj}(A) \in \mathbb{R}^{n \times n}, \quad (\operatorname{adj}(A))_{ij} = (-1)^{i+j} |A_{\setminus j, \setminus i}|
$$

(注意下标是 $A_{\setminus j, \setminus i}$)

> **伴随矩阵 = 代数余子式矩阵的转置**

对任意非奇异 $A$：

$$
A^{-1} = \frac{1}{|A|} \operatorname{adj}(A)
$$

(p.s: 公式好看但数值上千万别这么算)

### 11. 二次型与半正定矩阵 (Quadratic Forms and PSD Matrices)

给定方阵 $A \in \mathbb{R}^{n \times n}$ 和向量 $x \in \mathbb{R}^n$，标量值 $x^\top A x$ 称为**二次型**。展开来看：

$$
x^\top A x = \sum_{i=1}^n x_i (Ax)_i = \sum_{i=1}^n x_i \left( \sum_{j=1}^n A_{ij} x_j \right) = \sum_{i=1}^n \sum_{j=1}^n A_{ij} x_i x_j
$$

注意到：

$$
x^\top A x = (x^\top A x)^\top = x^\top A^\top x
$$

所以二次型的值等于：

$$
x^\top \left( \tfrac{1}{2} A + \tfrac{1}{2} A^\top \right) x
$$

> **二次型的值最终仅取决于 $A$ 的对称部分**（only the symmetric part of $A$ contributes to the quadratic form）

因此下面讨论的 $A$ 都默认是对称的。

**对称矩阵 $A$ 的 5 种分类**（按二次型的值域）：

- **正定 (PD)**：$x^\top A x > 0,\ \forall x \neq 0$
- **半正定 (PSD)**：$x^\top A x \geq 0,\ \forall x$
- **负定 (ND)**：$x^\top A x < 0,\ \forall x \neq 0$
- **半负定 (NSD)**：$x^\top A x \leq 0,\ \forall x$
- **不定 (Indefinite)**：存在 $x_1, x_2$ 使符号相反

**重要性质**：正定矩阵和负定矩阵都**一定是满秩的**，因此可逆。

证明（反证法）：假设第 $j$ 列是其余列的线性组合

$$
a_j = \sum_{i \neq j} x_i a_i
$$

令 $x_j = -1$，则

$$
Ax = \sum_{i=1}^n x_i a_i = 0
$$

这意味着存在非零向量 $x$ 使 $x^\top A x = 0$，所以 $A$ 既不正定也不负定。

> 因此，若 $A$ 是 PD 或 ND，则 $A$ 必满秩。

**Gram 矩阵 G**：任意矩阵 $A \in \mathbb{R}^{m \times n}$（不一定对称、不一定方阵），$G = A^\top A$ 一定是半正定的。如果 $m \geq n$ 且 $A$ 满秩，$G = A^\top A$ 是正定的。

(p.s: 简单证明：$x^\top G x = x^\top (A^\top A) x = (Ax)^\top (Ax) = \|Ax\|_2^2 \geq 0$，进一步如果 $x \neq 0$ 且 $A$ 满秩则 $> 0$)

### 12. 特征值与特征向量 (Eigenvalues and Eigenvectors)

给定方阵 $A \in \mathbb{R}^{n \times n}$，$\lambda \in \mathbb{C}$ 是 $A$ 的**特征值**，$x \in \mathbb{C}^n$ 是对应的**特征向量**，如果：

$$
Ax = \lambda x, \quad x \neq 0
$$

求特征值：

$$
|\lambda I - A| = 0
$$

(p.s: 实际中千万别直接展开这个行列式，$n$ 阶展开有 $n!$ 项，数值不稳定)

**特征值/特征向量的性质**（设 $A$ 有特征值 $\lambda_1, \dots, \lambda_n$ 和对应特征向量 $x_1, \dots, x_n$）：

① $\operatorname{tr} A = \sum_{i=1}^n \lambda_i$

② $\det(A) = \prod_{i=1}^n \lambda_i$

③ $A$ 的秩 = $A$ 的非零特征值个数

④ 若 $A$ 非奇异，$1/\lambda_i$ 是 $A^{-1}$ 的特征值，对应同一特征向量 $x_i$（证明：$Ax_i = \lambda_i x_i$，两边左乘 $A^{-1}$）

⑤ 对角矩阵 $D = \operatorname{diag}(d_1, \dots, d_n)$ 的特征值就是 $d_1, \dots, d_n$

**把特征向量方程合在一起写**：

$$
AX = X\Lambda
$$

其中 $X \in \mathbb{R}^{n \times n}$ 的列是 $A$ 的特征向量，$\Lambda$ 是以特征值为对角元的对角矩阵：

$$
X = \begin{bmatrix} \mid & \mid & & \mid \\ x_1 & x_2 & \cdots & x_n \\ \mid & \mid & & \mid \end{bmatrix}, \quad \Lambda = \operatorname{diag}(\lambda_1, \dots, \lambda_n)
$$

如果 $A$ 的特征向量线性无关，$X$ 可逆，于是 $A = X \Lambda X^{-1}$。能写成这种形式的矩阵 $A$ 称为**可对角化** (diagonalizable)。

### 13. 对称矩阵的特征值

两个神奇性质：

> ① $A$ 的所有特征值都是实数

> ② $A$ 的特征向量两两正交归一化（也就是上面定义的 $X$ 是正交矩阵）

因此对称矩阵可以分解为：

$$
\boxed{A = U \Lambda U^\top}
$$

其中 $U$ 是正交矩阵（$U^\top U = I$），$\Lambda$ 是对角矩阵。

**用它判断矩阵的正定性**：设 $A \in \mathcal{S}^n$ 且 $A = U \Lambda U^\top$：

$$
x^\top A x = x^\top U \Lambda U^\top x = y^\top \Lambda y = \sum_{i=1}^n \lambda_i y_i^2
$$

其中 $y = U^\top x$。

> 因为 $y_i^2$ 总是正的，**这个表达式的符号完全取决于 $\lambda_i$**。

所以：

- $\lambda_i > 0\ \forall i$ $\iff$ $A$ 正定
- $\lambda_i \geq 0\ \forall i$ $\iff$ $A$ 半正定
- 以此类推

**应用**：特征值/特征向量经常出现在"最大化某个矩阵函数"的问题里。

对称矩阵 $A \in \mathcal{S}^n$ 的优化问题：

$$
\max_{x \in \mathbb{R}^n} x^\top A x \quad \text{subject to} \quad \|x\|_2^2 = 1
$$

（即找单位向量，使二次型最大）

把特征值排序 $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_n$，最优 $x$ 是 $x_1$（$\lambda_1$ 对应的特征向量），最大值就是 $\lambda_1$。

类似的，最小化问题：

$$
\min_{x \in \mathbb{R}^n} x^\top A x \quad \text{subject to} \quad \|x\|_2^2 = 1
$$

最优 $x$ 是 $x_n$（$\lambda_n$ 对应的特征向量），最小值就是 $\lambda_n$。

(p.s: 简单证明)

$$
x^\top A x = (Uy)^\top (U \Lambda U^\top) (Uy) = y^\top U^\top U \Lambda U^\top U y = y^\top \Lambda y = \sum_{i=1}^n \lambda_i y_i^2
$$

其中 $y = U^\top x$。

### 14. 矩阵微积分 (Matrix Calculus)

#### 14.1 梯度 (The Gradient)

函数 $f: \mathbb{R}^{m \times n} \to \mathbb{R}$（输入矩阵，输出实数），对 $A \in \mathbb{R}^{m \times n}$ 的梯度定义为偏导数矩阵：

$$
\nabla_A f(A) \in \mathbb{R}^{m \times n} = \begin{bmatrix}
\frac{\partial f(A)}{\partial A_{11}} & \frac{\partial f(A)}{\partial A_{12}} & \cdots & \frac{\partial f(A)}{\partial A_{1n}} \\
\frac{\partial f(A)}{\partial A_{21}} & \frac{\partial f(A)}{\partial A_{22}} & \cdots & \frac{\partial f(A)}{\partial A_{2n}} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f(A)}{\partial A_{m1}} & \frac{\partial f(A)}{\partial A_{m2}} & \cdots & \frac{\partial f(A)}{\partial A_{mn}}
\end{bmatrix}
$$

即 $m \times n$ 矩阵，第 $(i,j)$ 项是 $\frac{\partial f(A)}{\partial A_{ij}}$。

> 注意 $\nabla_A f(A)$ 的大小永远跟 $A$ 一样。

如果 $A$ 是向量 $x \in \mathbb{R}^n$：

$$
\nabla_x f(x) = \begin{bmatrix} \frac{\partial f(x)}{\partial x_1} \\ \frac{\partial f(x)}{\partial x_2} \\ \vdots \\ \frac{\partial f(x)}{\partial x_n} \end{bmatrix}
$$

> **梯度只对实值函数有意义**，不能对向量值函数求梯度（比如 $\nabla_x (Ax)$ 无定义）。

(p.s: 关于下标记号)

- $\nabla_z f(Ax)$：把 $Ax$ 整体视为变量 $z$ 代入 $f$，对 $f$ 关于 $z$ 求导后代入 $z = Ax$
- $\nabla_x f(Ax)$：直接对 $x$ 求导

#### 14.2 Hessian 海森矩阵

函数 $f: \mathbb{R}^n \to \mathbb{R}$（输入向量，输出实数），对 $x$ 的 Hessian 矩阵：


$$
\nabla_x^2 f(x) \in \mathbb{R}^{n \times n} = \begin{bmatrix}
\frac{\partial^2 f(x)}{\partial x_1^2} & \frac{\partial^2 f(x)}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f(x)}{\partial x_1 \partial x_n} \\
\frac{\partial^2 f(x)}{\partial x_2 \partial x_1} & \frac{\partial^2 f(x)}{\partial x_2^2} & \cdots & \frac{\partial^2 f(x)}{\partial x_2 \partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f(x)}{\partial x_n \partial x_1} & \frac{\partial^2 f(x)}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f(x)}{\partial x_n^2}
\end{bmatrix}
$$

$(\nabla_x^2 f(x))_{ij} = \frac{\partial^2 f(x)}{\partial x_i \partial x_j}$

> **Hessian 总是对称的**，因为 $\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i}$ （二阶混合导数的性质）

(p.s: 区分高阶导数)

- 对向量函数的梯度是向量，**不能再对向量求梯度**（即 $\nabla_x \nabla_x f(x)$ 未定义）
- 但可以对梯度的**每一维**（这是一个实数）求梯度，得到 Hessian 的列（或行），对应记号为：

#### 14.3 二次型和线性函数的梯度与 Hessian

**线性函数的导数** $f(x) = b^\top x$（$b, x \in \mathbb{R}^n$，$f \in \mathbb{R}$）：

$$
\frac{\partial f(x)}{\partial x_k} = \frac{\partial}{\partial x_k} \sum_{i=1}^n b_i x_i = b_k
$$

所以 $\nabla_x b^\top x = b$。

**二次函数的导数** $f(x) = x^\top A x$，$A \in \mathcal{S}^n$（$\mathcal{S}^n$ 表示所有 $n \times n$ 实对称矩阵）：

$$
f(x) = \sum_{i=1}^n \sum_{j=1}^n A_{ij} x_i x_j
$$

对 $x_k$ 求偏导时，把含 $x_k$ 和 $x_k^2$ 的项分开考虑：

$$
\begin{aligned}
\frac{\partial f(x)}{\partial x_k}
&= \frac{\partial}{\partial x_k} \sum_{i=1}^n \sum_{j=1}^n A_{ij} x_i x_j \\
&= \frac{\partial}{\partial x_k} \left[ \sum_{i \neq k} \sum_{j \neq k} A_{ij} x_i x_j + \sum_{i \neq k} A_{ik} x_i x_k + \sum_{j \neq k} A_{kj} x_k x_j + A_{kk} x_k^2 \right] \\
&= \sum_{i \neq k} A_{ik} x_i + \sum_{j \neq k} A_{kj} x_j + 2 A_{kk} x_k \\
&= \sum_{i=1}^n A_{ik} x_i + \sum_{j=1}^n A_{kj} x_j = 2 \sum_{i=1}^n A_{ki} x_i
\end{aligned}
$$

最后一步用了 $A$ 对称（这是合理的假设，因为它出现在二次型里）。

> 注意 $\nabla_x f(x)$ 的第 $k$ 项就是 $A$ 的第 $k$ 行与 $x$ 的内积。

所以：

$$
\boxed{\nabla_x x^\top A x = 2Ax} \quad (A \text{ 对称})
$$

**Hessian**：

$$
\frac{\partial^2 f(x)}{\partial x_k \partial x_l} = \frac{\partial}{\partial x_k} \left[ \frac{\partial f(x)}{\partial x_l} \right] = \frac{\partial}{\partial x_k} \left[ 2 \sum_{i=1}^n A_{li} x_i \right] = 2 A_{lk} = 2 A_{kl}
$$

所以：

$$
\boxed{\nabla_x^2 x^\top A x = 2A} \quad (A \text{ 对称})
$$

(p.s: 类似单变量 $\frac{d^2}{dx^2} a x^2 = 2a$)

**总结**：

- $\nabla_x b^\top x = b$
- $\nabla_x x^\top A x = 2Ax$（$A$ 对称）
- $\nabla_x^2 x^\top A x = 2A$（$A$ 对称）

### 15. 最小二乘 (Least Squares)

给定 $A \in \mathbb{R}^{m \times n}$（假设满秩）和 $b \in \mathbb{R}^m$，且 $b \notin \mathcal{R}(A)$。

这种情况下不存在 $x \in \mathbb{R}^n$ 使 $Ax = b$，于是退而求其次：找 $x$ 使 $Ax$ 离 $b$ 最近，用欧氏范数的平方度量：

$$
\|Ax - b\|_2^2 = (Ax - b)^\top (Ax - b) = x^\top A^\top A x - 2b^\top A x + b^\top b
$$

对 $x$ 求梯度并用上一节的结论：

$$
\nabla_x (x^\top A^\top A x - 2b^\top A x + b^\top b) = 2A^\top A x - 2A^\top b
$$

令梯度为 0 解出 $x$：

$$
\boxed{x = (A^\top A)^{-1} A^\top b}
$$

这就是**正规方程**的解，跟我们之前直接投影推出的结果一样！！！

### 16. 行列式的梯度 (Gradients of the Determinant)

求 $\nabla_A |A|$（对 $A \in \mathbb{R}^{n \times n}$ 的梯度）：

回顾：$|A| = \sum_{i=1}^n (-1)^{i+j} A_{ij} |A_{\setminus i, \setminus j}|$（对任意 $j \in \{1, \dots, n\}$）

对 $A_{kl}$ 求偏导（注意梯度是逐项的）：

$$
\begin{aligned}
\frac{\partial}{\partial A_{kl}} |A| &= \frac{\partial}{\partial A_{kl}} \sum_{i=1}^n (-1)^{i+j} A_{ij} |A_{\setminus i, \setminus j}| \\
&= (-1)^{k+l} |A_{\setminus k, \setminus l}| \\
&= (\operatorname{adj}(A))_{lk}
\end{aligned}
$$

(注意 $\operatorname{adj}$ 的下标是 $lk$ 而不是 $kl$，所以要转置)

$$
\begin{aligned}
\nabla_A |A| &= (\operatorname{adj}(A))^\top \\
&= |A| A^{-\top}
\end{aligned}
$$

**结论**：

$$
\boxed{\nabla_A |A| = |A| A^{-\top}}
$$

**例**：$f(A) = \log |A|$（$A$ 必须是正定的，保证 $|A| > 0$）

用链式法则（就是普通的单变量链式法则）：

$$
\frac{\partial \log |A|}{\partial A_{ij}} = \frac{\partial \log |A|}{\partial |A|} \cdot \frac{\partial |A|}{\partial A_{ij}} = \frac{1}{|A|} \cdot \frac{\partial |A|}{\partial A_{ij}}
$$

所以：

$$
\nabla_A \log |A| = \frac{1}{|A|} \nabla_A |A| = A^{-1}
$$

最后一步可以去掉转置，因为 $A$ 对称。

(p.s: 类似单变量 $\frac{d}{dx} \log x = \frac{1}{x}$)

### 17. 特征值作为优化 (Eigenvalues as Optimization)

最后用矩阵微积分解一个优化问题，自然导出特征值/特征向量分析。

对称矩阵 $A \in \mathcal{S}^n$ 的等式约束优化问题：

$$
\max_{x \in \mathbb{R}^n} x^\top A x \quad \text{subject to} \quad \|x\|_2^2 = 1
$$

带等式约束的标准解法是构造**拉格朗日函数**：

$$
\mathcal{L}(x, \lambda) = x^\top A x - \lambda x^\top x
$$

最优解 $x^*$ 必满足拉格朗日函数的梯度为 0（这是必要不充要条件）：

$$
\nabla_x \mathcal{L}(x, \lambda) = \nabla_x (x^\top A x - \lambda x^\top x) = 2Ax - 2\lambda x = 0
$$

> 注意这就是线性方程 $Ax = \lambda x$。**说明满足约束 $\|x\|_2 = 1$ 使 $x^\top Ax$ 取极大（或极小）值的 $x$ 一定是 $A$ 的特征向量。**

## 参考资料

- [Stanford CS229 Linear Algebra Review](https://cs229.stanford.edu/section/cs229-linalg.pdf) — 原始讲义资料，本文是其中数学基础部分的中文整理
- [Strang, Introduction to Linear Algebra](https://math.mit.edu/~gs/linearalgebra/) — Strang老爷子的MIT经典公开课，主要用于线性代数系统性的入门学习
- [3Blue1Brown — Essence of Linear Algebra](https://www.bilibili.com/video/BV1ys411472E) — 系列视频，几何直觉的最佳入门，强烈推荐学习线性代数之前先看一遍便可豁然开朗
- [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) — 矩阵恒等式、matrix calculus速查表，本文涉及的所有公式都能在里面找到，是开源pdf
- [Boyd & Vandenberghe, Introduction to Applied Linear Algebra](https://vmls-book.stanford.edu/) — 偏应用 / 优化视角，是Stanford另一门课程教材，感觉可以拓展阅读