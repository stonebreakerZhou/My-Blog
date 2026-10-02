---
title: "CS231n : Lec 3 — 正则化（Regularization）+ 优化（Optimization）"
description: CS231n Lecture 3 学习笔记，覆盖正则化（L1/L2/Elastic Net）和优化（SGD + Momentum / Nesterov / AdaGrad / RMSProp / Adam / AdamW），以及学习率衰减策略与二阶优化。
pubDate: 2026-10-02
series: cs231n
subSeries: personal-lecture-notes
order: 3
categories:
  - CS231n
  - Regularization
  - Optimization
  - SGD
  - Adam
---

## 正则化 + 优化（Regularization + Optimization）


## 1. 正则化（Regularization）

$$
\begin{aligned}
\text{总损失} \quad L(W) &= \frac{1}{N} \sum_{i=1}^N L_i (f(x_i, W), y_i) + \lambda R(W) \\
&= \text{Data loss} + \text{regularization}
\end{aligned}
$$

**$\lambda$** : 正则化强度（regularization strength，超参数）

**奥卡姆剃刀（Occam's Razor）** : 在多个相互竞争的假设中，*最简单的是最好的*。


### 简单例子（Simple examples）

#### L2 正则化（L2 regularization）

$$
R_{L2} (W) = \sum_k \sum_l W_{(k,l)}^2 = \|W\|_2^2
$$

#### L1 正则化（L1 regularization）

$$
R_{L1} (W) = \sum_k \sum_l |W_{(k,l)}| = \|W\|_1
$$

#### 弹性网络（Elastic net，L1 + L2）

$$
R_{\text{ElasticNet}} (W) = \sum_k \sum_l \left( \beta (W_{(k,l)})^2 + |W_{k,l}| \right)
$$

实践中，使用 L1 正则化时，权重矩阵 $W$ 中会有更多零值；而 L2 正则化会让 $W$ 中的值是接近零但非零的小数，呈"分散"分布。

#### 更复杂的方法

dropout、batch normalization（批归一化）、stochastic depth（随机深度）、fractional pooling（分数池化）等。

> **Q：为什么要正则化？**
> **A：** ① 对权重表达偏好 ② 让模型更简单，从而在测试集上也 work ③ 通过加入曲率改善优化过程

---







## 2. 优化（Optimization）


**数值梯度（Numerical gradient）**：近似、慢、容易写
**解析梯度（Analytic gradient）**：精确、快、易出错
⇒
实践中：总是使用解析梯度（数学上推导出来的），但用数值梯度检查实现是否有 bug。这个过程叫 *gradient check（梯度检查）*。


### 随机梯度下降（Stochastic Gradient Descent，SGD）

#### SGD 的问题 1

场景：损失函数在一个方向变化快，另一个方向变化慢。

假设参数只有两个：$w_1$ 和 $w_2$。

损失函数 $L(w_1, w_2)$ 的等高线画出来，是一个狭长的椭圆（如下图）：

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs231n/Lec3_SGD_problem1_loss_contour_figure.webp" alt="loss contour" width="100%" loading="lazy" decoding="async" /></div>

沿着 $w_2$ 方向：等高线很密，这个方向曲率大；沿着 $w_1$ 方向：等高线很稀疏，这个方向*曲率小*。

当我们使用梯度下降，在"陡峭方向上"（即 $w_2$ 方向上）会来回震荡！

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs231n/Lec3_SGD_problem1_contour_route.webp" alt="SGD route" width="100%" loading="lazy" decoding="async" /></div>

并且，一个全局学习率无法同时适应所有方向：如果将学习率 $\alpha$ 调大，让平坦方向走得快一点，那么陡峭方向就会震荡得更厉害，甚至发散；如果将学习率 $\alpha$ 调小，让陡峭方向不震荡，那么平坦方向会进展非常缓慢。

数学上，我们说：Loss function has high condition number: ratio of largest to smallest singular value of the Hessian matrix is large.

> 补充说明 condition number ：引入 *Hessian 矩阵 $H$*。
> - $H$ 是损失函数的二阶导数矩阵：
>   $H_{ij} = \frac{\partial^2 L}{\partial w_i \partial w_j}$
> - 它描述了损失曲面在各个方向上的*曲率*。
> - Hessian 的特征值表示不同方向上的曲率大小：
>   - 大特征值 → 陡峭方向；
>   - 小特征值 → 平坦方向。
>
> **条件数（condition number）** 定义为：
>
> $\text{condition number} = \frac{\lambda_{\max}}{\lambda_{\min}}$
>
> 也就是最大曲率除以最小曲率。
> - 如果条件数接近 1，说明各个方向曲率差不多，曲面像个圆碗，梯度下降很好走。
> - 如果条件数很大，说明有的方向极陡，有的方向极平，曲面像个又长又窄的山谷。
>
> 所以条件数很大时优化很难。



#### SGD 的问题 2（Problem 2）

场景：梯度下降遇到损失函数局部最小值（local minima）或鞍点（saddle point）时，梯度为零，参数不再更新，优化停滞。

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs231n/Lec3_SGD_problem2_figure.webp" alt="SGD problem2 figure" width="60%" loading="lazy" decoding="async" /></div>

Saddle points are much more common in higher dimension（在高维空间中鞍点比局部极小值更常见）。



#### SGD 的问题 3（Problem 3）

Our gradients come from minibatches so they can be noisy!（我们的梯度来自 minibatch，所以是有噪声的！）






#### 解决方法 1：SGD + 动量（SGD + Momentum）

**更新规则（Update rule）**：（经典 Momentum 做法）

$$
\begin{aligned}
v_{t+1} &= \rho v_t + \nabla f(x_t) \\
w_{t+1} &= x_t - \alpha v_{t+1}
\end{aligned}
$$

① 把"速度（velocity）"作为梯度的运行均值累积起来
② $\rho$ 给系统加"摩擦力”（friction）；通常取 $\rho = 0.9$ 或 0.99

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs231n/Lec3_classic_momentum_update_figure.webp" alt="classic momentum update" width="60%" loading="lazy" decoding="async" /></div>

把当前点算出的梯度与速度结合，得到更新权重所用的步（是在当前点算梯度，再用速度累积）。

应用 Momentum 后的优化图示：

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs231n/Lec3_sol1_to_problem1.webp" alt="SGD+momentum → poor conditioning" width="100%" loading="lazy" decoding="async" /></div>

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs231n/Lec3_sol1_to_problem2.webp" alt="SGD+momentum → local minima+saddle points" width="85%" loading="lazy" decoding="async" /></div>

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs231n/Lec3_sol1_to_problem3.webp" alt="SGD+momentum → gradient noise" width="60%" loading="lazy" decoding="async" /></div>



**补充 1：Nesterov 动量（Nestrov Momentum）**

先用当前速度往前看一步，到那个预测位置算梯度，再把这个梯度和速度混合，得到实际更新方向。

原始 Nesterov 更新公式：

$$
\begin{aligned}
v_{t+1} &= \mu v_t - \alpha \nabla f(x_t + \mu v_t) \\
x_{t+1} &= x_t + v_{t+1}
\end{aligned}
$$

与 classic Momentum 的关键区别：
① classic Momentum：梯度在 $x_t$ 算；
② Nesterov：梯度在 $x_t + \mu v_t$ 算。

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs231n/Lec3_Nesterov_momentum_update.webp" alt="Nesterov momentum update" width="65%" loading="lazy" decoding="async" /></div>

直接使用这个公式不直观：梯度不在当前点 $x_t$ 算，而在 $x_t + \mu v_t$ 算；这导致更新公式里同时有 $x_t$ 和 $v_t$。

我们不妨考虑变量代换：

$$
\tilde{x}_t = x_t + \mu v_t
$$

于是写成：

$$
\begin{aligned}
v_{t+1} &= \mu v_t - \alpha \nabla f(\tilde{x}_t) \\
\tilde{x}_{t+1} &= x_{t+1} + \mu v_{t+1} = x_t + (1 + \mu) v_{t+1} \\
&= (\tilde{x}_t - \mu v_t) + (1 + \mu) v_{t+1} \\
&= \tilde{x}_t + v_{t+1} + \mu(v_{t+1} - v_t)
\end{aligned}
$$

这个形式和原始 Nesterov 完全等价。

实现时跟踪 $\tilde{x}_t$，只需保存旧速度 $v_t$；
梯度仍在 $\tilde{x}_t$ 点计算；
更新表达式更接近"梯度 + 速度 → 实际步"的统一框架；
若要回到原始参数，可用 $x_t = \tilde{x}_t - \mu v_t$ 换算。




**补充 2：AdaGrad**

前面的两种 Momentum 方法是在参数更新方向上进行优化，它们解决的是梯度方向来回震荡、收敛慢的问题。

但它们有一个共同点：所有参数维度共享同一个学习率 $\alpha$。

于是现在我们考虑参数不同维度各自具有不同的学习率：即自适应学习率方法。

AdaGrad 的做法：
① 对每个参数维度，单独累积历史梯度平方；
② 用这个累积量去逐元素缩放梯度。

设梯度：

$$
g_t = \nabla f(x_t)
$$

AdaGrad 维护一个和参数同维度的累积变量 $s_t$，初始为 0：

$$
s_t = s_{t-1} + g_t \odot g_t
$$

注意这里是累加，不是平均，也不是指数衰减。
所以 $s_t$ 的每个分量，就是对应维度从训练开始到现在的梯度平方和。

参数更新公式：

$$
x_{t+1} = x_t - \alpha \frac{g_t}{\sqrt{s_t} + \epsilon}
$$

逐维度写：

$$
x_{t+1,i} = x_{t,i} - \alpha \frac{g_{t,i}}{\sqrt{s_{t,i}} + \epsilon}
$$

因此每个维度的有效学习率是：

$$
\alpha_i = \frac{\alpha}{\sqrt{s_{t,i}} + \epsilon}
$$

这就是 per-parameter learning rates 或 adaptive learning rates。

最终使得陡峭方向（梯度大）的进展被抑制；平坦方向（梯度小）的进展被加速。

但由于 $s_t$ 是单调累积变量，所以最终更新步长会衰减到 0。这导致训练后期参数几乎不再更新，模型可能过早停止优化！



于是这便引出了 RMSProp 对于 AdaGrad 改进的核心思想：

① AdaGrad 的问题根源在于：它把所有历史梯度平方都累加起来，从不遗忘。

一个自然的改进思路是：

② 不要累加所有历史，而是只保留最近一段时间的梯度平方，让旧信息慢慢"漏掉"。（也就是引入一个"记忆衰减系数"！）

因此 RMSProp 也被称为 "Leaky AdaGrad"。

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs231n/Lec3_AdaGrad_to_RMSProp.webp" alt="AdaGrad → RMSProp" width="100%" loading="lazy" decoding="async" /></div>




#### 更复杂的优化器：RMSProp

RMSProp 不看梯度的"方向"，而是看每个参数维度上梯度的"历史大小"：
梯度一直大的维度，这个方向上容易震荡，于是减小其学习率；梯度一直小的维度，这个方向上较为平坦，于是增大其学习率。
这样每个参数都有自己独立的学习率。

RMSProp 的具体更新公式：

设：
$\beta$：衰减率，通常取 $0.9$ 或 $0.99$（$\beta$ 越大越关注较远的历史）；
$s_t$：缓存变量，和参数同维度，初始为 $\vec{0}$；
$\odot$：逐元素乘法。

$$
g_t = \nabla_w L(w_t)
$$

$$
s_t = \beta s_{t-1} + (1-\beta) g_t \odot g_t
$$

$$
w_{t+1} = w_t - \eta \frac{g_t}{\sqrt{s_t} + \epsilon}
$$

其中根号、除法都是逐元素操作。

```python
cache = 0
for t in range(steps):
    g = compute_gradient(w)
    cache = beta * cache + (1 - beta) * g * g
    w = w - learning_rate * g / (np.sqrt(cache) + eps)
```

普通 SGD 更新：

$$
w_{t+1} = w_t - \eta g_t
$$

所有参数共用一个学习率 $\eta$。

RMSProp 不同，它对每个维度单独缩放：

$$
w_{t+1,i} = w_{t,i} - \frac{\eta}{\sqrt{s_{t,i}} + \epsilon} g_{t,i}
$$

每个参数 $w_i$ 的有效学习率是：

$$
\eta_i = \frac{\eta}{\sqrt{s_{t,i}} + \epsilon}
$$





#### Adam $\approx$ Momentum + RMSProp

**1 ) Adam（almost form）：**

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs231n/Lec3_Adam(almost).webp" alt="Adam (almost)" width="100%" loading="lazy" decoding="async" /></div>

**Step 1：**

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

$g_t$ 是当前的梯度值，所以这一步是在更新动量 $m_t$ 的值（梯度的一阶矩），记录梯度的平均方向。

**Step 2：**

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

$v_t$ 记录的是每个参数维度上梯度过去有多大（梯度平方的二阶矩）。

**Step 3（update）：**

$$
x \leftarrow x - \alpha \frac{m_t}{\sqrt{v_t} + \epsilon}
$$

最后一步就是用 Momentum 的方向，除以 RMSProp 的缩放因子（逐维度除法来缩放），最后得到每个参数的实际更新量。



**2 ) Adam（full form）：**

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs231n/Lec3_Adam(full_form).webp" alt="Adam (full form)" width="100%" loading="lazy" decoding="async" /></div>

完整版 Adam 在 Momentum 与 RMSProp 之间新增关键一步：偏差校正

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
$$

$$
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

然后最后在 RMSProp 处用校正后的值更新参数：

$$
x_{t+1} = x_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

The full form of Adam adds *bias correction* for the fact that first and second moment estimates start at zero.
（Adam 的完整形式为了一阶矩和二阶矩估计从零起步这件事，加上了 *偏差校正*。）




Adam（full form）新增的偏差校正背后的原理：

**1. 指数加权平均本身带了偏差**

由于：

$$
m_t = \beta m_{t-1} + (1 - \beta) g_t
$$

且初始 $m_0 = 0$，展开得：

$$
\begin{aligned}
m_t &= (1 - \beta) g_t + \beta (1 - \beta) g_{t-1} + \beta^2 (1 - \beta) g_{t-2} + \dots \\
&= (1 - \beta) \sum_{i=1}^t \beta^{t-i} g_i
\end{aligned}
$$

而"真正的"加权平均应该是权重归一化的：

$$
\text{真实加权平均} = \frac{\sum_{i=1}^t \beta^{t-i} g_i}{\sum_{i=1}^t \beta^{t-i}}
$$

这里的分母：

$$
\sum_{i=1}^t \beta^{t-i} = \frac{1 - \beta^t}{1 - \beta}
$$

所以"真实的加权平均"为：

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
$$

这就是偏差的来源。
当 $t$ 很大时，$(1 - \beta^t) \to 1$，偏差消失。
当 $t$ 很小时，比如 $t = 1$，$(1 - \beta^1) = 1 - \beta = 0.1$，偏差非常严重。

$v_t$ 同理。

**2. 偏差校正就是除以这个系数**

$$
\begin{aligned}
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t} = \text{真实一阶矩} \\
\hat{v}_t &= \frac{v_t}{1 - \beta_2^t} = \text{真实二阶矩}
\end{aligned}
$$

偏差校正是为了在数学上恢复到无偏估计。

**3. 如果不做校正，会出什么问题？**

之前我们的参数更新量是：

$$
\Delta x = \alpha \frac{m_t}{\sqrt{v_t} + \epsilon}
$$

因为 $m_t$ 和 $v_t$ 相比真实的一阶、二阶矩都被缩小了，所以这个比例式的值会受到因子：

$$
\frac{1 - \beta_1^t}{\sqrt{1 - \beta_2^t}}
$$

的影响。

这个因子在训练初期可能很大（比如 $t=1, \beta_1 = 0.9, \beta_2 = 0.999$ 时约 3.16），也可能很小。
它完全取决于所选的 $\beta_1, \beta_2$。

这就带来几个问题：

① 早期步长不可控：实际第一步步长是 $3.16 \alpha$

② 对超参数过于敏感：不同 $\beta_1, \beta_2$ 会导致不同的初始偏差。没有校正，Adam 的表现会非常依赖这些超参数，调参困难。

③ 早期训练不稳定：如果某个维度的梯度很小，$v_1$ 可能比 $\epsilon$ 还小，分母被 $\epsilon$ 主导，步长又会变得非常小。
所以早期更新量可能一会儿很大、一会儿很小，不稳定。

**4. 校正后的好处**

校正后：

$$
\Delta x_i = \alpha \frac{g_{(1,i)}}{|g_{(1,i)}| + \epsilon} \approx \alpha \odot \text{sign}(g_{(1,i)})
$$

每个维度的第一步步长都接近 $\alpha$，与梯度大小无关，与 $\beta_1, \beta_2$ 无关。这就让：初期步长可控；训练稳定；对超参数不敏感。

> ***Adam with $\beta_1 = 0.9, \beta_2 = 0.999$ and $\alpha(\text{learning rate}) = 1e-3 \text{ or } 5e-4$ is a great starting point for many models!***
>
> （取 $\beta_1 = 0.9, \beta_2 = 0.999$，$\alpha = 1e-3$ 或 $5e-4$ 是很多模型不错的起点。）

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs231n/Lec3_Adam_optimization_route_figure.webp" alt="Adam optimization route" width="100%" loading="lazy" decoding="async" /></div>





#### AdamW（带 weight decay 的 Adam 变体）

**1. 什么是 *weight decay*？为什么要 weight decay？**

Weight decay 实际上就是我们希望训练得到的权重 $W$ 不要过大，过大的权重实际上是对某些数值特别敏感，也会导致数值不稳定（激活值、梯度爆炸），也会导致模型过拟合。所以 weight decay 要做的就是一直把模型权重向着原点方向拉动。

**2. SGD 中正则项与 weight decay 之间的关系？**

在 SGD 中如果我们加入了 L2 正则项：

$$
L = L_{\text{data}} + \lambda / 2 \|w\|^2
$$

此时训练梯度变为：

$$
\begin{aligned}
g &= \nabla_w L \\
&= \nabla_w L_{\text{data}} + \lambda w \\
&= g_{\text{data}} + \lambda w
\end{aligned}
$$

SGD 更新中：

$$
w \leftarrow w - \alpha g_{\text{data}} - \alpha \lambda w
$$

最后一项就是把 $w$ 向着 0 方向拉动，并且对每个参数 $w_t$ 都是 $- \alpha \lambda w_t$，和梯度大小无关，即 <u>每个参数按同样比例衰减！！！</u>

所以说 SGD 中的 L2 正则项就具有 weight decay 的作用！

**3. Standard Adam 中使用 L2 正则项？**

计算梯度为：

$$
g_t = \nabla_w L_{\text{data}} + \lambda w_t
$$

这个 $g_t$ 被送进 Adam 然后做参数更新：

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
$$

$$
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

$$
w_{t+1} = w_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

假设训练已经稳定，$m_t \approx g_t$，$v_t \approx g_t^2$。

那么更新量近似为：

$$
\Delta w \approx - \eta \frac{g_{\text{data}} + \lambda w}{\sqrt{(g_{\text{data}} + \lambda w)^2 + \epsilon}}
$$

如果 $|g_{\text{data}}|$ 远大于 $\lambda w$，那么：

$$
\begin{aligned}
\Delta w &\approx - \eta \frac{g_{\text{data}} + \lambda w}{|g_{\text{data}}|} \\
&\approx - \eta \frac{g_{\text{data}}}{|g_{\text{data}}|} - \eta \frac{\lambda w}{|g_{\text{data}}|}
\end{aligned}
$$

第二项就是在做 weight decay，但此时：

梯度大的参数，$|g_{\text{data}}|$ 大，分母大，weight decay 作用被压小；
梯度小的参数，$g_{\text{data}}$ 小，分母小，weight decay 作用被放大。

本来是要对参数均匀正则化，但现在"梯度小的参数被过度正则化，梯度大的参数几乎不正则化"，这完全违背了 weight decay 的初衷！

**4. AdamW 的做法**

AdamW 把 weight decay 从梯度里拿出来：

$$
g_t = g_{\text{data}}
$$

先做纯 Adam 更新：

$$
w_{t+1} = w_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

然后单独做 weight decay：

$$
w_{t+1} \leftarrow w_{t+1} - \eta \lambda w_t
$$

这样 weight decay 不进入 $m_t$、$v_t$，不受 $\sqrt{\hat{v}_t}$ 缩放。
每个参数都按同样的比例衰减，恢复到 SGD 里 weight decay 的行为。

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs231n/Lec3_Adam_AdamW_comparison.webp" alt="Standard Adam with L2 vs. AdamW" width="80%" loading="lazy" decoding="async" /></div>







### 超参数：学习率（Hyperparameter : Learning rate）

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs231n/Lec3_loss_figure_with_different_learning_rates.webp" alt="different learning rates" width="70%" loading="lazy" decoding="async" /></div>

事实上，这些都可以成为好的学习率，因为在现代深度学习中我们在训练时可以切换改变好几种学习率。

#### 学习率随时间衰减（Learning rate decays over time）

随着训练进行，当前学习率太高，无法再继续收敛。所以我们需要在训练过程中对学习率进行衰减。

**方案 1：在几个固定时刻降低学习率（Reduce learning rate at a few fixed points）**

e.g. 对于 ResNets，在第 30、60、90 个 epoch 后把学习率乘以 0.1。

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs231n/Lec3_learning_rate_decay_1.webp" alt="learning rate decay choice 1" width="70%" loading="lazy" decoding="async" /></div>

**方案 2：余弦函数（Cosine function）：**

$$
\alpha_t = \frac{1}{2} \alpha_0 \left( 1 + \cos\left(\frac{t \pi}{T}\right) \right)
$$

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs231n/Lec3_learning_rate_decay_2.webp" alt="learning rate decay choice 2" width="70%" loading="lazy" decoding="async" /></div>

对应的 loss 曲线会长这样：

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs231n/Lec3_training_loss_figure_under_learning_rate_decay_2.webp" alt="training loss figure under Cosine learning rate" width="70%" loading="lazy" decoding="async" /></div>

**方案 3：线性衰减（Linear decay）**

$$
\alpha_t = \alpha_0 (1 - t/T)
$$

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs231n/Lec3_learning_rate_decay_3.webp" alt="learning rate decay choice 3" width="70%" loading="lazy" decoding="async" /></div>

**方案 4：反平方根衰减（Inverse sqrt decay）**

$$
\alpha_t = \alpha_0 / \sqrt{t}
$$

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs231n/Lec3_learning_rate_decay_4.webp" alt="learning rate decay choice 4" width="70%" loading="lazy" decoding="async" /></div>

#### 线性 warmup（Linear warmup）

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs231n/Lec3_learning_rate_decay_linear_warmup.webp" alt="linear warmup" width="70%" loading="lazy" decoding="async" /></div>

初始学习率太高可能让 loss 爆炸；通过在前 ~5,000 次迭代里把学习率从 0 线性升上去可以避免这个问题。

> **经验法则（Empirical rule of thumb）**：
> *If you increase the batch size by $N$, then also scale the initial learning rate by $N$*
> （如果你把 batch size 增大 $N$ 倍，初始学习率也按 $N$ 倍线性放大。）








### 二阶优化（Second-order Optimization）

#### 用梯度构造线性近似（Use gradient form linear approximation）

在点 $w_t$ 处，我们用一阶泰勒展开对损失做局部线性近似：

$$
L(w) \approx L(w_t) + \nabla L(w_t)^\top (w - w_t)
$$

这个线性近似没有全局最小值，由于负梯度方向是当前点处使线性近似下降最快的方向，所以我们选择负梯度方向作为下降方向，并取一个步长 $\alpha$（线性近似只在局部有效，所以步长不能过大）：

$$
w_{t+1} = w_t - \alpha \nabla L(w_t)
$$

每一步只使用一阶导数信息，因此这类方法称为一阶优化（first-order optimization）。

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs231n/Lec3_first-order_optimization.webp" alt="first-order optimization" width="80%" loading="lazy" decoding="async" /></div>


#### 用梯度和 Hessian 构造二次近似（Use gradient and Hessian to form quadratic approximation）

Second-order Taylor expansion at $\theta_0$（在 $\theta_0$ 处做二阶泰勒展开）：

$$
J(\theta) \approx J(\theta_0) + (\theta - \theta_0) \nabla_\theta J(\theta_0) + \frac{1}{2} (\theta - \theta_0)^T H (\theta - \theta_0)
$$

其中 $H = \nabla_\theta^2 J(\theta_0)$ 是 Hessian 矩阵，满足：

$$
H_{ij} = \frac{\partial^2 J}{\partial \theta_i \partial \theta_j}
$$

对近似函数关于 $\theta$ 求梯度：

$$
\nabla_\theta \left[ J(\theta_0) + (\theta - \theta_0)^\top \nabla J(\theta_0) + \frac{1}{2} (\theta - \theta_0)^\top H (\theta - \theta_0) \right]
$$

逐项求导：

$J(\theta_0)$ 是常数，导数为 $0$；
$(\theta - \theta_0)^\top \nabla J(\theta_0)$ 对 $\theta$ 求导得 $\nabla J(\theta_0)$；
$\frac{1}{2} (\theta - \theta_0)^\top H (\theta - \theta_0)$ 对 $\theta$ 求导得 $H (\theta - \theta_0)$。

所以：

$$
\nabla J(\theta_0) + H (\theta - \theta_0) = 0
$$

解出：

$$
H (\theta - \theta_0) = - \nabla J(\theta_0)
$$

$$
\theta^* = \theta_0 - H^{-1} \nabla J(\theta_0)
$$

这就是牛顿法更新公式。

<div style="text-align: center;"><img src="/My-Blog/blog-images/cs231n/Lec3_second-order_optimization.webp" alt="second-order optimization" width="85%" loading="lazy" decoding="async" /></div>

二阶优化实际上就是按照二阶泰勒展开用一个二次曲面来近似当前点处的原损失函数曲面，然后优化结果就是直接使参数到达二次曲面的最低点。
这样看来，二阶优化仿佛与一阶优化不同，仿佛没有步长的限制！（一阶优化线性近似没有最低点，所以才有步长的限制）当然我们也可以人为加上一个步长限制，使用"阻尼牛顿法"。

相比之下，Adam 实际就是一种"廉价"的二阶近似。因为 Adam 优化中使用 $\frac{1}{\sqrt{v_t} + \epsilon}$ 逐维度缩放梯度 $\hat{m}_t$，而牛顿法是在使用 $H^{-1}$ 缩放梯度，自适应步长。

> However, this is *bad* for deep learning !
> Because Hessian $H$ has $O(N^2)$ elements, inverting takes $O(N^3)$, and $N$ is a very large number in deep learning!（参数过多）



**BFGS，L-BFGS**：

由于严格牛顿法的 $H^{-1}$ 取逆参数爆炸问题，我们换而维护一个近似矩阵 $B \approx H^{-1}$，然后用 $B$ 代替 $H^{-1}$ 做参数更新：

$$
\theta_{t+1} = \theta_t - B_t \nabla J(\theta_t)
$$

我们令以下符号：
$s_t = \theta_{t+1} - \theta_t$：参数变化；
$y_t = \nabla J(\theta_{t+1}) - \nabla J(\theta_t)$：梯度变化；
$y_t^\top s_t > 0$（曲率条件成立）。

我们的近似逆矩阵的计算（更新）公式为：

$$
B_{t+1} = \left( I - \frac{s_t y_t^\top}{y_t^\top s_t} \right) B_t \left( I - \frac{y_t s_t^\top}{y_t^\top s_t} \right) + \frac{s_t s_t^\top}{y_t^\top s_t}
$$

满足割线方程：

$$
B_{t+1} y_t = s_t
$$

这样每次矩阵取逆操作数从 $O(N^3)$ 降为 $O(N^2)$ 就可以得到一个近似的逆矩阵。

这个方法：BFGS 仍然要存一个 $N \times N$ 的近似逆 Hessian，参数量仍有 $O(N^2)$，当 $N$ 很大时，参数量也撑不住。

于是还有进一步的方法：L-BFGS（Limited memory BFGS），其思路是：不存完整的近似逆 Hessian，只存最近 $m$ 步的梯度差和参数差。用这些历史信息隐式地计算 $H^{-1} \nabla J$，而不显式构造矩阵。这样参数量降为 $O(m N)$。但注意 L-BFGS 主要适用于全批量模式（full-batch mode），对于 mini-batch 而言其数据噪声大，近似出的梯度不准确。


> **In practice（实践中）：**
>
> ① *Adam(W)* 是很多情况下的不错默认选择；即使学习率不变，它也通常 work 得不错。
>
> ② *SGD + Momentum* 可以超过 Adam，但需要对学习率和调度更细致的调参。
>
> ③ 如果你能负担全批量更新，可以试试 *L-BFGS*（别忘了关掉所有噪声来源）。

---




## 参考资料

- [CS231n Lecture 3 — Loss Functions and Optimization](https://cs231n.stanford.edu/slides/2026/lecture_3.pdf) — 2026 slide PDF
- [CS231n 2025 spring](https://www.bilibili.com/video/BV1YJ3PzLEiW) — CS231n Spring 2025 视频
- [Adam: A Method for Stochastic Optimization (Kingma & Ba, 2015)](https://arxiv.org/abs/1412.6980) — Adam 原始论文
- [Decoupled Weight Decay Regularization (Loshchilov & Hutter, 2019)](https://arxiv.org/abs/1711.05101) — AdamW 原始论文
- [An overview of gradient descent optimization algorithms (Ruder, 2016)](https://arxiv.org/abs/1609.04747) — 各类优化器演进综述
- [Hessian matrix (Wikipedia)](https://en.wikipedia.org/wiki/Hessian_matrix) — Hessian 矩阵的几何含义补充
