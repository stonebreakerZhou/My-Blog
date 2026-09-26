---
title: "CS229 : Lec 13 — 模型调试与误差分析"
description: CS229 Lecture 13 学习笔记。当模型表现不好时，怎么系统地诊断问题到底出在哪里：bias / variance 诊断、目标函数 vs 真正关心的指标、最优化算法是否收敛；进一步，对一个含多组件的 system pipeline 做 error analysis 与 ablative analysis 来定位改进部件。
pubDate: 2026-09-15
series: cs229
subSeries: personal-lecture-notes
order: 14
categories:
  - CS229
  - Model Debugging
  - Error Analysis
  - Ablative Analysis
  - Bias-Variance
  - Diagnostics
  - Premature Optimization
---


> **TL;DR**:
> - **调试学习算法**：当 test error 很高时，**不要盲目试**——先做 **bias-variance 诊断**，看 train / test error 的差距来区分 high bias vs high variance。
> - **High variance → overfit**：train error 低，test error 高，**两者差距大**；修正措施：增加训练样本 / 减少特征。
> - **High bias → underfit**：train error 也高，**两者error差距小**；修正措施：增加特征 / 减小正则化。
> - **诊断学习算法本身**：模型表现差的原因还可能是 (a) **优化算法没收敛**；(b) **目标函数不是你真正关心的指标**；(c) **模型族根本不对**。
> - **Error Analysis**：对含多组件的 pipeline，把每个组件换成绝对的标准答案，看 accuracy 提升多少 → 找到**最值得改进**的组件。
> - **Ablative Analysis**：与 error analysis 相反，从已有系统**逐步移除**组件，看 accuracy 跌多少 → 找到**贡献最大**的组件。




## 引子

前面的 Lec 讲了各种模型与训练技巧。但实际工程中**最常见的问题不是"模型不够强"，而是"模型效果差，到底差在哪"**——这正是 Lec 13 的主题：

- 模型调试 (Debugging Learning Algorithms)
- 误差分析 (Error Analysis)
- 消融分析 (Ablative Analysis)

> **核心思想**：用**系统化的诊断 (diagnostics)** 取代盲目的反复试错
> **Premature (Statistical) Optimization**：在搞清问题之前就闷头调参，常常浪费时间。

---




## 1. 调试学习算法 (Debugging Learning Algorithms)

### 1.1 Motivating Example

假如我们在做一个**反垃圾邮件分类器**，精心挑选了 100 个词作为特征，然后用带正则化的逻辑回归（即 Bayesian Logistic Regression）+ 梯度上升法进行训练，结果 test error 非常高——怎么办？

$$
\max_\theta \sum_{i=1}^{m} \log p(y^{(i)} \mid x^{(i)}, \theta) - \lambda \|\theta\|^2
$$

通常我们可能会盲目地试这些：

> **常见（但不系统）的尝试**：
> - 获取更多训练样本
> - 减少 / 增加特征集
> - 改用不同特征：邮件 header vs. 邮件 body
> - 增加梯度法的迭代次数
> - 改用 Newton's method
> - 调整 $\lambda$ 值
> - 改用 SVM / 神经网络

> **一种常见的更有效的做法**：用 **bias-variance 诊断 (diagnostic)**——看看到底是 high bias (underfit) 还是 high variance (overfit)。



### 1.2 Bias-Variance 诊断

**① High Variance**：training error 远小于 test error（两者差距大）
**② High Bias**：training error 本身也很高（两者差距小）

#### 1) High Variance 的典型学习曲线

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec13_learning-curve_high-variance.webp" alt="high variance learning curve" width="70%" loading="lazy" decoding="async" /></div>

对应**过拟合**：

- ① Test error 仍在随 $m$ 增加而下降 → **增加训练样本会有帮助**
- ② **Train error 和 test error 差距很大**（这是更重要的信号！）

#### 2) High Bias 的典型学习曲线

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec13_learning-curve_high-bias.webp" alt="high bias learning curve" width="70%" loading="lazy" decoding="async" /></div>

- ① 即使是 training error 也高得不可接受
- ② Train / test error **差距很小**

#### 3) 各问题对应的修复手段

| 修复手段                       | 对应修正的问题                                                                |
| -------------------------- | ---------------------------------------------------------------------- |
| 获取更多训练样本                   | <span style="color:red">**Fixes high variance**</span>                  |
| 减小特征集                      | <span style="color:red">**Fixes high variance**</span>                  |
| 增大特征集                      | <span style="color:red">**Fixes high bias**</span>                      |
| 加入邮件 header 作为特征（相当于增加特征集） | <span style="color:red">**Fixes high bias**</span>                      |
| 调整 $\lambda$ 的值            | <span style="color:red">bias-variance trade-off</span>（$\lambda$ 越大，正则化越强 → bias 越大、variance 越小）|

---



### 1.3 另一个例子：模型与目标函数不匹配

> **场景**：Logistic Regression 在 spam 上 2% error、non-spam 上 2% error（non-spam 上的高误分类不可接受）；改用 SVM (linear kernel)，spam 上 10% error、non-spam 上 0.01% error（non-spam 上的性能可接受）。但因为计算效率考虑，我们还是想用 LR。

**面对这种情况怎么分析问题？**

#### ① 算法（LR 的 gradient ascent）是否收敛了？

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec13_maximize_J(theta).webp" alt="objective optimization" width="60%" loading="lazy" decoding="async" /></div>

> 仅看目标函数 $J(\theta)$ 曲线**很难判断算法是否真的收敛**了——这是不收敛的常见隐忧。

#### ② 你优化的目标函数对不对？

例如，如果我们更关心 non-spam 的分类，可以对每个样本加权重得到一个加权和作为正确率的目标函数：

$$
a(\theta) = \sum_i w^{(i)} \mathbb{1}\{h_\theta(x^{(i)}) = y^{(i)}\}
$$

> 注意 $a(\theta)$ 可能才是我们**真正关心的指标**——但 LR 在优化的却不是它（逻辑回归的目标函数不一样）。

#### ③ Logistic Regression 是不是对的模型？$\lambda$ 是不是对的值？

$$
\max_\theta J(\theta) = \sum_{i=1}^{m} \log p(y^{(i)} \mid x^{(i)}, \theta) - \lambda \|\theta\|^2
$$

#### ④ SVM 是不是对的模型？$C$ 是不是对的值？

$$
\begin{aligned}
\min_{w, b} \quad & \|w\|^2 + C \sum_{i=1}^{m} \xi_i \\
\text{s.t.} \quad & y^{(i)}(w^T x^{(i)} - b) \ge 1 - \xi_i
\end{aligned}
$$

---



### 1.4 诊断区分"算法没收敛" vs "目标函数不对"

在上面的例子中我们已经训出了两组参数：$\theta_{\text{SVM}}$ 和 $\theta_{\text{BLR}}$（BLR = Bayesian Logistic Regression）。我们真正关心的指标是上面的 $a(\theta)$。

$$
a(\theta) = \sum_i w^{(i)} \mathbb{1}\{h_\theta(x^{(i)}) = y^{(i)}\}
$$

并且已知：

$$
a(\theta_{\text{SVM}}) > a(\theta_{\text{BLR}})
$$

> 记住：$\theta_{\text{BLR}}$ 来自最大化 $J(\theta)$（在 BLR 框架下）。



#### Optimization Algorithm 诊断

诊断这一项，看 BLR 在自己的目标函数 $J(\theta)$ 上有没有输给 SVM：

$$
J(\theta_{\text{SVM}}) \stackrel{?}{>} J(\theta_{\text{BLR}})
$$


**Case 1**：$\,a(\theta_{\text{SVM}}) > a(\theta_{\text{BLR}})$ 且 $J(\theta_{\text{SVM}}) > J(\theta_{\text{BLR}})$

> BLR 本身目的就是最大化 $J(\theta)$，但 SVM 在 $J(\theta)$ 上反而更高——说明 $\theta_{\text{BLR}}$ **没能最大化** $J(\theta)$。
>
> **Problem: optimization algorithm**（算法没收敛 / 跑得不够久）


**Case 2**：$\,a(\theta_{\text{SVM}}) > a(\theta_{\text{BLR}})$ 且 $J(\theta_{\text{SVM}}) \le J(\theta_{\text{BLR}})$

> BLR 成功地最大化 $J(\theta)$，但 SVM 在真正关心的 $a(\theta)$ 上更好——这说明 $J(\theta)$ 不是合适的优化目标。
>
> **Problem: objective function**。修复方法：调整当前模型的 $\lambda, C$ 等；或者直接换模型（因为模型不同，目标函数天然不同）。



#### 各问题对应的修正手段

| 修复手段 | 对应修正的问题 |
| --- | --- |
| 增加 gradient descent 迭代次数 | <span style="color:red">**Fixes optimization algorithm**</span> |
| 改用 Newton's method | <span style="color:red">**Fixes optimization algorithm**</span> |
| 调整 $\lambda$ | <span style="color:red">**Fixes optimization objective**</span>（也更常用于 bias-variance trade-off）|
| 改用 SVM / 神经网络 | <span style="color:red">**Fixes optimization objective 或换模型**</span> |

> 同样的诊断思路常常用于调试 **RL** 算法。

---






## 2. 误差分析 (Error Analysis)

很多实际系统会把多个学习组件拼成一条**流水线 (pipeline)**：

<div style="text-align: center;"><img src="/My-Blog/blog-images/Lec13_system-pipeline.webp" alt="system pipeline" width="100%" loading="lazy" decoding="async" /></div>

> **目标**：把最后得到的误差**归因**到每个组件——看看每个组件到底贡献了多少错误，方便我们决定**下一个该改进哪个组件**。

**做法**：把每个组件**依次替换为 ground-truth**，看 accuracy 提升了多少。提升最大的那个组件就是最值得改进的。

> **Error 分析的本质**：找出**当前点**与**目标（完美性能）** 之间的差距。

---





## 3. 消融分析 (Ablative Analysis)

与 Error Analysis 相反的方向——**逐步移除**系统中的每个组件，看 accuracy 跌了多少：

> 从现有系统中**逐个去掉**组件，看哪一个去掉后掉得最厉害——这就是**贡献最大**的组件。

> **Ablative Analysis 的本质**：找出**当前性能**与**远差于此的性能**之间的差距。

---





## 参考资料

- [CS229 Error Analysis Note](https://cs229.stanford.edu/notes2021fall/error-analysis.pdf) — Lec 13 主讲义：Model Debugging / Error Analysis / Ablative Analysis
- [CS229 2018](https://www.bilibili.com/video/BV1b4anzMEUv) — B 站上 CS229 Lec 13 讲课视频
- [Deep Learning](https://www.deeplearningbook.org/) — Goodfellow, Bengio, Courville；Ch.8 Optimization for Training（含 SGD / Mini-batch / 收敛诊断）