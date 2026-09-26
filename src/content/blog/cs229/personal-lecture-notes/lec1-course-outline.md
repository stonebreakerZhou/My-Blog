---
title: "CS229 : Lec 1 — 课程大纲"
description: CS229 Lecture 1 ：课程简介与整体概览
pubDate: 2026-08-01
series: cs229
subSeries: personal-lecture-notes
order: 2
categories:
  - CS229
  - Course Outline
---

## 引子

CS229 的第一堂课是一个**总览性 (course outline)**——在进入具体算法前，先给整门课画一张路线图：本课程会把**机器学习**大致分成几类范式，并给出每类的代表问题。

---



## 1. What is ML ?

> **Field of study that gives computers the ability to learn without being explicitly programmed.** (Arthur Samuel, 1959)

更形式化的定义（Mitchell）：

> **Well-posed Learning Problem**: A computer program is said to **learn from experience $E$** with respect to **some task $T$** and **some performance measure $P$**, if its performance on $T$, as measured by $P$, **improves with experience $E$**.

> 三个关键词：$E$（经验 / 数据）、$T$（任务）、$P$（性能度量）

### 学科层级关系

> $\text{AI} \supset \text{ML} \supset \text{DL}$

---



## 2. Supervised Learning (监督学习)

> Given a dataset of $(X, Y)$ (inputs $X$ and labels $Y$), the goal is to learn a mapping from $X$ to $Y$.

监督学习的核心是**有标注数据**：每个样本 $(x^{(i)}, y^{(i)})$ 都成对给出。

### 2.1 Regression (回归)

> The value $y$ that we're trying to predict is **continuous**.

### 2.2 Classification (分类)

> Similar to above, but the term classification refers to that $Y$ takes on a **discrete** number of values.

---



## 3. Unsupervised Learning (无监督学习)

> Only inputs $X$ and **no outputs $Y$** are given — asked to figure out interesting structure in the given data.

无监督学习没有标注——算法需要**自己发现数据中的结构**。

### 典型例子

- **Cocktail Party Problem**：从混合录音中分离独立声源
- **ICA** (Independent Component Analysis)：独立成分分析

---



## 4. Reinforcement Learning (强化学习)

> Widely used in **game playing** and **robotic applications**.

强化学习的核心是 **reward-driven**：智能体通过与环境交互，根据奖励信号学会最优策略。CS229 后半段（Lec 18–21）会展开 RL，本博客系列在此处已用公告形式收尾——详见 [[lec18-21-rl]]。

---




## 参考资料

- [CS229](https://cs229.stanford.edu/) — CS229 课程官方网站
- [CS229 2018](https://www.bilibili.com/video/BV1b4anzMEUv) — B 站上 CS229 Lec 1 讲课视频