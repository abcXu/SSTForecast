# 🌊 基于交叉注意力机制的海表温度预测算法设计与实现

本代码仓库为我的本科毕业设计的代码实现，本项目提出了一种融合 **交叉注意力机制（Cross-Attention）** 和 **二维门控循环单元（GRU2D）** 的海表温度（SST）预测算法，通过深度时空特征建模与多模态信息融合，有效提升了 SST 预测的精度与稳定性。该模型适用于遥感数据建模、气候预测、海洋科学等领域。

---

## 📘 项目简介

海表温度（SST）是反映气候变化与海洋动力过程的重要指标，传统的预测方法如统计模型或数值模拟存在建模复杂度高、非线性表达能力弱等问题。为了解决上述瓶颈，本项目引入深度学习技术，结合 GRU2D 与 Cross-Attention 模块，建立一个适用于时空序列的高精度预测模型。

---

## 🧱 模型结构

### 模型整体架构

<p align="center">
  <img src="https://gitee.com/xu_yan_peng/picgo-typro/raw/master/image-20250525133343623.png" alt="模型整体结构" width="600"/>
</p>

### 交叉注意力模块

<p align="center">
  <img src="https://gitee.com/xu_yan_peng/picgo-typro/raw/master/image-20250525133426593.png" alt="交叉注意力结构" width="600"/>
</p>

### GRU2D 模块

<p align="center">
  <img src="https://gitee.com/xu_yan_peng/picgo-typro/raw/master/image-20250525133445332.png" alt="GRU2D结构" width="600"/>
</p>

## 📈 评估指标

训练与验证过程中支持以下指标：

- 均方根误差（RMSE）
- 结构相似性（SSIM）
- 平均绝对误差（MAE）
- 决定系数（$R^2$）

