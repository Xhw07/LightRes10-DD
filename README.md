# 轻量化改进ResNet10图像分类模型
基于PyTorch实现的改进型轻量级ResNet10模型，集成**双重Dropout**优化，适用于二分类图像分类任务。

## 核心特性
- 轻量化ResNet10网络结构（简化版ResNet）
- 双重Dropout机制（空间Dropout + 全连接层Dropout）
- 固定100轮训练，无早停
- 支持训练/验证集独立划分
- 输出准确率、F1、精确率、召回率等评估指标

## 环境依赖
- Python 3.x
- PyTorch
- torchvision
- scikit-learn
- numpy

## 数据集格式
数据集根目录需包含 `train` 和 `val` 文件夹，按类别分二级子文件夹存放图像。

## 主要配置
- 图像尺寸：224×224
- 批次大小：16
- 学习率：8e-5
- 训练轮数：100
- 分类数：2
