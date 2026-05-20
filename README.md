# Deep Residual Learning for Image Recognition

基于论文 *Deep Residual Learning for Image Recognition* 的课程实验级简化复现项目，使用 `Python + PyTorch` 在 `CIFAR-10` 数据集上实现并训练一个小型 `ResNet-20` 模型。

## 项目说明

原论文由 Kaiming He 等人发表于 CVPR 2016，核心思想是通过残差连接（Residual Connection）缓解深层神经网络训练中的退化问题。本项目没有严格复现论文在 ImageNet 上的大规模实验，而是选择更适合课程作业与普通电脑运行环境的 `CIFAR-10` 数据集，完成对残差网络核心思想的实现与验证。

因此，本项目更准确的定位是：

- 基于论文思想的简化复现
- 面向课程实验的可运行实现
- 用于学习 ResNet 结构、训练流程和实验分析

## 项目结构

```text
.
├── install-deps.ps1         # Windows PowerShell 依赖安装脚本
├── model.py                 # ResNet-20 模型定义
├── train.py                 # 训练与测试脚本
├── requirements-cpu.txt     # CPU 环境依赖列表
├── RESULT.md                # 本次实验结果与说明
├── run_summary.json         # 运行摘要
└── README.md                # 项目说明文档
```

## 环境要求

- Python 3.8
- PyTorch
- torchvision
- numpy
- matplotlib
- scikit-learn
- pandas
- tqdm

推荐使用项目虚拟环境 `venv`。

## 安装依赖

### 方式一：使用 PowerShell 脚本

在项目目录下运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\install-deps.ps1
```

### 方式二：手动安装

```powershell
python -m pip install -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## 运行方法

直接执行训练脚本：

```powershell
python train.py
```

脚本会自动完成以下工作：

- 自动下载 `CIFAR-10` 数据集
- 构建小型 `ResNet-20`
- 完成训练与测试
- 生成结果文件 `RESULT.md`
- 保存模型权重 `resnet20_cifar10_demo.pth`
- 生成运行摘要 `run_summary.json`

## 当前实验配置

本项目默认采用轻量实验配置，便于在普通 CPU 环境中运行：

- 模型：ResNet-20
- 数据集：CIFAR-10
- Epochs：5
- Batch Size：128
- Learning Rate：0.01
- 训练样本数：5000
- 测试样本数：1000
- 优化器：SGD(momentum=0.9, weight_decay=1e-4)

## 当前实验结果

本次实际运行结果如下：

- Test Loss：1.8212
- Test Accuracy：31.80%

更详细的训练过程、分类报告与实验分析见 [RESULT.md](./RESULT.md)。

## 结果说明

由于本项目使用的是课程实验级简化配置，而不是论文中的完整 ImageNet 训练方案，因此当前结果不能与原论文最终指标直接对比。该项目的主要目标是：

- 理解 ResNet 的基本结构
- 复现残差连接的核心思想
- 跑通图像分类实验的完整流程
- 为课程实验报告提供代码和结果支撑

如果需要更接近论文结果，可以进一步：

- 使用完整 CIFAR-10 训练集
- 增加训练轮数
- 使用 GPU 训练
- 采用更完整的学习率调度策略
- 对比普通 CNN 与 ResNet 的训练效果

## 参考论文

- He, K., Zhang, X., Ren, S., Sun, J. *Deep Residual Learning for Image Recognition*. CVPR 2016.
- 论文链接：https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html

## 备注

- `data/`、`.venv/`、`__pycache__/`、`.idea/` 等目录已加入 `.gitignore`，不会上传到 GitHub。
- 第一次运行时会自动下载 `CIFAR-10` 数据集，请保持网络可用。
