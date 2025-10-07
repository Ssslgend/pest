# BiLSTM害虫风险预测模型

这是一个基于BiLSTM的害虫风险预测模型，用于预测水稻害虫的风险等级。

## 项目结构

```
baseline/
├── model.py           # 模型定义
├── data_processor.py  # 数据处理
├── train.py          # 训练脚本
├── evaluate.py       # 评估脚本
├── config.py         # 配置文件
└── README.md         # 项目说明
```

## 环境要求

- Python 3.7+
- PyTorch 1.7+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

## 使用方法

1. 训练模型：
```bash
python train.py
```

2. 评估模型：
```bash
python evaluate.py
```

## 模型架构

- 双向LSTM层：2层，隐藏单元数128
- Dropout：0.5
- 优化器：Adam
- 学习率：0.001
- 早停：10个epoch无改善

## 评估指标

- 准确率（Accuracy）
- F1分数（F1 Score）
- 精确率（Precision）
- 召回率（Recall）
- ROC AUC

## 输出结果

- 训练过程中的损失和准确率
- 测试集的各项评估指标
- 混淆矩阵图
- 最佳模型保存为 `best_model.pth` 