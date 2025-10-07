# 美国白蛾预测系统 (重构版本)

## 📋 项目概述

基于BiLSTM的山东省美国白蛾病虫害风险预测系统，使用气象数据、地理数据和发病历史数据进行预测。

## 🚀 快速开始

### 1. 环境安装

```bash
# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

确保以下数据文件存在：
- `datas/shandong_pest_data/发病情况.xlsx` - 发病数据
- `datas/shandong_pest_data/shandong.json` - 县边界数据
- `datas/shandong_pest_data/real_occurrence_*.csv` - 训练数据集

### 3. 训练模型

```bash
python scripts/training/train_pest_prediction_model.py
```

### 4. 模型预测

```bash
# 县级预测
python scripts/prediction/predict_county.py

# 栅格预测
python scripts/prediction/predict_raster.py
```

## 📁 项目结构

```
pest/
├── src/                          # 核心源代码
│   ├── config/                   # 配置管理
│   ├── data/                     # 数据处理模块
│   ├── models/                   # 模型定义
│   ├── training/                 # 训练模块
│   ├── prediction/               # 预测模块
│   ├── evaluation/               # 评估模块
│   └── utils/                    # 通用工具
├── scripts/                      # 执行脚本
├── experiments/                  # 实验和研究
├── tests/                        # 测试代码
├── docs/                         # 文档
├── data/                         # 数据目录
├── results/                      # 结果输出
├── datas/                        # 源数据
└── oldPestBlstem/                # 原始代码归档
```

## 📊 数据说明

### 发病程度说明
- **程度1**: 低度发生 (87.5%)
- **程度2**: 中度发生 (11.4%)
- **程度3**: 高度发生 (1.2%)

**重要**: 所有县都有美国白蛾发生，程度1-3表示严重程度，不是是否发生。

### 数据覆盖
- **时间范围**: 2019-2023年
- **空间覆盖**: 112个县区
- **特征维度**: 31个气象特征
- **样本总数**: 6,735条月度记录

## 🎯 主要功能模块

### 1. 数据处理 (src/data/)
- 数据加载器: Excel, GeoJSON, 栅格数据
- 数据处理器: 发病数据, 气象数据, 空间数据
- 数据整合器: 多源数据整合

### 2. 模型 (src/models/)
- 基础模型: BiLSTM with Attention
- 专用模型: 县级模型, 栅格模型

### 3. 训练 (src/training/)
- 训练器: 基础训练器, 专用训练器
- 流水线: 数据流水线, 训练流水线

### 4. 预测 (src/prediction/)
- 预测器: 县级预测, 栅格预测
- 后处理: 空间平滑, 可视化

### 5. 评估 (src/evaluation/)
- 评估器: 模型评估, 性能分析
- 指标: 分类指标, 回归指标

## 📈 使用示例

### 训练新模型

```python
from src.config.base_config import *
from src.models.base.bilstm import BiLSTMWithAttention
from src.training.trainers.base_trainer import BaseTrainer

# 创建模型
model = BiLSTMWithAttention(
    input_size=len(DATA_CONFIG['feature_columns']),
    hidden_size=256,
    num_layers=4,
    num_classes=3
)

# 创建训练器
trainer = BaseTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters()),
    config=TRAINING_CONFIG
)

# 开始训练
trainer.train()
```

## 📚 文档

- [项目重构方案](项目重构方案.md) - 详细的重构设计文档
- [重构总结](项目重构总结.md) - 重构完成总结
- [README_重构版本.md](README_重构版本.md) - 详细使用指南

## 🧪 实验

```bash
# 运行测试
pytest tests/

# 运行消融实验
python experiments/ablation_studies/feature_ablation.py

# 运行模型比较
python experiments/model_comparison/comparative_analysis.py
```

## 📈 性能指标

当前模型性能 (基于真实发病数据):
- **训练集准确率**: ~90%
- **验证集准确率**: ~85%
- **测试集准确率**: ~88%

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证

## 📞 历史版本

- **oldPestBlstem/** - 原始项目代码归档
- **当前版本** - 重构后的模块化版本

---

**注意**: 这是重构后的版本，具有更好的模块化结构。如需查看旧版本代码，请查看 `oldPestBlstem/` 目录。