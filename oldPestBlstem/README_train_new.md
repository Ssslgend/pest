# 害虫风险预测 BiLSTM 栅格模型训练工具

本项目为害虫风险预测BiLSTM模型训练工具，使用改进的配置和数据处理机制，可以直接处理train.csv数据集进行训练。

## 功能特点

- 支持处理标准CSV格式数据
- 使用BiLSTM深度学习模型进行害虫风险预测
- 自动进行特征标准化和数据拆分
- 生成风险等级分类图和特征重要性分析
- 支持多种评估指标和早停策略
- 可视化训练过程与模型性能

## 环境要求

- Python 3.7+
- PyTorch 1.8+
- pandas, numpy, scikit-learn, matplotlib, joblib
- rasterio (用于栅格处理)
- geopandas (用于地理数据处理)

## 文件结构

```
pestBIstm/
├── datas/
│   └── train.csv             # 训练数据集
├── sd_raster_prediction/
│   ├── config_raster_new.py  # 新的配置文件
│   ├── data_processor_raster_new.py  # 新的数据处理器
│   ├── train_raster_new.py   # 新的训练脚本
│   └── test_config_new.py    # 配置测试脚本
├── model/
│   └── bilstm.py             # BiLSTM模型定义
├── utils/
│   └── ...                   # 辅助工具函数
├── results/                  # 训练结果保存目录
│   ├── trained_model/        # 模型权重文件
│   └── analysis/             # 分析结果与图表
└── run_train_new.py          # 运行训练的主脚本
```

## 使用方法

### 1. 使用运行脚本

最简单的使用方式是直接运行主脚本：

```bash
python run_train_new.py
```

这将显示一个菜单，您可以选择：
1. 测试配置 - 检查数据集和配置是否正确
2. 开始训练 - 使用train.csv数据开始训练过程
3. 退出

### 2. 手动运行各个脚本

您也可以单独运行每个脚本：

```bash
# 测试配置
python sd_raster_prediction/test_config_new.py

# 开始训练
python sd_raster_prediction/train_raster_new.py
```

## 数据格式要求

train.csv应包含以下列：
- `发生样点纬度`,`发生样点经度` - 地理坐标
- `label` - 标签列（1表示害虫发生点，0表示无害虫点）
- 其他列将被作为特征，除了`year`列会被排除

## 配置说明

配置文件`config_raster_new.py`包含以下主要设置：

- 文件路径设置（数据集、模型保存位置等）
- 数据处理参数（测试集比例、验证集比例等）
- 模型超参数（隐藏层大小、层数、Dropout等）
- 训练参数（批次大小、学习率、早停耐心值等）
- 预测参数（输出格式、风险等级阈值等）

## 训练输出

训练完成后，以下文件会被保存：

- 训练好的模型权重文件 (`.pth`)
- 特征标准化器 (`.joblib`)
- 训练历史图表 (`.png`)
- 特征重要性分析 (`.png` 和 `.csv`)
- 训练过程日志 (`.csv`)

## 联系与支持

如有问题或建议，请联系项目维护者。 