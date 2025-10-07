# 山东省县域美国白蛾第一代发病情况训练数据集

## 概述

本数据集是针对山东省县域粒度美国白蛾第一代（5-6月）发病情况的预测训练数据集，整合了真实发病程度数据、县边界地理信息和栅格气象数据。

## 数据源

### 1. 真实发病程度数据
- **文件**: `datas/shandong_pest_data/发病情况.xlsx`
- **内容**: 山东省各县2019-2024年美国白蛾发病程度记录
- **结构**:
  - Year: 年份
  - City: 城市
  - County: 县/区
  - FirstGen_Severity_MayJun: 第一代发病程度（5-6月）
    - 1: 轻度发病
    - 2: 中度发病
    - 3: 重度发病
  - SecondGen_Severity_JulAug: 第二代发病程度（7-8月）
  - ThirdGen_Severity_SepOct: 第三代发病程度（9-10月）

### 2. 县边界地理数据
- **文件**: `datas/shandong_pest_data/shandong.json`
- **格式**: GeoJSON
- **内容**: 山东省136个县的边界数据
- **字段**: adcode, name, center, centroid, level等

### 3. 栅格气象数据
- **文件**: `datas/shandong_pest_data/shandong_spatial_meteorological_data.csv`
- **时间范围**: 2019-2023年
- **空间覆盖**: 山东省全域栅格数据
- **时间分辨率**: 日数据
- **主要特征**:
  - 温度 (Temperature)
  - 湿度 (Humidity)
  - 降雨量 (Rainfall)
  - 风速 (WS)
  - 气压 (Pressure)
  - 日照 (Sunshine)
  - 能见度 (Visibility)
  - 温湿度指数 (Temp_Humidity_Index)
  - 移动平均特征 (7天, 30天)
  - 累积降雨量等

## 数据处理流程

### 1. 第一代发病数据提取
- 从原始数据中提取第一代（5-6月）发病程度记录
- 创建二分类标签：有发病（1-3级）vs 无发病（0级）
- 统计分析显示：数据中所有记录都有发病（1-3级）

### 2. 气象特征提取
- 筛选5-6月气象数据（第一代发生期）
- 按县、年分组计算统计特征：
  - 均值 (mean)
  - 标准差 (std)
  - 最小值 (min)
  - 最大值 (max)
  - 中位数 (median)

### 3. 空间特征整合
- 基于县边界数据计算县质心坐标
- 添加纬度、经度作为空间特征
- 提供地理空间上下文信息

### 4. 数据集整合
- 以发病数据为基础，整合气象和空间特征
- 确保县名匹配和年份对应
- 处理缺失值和异常值

## 最终数据集

### 文件结构
```
datas/shandong_pest_data/
├── county_level_firstgen_train.csv      # 训练集 (2019-2022, 452条)
├── county_level_firstgen_val.csv        # 验证集 (2023, 113条)
├── county_level_firstgen_test.csv       # 测试集 (2024, 0条)
├── county_level_firstgen_complete.csv   # 完整数据集 (565条)
└── county_level_firstgen_dataset_info.json # 数据集信息
```

### 数据集统计
- **总记录数**: 565条
- **时间范围**: 2019-2023年
- **覆盖县数**: 110个县
- **覆盖市数**: 16个市
- **年份分布**: 每年约113条记录

### 发病程度分布
- 轻度发病 (1级): 363条 (64.2%)
- 中度发病 (2级): 201条 (35.6%)
- 重度发病 (3级): 1条 (0.2%)

### 特征列表 (共27个)
#### 气象特征 (25个)
- **温度相关**: Temperature_mean, Temperature_std, Temperature_min, Temperature_max, Temperature_median
- **湿度相关**: Humidity_mean, Humidity_std, Humidity_min, Humidity_max, Humidity_median
- **降雨相关**: Rainfall_mean, Rainfall_std, Rainfall_min, Rainfall_max, Rainfall_median
- **气压相关**: Pressure_mean, Pressure_std, Pressure_min, Pressure_max, Pressure_median
- **温湿度指数**: Temp_Humidity_Index_mean, Temp_Humidity_Index_std, Temp_Humidity_Index_min, Temp_Humidity_Index_max, Temp_Humidity_Index_median

#### 空间特征 (2个)
- **Latitude**: 纬度（县质心）
- **Longitude**: 经度（县质心）

#### 标签变量
- **Severity_Level**: 发病程度 (1-3级)
- **Has_Occurrence**: 是否发病 (1: 有, 0: 无)

## 主要城市数据量
- 聊城市: 55条
- 济南市: 55条
- 淄博市: 50条
- 潍坊市: 50条
- 烟台市: 45条
- 威海市: 45条
- 青岛市: 40条
- 临沂市: 40条
- 淄博市: 30条
- 临沂市: 30条

## 使用方法

### 1. 数据加载
```python
import pandas as pd

# 加载训练数据
train_data = pd.read_csv('datas/shandong_pest_data/county_level_firstgen_train.csv')
val_data = pd.read_csv('datas/shandong_pest_data/county_level_firstgen_val.csv')
```

### 2. 特征选择
```python
from county_level_config import CountyLevelConfig

# 气象特征
meteo_features = CountyLevelConfig.METEOROLOGICAL_FEATURES

# 空间特征
spatial_features = CountyLevelConfig.SPATIAL_FEATURES

# 所有特征
all_features = CountyLevelConfig.ALL_FEATURES
```

### 3. 模型训练
```python
from sklearn.ensemble import RandomForestClassifier

# 准备特征和标签
X_train = train_data[all_features]
y_train_class = train_data['Has_Occurrence']  # 分类任务
y_train_reg = train_data['Severity_Level']   # 回归任务

# 训练分类模型
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train_class)
```

## 特点与限制

### 优势
- **真实数据**: 基于实际发病记录
- **时空覆盖**: 覆盖全省5年数据
- **多维特征**: 整合气象、空间多重特征
- **统计丰富**: 提供多种统计量的气象特征

### 限制
- **样本不均衡**: 重度发病样本较少
- **负样本缺失**: 数据中所有县都有发病记录
- **时间跨度**: 2024年数据缺失，影响测试集
- **县名匹配**: 部分县名可能存在不匹配问题

## 应用场景

1. **发病预测**: 预测下一年度各县发病程度
2. **风险评估**: 评估不同地区发病风险
3. **防控决策**: 指导病虫害防控资源分配
4. **科学研究**: 研究气象因子与发病关系

## 技术规格

- **数据格式**: CSV (UTF-8编码)
- **特征数量**: 27个
- **样本数量**: 565条
- **时间粒度**: 年度
- **空间粒度**: 县级
- **预测目标**: 美国白蛾第一代发病情况

## 版本信息

- **创建日期**: 2025-10-03
- **数据版本**: v1.0
- **处理脚本**: `create_county_level_training_data.py`
- **配置文件**: `county_level_config.py`