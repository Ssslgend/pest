# 山东省美国白蛾发生预测增强数据集

生成时间: 2025-10-06 12:22:33

## 数据集概述

- 总样本数: 690
- 覆盖县数: 135
- 年份范围: 2019-2023
- 特征数: 105

## 发病程度分布

| 发病等级 | 样本数 | 占比 |
|---------|--------|------|
| 0级 | 125 | 18.1% |
| 1级 | 363 | 52.6% |
| 2级 | 201 | 29.1% |
| 3级 | 1 | 0.1% |

## 特征分类

### 基础信息
- County: 县级行政区名称
- Year: 年份
- Severity_Level: 发病程度等级 (0-3级)

### 气象特征 (47个)
- Temperature_mean
- Temperature_std
- Temperature_min
- Temperature_max
- Temperature_median
- Humidity_mean
- Humidity_std
- Humidity_min
- Humidity_max
- Humidity_median
- Temp_Humidity_Index_mean
- Temp_Humidity_Index_std
- Temp_Humidity_Index_min
- Temp_Humidity_Index_max
- Temp_Humidity_Index_median
- Spring_Temp
- Summer_Temp
- Autumn_Temp
- Winter_Temp
- Annual_Temp
- Temp_Range
- Spring_Precip
- Summer_Precip
- Autumn_Precip
- Winter_Precip
- Annual_Precip
- Precip_Days
- Max_Daily_Precip
- Spring_Humidity
- Summer_Humidity
- Autumn_Humidity
- Winter_Humidity
- Annual_Humidity
- Spring_Wind
- Summer_Wind
- Autumn_Wind
- Winter_Wind
- Annual_Wind
- Max_Wind_Speed
- Spring_Sunshine
- Summer_Sunshine
- Autumn_Sunshine
- Winter_Sunshine
- Annual_Sunshine
- Frost_Free_Days
- Heat_Wave_Days
- Drought_Index

### 遥感特征 (33个)
- Spring_NDVI
- Summer_NDVI
- Autumn_NDVI
- Annual_NDVI
- Spring_EVI
- Summer_EVI
- Autumn_EVI
- Annual_EVI
- Spring_LST
- Summer_LST
- Autumn_LST
- Winter_LST
- Annual_LST
- Forest_Cover_Percent
- Farmland_Percent
- Urban_Percent
- Water_Percent
- Other_Land_Percent
- Vegetation_Cover_Percent
- Elevation_Mean
- Elevation_STD
- Slope_Mean
- River_Density
- Lake_Distance
- TRMM_Spring
- TRMM_Summer
- TRMM_Autumn
- TRMM_Annual
- Soil_Moisture_Spring
- Soil_Moisture_Summer
- Soil_Moisture_Autumn
- Soil_Moisture_Annual
- County_Elevation

### 地理特征 (8个)
- County
- Forest_Cover_Percent
- Coastal_Distance
- County_Elevation
- Forest_Cover_Base
- Coastal_Influence_Index
- Mountain_Influence_Index
- Forest_Ecology_Index

## 数据增强说明

1. **健康县数据**: 为缺失的25个县生成了历史健康县数据（0级发病程度）
2. **遥感数据**: 整合了NDVI、EVI、LST、土地利用、植被覆盖度等遥感特征
3. **地理数据**: 添加了海岸线距离、海拔、森林覆盖率等地理环境特征
4. **数据覆盖**: 实现了山东省135个县级行政区的完整覆盖

## 数据质量

- 所有特征均经过合理性检查
- 异常值使用3σ规则检测和处理
- 数据分布符合山东省地理气候特征
- 保证了数据的可重复性和一致性
