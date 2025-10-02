# 山东美国白蛾病虫害风险预测系统 - 多年份预测功能

本系统可对2019-2024年山东省美国白蛾病虫害风险进行预测分析，并生成相应的风险分布图。

## 新增功能：多年份预测

现已支持根据年份自动调整输入输出路径，可以一次性预测多个年份的美国白蛾病虫害风险分布情况。

### 使用方法

#### 方法一：运行多年份批量预测脚本

```bash
python run_multi_year_prediction.py
```

运行后，系统将提示您输入要预测的年份，多个年份用空格分隔，例如：`2019 2020 2023`
也可以输入 `all` 来预测所有年份（2019-2024）

#### 方法二：使用命令行参数

```bash
# 预测指定年份
python run_multi_year_prediction.py -y 2019 2020 2023

# 预测所有年份
python run_multi_year_prediction.py --all

# 预测指定年份但跳过可视化步骤
python run_multi_year_prediction.py -y 2019 --skip-vis
```

#### 方法三：在其他脚本中调用

```python
# 单年份预测示例
from sd_raster_prediction.config_raster_new import get_config
from sd_raster_prediction.predict_raster_new import predict_raster

# 获取2022年的配置
config_2022 = get_config(prediction_year=2022)

# 执行2022年预测
predict_raster(config_2022)
```

### 输出结果

对每个年份，系统会在对应的输出目录中生成以下文件：

1. `sd_predicted_probability.tif` - 美国白蛾发生概率栅格图
2. `sd_risk_classification.tif` - 风险等级分类栅格图
3. `sd_raw_probability.tif` - 原始概率值栅格图
4. `probability_map.png` - 概率分布可视化图
5. `risk_map.png` - 风险等级分布可视化图
6. `probability_histogram.png` - 概率分布直方图
7. `risk_distribution_pie.png` - 风险等级比例饼图

### 注意事项

1. 请确保对应年份的输入数据已经准备好，存放在正确的目录结构中
2. 输入路径格式：`H:/data_new2025/2019_2024_sd/prediction_year/{年份}`
3. 输出路径格式：`H:/data_new2025/2019_2024_sd/prediction_year/results/{年份}`
4. 如果某年份的输入目录不存在，该年份将被跳过预测 