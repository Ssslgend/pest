# sd_raster_prediction/feature_extrapolation.py
import numpy as np
import rasterio
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

class FeatureExtrapolator:
    """特征时间序列外推器，用于预测未来时期的特征值"""
    
    def __init__(self, config):
        """
        初始化特征外推器
        
        参数:
            config: 配置字典，包含特征变化率和方向
        """
        self.config = config
        self.future_config = config.get('future', {})
        self.feature_change_rates = self.future_config.get('feature_monthly_change_rates', {})
        self.feature_directions = self.future_config.get('feature_change_direction', {})
        self.prediction_method = self.future_config.get('feature_prediction', {}).get('method', 'trend_extrapolation')
        self.seasonal_factor = self.future_config.get('feature_prediction', {}).get('seasonal_factor', True)
        self.smoothing_factor = self.future_config.get('feature_prediction', {}).get('smoothing_factor', 0.3)
        
        # 获取当前月份（用于季节性调整）
        self.current_month = datetime.now().month
        
        # 记录已加载的栅格数据
        self.cached_rasters = {}
        
    def get_seasonal_factor(self, feature_name, target_month):
        """
        获取季节性调整因子
        
        参数:
            feature_name: 特征名称
            target_month: 目标月份
            
        返回:
            季节性调整因子
        """
        # 这是一个简化的季节性模型，根据特征类型和月份返回不同的调整因子
        # 实际应用中，应该基于历史数据建立更精确的季节性模型
        
        # 植被相关特征有较强的季节性
        vegetation_features = ['evi_Band_1', 'lai_Band_1', 'ndvi_Band_']
        temperature_features = ['lst_Band_1', '最热月最高温', '最冷月最低温', '年均温_', '平均日较差']
        precipitation_features = ['最干月降水量', '年降水量', '降水季节性']
        
        # 北半球季节模型（简化版）
        season_factors = {
            # 月份: [植被因子, 温度因子, 降水因子]
            1: [-0.8, -0.9, 0.2],   # 1月：植被减少，温度低，降水适中
            2: [-0.6, -0.7, 0.3],   # 2月
            3: [0.2, -0.3, 0.5],    # 3月：春季，植被开始生长，温度回升，降水增加
            4: [0.6, 0.1, 0.7],     # 4月
            5: [0.8, 0.4, 0.6],     # 5月
            6: [0.9, 0.7, 0.4],     # 6月：夏初，植被旺盛，温度高，降水变化
            7: [0.7, 0.9, 0.2],     # 7月：夏季，植被稳定，温度最高，降水变化
            8: [0.5, 0.8, 0.1],     # 8月
            9: [0.2, 0.5, 0.3],     # 9月：秋季，植被开始减少，温度下降，降水变化
            10: [-0.3, 0.0, 0.4],   # 10月
            11: [-0.6, -0.4, 0.3],  # 11月
            12: [-0.7, -0.7, 0.2],  # 12月：冬季，植被少，温度低，降水变化
        }
        
        # 根据特征类型选择适当的季节性因子
        if feature_name in vegetation_features:
            return season_factors[target_month][0]
        elif feature_name in temperature_features:
            return season_factors[target_month][1]
        elif feature_name in precipitation_features:
            return season_factors[target_month][2]
        else:
            return 0.0  # 默认无季节性影响
    
    def extrapolate_feature(self, feature_name, base_raster, periods):
        """
        外推特征值
        
        参数:
            feature_name: 特征名称
            base_raster: 基础栅格数据
            periods: 外推的时期数
            
        返回:
            外推后的栅格数据列表
        """
        # 获取特征变化率和方向
        change_rate = self.feature_change_rates.get(feature_name, 0.0)
        direction = self.feature_directions.get(feature_name, 0)
        
        # 如果变化率为0或方向为0，则所有时期的值都相同
        if change_rate == 0.0 or direction == 0:
            return [base_raster.copy() for _ in range(periods)]
        
        # 准备存储外推结果
        extrapolated_features = []
        
        # 对每个时期进行外推
        for period in range(1, periods + 1):
            # 基于方法选择外推算法
            if self.prediction_method == 'trend_extrapolation':
                # 基于趋势的简单外推
                extrapolated = self._trend_extrapolation(
                    base_raster, 
                    feature_name,
                    change_rate, 
                    direction, 
                    period
                )
            else:
                # 默认使用趋势外推
                extrapolated = self._trend_extrapolation(
                    base_raster, 
                    feature_name,
                    change_rate, 
                    direction, 
                    period
                )
            
            extrapolated_features.append(extrapolated)
        
        return extrapolated_features
    
    def _trend_extrapolation(self, base_raster, feature_name, change_rate, direction, period):
        """
        基于趋势的简单外推算法
        
        参数:
            base_raster: 基础栅格数据
            feature_name: 特征名称
            change_rate: 变化率
            direction: 变化方向
            period: 外推时期
            
        返回:
            外推后的栅格数据
        """
        # 复制基础栅格
        extrapolated = base_raster.copy()
        
        # 计算基本变化量
        change_factor = 1.0 + (direction * change_rate * period)
        
        # 如果考虑季节性因素
        if self.seasonal_factor:
            # 计算目标月份
            target_month = ((self.current_month - 1 + period) % 12) + 1
            seasonal_adjust = self.get_seasonal_factor(feature_name, target_month)
            
            # 应用季节性调整
            change_factor = change_factor * (1.0 + seasonal_adjust * self.smoothing_factor)
        
        # 应用变化因子（确保值在合理范围内）
        if np.isnan(base_raster).any():
            # 如果有NaN值，保留NaN
            mask = ~np.isnan(base_raster)
            extrapolated[mask] = base_raster[mask] * change_factor
        else:
            # 如果没有NaN值，直接应用变化因子
            extrapolated = base_raster * change_factor
        
        return extrapolated
    
    def extrapolate_all_features(self, feature_raster_map, periods):
        """
        外推所有特征
        
        参数:
            feature_raster_map: 特征到栅格文件路径的映射
            periods: 外推的时期数
            
        返回:
            外推后的特征栅格数据，格式为 {feature_name: [raster_1, raster_2, ..., raster_n]}
        """
        extrapolated_features = {}
        
        print(f"Extrapolating features for {periods} future periods...")
        for feature_name, raster_path in tqdm(feature_raster_map.items(), desc="Extrapolating Features"):
            # 检查是否已缓存该栅格
            if feature_name not in self.cached_rasters:
                try:
                    # 打开栅格文件
                    with rasterio.open(raster_path) as src:
                        # 读取数据
                        raster_data = src.read(1).astype(np.float32)
                        # 缓存数据
                        self.cached_rasters[feature_name] = raster_data
                except Exception as e:
                    print(f"Error opening raster {feature_name} ({raster_path}): {e}")
                    continue
            
            # 获取基础栅格数据
            base_raster = self.cached_rasters[feature_name]
            
            # 外推特征
            extrapolated = self.extrapolate_feature(feature_name, base_raster, periods)
            
            # 存储结果
            extrapolated_features[feature_name] = extrapolated
        
        return extrapolated_features
    
    def visualize_extrapolation(self, feature_name, extrapolated_data, output_dir):
        """
        可视化特征外推结果
        
        参数:
            feature_name: 特征名称
            extrapolated_data: 外推后的数据列表
            output_dir: 输出目录
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 计算值范围，以保持一致的色标
        all_data = np.concatenate([data.flatten() for data in extrapolated_data])
        vmin, vmax = np.nanpercentile(all_data, [2, 98])
        
        # 为每个时期创建图像
        for i, data in enumerate(extrapolated_data):
            plt.figure(figsize=(10, 8))
            im = plt.imshow(data, cmap='viridis', vmin=vmin, vmax=vmax)
            plt.colorbar(im, label=f'{feature_name} 值')
            plt.title(f'{feature_name} - 未来第 {i+1} 个月')
            plt.tight_layout()
            
            # 保存图像
            output_path = os.path.join(output_dir, f'{feature_name}_period_{i+1}.png')
            plt.savefig(output_path, dpi=200)
            plt.close()
            
        print(f"Visualization for feature {feature_name} saved to {output_dir}")
        
    def save_extrapolated_rasters(self, feature_name, extrapolated_data, output_dir, template_raster_path):
        """
        保存外推后的栅格数据为GeoTIFF文件
        
        参数:
            feature_name: 特征名称
            extrapolated_data: 外推后的数据列表
            output_dir: 输出目录
            template_raster_path: 模板栅格文件路径，用于获取元数据
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 打开模板栅格以获取元数据
        with rasterio.open(template_raster_path) as src:
            profile = src.profile.copy()
            
            # 为每个时期保存栅格
            for i, data in enumerate(extrapolated_data):
                output_path = os.path.join(output_dir, f'{feature_name}_period_{i+1}.tif')
                
                # 更新数据类型
                profile.update(dtype=rasterio.float32)
                
                # 保存栅格
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(data.astype(rasterio.float32), 1)
        
        print(f"Extrapolated rasters for feature {feature_name} saved to {output_dir}") 