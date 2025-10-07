#!/usr/bin/env python3
"""
山东县域美国白蛾风险预测结果输出格式管理
定义标准化的预测结果输出格式
"""

import pandas as pd
import numpy as np
import json
import geopandas as gpd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from enum import Enum
import os

class RiskLevel(Enum):
    """风险等级枚举"""
    LOW = 1        # 低风险
    MEDIUM = 2     # 中风险  
    HIGH = 3       # 高风险
    EXTREME = 4    # 极高风险

class PredictionOutputFormat:
    """预测结果输出格式管理器"""
    
    def __init__(self, data_dir: str = "pestBIstm"):
        self.data_dir = data_dir
        self.output_dir = os.path.join(data_dir, "results", "predictions")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def create_prediction_result_template(self) -> Dict:
        """
        创建预测结果模板
        Returns:
            预测结果字典模板
        """
        return {
            "prediction_info": {
                "model_name": "山东县域美国白蛾风险预测模型",
                "prediction_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "target_year": 2024,
                "target_counties": [],
                "feature_types": [
                    "avg_tmp", "precipitation", "rel_humidity", 
                    "ndvi", "dem", "soil_moisture"
                ],
                "model_version": "v1.0",
                "prediction_confidence_threshold": 0.7
            },
            "county_predictions": [],
            "summary_statistics": {
                "total_counties": 0,
                "high_risk_counties": 0,
                "medium_risk_counties": 0,
                "low_risk_counties": 0,
                "average_probability": 0.0,
                "max_probability": 0.0,
                "min_probability": 0.0
            },
            "risk_distribution": {
                "extreme_risk": [],
                "high_risk": [],
                "medium_risk": [],
                "low_risk": []
            }
        }
    
    def classify_risk_level(self, probability: float) -> Tuple[str, RiskLevel]:
        """
        根据发生概率分类风险等级
        Args:
            probability: 发生概率 (0-1)
        Returns:
            (风险等级名称, 风险等级枚举)
        """
        if probability >= 0.8:
            return "极高风险", RiskLevel.EXTREME
        elif probability >= 0.6:
            return "高风险", RiskLevel.HIGH
        elif probability >= 0.3:
            return "中风险", RiskLevel.MEDIUM
        else:
            return "低风险", RiskLevel.LOW
    
    def calculate_confidence_score(self, probability: float, model_uncertainty: float = 0.1) -> float:
        """
        计算预测置信度
        Args:
            probability: 发生概率
            model_uncertainty: 模型不确定性
        Returns:
            置信度分数 (0-1)
        """
        # 基于概率和不确定性计算置信度
        if probability < 0.2 or probability > 0.8:
            # 极端概率置信度较高
            base_confidence = 0.9
        elif 0.4 <= probability <= 0.6:
            # 中等概率置信度较低
            base_confidence = 0.6
        else:
            # 其他概率中等置信度
            base_confidence = 0.75
        
        # 调整置信度
        confidence = base_confidence * (1 - model_uncertainty)
        return max(0.1, min(1.0, confidence))
    
    def create_county_prediction(self, county_name: str, probability: float, 
                               geometry=None, additional_info: Dict = None) -> Dict:
        """
        创建县域预测结果
        Args:
            county_name: 县域名称
            probability: 发生概率
            geometry: 县域几何信息
            additional_info: 额外信息
        Returns:
            县域预测结果字典
        """
        # 分类风险等级
        risk_level_name, risk_level_enum = self.classify_risk_level(probability)
        
        # 计算置信度
        confidence = self.calculate_confidence_score(probability)
        
        # 创建预测结果
        prediction = {
            "county_name": county_name,
            "occurrence_probability": round(probability, 4),
            "risk_level": risk_level_name,
            "risk_level_code": risk_level_enum.value,
            "confidence_score": round(confidence, 4),
            "prediction_reliable": confidence >= 0.7,
            "prediction_date": datetime.now().strftime("%Y-%m-%d"),
            "geometry_wkt": geometry.wkt if geometry else None
        }
        
        # 添加额外信息
        if additional_info:
            prediction.update(additional_info)
        
        return prediction
    
    def create_prediction_dataframe(self, predictions: List[Dict]) -> pd.DataFrame:
        """
        创建预测结果DataFrame
        Args:
            predictions: 预测结果列表
        Returns:
            预测结果DataFrame
        """
        df_data = []
        
        for pred in predictions:
            row = {
                'county_name': pred['county_name'],
                'occurrence_probability': pred['occurrence_probability'],
                'risk_level': pred['risk_level'],
                'risk_level_code': pred['risk_level_code'],
                'confidence_score': pred['confidence_score'],
                'prediction_reliable': pred['prediction_reliable'],
                'prediction_date': pred['prediction_date']
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # 按风险等级和概率排序
        df['risk_level_order'] = df['risk_level_code']
        df = df.sort_values(['risk_level_order', 'occurrence_probability'], ascending=[False, False])
        df = df.drop('risk_level_order', axis=1)
        
        return df
    
    def create_summary_statistics(self, predictions: List[Dict]) -> Dict:
        """
        创建统计摘要
        Args:
            predictions: 预测结果列表
        Returns:
            统计摘要字典
        """
        if not predictions:
            return {}
        
        probabilities = [p['occurrence_probability'] for p in predictions]
        risk_levels = [p['risk_level'] for p in predictions]
        
        # 统计各风险等级数量
        risk_counts = {
            'extreme_risk': risk_levels.count('极高风险'),
            'high_risk': risk_levels.count('高风险'),
            'medium_risk': risk_levels.count('中风险'),
            'low_risk': risk_levels.count('低风险')
        }
        
        # 创建风险分布
        risk_distribution = {
            'extreme_risk': [p['county_name'] for p in predictions if p['risk_level'] == '极高风险'],
            'high_risk': [p['county_name'] for p in predictions if p['risk_level'] == '高风险'],
            'medium_risk': [p['county_name'] for p in predictions if p['risk_level'] == '中风险'],
            'low_risk': [p['county_name'] for p in predictions if p['risk_level'] == '低风险']
        }
        
        summary = {
            "total_counties": len(predictions),
            "high_risk_counties": risk_counts['high_risk'] + risk_counts['extreme_risk'],
            "medium_risk_counties": risk_counts['medium_risk'],
            "low_risk_counties": risk_counts['low_risk'],
            "average_probability": round(np.mean(probabilities), 4),
            "max_probability": round(np.max(probabilities), 4),
            "min_probability": round(np.min(probabilities), 4),
            "std_probability": round(np.std(probabilities), 4),
            "risk_counts": risk_counts,
            "risk_distribution": risk_distribution
        }
        
        return summary
    
    def save_predictions_csv(self, predictions: List[Dict], filename: str = None) -> str:
        """
        保存预测结果为CSV格式
        Args:
            predictions: 预测结果列表
            filename: 文件名
        Returns:
            文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"shandong_county_predictions_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # 创建DataFrame
        df = self.create_prediction_dataframe(predictions)
        
        # 保存CSV
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        print(f"预测结果CSV已保存至: {filepath}")
        return filepath
    
    def save_predictions_json(self, complete_result: Dict, filename: str = None) -> str:
        """
        保存完整预测结果为JSON格式
        Args:
            complete_result: 完整预测结果
            filename: 文件名
        Returns:
            文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"shandong_county_predictions_complete_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # 保存JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(complete_result, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"完整预测结果JSON已保存至: {filepath}")
        return filepath
    
    def save_predictions_geojson(self, predictions: List[Dict], boundaries_gdf: gpd.GeoDataFrame, 
                               filename: str = None) -> str:
        """
        保存预测结果为GeoJSON格式（包含地理信息）
        Args:
            predictions: 预测结果列表
            boundaries_gdf: 县域边界GeoDataFrame
            filename: 文件名
        Returns:
            文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"shandong_county_predictions_geojson_{timestamp}.geojson"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # 合并预测结果和边界数据
        pred_df = self.create_prediction_dataframe(predictions)
        
        # 创建预测结果字典
        pred_dict = {row['county_name']: row for _, row in pred_df.iterrows()}
        
        # 为边界数据添加预测结果
        geojson_data = []
        for _, row in boundaries_gdf.iterrows():
            county_name = row['county_name']
            
            if county_name in pred_dict:
                # 创建特征属性
                properties = {
                    'county_name': county_name,
                    'occurrence_probability': pred_dict[county_name]['occurrence_probability'],
                    'risk_level': pred_dict[county_name]['risk_level'],
                    'risk_level_code': pred_dict[county_name]['risk_level_code'],
                    'confidence_score': pred_dict[county_name]['confidence_score'],
                    'prediction_reliable': pred_dict[county_name]['prediction_reliable'],
                    'prediction_date': pred_dict[county_name]['prediction_date']
                }
                
                # 添加原有属性
                for col in boundaries_gdf.columns:
                    if col != 'geometry' and col != 'county_name':
                        properties[col] = row[col]
                
                geojson_data.append({
                    'type': 'Feature',
                    'properties': properties,
                    'geometry': row['geometry'].__geo_interface__
                })
        
        # 创建GeoJSON
        geojson_result = {
            'type': 'FeatureCollection',
            'features': geojson_data,
            'metadata': {
                'prediction_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'model_name': '山东县域美国白蛾风险预测模型',
                'total_features': len(geojson_data)
            }
        }
        
        # 保存GeoJSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(geojson_result, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"预测结果GeoJSON已保存至: {filepath}")
        return filepath
    
    def save_risk_report(self, complete_result: Dict, filename: str = None) -> str:
        """
        保存风险分析报告
        Args:
            complete_result: 完整预测结果
            filename: 文件名
        Returns:
            文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"shandong_county_risk_report_{timestamp}.txt"
        
        filepath = os.path.join(self.output_dir, filename)
        
        summary = complete_result['summary_statistics']
        risk_dist = complete_result['risk_distribution']
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("山东县域美国白蛾风险预测报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"预测日期: {complete_result['prediction_info']['prediction_date']}\n")
            f.write(f"预测年份: {complete_result['prediction_info']['target_year']}\n")
            f.write(f"模型版本: {complete_result['prediction_info']['model_version']}\n\n")
            
            f.write("一、总体统计\n")
            f.write("-" * 30 + "\n")
            f.write(f"预测县域总数: {summary['total_counties']}\n")
            f.write(f"高风险县域数: {summary['high_risk_counties']}\n")
            f.write(f"中风险县域数: {summary['medium_risk_counties']}\n")
            f.write(f"低风险县域数: {summary['low_risk_counties']}\n")
            f.write(f"平均发生概率: {summary['average_probability']}\n")
            f.write(f"最高发生概率: {summary['max_probability']}\n")
            f.write(f"最低发生概率: {summary['min_probability']}\n\n")
            
            f.write("二、风险等级分布\n")
            f.write("-" * 30 + "\n")
            f.write(f"极高风险({len(risk_dist['extreme_risk'])}个): {', '.join(risk_dist['extreme_risk'])}\n")
            f.write(f"高风险({len(risk_dist['high_risk'])}个): {', '.join(risk_dist['high_risk'])}\n")
            f.write(f"中风险({len(risk_dist['medium_risk'])}个): {', '.join(risk_dist['medium_risk'])}\n")
            f.write(f"低风险({len(risk_dist['low_risk'])}个): {', '.join(risk_dist['low_risk'])}\n\n")
            
            f.write("三、防控建议\n")
            f.write("-" * 30 + "\n")
            f.write("1. 极高风险区域: 立即开展全面监测和应急防控\n")
            f.write("2. 高风险区域: 加强监测力度，做好防控准备\n")
            f.write("3. 中风险区域: 定期监测，保持警惕\n")
            f.write("4. 低风险区域: 常规监测，关注环境变化\n")
        
        print(f"风险分析报告已保存至: {filepath}")
        return filepath
    
    def generate_complete_output(self, predictions: List[Dict], boundaries_gdf: gpd.GeoDataFrame = None) -> Dict:
        """
        生成完整的预测结果输出
        Args:
            predictions: 预测结果列表
            boundaries_gdf: 县域边界数据
        Returns:
            完整预测结果字典
        """
        # 创建模板
        complete_result = self.create_prediction_result_template()
        
        # 更新基本信息
        complete_result['prediction_info']['target_counties'] = [p['county_name'] for p in predictions]
        complete_result['county_predictions'] = predictions
        
        # 创建统计摘要
        summary = self.create_summary_statistics(predictions)
        complete_result['summary_statistics'] = summary
        complete_result['risk_distribution'] = summary['risk_distribution']
        
        # 保存各种格式的输出
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. CSV格式
        self.save_predictions_csv(predictions, f"shandong_county_predictions_{timestamp}.csv")
        
        # 2. JSON格式
        self.save_predictions_json(complete_result, f"shandong_county_predictions_complete_{timestamp}.json")
        
        # 3. GeoJSON格式（如果有边界数据）
        if boundaries_gdf is not None:
            self.save_predictions_geojson(predictions, boundaries_gdf, 
                                        f"shandong_county_predictions_geojson_{timestamp}.geojson")
        
        # 4. 风险报告
        self.save_risk_report(complete_result, f"shandong_county_risk_report_{timestamp}.txt")
        
        return complete_result

def main():
    """演示函数"""
    # 创建输出格式管理器
    output_manager = PredictionOutputFormat()
    
    # 示例预测结果
    sample_predictions = [
        output_manager.create_county_prediction("历下区", 0.85),
        output_manager.create_county_prediction("市中区", 0.72),
        output_manager.create_county_prediction("槐荫区", 0.45),
        output_manager.create_county_prediction("天桥区", 0.23),
        output_manager.create_county_prediction("历城区", 0.91)
    ]
    
    print("示例预测结果:")
    for pred in sample_predictions:
        print(f"{pred['county_name']}: {pred['risk_level']} ({pred['occurrence_probability']:.2f})")
    
    # 创建统计摘要
    summary = output_manager.create_summary_statistics(sample_predictions)
    print(f"\n统计摘要:")
    print(f"总县域数: {summary['total_counties']}")
    print(f"高风险县域: {summary['high_risk_counties']}")
    print(f"平均概率: {summary['average_probability']}")
    
    # 保存结果
    complete_result = output_manager.generate_complete_output(sample_predictions)
    print(f"\n完整结果生成完成!")

if __name__ == "__main__":
    main()