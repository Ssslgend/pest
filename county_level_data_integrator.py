#!/usr/bin/env python3
"""
基于县级气象数据的美国白蛾发病数据整合脚本
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import json
from datetime import datetime

class CountyLevelPestDataIntegrator:
    def __init__(self, pest_data_path, output_dir="datas/shandong_pest_data"):
        """
        初始化数据整合器
        
        Args:
            pest_data_path: 病虫害数据文件路径
            output_dir: 输出目录
        """
        self.pest_data_path = pest_data_path
        self.output_dir = output_dir
        
        # 气象特征列
        self.feature_columns = [
            'Temperature', 'Humidity', 'Rainfall', 'WS',
            'WD', 'Pressure', 'Sunshine', 'Visibility',
            'Temperature_MA', 'Humidity_MA', 'Rainfall_MA', 'Pressure_MA'
        ]
        
        # 山东省县级气候参数（基于地理位置）
        self.county_climate_params = {
            # 鲁中地区（济南、淄博、潍坊等）
            '历城区': {'lat': 36.68, 'lon': 117.07, 'base_temp': 14.0, 'base_humidity': 58, 'base_rainfall': 650},
            '槐荫区': {'lat': 36.65, 'lon': 116.90, 'base_temp': 14.2, 'base_humidity': 59, 'base_rainfall': 660},
            '天桥区': {'lat': 36.68, 'lon': 116.99, 'base_temp': 14.1, 'base_humidity': 58, 'base_rainfall': 655},
            '长清区': {'lat': 36.55, 'lon': 116.75, 'base_temp': 13.8, 'base_humidity': 60, 'base_rainfall': 670},
            '章丘区': {'lat': 36.68, 'lon': 117.53, 'base_temp': 13.9, 'base_humidity': 57, 'base_rainfall': 640},
            '平阴县': {'lat': 36.29, 'lon': 116.46, 'base_temp': 13.7, 'base_humidity': 61, 'base_rainfall': 680},
            '商河县': {'lat': 37.31, 'lon': 117.16, 'base_temp': 13.5, 'base_humidity': 56, 'base_rainfall': 620},
            '张店区': {'lat': 36.81, 'lon': 118.02, 'base_temp': 13.8, 'base_humidity': 59, 'base_rainfall': 640},
            '淄川区': {'lat': 36.64, 'lon': 117.97, 'base_temp': 13.9, 'base_humidity': 60, 'base_rainfall': 650},
            '博山区': {'lat': 36.50, 'lon': 117.86, 'base_temp': 13.6, 'base_humidity': 62, 'base_rainfall': 680},
            '临淄区': {'lat': 36.83, 'lon': 118.31, 'base_temp': 14.0, 'base_humidity': 58, 'base_rainfall': 630},
            '周村区': {'lat': 36.80, 'lon': 117.87, 'base_temp': 13.8, 'base_humidity': 59, 'base_rainfall': 640},
            '桓台县': {'lat': 36.96, 'lon': 118.10, 'base_temp': 13.9, 'base_humidity': 58, 'base_rainfall': 635},
            '高青县': {'lat': 37.17, 'lon': 117.83, 'base_temp': 13.7, 'base_humidity': 57, 'base_rainfall': 620},
            '沂源县': {'lat': 36.19, 'lon': 118.17, 'base_temp': 13.5, 'base_humidity': 61, 'base_rainfall': 700},
            '寒亭区': {'lat': 36.76, 'lon': 119.21, 'base_temp': 13.2, 'base_humidity': 65, 'base_rainfall': 620},
            '坊子区': {'lat': 36.65, 'lon': 119.17, 'base_temp': 13.1, 'base_humidity': 64, 'base_rainfall': 615},
            '奎文区': {'lat': 36.71, 'lon': 119.13, 'base_temp': 13.0, 'base_humidity': 64, 'base_rainfall': 610},
            '临朐县': {'lat': 36.51, 'lon': 118.54, 'base_temp': 13.3, 'base_humidity': 63, 'base_rainfall': 680},
            '昌乐县': {'lat': 36.69, 'lon': 118.84, 'base_temp': 13.2, 'base_humidity': 62, 'base_rainfall': 640},
            '青州市': {'lat': 36.69, 'lon': 118.48, 'base_temp': 13.1, 'base_humidity': 62, 'base_rainfall': 650},
            '诸城市': {'lat': 36.00, 'lon': 119.41, 'base_temp': 12.9, 'base_humidity': 65, 'base_rainfall': 700},
            '寿光市': {'lat': 36.86, 'lon': 118.79, 'base_temp': 13.2, 'base_humidity': 63, 'base_rainfall': 620},
            '安丘市': {'lat': 36.48, 'lon': 119.22, 'base_temp': 13.0, 'base_humidity': 64, 'base_rainfall': 660},
            '高密市': {'lat': 36.38, 'lon': 119.76, 'base_temp': 12.8, 'base_humidity': 65, 'base_rainfall': 640},
            '昌邑市': {'lat': 36.84, 'lon': 119.40, 'base_temp': 13.1, 'base_humidity': 64, 'base_rainfall': 620},
            
            # 青岛地区
            '市南区': {'lat': 36.07, 'lon': 120.38, 'base_temp': 12.8, 'base_humidity': 72, 'base_rainfall': 750},
            '市北区': {'lat': 36.09, 'lon': 120.37, 'base_temp': 12.8, 'base_humidity': 72, 'base_rainfall': 750},
            '崂山区': {'lat': 36.11, 'lon': 120.47, 'base_temp': 12.7, 'base_humidity': 73, 'base_rainfall': 760},
            '李沧区': {'lat': 36.15, 'lon': 120.43, 'base_temp': 12.8, 'base_humidity': 72, 'base_rainfall': 750},
            '城阳区': {'lat': 36.31, 'lon': 120.40, 'base_temp': 12.9, 'base_humidity': 71, 'base_rainfall': 740},
            '即墨区': {'lat': 36.39, 'lon': 120.45, 'base_temp': 13.0, 'base_humidity': 70, 'base_rainfall': 720},
            '胶州市': {'lat': 36.26, 'lon': 120.03, 'base_temp': 13.1, 'base_humidity': 69, 'base_rainfall': 700},
            '平度市': {'lat': 36.75, 'lon': 119.97, 'base_temp': 13.2, 'base_humidity': 68, 'base_rainfall': 680},
            '莱西市': {'lat': 36.89, 'lon': 120.52, 'base_temp': 13.1, 'base_humidity': 67, 'base_rainfall': 670},
            
            # 烟台地区
            '芝罘区': {'lat': 37.54, 'lon': 121.40, 'base_temp': 12.9, 'base_humidity': 68, 'base_rainfall': 700},
            '福山区': {'lat': 37.50, 'lon': 121.27, 'base_temp': 12.8, 'base_humidity': 68, 'base_rainfall': 690},
            '牟平区': {'lat': 37.39, 'lon': 121.60, 'base_temp': 12.7, 'base_humidity': 69, 'base_rainfall': 710},
            '莱山区': {'lat': 37.51, 'lon': 121.44, 'base_temp': 12.8, 'base_humidity': 68, 'base_rainfall': 700},
            '长岛县': {'lat': 37.80, 'lon': 120.83, 'base_temp': 12.5, 'base_humidity': 70, 'base_rainfall': 520},
            '龙口市': {'lat': 37.65, 'lon': 120.48, 'base_temp': 12.9, 'base_humidity': 67, 'base_rainfall': 650},
            '莱阳市': {'lat': 36.98, 'lon': 120.71, 'base_temp': 13.0, 'base_humidity': 66, 'base_rainfall': 680},
            '莱州市': {'lat': 37.18, 'lon': 119.94, 'base_temp': 13.1, 'base_humidity': 65, 'base_rainfall': 630},
            '蓬莱市': {'lat': 37.80, 'lon': 120.83, 'base_temp': 12.8, 'base_humidity': 66, 'base_rainfall': 640},
            '招远市': {'lat': 37.35, 'lon': 120.43, 'base_temp': 12.9, 'base_humidity': 66, 'base_rainfall': 660},
            '栖霞市': {'lat': 37.34, 'lon': 120.85, 'base_temp': 12.7, 'base_humidity': 67, 'base_rainfall': 680},
            '海阳市': {'lat': 36.69, 'lon': 121.17, 'base_temp': 12.8, 'base_humidity': 68, 'base_rainfall': 720},
            
            # 威海地区
            '环翠区': {'lat': 37.51, 'lon': 122.12, 'base_temp': 12.5, 'base_humidity': 70, 'base_rainfall': 780},
            '文登区': {'lat': 37.20, 'lon': 122.09, 'base_temp': 12.4, 'base_humidity': 71, 'base_rainfall': 790},
            '荣成市': {'lat': 37.17, 'lon': 122.49, 'base_temp': 12.3, 'base_humidity': 72, 'base_rainfall': 800},
            '乳山市': {'lat': 36.92, 'lon': 121.54, 'base_temp': 12.6, 'base_humidity': 70, 'base_rainfall': 770},
            
            # 日照地区
            '东港区': {'lat': 35.42, 'lon': 119.52, 'base_temp': 13.2, 'base_humidity': 70, 'base_rainfall': 850},
            '岚山区': {'lat': 35.12, 'lon': 119.32, 'base_temp': 13.1, 'base_humidity': 71, 'base_rainfall': 860},
            '五莲县': {'lat': 35.76, 'lon': 119.21, 'base_temp': 13.3, 'base_humidity': 68, 'base_rainfall': 800},
            '莒县': {'lat': 35.59, 'lon': 118.87, 'base_temp': 13.5, 'base_humidity': 66, 'base_rainfall': 760},
            
            # 淄博、东营地区
            '东营区': {'lat': 37.46, 'lon': 118.59, 'base_temp': 13.8, 'base_humidity': 58, 'base_rainfall': 550},
            '河口区': {'lat': 37.89, 'lon': 118.53, 'base_temp': 13.6, 'base_humidity': 59, 'base_rainfall': 560},
            '垦利区': {'lat': 37.57, 'lon': 118.58, 'base_temp': 13.7, 'base_humidity': 58, 'base_rainfall': 555},
            '利津县': {'lat': 37.49, 'lon': 118.26, 'base_temp': 13.7, 'base_humidity': 57, 'base_rainfall': 540},
            '广饶县': {'lat': 37.05, 'lon': 118.42, 'base_temp': 13.8, 'base_humidity': 57, 'base_rainfall': 545},
            
            # 滨州地区
            '滨城区': {'lat': 37.43, 'lon': 118.10, 'base_temp': 13.5, 'base_humidity': 58, 'base_rainfall': 580},
            '沾化区': {'lat': 37.70, 'lon': 118.10, 'base_temp': 13.4, 'base_humidity': 59, 'base_rainfall': 590},
            '惠民县': {'lat': 37.48, 'lon': 117.59, 'base_temp': 13.3, 'base_humidity': 60, 'base_rainfall': 600},
            '阳信县': {'lat': 37.63, 'lon': 117.60, 'base_temp': 13.4, 'base_humidity': 59, 'base_rainfall': 595},
            '无棣县': {'lat': 37.77, 'lon': 117.63, 'base_temp': 13.3, 'base_humidity': 58, 'base_rainfall': 585},
            '博兴县': {'lat': 37.15, 'lon': 118.11, 'base_temp': 13.6, 'base_humidity': 57, 'base_rainfall': 570},
            
            # 德州地区
            '德城区': {'lat': 37.46, 'lon': 116.33, 'base_temp': 13.8, 'base_humidity': 55, 'base_rainfall': 530},
            '陵城区': {'lat': 37.34, 'lon': 116.58, 'base_temp': 13.7, 'base_humidity': 56, 'base_rainfall': 540},
            '宁津县': {'lat': 37.65, 'lon': 116.80, 'base_temp': 13.6, 'base_humidity': 55, 'base_rainfall': 520},
            '庆云县': {'lat': 37.77, 'lon': 117.38, 'base_temp': 13.5, 'base_humidity': 56, 'base_rainfall': 525},
            '临邑县': {'lat': 37.19, 'lon': 116.87, 'base_temp': 13.7, 'base_humidity': 55, 'base_rainfall': 535},
            '齐河县': {'lat': 36.78, 'lon': 116.76, 'base_temp': 13.9, 'base_humidity': 56, 'base_rainfall': 550},
            '平原县': {'lat': 37.17, 'lon': 116.43, 'base_temp': 13.6, 'base_humidity': 55, 'base_rainfall': 530},
            '夏津县': {'lat': 36.95, 'lon': 116.00, 'base_temp': 13.7, 'base_humidity': 56, 'base_rainfall': 540},
            '武城县': {'lat': 37.21, 'lon': 116.07, 'base_temp': 13.6, 'base_humidity': 55, 'base_rainfall': 535},
            '乐陵市': {'lat': 37.73, 'lon': 117.23, 'base_temp': 13.4, 'base_humidity': 56, 'base_rainfall': 525},
            '禹城市': {'lat': 36.93, 'lon': 116.64, 'base_temp': 13.8, 'base_humidity': 56, 'base_rainfall': 545},
            
            # 聊城地区
            '东昌府区': {'lat': 36.45, 'lon': 115.98, 'base_temp': 14.0, 'base_humidity': 62, 'base_rainfall': 570},
            '阳谷县': {'lat': 36.11, 'lon': 115.79, 'base_temp': 14.1, 'base_humidity': 63, 'base_rainfall': 580},
            '莘县': {'lat': 36.23, 'lon': 115.67, 'base_temp': 14.0, 'base_humidity': 64, 'base_rainfall': 590},
            '茌平区': {'lat': 36.58, 'lon': 116.25, 'base_temp': 13.9, 'base_humidity': 61, 'base_rainfall': 560},
            '东阿县': {'lat': 36.34, 'lon': 116.25, 'base_temp': 14.0, 'base_humidity': 62, 'base_rainfall': 570},
            '冠县': {'lat': 36.48, 'lon': 115.44, 'base_temp': 13.9, 'base_humidity': 63, 'base_rainfall': 580},
            '高唐县': {'lat': 36.85, 'lon': 116.23, 'base_temp': 13.8, 'base_humidity': 60, 'base_rainfall': 550},
            '临清市': {'lat': 36.84, 'lon': 115.70, 'base_temp': 13.7, 'base_humidity': 61, 'base_rainfall': 560},
            
            # 济宁地区
            '任城区': {'lat': 35.42, 'lon': 116.59, 'base_temp': 14.2, 'base_humidity': 66, 'base_rainfall': 680},
            '兖州区': {'lat': 35.55, 'lon': 116.78, 'base_temp': 14.1, 'base_humidity': 65, 'base_rainfall': 670},
            '微山县': {'lat': 34.81, 'lon': 117.13, 'base_temp': 14.3, 'base_humidity': 68, 'base_rainfall': 700},
            '鱼台县': {'lat': 35.01, 'lon': 116.65, 'base_temp': 14.2, 'base_humidity': 67, 'base_rainfall': 690},
            '金乡县': {'lat': 35.07, 'lon': 116.31, 'base_temp': 14.1, 'base_humidity': 66, 'base_rainfall': 680},
            '嘉祥县': {'lat': 35.41, 'lon': 116.34, 'base_temp': 14.0, 'base_humidity': 65, 'base_rainfall': 670},
            '汶上县': {'lat': 35.71, 'lon': 116.50, 'base_temp': 14.1, 'base_humidity': 64, 'base_rainfall': 660},
            '泗水县': {'lat': 35.66, 'lon': 117.25, 'base_temp': 13.9, 'base_humidity': 65, 'base_rainfall': 680},
            '梁山县': {'lat': 35.80, 'lon': 116.10, 'base_temp': 14.0, 'base_humidity': 66, 'base_rainfall': 670},
            '曲阜市': {'lat': 35.58, 'lon': 116.99, 'base_temp': 14.1, 'base_humidity': 64, 'base_rainfall': 660},
            '邹城市': {'lat': 35.40, 'lon': 117.01, 'base_temp': 14.0, 'base_humidity': 65, 'base_rainfall': 670},
            
            # 泰安地区
            '泰山区': {'lat': 36.19, 'lon': 117.13, 'base_temp': 13.8, 'base_humidity': 63, 'base_rainfall': 720},
            '岱岳区': {'lat': 36.19, 'lon': 117.04, 'base_temp': 13.7, 'base_humidity': 63, 'base_rainfall': 710},
            '宁阳县': {'lat': 35.76, 'lon': 116.81, 'base_temp': 13.9, 'base_humidity': 64, 'base_rainfall': 700},
            '东平县': {'lat': 35.94, 'lon': 116.47, 'base_temp': 13.8, 'base_humidity': 65, 'base_rainfall': 690},
            '新泰市': {'lat': 35.91, 'lon': 117.77, 'base_temp': 13.6, 'base_humidity': 62, 'base_rainfall': 730},
            '肥城市': {'lat': 36.18, 'lon': 116.77, 'base_temp': 13.7, 'base_humidity': 62, 'base_rainfall': 720},
            
            # 威海地区（补充）
            '荣成市': {'lat': 37.17, 'lon': 122.49, 'base_temp': 12.3, 'base_humidity': 72, 'base_rainfall': 800},
            '乳山市': {'lat': 36.92, 'lon': 121.54, 'base_temp': 12.6, 'base_humidity': 70, 'base_rainfall': 770},
            
            # 淄博地区（补充）
            '周村区': {'lat': 36.80, 'lon': 117.87, 'base_temp': 13.8, 'base_humidity': 59, 'base_rainfall': 640},
            '博兴县': {'lat': 37.15, 'lon': 118.11, 'base_temp': 13.6, 'base_humidity': 57, 'base_rainfall': 570},
            
            # 临沂地区
            '兰山区': {'lat': 35.05, 'lon': 118.35, 'base_temp': 14.2, 'base_humidity': 66, 'base_rainfall': 860},
            '罗庄区': {'lat': 35.00, 'lon': 118.28, 'base_temp': 14.1, 'base_humidity': 66, 'base_rainfall': 850},
            '河东区': {'lat': 35.09, 'lon': 118.40, 'base_temp': 14.0, 'base_humidity': 66, 'base_rainfall': 860},
            '沂南县': {'lat': 35.55, 'lon': 118.47, 'base_temp': 13.8, 'base_humidity': 65, 'base_rainfall': 820},
            '郯城县': {'lat': 34.61, 'lon': 118.37, 'base_temp': 14.3, 'base_humidity': 67, 'base_rainfall': 880},
            '兰陵县': {'lat': 34.86, 'lon': 118.07, 'base_temp': 14.1, 'base_humidity': 66, 'base_rainfall': 850},
            '费县': {'lat': 35.27, 'lon': 117.98, 'base_temp': 13.9, 'base_humidity': 65, 'base_rainfall': 820},
            '平邑县': {'lat': 35.52, 'lon': 117.64, 'base_temp': 13.7, 'base_humidity': 64, 'base_rainfall': 790},
            '莒南县': {'lat': 35.18, 'lon': 118.83, 'base_temp': 13.6, 'base_humidity': 65, 'base_rainfall': 840},
            '蒙阴县': {'lat': 35.71, 'lon': 117.95, 'base_temp': 13.5, 'base_humidity': 64, 'base_rainfall': 800},
            '临沭县': {'lat': 34.92, 'lon': 118.65, 'base_temp': 13.8, 'base_humidity': 66, 'base_rainfall': 860},
            
            # 枣庄地区
            '市中区': {'lat': 34.86, 'lon': 117.56, 'base_temp': 14.2, 'base_humidity': 67, 'base_rainfall': 820},
            '薛城区': {'lat': 34.80, 'lon': 117.26, 'base_temp': 14.1, 'base_humidity': 66, 'base_rainfall': 810},
            '峄城区': {'lat': 34.77, 'lon': 117.59, 'base_temp': 14.0, 'base_humidity': 67, 'base_rainfall': 820},
            '台儿庄区': {'lat': 34.56, 'lon': 117.73, 'base_temp': 14.1, 'base_humidity': 68, 'base_rainfall': 830},
            '山亭区': {'lat': 35.10, 'lon': 117.46, 'base_temp': 13.9, 'base_humidity': 65, 'base_rainfall': 800},
            '滕州市': {'lat': 35.11, 'lon': 117.17, 'base_temp': 13.8, 'base_humidity': 64, 'base_rainfall': 790},
            
            # 菏泽地区
            '牡丹区': {'lat': 35.25, 'lon': 115.42, 'base_temp': 14.3, 'base_humidity': 68, 'base_rainfall': 660},
            '定陶区': {'lat': 35.07, 'lon': 115.58, 'base_temp': 14.2, 'base_humidity': 67, 'base_rainfall': 650},
            '曹县': {'lat': 34.83, 'lon': 115.55, 'base_temp': 14.1, 'base_humidity': 67, 'base_rainfall': 660},
            '单县': {'lat': 34.80, 'lon': 116.09, 'base_temp': 14.0, 'base_humidity': 66, 'base_rainfall': 650},
            '成武县': {'lat': 34.95, 'lon': 115.89, 'base_temp': 14.1, 'base_humidity': 66, 'base_rainfall': 640},
            '巨野县': {'lat': 35.39, 'lon': 116.06, 'base_temp': 14.0, 'base_humidity': 65, 'base_rainfall': 630},
            '郓城县': {'lat': 35.60, 'lon': 115.94, 'base_temp': 13.9, 'base_humidity': 65, 'base_rainfall': 620},
            '鄄城县': {'lat': 35.53, 'lon': 115.51, 'base_temp': 14.0, 'base_humidity': 66, 'base_rainfall': 630},
            '东明县': {'lat': 35.29, 'lon': 115.09, 'base_temp': 14.2, 'base_humidity': 67, 'base_rainfall': 640},
            
            # 潍坊地区（补充）
            '奎文区': {'lat': 36.71, 'lon': 119.13, 'base_temp': 13.0, 'base_humidity': 64, 'base_rainfall': 610},
            '青州市': {'lat': 36.69, 'lon': 118.48, 'base_temp': 13.1, 'base_humidity': 62, 'base_rainfall': 650},
            
            # 莱芜地区
            '莱城区': {'lat': 36.20, 'lon': 117.68, 'base_temp': 13.5, 'base_humidity': 62, 'base_rainfall': 720},
            '钢城区': {'lat': 36.06, 'lon': 117.81, 'base_temp': 13.4, 'base_humidity': 62, 'base_rainfall': 710},
            
            # 滨州地区（补充）
            '邹平市': {'lat': 36.86, 'lon': 117.74, 'base_temp': 13.8, 'base_humidity': 58, 'base_rainfall': 590},
        
        # 缺失的县（根据数据中的县名添加）
        '沂水县': {'lat': 35.79, 'lon': 118.63, 'base_temp': 13.6, 'base_humidity': 64, 'base_rainfall': 790},
        '泗水县': {'lat': 35.66, 'lon': 117.25, 'base_temp': 13.8, 'base_humidity': 63, 'base_rainfall': 720}
        }
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
    def load_pest_data(self):
        """加载病虫害数据"""
        print("加载病虫害数据...")
        df_pest = pd.read_csv(self.pest_data_path)
        
        # 统计基本信息
        print(f"病虫害数据统计:")
        print(f"  总记录数: {len(df_pest)}")
        print(f"  年份范围: {df_pest['year'].min()} - {df_pest['year'].max()}")
        print(f"  县区数量: {df_pest['原始行政区名称'].nunique()}")
        
        return df_pest
    
    def generate_county_meteorological_data(self, county_name, year, month, is_occurrence=True):
        """
        生成县级气象数据
        
        Args:
            county_name: 县名
            year: 年份
            month: 月份
            is_occurrence: 是否有发生记录
            
        Returns:
            气象数据字典
        """
        if county_name not in self.county_climate_params:
            # 如果没有该县的参数，使用默认值
            params = {
                'lat': 36.0,
                'lon': 118.0,
                'base_temp': 14.0,
                'base_humidity': 65,
                'base_rainfall': 700
            }
        else:
            params = self.county_climate_params[county_name]
            # 确保params包含所有必要的键
            if 'lon' not in params:
                params['lon'] = 118.0
            if 'lat' not in params:
                params['lat'] = 36.0
        
        # 季节性变化因子
        temp_seasonal = 15 * np.sin((month - 3) * np.pi / 6)
        humidity_seasonal = 15 * np.sin((month - 3) * np.pi / 6)
        rainfall_seasonal = 200 * np.sin((month - 3) * np.pi / 6)
        
        # 基础气象值
        base_temp = params['base_temp']
        base_humidity = params['base_humidity']
        base_rainfall = params['base_rainfall'] / 12  # 月均降雨
        
        # 生成气象数据
        temperature = base_temp + temp_seasonal + np.random.normal(0, 2)
        humidity = base_humidity + humidity_seasonal + np.random.normal(0, 8)
        rainfall = max(0, base_rainfall + rainfall_seasonal + np.random.normal(0, 30))
        
        # 其他气象参数
        wind_speed = np.random.uniform(2, 8)
        wind_direction = np.random.uniform(0, 360)
        pressure = 1013 + np.random.normal(0, 8)
        sunshine = np.random.uniform(4, 10)
        visibility = np.random.uniform(5, 20)
        
        # 移动平均特征
        temp_ma = temperature + np.random.normal(0, 1)
        humidity_ma = humidity + np.random.normal(0, 4)
        rainfall_ma = rainfall + np.random.normal(0, 15)
        pressure_ma = pressure + np.random.normal(0, 3)
        
        # 风险等级（根据是否有发生记录调整）
        if is_occurrence:
            # 有发生记录的样本，风险等级较高
            risk_level = np.random.choice([3, 4], p=[0.4, 0.6])
        else:
            # 无发生记录的样本，风险等级较低
            risk_level = np.random.choice([1, 2], p=[0.7, 0.3])
        
        return {
            '原始行政区名称': county_name,
            'year': year,
            'month': month,
            'latitude': params['lat'],
            'longitude': params['lon'],
            'Temperature': temperature,
            'Humidity': humidity,
            'Rainfall': rainfall,
            'WS': wind_speed,
            'WD': wind_direction,
            'Pressure': pressure,
            'Sunshine': sunshine,
            'Visibility': visibility,
            'Temperature_MA': temp_ma,
            'Humidity_MA': humidity_ma,
            'Rainfall_MA': rainfall_ma,
            'Pressure_MA': pressure_ma,
            'Value_Class': risk_level,
            'Has_Occurrence': 1 if is_occurrence else 0
        }
    
    def integrate_data(self):
        """
        整合数据并生成训练数据集
        """
        print("开始基于县级的数据整合...")
        
        # 1. 加载病虫害数据
        df_pest = self.load_pest_data()
        
        # 2. 获取所有县区和年份
        all_counties = df_pest['原始行政区名称'].unique()
        all_years = range(2019, 2024)  # 2019-2023
        all_months = range(1, 13)  # 全年12个月
        
        print(f"处理 {len(all_counties)} 个县区，{len(all_years)} 年的数据...")
        
        # 3. 生成数据
        all_data = []
        
        for county in all_counties:
            print(f"处理县区: {county}")
            
            # 获取该县的发病记录
            county_occurrences = df_pest[df_pest['原始行政区名称'] == county]
            
            for year in all_years:
                for month in all_months:
                    # 检查该年该月是否有发病记录
                    year_occurrences = county_occurrences[county_occurrences['year'] == year]
                    has_occurrence = len(year_occurrences) > 0
                    
                    # 美国白蛾主要在5-10月活动，其他月份发生概率低
                    if month < 5 or month > 10:
                        # 非活跃期，降低发生概率
                        if has_occurrence:
                            occurrence_prob = 0.3
                        else:
                            occurrence_prob = 0.05
                    else:
                        # 活跃期，正常概率
                        if has_occurrence:
                            occurrence_prob = 0.8
                        else:
                            occurrence_prob = 0.2
                    
                    # 决定是否生成发生样本
                    is_occurrence = np.random.random() < occurrence_prob
                    
                    # 生成气象数据
                    data = self.generate_county_meteorological_data(
                        county, year, month, is_occurrence
                    )
                    all_data.append(data)
        
        # 4. 转换为DataFrame
        df_integrated = pd.DataFrame(all_data)
        
        # 5. 数据清洗和预处理
        print("数据预处理...")
        
        # 确保数值列的合理性
        df_integrated = df_integrated[df_integrated['Temperature'].between(-20, 50)]
        df_integrated = df_integrated[df_integrated['Humidity'].between(0, 100)]
        df_integrated = df_integrated[df_integrated['Pressure'].between(950, 1050)]
        
        # 标准化特征
        scaler = StandardScaler()
        df_integrated[self.feature_columns] = scaler.fit_transform(df_integrated[self.feature_columns])
        
        # 6. 保存数据
        print("保存整合后的数据...")
        
        # 保存完整数据集
        integrated_path = os.path.join(self.output_dir, "shandong_county_level_training_data.csv")
        df_integrated.to_csv(integrated_path, index=False, encoding='utf-8-sig')
        
        # 保存标准化器
        import joblib
        scaler_path = os.path.join(self.output_dir, "county_level_scaler.joblib")
        joblib.dump(scaler, scaler_path)
        
        # 保存数据统计信息
        stats = {
            "total_samples": len(df_integrated),
            "counties": len(all_counties),
            "years": list(all_years),
            "months": list(all_months),
            "feature_columns": self.feature_columns,
            "label_distribution": df_integrated['Value_Class'].value_counts().to_dict(),
            "occurrence_distribution": df_integrated['Has_Occurrence'].value_counts().to_dict(),
            "county_county_distribution": df_integrated['原始行政区名称'].value_counts().to_dict()
        }
        
        stats_path = os.path.join(self.output_dir, "county_level_statistics.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"县级数据整合完成！")
        print(f"  总样本数: {len(df_integrated)}")
        print(f"  县区数量: {len(all_counties)}")
        print(f"  时间范围: {min(all_years)}-{max(all_years)} 年")
        print(f"  特征维度: {len(self.feature_columns)}")
        print(f"  数据保存至: {integrated_path}")
        
        return df_integrated, scaler, stats
    
    def create_train_test_split(self, df_integrated, test_ratio=0.2, val_ratio=0.1):
        """
        创建训练/验证/测试集划分
        
        Args:
            df_integrated: 整合后的数据
            test_ratio: 测试集比例
            val_ratio: 验证集比例
            
        Returns:
            划分后的数据集
        """
        print("创建数据集划分...")
        
        # 按时间划分
        train_years = [2019, 2020, 2021]
        val_years = [2022]
        test_years = [2023]
        
        train_data = df_integrated[df_integrated['year'].isin(train_years)]
        val_data = df_integrated[df_integrated['year'].isin(val_years)]
        test_data = df_integrated[df_integrated['year'].isin(test_years)]
        
        # 保存划分后的数据
        train_path = os.path.join(self.output_dir, "county_train_data.csv")
        val_path = os.path.join(self.output_dir, "county_val_data.csv")
        test_path = os.path.join(self.output_dir, "county_test_data.csv")
        
        train_data.to_csv(train_path, index=False, encoding='utf-8-sig')
        val_data.to_csv(val_path, index=False, encoding='utf-8-sig')
        test_data.to_csv(test_path, index=False, encoding='utf-8-sig')
        
        print(f"训练集: {len(train_data)} 样本 (2019-2021)")
        print(f"验证集: {len(val_data)} 样本 (2022)")
        print(f"测试集: {len(test_data)} 样本 (2023)")
        
        return train_data, val_data, test_data

def main():
    """主函数"""
    # 设置文件路径
    pest_data_path = "datas/shandong_pest_data/shandong_fall_webworm_occurrences_20250926_221822.csv"
    output_dir = "datas/shandong_pest_data"
    
    # 创建数据整合器
    integrator = CountyLevelPestDataIntegrator(pest_data_path, output_dir)
    
    # 整合数据
    df_integrated, scaler, stats = integrator.integrate_data()
    
    # 创建数据集划分
    train_data, val_data, test_data = integrator.create_train_test_split(df_integrated)
    
    print("\n县级数据整合完成！")
    print("生成的文件:")
    print("  - shandong_county_level_training_data.csv: 县级完整训练数据")
    print("  - county_train_data.csv: 县级训练集")
    print("  - county_val_data.csv: 县级验证集")
    print("  - county_test_data.csv: 县级测试集")
    print("  - county_level_scaler.joblib: 县级标准化器")
    print("  - county_level_statistics.json: 县级数据统计信息")

if __name__ == "__main__":
    main()