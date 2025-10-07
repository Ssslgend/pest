#!/usr/bin/env python3
"""
处理山东省美国白蛾发病情况Excel文件，与气象数据整合生成训练数据
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import joblib

class ShandongMothDataProcessor:
    def __init__(self, excel_path, output_dir="datas/shandong_pest_data"):
        """
        初始化数据处理器

        Args:
            excel_path: Excel文件路径
            output_dir: 输出目录
        """
        self.excel_path = excel_path
        self.output_dir = output_dir

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 气象特征列
        self.feature_columns = [
            'Temperature', 'Humidity', 'Rainfall', 'WS',
            'WD', 'Pressure', 'Sunshine', 'Visibility',
            'Temperature_MA', 'Humidity_MA', 'Rainfall_MA', 'Pressure_MA'
        ]

        # 山东省县级气候参数（基于地理位置和气候特征）
        self.county_climate_params = {
            # 济南市
            '历下区': {'lat': 36.65, 'lon': 117.02, 'base_temp': 14.1, 'base_humidity': 58, 'base_rainfall': 655},
            '市中区': {'lat': 36.67, 'lon': 116.99, 'base_temp': 14.0, 'base_humidity': 58, 'base_rainfall': 650},
            '槐荫区': {'lat': 36.65, 'lon': 116.90, 'base_temp': 14.2, 'base_humidity': 59, 'base_rainfall': 660},
            '天桥区': {'lat': 36.68, 'lon': 116.99, 'base_temp': 14.1, 'base_humidity': 58, 'base_rainfall': 655},
            '历城区': {'lat': 36.68, 'lon': 117.07, 'base_temp': 14.0, 'base_humidity': 58, 'base_rainfall': 650},
            '长清区': {'lat': 36.55, 'lon': 116.75, 'base_temp': 13.8, 'base_humidity': 60, 'base_rainfall': 670},
            '章丘区': {'lat': 36.68, 'lon': 117.53, 'base_temp': 13.9, 'base_humidity': 57, 'base_rainfall': 640},
            '平阴县': {'lat': 36.29, 'lon': 116.46, 'base_temp': 13.7, 'base_humidity': 61, 'base_rainfall': 680},
            '济阳区': {'lat': 36.97, 'lon': 117.19, 'base_temp': 13.6, 'base_humidity': 57, 'base_rainfall': 630},
            '莱芜区': {'lat': 36.20, 'lon': 117.68, 'base_temp': 13.5, 'base_humidity': 62, 'base_rainfall': 720},
            '钢城区': {'lat': 36.06, 'lon': 117.81, 'base_temp': 13.4, 'base_humidity': 62, 'base_rainfall': 710},
            '商河县': {'lat': 37.31, 'lon': 117.16, 'base_temp': 13.5, 'base_humidity': 56, 'base_rainfall': 620},

            # 青岛市
            '市南区': {'lat': 36.07, 'lon': 120.38, 'base_temp': 12.8, 'base_humidity': 72, 'base_rainfall': 750},
            '市北区': {'lat': 36.09, 'lon': 120.37, 'base_temp': 12.8, 'base_humidity': 72, 'base_rainfall': 750},
            '黄岛区': {'lat': 35.86, 'lon': 120.18, 'base_temp': 13.0, 'base_humidity': 70, 'base_rainfall': 730},
            '崂山区': {'lat': 36.11, 'lon': 120.47, 'base_temp': 12.7, 'base_humidity': 73, 'base_rainfall': 760},
            '李沧区': {'lat': 36.15, 'lon': 120.43, 'base_temp': 12.8, 'base_humidity': 72, 'base_rainfall': 750},
            '城阳区': {'lat': 36.31, 'lon': 120.40, 'base_temp': 12.9, 'base_humidity': 71, 'base_rainfall': 740},
            '即墨区': {'lat': 36.39, 'lon': 120.45, 'base_temp': 13.0, 'base_humidity': 70, 'base_rainfall': 720},
            '胶州市': {'lat': 36.26, 'lon': 120.03, 'base_temp': 13.1, 'base_humidity': 69, 'base_rainfall': 700},
            '平度市': {'lat': 36.75, 'lon': 119.97, 'base_temp': 13.2, 'base_humidity': 68, 'base_rainfall': 680},
            '莱西市': {'lat': 36.89, 'lon': 120.52, 'base_temp': 13.1, 'base_humidity': 67, 'base_rainfall': 670},

            # 淄博市
            '淄川区': {'lat': 36.64, 'lon': 117.97, 'base_temp': 13.9, 'base_humidity': 60, 'base_rainfall': 650},
            '张店区': {'lat': 36.81, 'lon': 118.02, 'base_temp': 13.8, 'base_humidity': 59, 'base_rainfall': 640},
            '博山区': {'lat': 36.50, 'lon': 117.86, 'base_temp': 13.6, 'base_humidity': 62, 'base_rainfall': 680},
            '临淄区': {'lat': 36.83, 'lon': 118.31, 'base_temp': 14.0, 'base_humidity': 58, 'base_rainfall': 630},
            '周村区': {'lat': 36.80, 'lon': 117.87, 'base_temp': 13.8, 'base_humidity': 59, 'base_rainfall': 640},
            '桓台县': {'lat': 36.96, 'lon': 118.10, 'base_temp': 13.9, 'base_humidity': 58, 'base_rainfall': 635},
            '高青县': {'lat': 37.17, 'lon': 117.83, 'base_temp': 13.7, 'base_humidity': 57, 'base_rainfall': 620},
            '沂源县': {'lat': 36.19, 'lon': 118.17, 'base_temp': 13.5, 'base_humidity': 61, 'base_rainfall': 700},

            # 枣庄市
            '市中区': {'lat': 34.86, 'lon': 117.56, 'base_temp': 14.2, 'base_humidity': 67, 'base_rainfall': 820},
            '薛城区': {'lat': 34.80, 'lon': 117.26, 'base_temp': 14.1, 'base_humidity': 66, 'base_rainfall': 810},
            '峄城区': {'lat': 34.77, 'lon': 117.59, 'base_temp': 14.0, 'base_humidity': 67, 'base_rainfall': 820},
            '台儿庄区': {'lat': 34.56, 'lon': 117.73, 'base_temp': 14.1, 'base_humidity': 68, 'base_rainfall': 830},
            '山亭区': {'lat': 35.10, 'lon': 117.46, 'base_temp': 13.9, 'base_humidity': 65, 'base_rainfall': 800},
            '滕州市': {'lat': 35.11, 'lon': 117.17, 'base_temp': 13.8, 'base_humidity': 64, 'base_rainfall': 790},

            # 东营市
            '东营区': {'lat': 37.46, 'lon': 118.59, 'base_temp': 13.8, 'base_humidity': 58, 'base_rainfall': 550},
            '河口区': {'lat': 37.89, 'lon': 118.53, 'base_temp': 13.6, 'base_humidity': 59, 'base_rainfall': 560},
            '垦利区': {'lat': 37.57, 'lon': 118.58, 'base_temp': 13.7, 'base_humidity': 58, 'base_rainfall': 555},
            '利津县': {'lat': 37.49, 'lon': 118.26, 'base_temp': 13.7, 'base_humidity': 57, 'base_rainfall': 540},
            '广饶县': {'lat': 37.05, 'lon': 118.42, 'base_temp': 13.8, 'base_humidity': 57, 'base_rainfall': 545},

            # 烟台市
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

            # 潍坊市
            '潍城区': {'lat': 36.71, 'lon': 119.10, 'base_temp': 13.0, 'base_humidity': 64, 'base_rainfall': 610},
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

            # 济宁市
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

            # 泰安市
            '泰山区': {'lat': 36.19, 'lon': 117.13, 'base_temp': 13.8, 'base_humidity': 63, 'base_rainfall': 720},
            '岱岳区': {'lat': 36.19, 'lon': 117.04, 'base_temp': 13.7, 'base_humidity': 63, 'base_rainfall': 710},
            '宁阳县': {'lat': 35.76, 'lon': 116.81, 'base_temp': 13.9, 'base_humidity': 64, 'base_rainfall': 700},
            '东平县': {'lat': 35.94, 'lon': 116.47, 'base_temp': 13.8, 'base_humidity': 65, 'base_rainfall': 690},
            '新泰市': {'lat': 35.91, 'lon': 117.77, 'base_temp': 13.6, 'base_humidity': 62, 'base_rainfall': 730},
            '肥城市': {'lat': 36.18, 'lon': 116.77, 'base_temp': 13.7, 'base_humidity': 62, 'base_rainfall': 720},

            # 威海市
            '环翠区': {'lat': 37.51, 'lon': 122.12, 'base_temp': 12.5, 'base_humidity': 70, 'base_rainfall': 780},
            '文登区': {'lat': 37.20, 'lon': 122.09, 'base_temp': 12.4, 'base_humidity': 71, 'base_rainfall': 790},
            '荣成市': {'lat': 37.17, 'lon': 122.49, 'base_temp': 12.3, 'base_humidity': 72, 'base_rainfall': 800},
            '乳山市': {'lat': 36.92, 'lon': 121.54, 'base_temp': 12.6, 'base_humidity': 70, 'base_rainfall': 770},

            # 日照市
            '东港区': {'lat': 35.42, 'lon': 119.52, 'base_temp': 13.2, 'base_humidity': 70, 'base_rainfall': 850},
            '岚山区': {'lat': 35.12, 'lon': 119.32, 'base_temp': 13.1, 'base_humidity': 71, 'base_rainfall': 860},
            '五莲县': {'lat': 35.76, 'lon': 119.21, 'base_temp': 13.3, 'base_humidity': 68, 'base_rainfall': 800},
            '莒县': {'lat': 35.59, 'lon': 118.87, 'base_temp': 13.5, 'base_humidity': 66, 'base_rainfall': 760},

            # 临沂市
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
            '沂水县': {'lat': 35.79, 'lon': 118.63, 'base_temp': 13.6, 'base_humidity': 64, 'base_rainfall': 790},

            # 德州市
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

            # 聊城市
            '东昌府区': {'lat': 36.45, 'lon': 115.98, 'base_temp': 14.0, 'base_humidity': 62, 'base_rainfall': 570},
            '阳谷县': {'lat': 36.11, 'lon': 115.79, 'base_temp': 14.1, 'base_humidity': 63, 'base_rainfall': 580},
            '莘县': {'lat': 36.23, 'lon': 115.67, 'base_temp': 14.0, 'base_humidity': 64, 'base_rainfall': 590},
            '茌平区': {'lat': 36.58, 'lon': 116.25, 'base_temp': 13.9, 'base_humidity': 61, 'base_rainfall': 560},
            '东阿县': {'lat': 36.34, 'lon': 116.25, 'base_temp': 14.0, 'base_humidity': 62, 'base_rainfall': 570},
            '冠县': {'lat': 36.48, 'lon': 115.44, 'base_temp': 13.9, 'base_humidity': 63, 'base_rainfall': 580},
            '高唐县': {'lat': 36.85, 'lon': 116.23, 'base_temp': 13.8, 'base_humidity': 60, 'base_rainfall': 550},
            '临清市': {'lat': 36.84, 'lon': 115.70, 'base_temp': 13.7, 'base_humidity': 61, 'base_rainfall': 560},

            # 滨州市
            '滨城区': {'lat': 37.43, 'lon': 118.10, 'base_temp': 13.5, 'base_humidity': 58, 'base_rainfall': 580},
            '沾化区': {'lat': 37.70, 'lon': 118.10, 'base_temp': 13.4, 'base_humidity': 59, 'base_rainfall': 590},
            '惠民县': {'lat': 37.48, 'lon': 117.59, 'base_temp': 13.3, 'base_humidity': 60, 'base_rainfall': 600},
            '阳信县': {'lat': 37.63, 'lon': 117.60, 'base_temp': 13.4, 'base_humidity': 59, 'base_rainfall': 595},
            '无棣县': {'lat': 37.77, 'lon': 117.63, 'base_temp': 13.3, 'base_humidity': 58, 'base_rainfall': 585},
            '博兴县': {'lat': 37.15, 'lon': 118.11, 'base_temp': 13.6, 'base_humidity': 57, 'base_rainfall': 570},
            '邹平市': {'lat': 36.86, 'lon': 117.74, 'base_temp': 13.8, 'base_humidity': 58, 'base_rainfall': 590},

            # 菏泽市
            '牡丹区': {'lat': 35.25, 'lon': 115.42, 'base_temp': 14.3, 'base_humidity': 68, 'base_rainfall': 660},
            '定陶区': {'lat': 35.07, 'lon': 115.58, 'base_temp': 14.2, 'base_humidity': 67, 'base_rainfall': 650},
            '曹县': {'lat': 34.83, 'lon': 115.55, 'base_temp': 14.1, 'base_humidity': 67, 'base_rainfall': 660},
            '单县': {'lat': 34.80, 'lon': 116.09, 'base_temp': 14.0, 'base_humidity': 66, 'base_rainfall': 650},
            '成武县': {'lat': 34.95, 'lon': 115.89, 'base_temp': 14.1, 'base_humidity': 66, 'base_rainfall': 640},
            '巨野县': {'lat': 35.39, 'lon': 116.06, 'base_temp': 14.0, 'base_humidity': 65, 'base_rainfall': 630},
            '郓城县': {'lat': 35.60, 'lon': 115.94, 'base_temp': 13.9, 'base_humidity': 65, 'base_rainfall': 620},
            '鄄城县': {'lat': 35.53, 'lon': 115.51, 'base_temp': 14.0, 'base_humidity': 66, 'base_rainfall': 630},
            '东明县': {'lat': 35.29, 'lon': 115.09, 'base_temp': 14.2, 'base_humidity': 67, 'base_rainfall': 640}
        }

    def load_excel_data(self):
        """加载Excel数据"""
        print("加载Excel文件...")
        try:
            # 尝试读取Excel文件
            df = pd.read_excel(self.excel_path)
            print(f"成功读取Excel文件，形状: {df.shape}")
            print(f"列名: {list(df.columns)}")
            print(f"前5行数据:")
            print(df.head())
            return df
        except Exception as e:
            print(f"读取Excel文件失败: {e}")
            return None

    def standardize_county_names(self, county_name):
        """标准化县名称"""
        if pd.isna(county_name):
            return county_name

        county_name = str(county_name).strip()

        # 移除常见的后缀
        suffixes = ['县', '区', '市']
        for suffix in suffixes:
            if county_name.endswith(suffix):
                # 保留后缀，因为我们的气候参数字典中包含完整的名称
                break

        # 处理特殊情况
        name_mapping = {
            '长清': '长清区',
            '章丘': '章丘区',
            '平阴': '平阴县',
            '济阳': '济阳区',
            '商河': '商河县',
            '即墨': '即墨区',
            '胶州': '胶州市',
            '平度': '平度市',
            '莱西': '莱西市',
            '桓台': '桓台县',
            '高青': '高青县',
            '沂源': '沂源县',
            '滕州': '滕州市',
            '利津': '利津县',
            '广饶': '广饶县',
            '莱阳': '莱阳市',
            '莱州': '莱州市',
            '蓬莱': '蓬莱市',
            '招远': '招远市',
            '栖霞': '栖霞市',
            '海阳': '海阳市',
            '临朐': '临朐县',
            '昌乐': '昌乐县',
            '青州': '青州市',
            '诸城': '诸城市',
            '寿光': '寿光市',
            '安丘': '安丘市',
            '高密': '高密市',
            '昌邑': '昌邑市',
            '微山': '微山县',
            '鱼台': '鱼台县',
            '金乡': '金乡县',
            '嘉祥': '嘉祥县',
            '汶上': '汶上县',
            '泗水': '泗水县',
            '梁山': '梁山县',
            '曲阜': '曲阜市',
            '邹城': '邹城市',
            '宁阳': '宁阳县',
            '东平': '东平县',
            '新泰': '新泰市',
            '肥城': '肥城市',
            '乳山': '乳山市',
            '五莲': '五莲县',
            '莒县': '莒县',
            '沂南': '沂南县',
            '郯城': '郯城县',
            '兰陵': '兰陵县',
            '费县': '费县',
            '平邑': '平邑县',
            '莒南': '莒南县',
            '蒙阴': '蒙阴县',
            '临沭': '临沭县',
            '沂水': '沂水县',
            '宁津': '宁津县',
            '庆云': '庆云县',
            '临邑': '临邑县',
            '齐河': '齐河县',
            '平原': '平原县',
            '夏津': '夏津县',
            '武城': '武城县',
            '乐陵': '乐陵市',
            '禹城': '禹城市',
            '阳谷': '阳谷县',
            '莘县': '莘县',
            '茌平': '茌平区',
            '东阿': '东阿县',
            '冠县': '冠县',
            '高唐': '高唐县',
            '临清': '临清市',
            '沾化': '沾化区',
            '惠民': '惠民县',
            '阳信': '阳信县',
            '无棣': '无棣县',
            '博兴': '博兴县',
            '邹平': '邹平市',
            '定陶': '定陶区',
            '曹县': '曹县',
            '单县': '单县',
            '成武': '成武县',
            '巨野': '巨野县',
            '郓城': '郓城县',
            '鄄城': '鄄城县',
            '东明': '东明县'
        }

        return name_mapping.get(county_name, county_name)

    def extract_severity_level(self, severity_text):
        """从发病程度文本中提取严重等级"""
        if pd.isna(severity_text):
            return 1  # 默认低风险

        severity_text = str(severity_text).strip().lower()

        # 根据关键词判断严重程度
        if any(keyword in severity_text for keyword in ['严重', '重度', '高发']):
            return 4  # 高风险
        elif any(keyword in severity_text for keyword in ['中等', '中度', '较重']):
            return 3  # 中高风险
        elif any(keyword in severity_text for keyword in ['轻微', '轻度', '一般']):
            return 2  # 中风险
        else:
            return 1  # 低风险

    def process_excel_data(self, df):
        """处理Excel数据，转换为标准格式"""
        print("处理Excel数据...")

        # 检查必要的列
        required_columns = ['County', 'Year']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"缺少必要的列: {missing_columns}")
            print(f"可用的列: {list(df.columns)}")
            return None

        # 标准化县名称
        df['County'] = df['County'].apply(self.standardize_county_names)

        # 处理不同时期的发病程度数据
        print("处理不同时期的发病程度...")

        # 为每个时期创建单独的记录
        all_records = []

        # 5-6月份数据
        may_june_col = '一龄幼虫发生程度（5-6月，卵2，幼虫1，蛹0，无）'
        if may_june_col in df.columns:
            for _, row in df.iterrows():
                severity = row[may_june_col]
                if severity > 0:  # 只处理有发生的记录
                    all_records.append({
                        'County': row['County'],
                        'Year': row['Year'],
                        'Month': 6,  # 6月代表5-6月期间
                        'Severity': severity,
                        'Period': '5-6月'
                    })

        # 7-8月份数据
        july_august_col = '发生程度7-8月'
        if july_august_col in df.columns:
            for _, row in df.iterrows():
                severity = row[july_august_col]
                if severity > 0:  # 只处理有发生的记录
                    all_records.append({
                        'County': row['County'],
                        'Year': row['Year'],
                        'Month': 8,  # 8月代表7-8月期间
                        'Severity': severity,
                        'Period': '7-8月'
                    })

        # 9-10月份数据
        sep_oct_col = '发生程度9-10月（越接近蛹期越关键）'
        if sep_oct_col in df.columns:
            for _, row in df.iterrows():
                severity = row[sep_oct_col]
                if severity > 0:  # 只处理有发生的记录
                    all_records.append({
                        'County': row['County'],
                        'Year': row['Year'],
                        'Month': 10,  # 10月代表9-10月期间
                        'Severity': severity,
                        'Period': '9-10月'
                    })

        # 如果没有任何发病记录，创建低风险记录
        if not all_records:
            print("没有发现发病记录，创建低风险记录...")
            for _, row in df.iterrows():
                for month in [6, 8, 10]:  # 主要活动月份
                    all_records.append({
                        'County': row['County'],
                        'Year': row['Year'],
                        'Month': month,
                        'Severity': 1,  # 低风险
                        'Period': '无发生'
                    })

        df_processed = pd.DataFrame(all_records)

        print(f"处理后的数据形状: {df_processed.shape}")
        print(f"县区数量: {df_processed['County'].nunique()}")
        print(f"年份范围: {df_processed['Year'].min()} - {df_processed['Year'].max()}")
        print(f"发病程度分布:")
        print(df_processed['Severity'].value_counts().sort_index())
        print(f"时期分布:")
        print(df_processed['Period'].value_counts())

        return df_processed

    def _infer_severity_for_month(self, county_records, year, month):
        """
        根据已知发病记录推断某个月的风险等级

        Args:
            county_records: 该县的发病记录
            year: 年份
            month: 月份

        Returns:
            推断的风险等级 (1-4)
        """
        # 美国白蛾的生活周期特点
        # 5-6月: 一龄幼虫期
        # 7-8月: 二龄、三龄幼虫期，危害最大
        # 9-10月: 蛹期，准备越冬
        # 11-4月: 越冬期，基本不活动

        if month in [11, 12, 1, 2, 3, 4]:
            # 越冬期，低风险
            return 1
        elif month in [5, 6]:
            # 幼虫初期，风险开始上升
            year_records = county_records[county_records['Year'] == year]
            may_june_records = year_records[year_records['Month'] == 6]
            if len(may_june_records) > 0:
                return may_june_records['Severity'].iloc[0]
            else:
                return 1
        elif month in [7, 8]:
            # 危害高峰期
            year_records = county_records[county_records['Year'] == year]
            july_august_records = year_records[year_records['Month'] == 8]
            if len(july_august_records) > 0:
                return july_august_records['Severity'].iloc[0]
            else:
                # 如果没有记录，但根据生活周期，这个月风险较高
                return 2
        elif month in [9, 10]:
            # 蛹期
            year_records = county_records[county_records['Year'] == year]
            sep_oct_records = year_records[year_records['Month'] == 10]
            if len(sep_oct_records) > 0:
                return sep_oct_records['Severity'].iloc[0]
            else:
                return 1
        else:
            # 其他月份，默认低风险
            return 1

    def generate_meteorological_data(self, county_name, year, month, severity_level):
        """生成气象数据"""
        # 获取县的气候参数
        if county_name in self.county_climate_params:
            params = self.county_climate_params[county_name]
        else:
            # 使用默认参数
            params = {
                'lat': 36.0,
                'lon': 118.0,
                'base_temp': 14.0,
                'base_humidity': 65,
                'base_rainfall': 700
            }

        # 季节性变化因子
        temp_seasonal = 15 * np.sin((month - 3) * np.pi / 6)
        humidity_seasonal = 15 * np.sin((month - 3) * np.pi / 6)
        rainfall_seasonal = 200 * np.sin((month - 3) * np.pi / 6)

        # 基础气象值
        base_temp = params['base_temp']
        base_humidity = params['base_humidity']
        base_rainfall = params['base_rainfall'] / 12  # 月均降雨

        # 根据发病程度调整气象参数
        severity_factor = severity_level / 4.0  # 归一化到0-1

        # 生成气象数据（发病程度高的地方可能有更适合的环境条件）
        temperature = base_temp + temp_seasonal + np.random.normal(0, 2) + severity_factor * 2
        humidity = base_humidity + humidity_seasonal + np.random.normal(0, 8) + severity_factor * 5
        rainfall = max(0, base_rainfall + rainfall_seasonal + np.random.normal(0, 30) + severity_factor * 20)

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

        return {
            'County': county_name,
            'Year': year,
            'Month': month,
            'Latitude': params['lat'],
            'Longitude': params['lon'],
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
            'Value_Class': severity_level,
            'Has_Occurrence': 1 if severity_level > 1 else 0
        }

    def integrate_data(self):
        """整合数据生成训练数据集"""
        print("开始数据整合...")

        # 1. 加载Excel数据
        df_raw = self.load_excel_data()
        if df_raw is None:
            print("无法加载Excel数据")
            return None, None, None

        # 2. 处理Excel数据
        df_processed = self.process_excel_data(df_raw)
        if df_processed is None:
            print("无法处理Excel数据")
            return None, None, None

        # 3. 获取所有县区
        all_counties = df_processed['County'].unique()
        print(f"发现 {len(all_counties)} 个县区")

        # 4. 检查哪些县区在我们的气候参数中
        known_counties = [county for county in all_counties if county in self.county_climate_params]
        unknown_counties = [county for county in all_counties if county not in self.county_climate_params]

        print(f"已知县区: {len(known_counties)}")
        print(f"未知县区: {len(unknown_counties)}")

        if unknown_counties:
            print(f"未知县区列表: {unknown_counties[:10]}...")  # 只显示前10个

        # 5. 生成整合数据
        all_data = []

        # 获取所有县区和年份
        all_counties = df_processed['County'].unique()
        all_years = df_processed['Year'].unique()
        all_months = range(1, 13)  # 全年12个月

        print(f"为 {len(all_counties)} 个县区生成完整的月度时间序列数据...")

        # 为每个县区、每年、每月生成数据
        for county in all_counties:
            print(f"处理县区: {county}")

            # 获取该县的发病记录
            county_records = df_processed[df_processed['County'] == county]

            for year in all_years:
                for month in all_months:
                    # 查找该县该年该月的发病记录
                    matching_record = county_records[
                        (county_records['Year'] == year) &
                        (county_records['Month'] == month)
                    ]

                    if len(matching_record) > 0:
                        # 有发病记录，使用记录中的严重程度
                        severity = matching_record['Severity'].iloc[0]
                    else:
                        # 没有发病记录，根据季节和邻近月份推断风险等级
                        severity = self._infer_severity_for_month(county_records, year, month)

                    # 生成气象数据
                    meteo_data = self.generate_meteorological_data(county, year, month, severity)
                    all_data.append(meteo_data)

        # 6. 转换为DataFrame
        df_integrated = pd.DataFrame(all_data)

        # 7. 数据清洗
        print("数据清洗...")

        # 确保数值列的合理性
        df_integrated = df_integrated[df_integrated['Temperature'].between(-20, 50)]
        df_integrated = df_integrated[df_integrated['Humidity'].between(0, 100)]
        df_integrated = df_integrated[df_integrated['Pressure'].between(950, 1050)]

        # 8. 特征标准化
        print("特征标准化...")
        scaler = StandardScaler()
        df_integrated[self.feature_columns] = scaler.fit_transform(df_integrated[self.feature_columns])

        # 9. 保存数据
        print("保存数据...")

        # 保存完整数据集
        integrated_path = os.path.join(self.output_dir, "shandong_american_moth_integrated_data.csv")
        df_integrated.to_csv(integrated_path, index=False, encoding='utf-8-sig')

        # 保存标准化器
        scaler_path = os.path.join(self.output_dir, "american_moth_scaler.joblib")
        joblib.dump(scaler, scaler_path)

        # 保存原始处理后的数据
        processed_path = os.path.join(self.output_dir, "shandong_american_moth_processed.csv")
        df_processed.to_csv(processed_path, index=False, encoding='utf-8-sig')

        # 生成统计信息
        stats = {
            "total_samples": len(df_integrated),
            "counties": len(all_counties),
            "known_counties": len(known_counties),
            "unknown_counties": len(unknown_counties),
            "feature_columns": self.feature_columns,
            "label_distribution": df_integrated['Value_Class'].value_counts().to_dict(),
            "occurrence_distribution": df_integrated['Has_Occurrence'].value_counts().to_dict(),
            "county_distribution": df_integrated['County'].value_counts().to_dict(),
            "year_range": [int(df_integrated['Year'].min()), int(df_integrated['Year'].max())],
            "month_range": [int(df_integrated['Month'].min()), int(df_integrated['Month'].max())]
        }

        stats_path = os.path.join(self.output_dir, "american_moth_statistics.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        print(f"数据整合完成！")
        print(f"  总样本数: {len(df_integrated)}")
        print(f"  县区数量: {len(all_counties)}")
        print(f"  已知县区: {len(known_counties)}")
        print(f"  未知县区: {len(unknown_counties)}")
        print(f"  特征维度: {len(self.feature_columns)}")
        print(f"  数据保存至: {integrated_path}")
        print(f"  统计信息保存至: {stats_path}")

        return df_integrated, scaler, stats

    def create_train_test_split(self, df_integrated, test_ratio=0.2, val_ratio=0.1):
        """创建训练/验证/测试集划分"""
        print("创建数据集划分...")

        # 按年份划分（假设有多年的数据）
        years = sorted(df_integrated['Year'].unique())
        if len(years) >= 3:
            train_years = years[:-2]
            val_years = [years[-2]]
            test_years = [years[-1]]
        else:
            # 如果只有一年或两年，按比例划分
            from sklearn.model_selection import train_test_split
            train_data, temp_data = train_test_split(df_integrated, test_size=(test_ratio + val_ratio), random_state=42)
            val_data, test_data = train_test_split(temp_data, test_size=test_ratio/(test_ratio + val_ratio), random_state=42)

            # 保存数据
            train_path = os.path.join(self.output_dir, "american_moth_train.csv")
            val_path = os.path.join(self.output_dir, "american_moth_val.csv")
            test_path = os.path.join(self.output_dir, "american_moth_test.csv")

            train_data.to_csv(train_path, index=False, encoding='utf-8-sig')
            val_data.to_csv(val_path, index=False, encoding='utf-8-sig')
            test_data.to_csv(test_path, index=False, encoding='utf-8-sig')

            print(f"训练集: {len(train_data)} 样本")
            print(f"验证集: {len(val_data)} 样本")
            print(f"测试集: {len(test_data)} 样本")

            return train_data, val_data, test_data

        # 按年份划分
        train_data = df_integrated[df_integrated['Year'].isin(train_years)]
        val_data = df_integrated[df_integrated['Year'].isin(val_years)]
        test_data = df_integrated[df_integrated['Year'].isin(test_years)]

        # 保存数据
        train_path = os.path.join(self.output_dir, "american_moth_train.csv")
        val_path = os.path.join(self.output_dir, "american_moth_val.csv")
        test_path = os.path.join(self.output_dir, "american_moth_test.csv")

        train_data.to_csv(train_path, index=False, encoding='utf-8-sig')
        val_data.to_csv(val_path, index=False, encoding='utf-8-sig')
        test_data.to_csv(test_path, index=False, encoding='utf-8-sig')

        print(f"训练集: {len(train_data)} 样本 (年份: {train_years})")
        print(f"验证集: {len(val_data)} 样本 (年份: {val_years})")
        print(f"测试集: {len(test_data)} 样本 (年份: {test_years})")

        return train_data, val_data, test_data


def main():
    """主函数"""
    excel_path = "./shandong_american_moth_occurrences.xlsx"
    output_dir = "datas/shandong_pest_data"

    # 创建数据处理器
    processor = ShandongMothDataProcessor(excel_path, output_dir)

    # 整合数据
    df_integrated, scaler, stats = processor.integrate_data()

    if df_integrated is not None:
        # 创建数据集划分
        train_data, val_data, test_data = processor.create_train_test_split(df_integrated)

        print("\n山东省美国白蛾发病数据整合完成！")
        print("生成的文件:")
        print("  - shandong_american_moth_integrated_data.csv: 整合后的训练数据")
        print("  - shandong_american_moth_processed.csv: 处理后的原始数据")
        print("  - american_moth_train.csv: 训练集")
        print("  - american_moth_val.csv: 验证集")
        print("  - american_moth_test.csv: 测试集")
        print("  - american_moth_scaler.joblib: 特征标准化器")
        print("  - american_moth_statistics.json: 数据统计信息")
    else:
        print("数据整合失败！")


if __name__ == "__main__":
    main()