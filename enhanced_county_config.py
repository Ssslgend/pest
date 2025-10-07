#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强数据集配置文件
包含完整的山东省县级数据配置，支持健康县和遥感特征
"""

import os
import json
from typing import List, Dict, Any

class EnhancedCountyLevelConfig:
    """增强县级配置类"""

    # 数据路径配置
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'datas', 'shandong_pest_data')

    # 增强数据路径
    ENHANCED_COMPLETE_DATA_PATH = os.path.join(DATA_DIR, 'enhanced_complete_data.csv')
    ENHANCED_DATA_CONFIG_PATH = os.path.join(DATA_DIR, 'enhanced_data_config.json')

    # 原始数据路径（保持兼容性）
    COMPLETE_DATA_PATH = os.path.join(DATA_DIR, 'county_level_firstgen_complete.csv')

    # 模型保存路径
    MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'results', 'enhanced_models')
    VISUALIZATION_DIR = os.path.join(BASE_DIR, 'results', 'enhanced_visualizations')

    # 结果保存路径
    RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'enhanced_predictions')

    # 数据集配置
    TRAIN_YEARS = [2019, 2020, 2021]  # 训练年份
    VAL_YEARS = [2022]               # 验证年份
    TEST_YEARS = [2023]              # 测试年份

    # 发病程度定义
    SEVERITY_LEVELS = {
        0: {'name': '健康', 'description': '无发病迹象', 'color': '#2E8B57'},
        1: {'name': '轻度', 'description': '轻度发病', 'color': '#FFD700'},
        2: {'name': '中度', 'description': '中度发病', 'color': '#FF8C00'},
        3: {'name': '重度', 'description': '重度发病', 'color': '#DC143C'}
    }

    # 山东省完整县级行政区划
    SHANDONG_COUNTIES = [
        # 济南市 (12个区县)
        '历下区', '市中区', '槐荫区', '天桥区', '历城区', '长清区',
        '章丘区', '济阳区', '莱芜区', '钢城区', '平阴县', '商河县',

        # 青岛市 (10个区县)
        '市南区', '市北区', '黄岛区', '崂山区', '李沧区', '城阳区',
        '即墨区', '胶州市', '平度市', '莱西市',

        # 淄博市 (8个区县)
        '淄川区', '张店区', '博山区', '临淄区', '周村区', '桓台县',
        '高青县', '沂源县',

        # 枣庄市 (6个区县)
        '市中区', '薛城区', '峄城区', '台儿庄区', '山亭区', '滕州市',

        # 东营市 (5个区县)
        '东营区', '河口区', '垦利区', '利津县', '广饶县',

        # 烟台市 (12个区县)
        '芝罘区', '福山区', '牟平区', '莱山区', '长岛县', '龙口市',
        '莱阳市', '莱州市', '蓬莱市', '招远市', '栖霞市', '海阳市',

        # 潍坊市 (12个区县)
        '潍城区', '寒亭区', '坊子区', '奎文区', '临朐县', '昌乐县',
        '青州市', '诸城市', '寿光市', '安丘市', '高密市', '昌邑市',

        # 济宁市 (11个区县)
        '任城区', '兖州区', '微山县', '鱼台县', '金乡县', '嘉祥县',
        '汶上县', '泗水县', '梁山县', '曲阜市', '邹城市',

        # 泰安市 (6个区县)
        '泰山区', '岱岳区', '宁阳县', '东平县', '新泰市', '肥城市',

        # 威海市 (4个区县)
        '环翠区', '文登区', '荣成市', '乳山市',

        # 日照市 (4个区县)
        '东港区', '岚山区', '五莲县', '莒县',

        # 临沂市 (12个区县)
        '兰山区', '罗庄区', '河东区', '沂南县', '郯城县', '沂水县',
        '兰陵县', '费县', '平邑县', '莒南县', '蒙阴县', '临沭县',

        # 德州市 (11个区县)
        '德城区', '陵城区', '宁津县', '庆云县', '临邑县', '齐河县',
        '平原县', '夏津县', '武城县', '乐陵市', '禹城市',

        # 聊城市 (8个区县)
        '东昌府区', '阳谷县', '莘县', '东阿县', '冠县', '高唐县', '临清市',

        # 滨州市 (7个区县)
        '滨城区', '沾化区', '惠民县', '阳信县', '无棣县', '博兴县', '邹平市',

        # 菏泽市 (9个区县)
        '牡丹区', '定陶区', '曹县', '单县', '成武县', '巨野县', '郓城县',
        '鄄城县', '东明县'
    ]

    def __init__(self):
        self.load_enhanced_config()
        self.setup_directories()

    def load_enhanced_config(self):
        """加载增强数据配置"""
        try:
            if os.path.exists(self.ENHANCED_DATA_CONFIG_PATH):
                with open(self.ENHANCED_DATA_CONFIG_PATH, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    all_features = config.get('ENHANCED_FEATURES', [])
                    # 过滤掉非特征列
                    exclude_columns = ['Year', 'County', 'City', 'Severity_Level', 'Has_Occurrence', 'Latitude', 'Longitude']
                    self.ENHANCED_FEATURES = [f for f in all_features if f not in exclude_columns]
                    self.ENHANCED_FEATURE_COUNT = len(self.ENHANCED_FEATURES)
            else:
                # 如果配置文件不存在，使用默认配置
                self.ENHANCED_FEATURES = self.get_default_enhanced_features()
                self.ENHANCED_FEATURE_COUNT = len(self.ENHANCED_FEATURES)
        except Exception as e:
            print(f"加载增强配置失败: {e}")
            self.ENHANCED_FEATURES = self.get_default_enhanced_features()
            self.ENHANCED_FEATURE_COUNT = len(self.ENHANCED_FEATURES)

    def get_default_enhanced_features(self) -> List[str]:
        """获取默认增强特征列表"""
        features = []

        # 气象特征 (25个)
        weather_features = [
            'Spring_Temp', 'Summer_Temp', 'Autumn_Temp', 'Winter_Temp', 'Annual_Temp',
            'Temp_Range', 'Growing_Degree_Days', 'Spring_Precip', 'Summer_Precip',
            'Autumn_Precip', 'Winter_Precip', 'Annual_Precip', 'Precip_Days', 'Max_Daily_Precip',
            'Spring_Humidity', 'Summer_Humidity', 'Autumn_Humidity', 'Winter_Humidity', 'Annual_Humidity',
            'Spring_Wind', 'Summer_Wind', 'Autumn_Wind', 'Winter_Wind', 'Annual_Wind', 'Max_Wind_Speed',
            'Spring_Sunshine', 'Summer_Sunshine', 'Autumn_Sunshine', 'Winter_Sunshine', 'Annual_Sunshine',
            'Frost_Free_Days', 'Heat_Wave_Days', 'Drought_Index'
        ]
        features.extend(weather_features)

        # 遥感特征 (34个)
        remote_sensing_features = [
            'Spring_NDVI', 'Summer_NDVI', 'Autumn_NDVI', 'Annual_NDVI',
            'Spring_EVI', 'Summer_EVI', 'Autumn_EVI', 'Annual_EVI',
            'Spring_LST', 'Summer_LST', 'Autumn_LST', 'Winter_LST', 'Annual_LST',
            'Forest_Cover_Percent', 'Farmland_Percent', 'Urban_Percent', 'Water_Percent', 'Other_Land_Percent',
            'Vegetation_Cover_Percent', 'Elevation_Mean', 'Elevation_STD', 'Slope_Mean', 'Terrain_Roughness',
            'River_Density', 'Lake_Distance', 'Water_Proximity',
            'TRMM_Spring', 'TRMM_Summer', 'TRMM_Autumn', 'TRMM_Annual',
            'Soil_Moisture_Spring', 'Soil_Moisture_Summer', 'Soil_Moisture_Autumn', 'Soil_Moisture_Annual'
        ]
        features.extend(remote_sensing_features)

        # 地理环境特征 (6个)
        geographical_features = [
            'Coastal_Distance', 'County_Elevation', 'Forest_Cover_Base',
            'Coastal_Influence_Index', 'Mountain_Influence_Index', 'Forest_Ecology_Index'
        ]
        features.extend(geographical_features)

        return features

    def setup_directories(self):
        """创建必要的目录"""
        directories = [
            self.MODEL_SAVE_DIR,
            self.VISUALIZATION_DIR,
            self.RESULTS_DIR
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    @property
    def ALL_FEATURES(self) -> List[str]:
        """所有增强特征列表"""
        return self.ENHANCED_FEATURES

    @property
    def NUM_FEATURES(self) -> int:
        """特征数量"""
        return self.ENHANCED_FEATURE_COUNT

    @property
    def NUM_CLASSES(self) -> int:
        """类别数量（包含健康县0级）"""
        return 4

    @property
    def CLASS_NAMES(self) -> List[str]:
        """类别名称"""
        return ['健康', '轻度', '中度', '重度']

    @property
    def CLASS_COLORS(self) -> List[str]:
        """类别颜色"""
        return ['#2E8B57', '#FFD700', '#FF8C00', '#DC143C']

    def get_severity_info(self, level: int) -> Dict[str, Any]:
        """获取发病程度信息"""
        return self.SEVERITY_LEVELS.get(level, {'name': '未知', 'description': '', 'color': '#808080'})

    def county_exists(self, county: str) -> bool:
        """检查县是否在配置中"""
        return county in self.SHANDONG_COUNTIES

    def get_county_by_region(self, region: str) -> List[str]:
        """根据地区获取县列表"""
        region_mapping = {
            '济南': ['历下区', '市中区', '槐荫区', '天桥区', '历城区', '长清区', '章丘区', '济阳区', '莱芜区', '钢城区', '平阴县', '商河县'],
            '青岛': ['市南区', '市北区', '黄岛区', '崂山区', '李沧区', '城阳区', '即墨区', '胶州市', '平度市', '莱西市'],
            '淄博': ['淄川区', '张店区', '博山区', '临淄区', '周村区', '桓台县', '高青县', '沂源县'],
            '枣庄': ['市中区', '薛城区', '峄城区', '台儿庄区', '山亭区', '滕州市'],
            '东营': ['东营区', '河口区', '垦利区', '利津县', '广饶县'],
            '烟台': ['芝罘区', '福山区', '牟平区', '莱山区', '长岛县', '龙口市', '莱阳市', '莱州市', '蓬莱市', '招远市', '栖霞市', '海阳市'],
            '潍坊': ['潍城区', '寒亭区', '坊子区', '奎文区', '临朐县', '昌乐县', '青州市', '诸城市', '寿光市', '安丘市', '高密市', '昌邑市'],
            '济宁': ['任城区', '兖州区', '微山县', '鱼台县', '金乡县', '嘉祥县', '汶上县', '泗水县', '梁山县', '曲阜市', '邹城市'],
            '泰安': ['泰山区', '岱岳区', '宁阳县', '东平县', '新泰市', '肥城市'],
            '威海': ['环翠区', '文登区', '荣成市', '乳山市'],
            '日照': ['东港区', '岚山区', '五莲县', '莒县'],
            '临沂': ['兰山区', '罗庄区', '河东区', '沂南县', '郯城县', '沂水县', '兰陵县', '费县', '平邑县', '莒南县', '蒙阴县', '临沭县'],
            '德州': ['德城区', '陵城区', '宁津县', '庆云县', '临邑县', '齐河县', '平原县', '夏津县', '武城县', '乐陵市', '禹城市'],
            '聊城': ['东昌府区', '阳谷县', '莘县', '东阿县', '冠县', '高唐县', '临清市'],
            '滨州': ['滨城区', '沾化区', '惠民县', '阳信县', '无棣县', '博兴县', '邹平市'],
            '菏泽': ['牡丹区', '定陶区', '曹县', '单县', '成武县', '巨野县', '郓城县', '鄄城县', '东明县']
        }

        return region_mapping.get(region, [])

    def get_feature_categories(self) -> Dict[str, List[str]]:
        """获取特征分类"""
        categories = {}

        if self.ENHANCED_FEATURES:
            # 气象特征
            weather_features = [f for f in self.ENHANCED_FEATURES if any(x in f for x in ['Temp', 'Precip', 'Humidity', 'Wind', 'Sunshine', 'Frost', 'Heat', 'Drought'])]
            if weather_features:
                categories['气象特征'] = weather_features

            # 遥感特征
            rs_features = [f for f in self.ENHANCED_FEATURES if any(x in f for x in ['NDVI', 'EVI', 'LST', 'Land', 'Percent', 'Elevation', 'Slope', 'River', 'Lake', 'TRMM', 'Soil'])]
            if rs_features:
                categories['遥感特征'] = rs_features

            # 地理特征
            geo_features = [f for f in self.ENHANCED_FEATURES if any(x in f for x in ['Coastal', 'County', 'Forest', 'Influence', 'Ecology'])]
            if geo_features:
                categories['地理特征'] = geo_features

        return categories

    def print_config_summary(self):
        """打印配置摘要"""
        print("=== 增强数据集配置摘要 ===")
        print(f"数据路径: {self.ENHANCED_COMPLETE_DATA_PATH}")
        print(f"特征数量: {self.NUM_FEATURES}")
        print(f"类别数量: {self.NUM_CLASSES}")
        print(f"覆盖县数: {len(self.SHANDONG_COUNTIES)}")
        print(f"训练年份: {self.TRAIN_YEARS}")
        print(f"验证年份: {self.VAL_YEARS}")
        print(f"测试年份: {self.TEST_YEARS}")

        print("\n特征分类:")
        categories = self.get_feature_categories()
        for category, features in categories.items():
            print(f"  {category}: {len(features)}个")

        print("\n类别分布:")
        for level, info in self.SEVERITY_LEVELS.items():
            print(f"  {level}级 ({info['name']}): {info['description']}")

# 全局配置实例
enhanced_config = EnhancedCountyLevelConfig()

if __name__ == "__main__":
    config = EnhancedCountyLevelConfig()
    config.print_config_summary()