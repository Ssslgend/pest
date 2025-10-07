#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据增强分析
分析现有数据缺陷，识别需要补充的健康县数据
规划完整的数据整合方案
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from county_level_config import CountyLevelConfig
import json
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class DataEnhancementAnalyzer:
    """数据增强分析器"""

    def __init__(self):
        self.config = CountyLevelConfig()
        self.load_current_data()

    def load_current_data(self):
        """加载当前数据"""
        print("=== 当前数据分析 ===")

        # 加载主要数据
        self.current_data = pd.read_csv(self.config.COMPLETE_DATA_PATH)

        print(f"当前数据: {len(self.current_data)} 样本")
        print(f"县数: {self.current_data['County'].nunique()}")
        print(f"年份范围: {self.current_data['Year'].min()}-{self.current_data['Year'].max()}")

        # 分析发病程度分布
        print("\n发病程度分布:")
        severity_dist = self.current_data['Severity_Level'].value_counts().sort_index()
        for level, count in severity_dist.items():
            print(f"  {level}级: {count} 样本 ({count/len(self.current_data)*100:.1f}%)")

        # 分析县覆盖情况
        print(f"\n当前覆盖的县: {self.current_data['County'].nunique()} 个")
        self.current_counties = set(self.current_data['County'].unique())

    def analyze_missing_counties(self):
        """分析缺失的县数据"""
        print("\n=== 山东省完整县级行政区划分析 ===")

        # 山东省完整县级行政区划（根据最新的行政区划）
        shandong_all_counties = [
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

            # 威海市 (7个区县)
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

        # 去重
        shandong_all_counties = list(set(shandong_all_counties))

        print(f"山东省总县级行政区划数: {len(shandong_all_counties)}")
        print(f"当前数据覆盖县数: {len(self.current_counties)}")

        # 找出缺失的县
        missing_counties = set(shandong_all_counties) - self.current_counties
        print(f"缺失的县数: {len(missing_counties)}")

        if missing_counties:
            print(f"缺失的县: {sorted(list(missing_counties))}")

        self.shandong_all_counties = shandong_all_counties
        self.missing_counties = missing_counties

        # 创建数据覆盖热力图
        self.create_coverage_heatmap()

    def create_coverage_heatmap(self):
        """创建数据覆盖热力图"""
        print("\n创建数据覆盖情况可视化...")

        # 创建覆盖情况矩阵
        years = sorted(self.current_data['Year'].unique())
        coverage_matrix = pd.DataFrame(index=years, columns=self.shandong_all_counties)

        for year in years:
            year_data = self.current_data[self.current_data['Year'] == year]
            covered_counties = set(year_data['County'].unique())

            for county in self.shandong_all_counties:
                if county in covered_counties:
                    coverage_matrix.loc[year, county] = 1
                else:
                    coverage_matrix.loc[year, county] = 0

        # 转换为数值类型
        coverage_matrix = coverage_matrix.fillna(0).astype(int)

        # 统计每年的覆盖率
        yearly_coverage = coverage_matrix.sum(axis=1) / len(self.shandong_all_counties)

        # 可视化
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))

        # 1. 年度覆盖率趋势
        axes[0, 0].plot(years, yearly_coverage, 'o-', linewidth=3, markersize=8, color='red')
        axes[0, 0].set_title('山东省县级数据覆盖率年度变化', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('年份')
        axes[0, 0].set_ylabel('覆盖率')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)

        # 添加数值标签
        for i, (year, coverage) in enumerate(zip(years, yearly_coverage)):
            axes[0, 0].text(year, coverage + 0.02, f'{coverage:.1%}',
                           ha='center', va='bottom', fontweight='bold')

        # 2. 县覆盖频率热力图（显示前20个县）
        county_coverage = coverage_matrix.sum(axis=0).sort_values(ascending=True)
        bottom_20 = county_coverage.head(20)

        sns.heatmap(bottom_20.values.reshape(-1, 1),
                   annot=True, fmt='d', cmap='Reds',
                   yticklabels=bottom_20.index,
                   xticklabels=['覆盖年数'],
                   ax=axes[0, 1])
        axes[0, 1].set_title('覆盖年数最少的20个县', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('')

        # 3. 整体覆盖热力图（按地区分组）
        # 将县按地区分组显示
        regions = {
            '济南': ['历下区', '市中区', '槐荫区', '天桥区', '历城区', '长清区', '章丘区', '济阳区', '莱芜区', '钢城区', '平阴县', '商河县'],
            '青岛': ['市南区', '市北区', '黄岛区', '崂山区', '李沧区', '城阳区', '即墨区', '胶州市', '平度市', '莱西市'],
            '烟台': ['芝罘区', '福山区', '牟平区', '莱山区', '长岛县', '龙口市', '莱阳市', '莱州市', '蓬莱市', '招远市', '栖霞市', '海阳市'],
            '潍坊': ['潍城区', '寒亭区', '坊子区', '奎文区', '临朐县', '昌乐县', '青州市', '诸城市', '寿光市', '安丘市', '高密市', '昌邑市'],
            '其他': [c for c in self.shandong_all_counties if c not in
                    ['历下区', '市中区', '槐荫区', '天桥区', '历城区', '长清区', '章丘区', '济阳区', '莱芜区', '钢城区', '平阴县', '商河县',
                     '市南区', '市北区', '黄岛区', '崂山区', '李沧区', '城阳区', '即墨区', '胶州市', '平度市', '莱西市',
                     '芝罘区', '福山区', '牟平区', '莱山区', '长岛县', '龙口市', '莱阳市', '莱州市', '蓬莱市', '招远市', '栖霞市', '海阳市',
                     '潍城区', '寒亭区', '坊子区', '奎文区', '临朐县', '昌乐县', '青州市', '诸城市', '寿光市', '安丘市', '高密市', '昌邑市']]
        }

        region_coverage = {}
        for region, counties in regions.items():
            region_counties = [c for c in counties if c in coverage_matrix.columns]
            if region_counties:
                region_data = coverage_matrix[region_counties]
                region_coverage[region] = region_data.sum().sum() / (len(region_counties) * len(years))

        # 地区覆盖率对比
        region_names = list(region_coverage.keys())
        coverage_rates = list(region_coverage.values())

        bars = axes[1, 0].bar(region_names, coverage_rates, color=['skyblue', 'lightgreen', 'salmon', 'orange', 'purple'])
        axes[1, 0].set_title('不同地区数据覆盖率对比', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('覆盖率')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].tick_params(axis='x', rotation=45)

        for bar, rate in zip(bars, coverage_rates):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                           f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')

        # 4. 缺失数据统计
        missing_stats = {
            '完全缺失的县': len(self.missing_counties),
            '部分缺失的县': len([c for c in self.current_counties
                               if len(self.current_data[self.current_data['County'] == c]) < len(years)]),
            '完整覆盖的县': len([c for c in self.current_counties
                               if len(self.current_data[self.current_data['County'] == c]) == len(years)])
        }

        colors = ['red', 'orange', 'green']
        wedges, texts, autotexts = axes[1, 1].pie(missing_stats.values(),
                                                 labels=missing_stats.keys(),
                                                 colors=colors,
                                                 autopct='%1.1f%%',
                                                 startangle=90)
        axes[1, 1].set_title('县数据覆盖情况分类', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig('results/data_enhancement/coverage_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("数据覆盖分析图保存到: results/data_enhancement/coverage_analysis.png")

    def plan_data_enhancement(self):
        """规划数据增强方案"""
        print("\n=== 数据增强规划 ===")

        enhancement_plan = {
            'current_status': {
                'total_samples': len(self.current_data),
                'covered_counties': len(self.current_counties),
                'missing_counties': len(self.missing_counties),
                'total_shandong_counties': len(self.shandong_all_counties),
                'coverage_rate': len(self.current_counties) / len(self.shandong_all_counties)
            },
            'enhancement_tasks': {
                'healthy_county_data': {
                    'description': '添加健康县数据（0级发病程度）',
                    'target_counties': len(self.missing_counties) + len(self.current_counties),
                    'estimated_samples': (len(self.missing_counties) + len(self.current_counties)) * 5,  # 5年数据
                    'priority': 'HIGH'
                },
                'remote_sensing_features': {
                    'description': '整合遥感数据特征',
                    'features': [
                        'NDVI (植被指数)',
                        'EVI (增强植被指数)',
                        'LST (地表温度)',
                        '土地利用类型',
                        '植被覆盖度',
                        '降水遥感数据',
                        '土壤湿度'
                    ],
                    'data_sources': [
                        'Landsat 8/9',
                        'Sentinel-2',
                        'MODIS',
                        'Google Earth Engine'
                    ],
                    'priority': 'HIGH'
                },
                'additional_meteorological': {
                    'description': '补充气象数据',
                    'features': [
                        '极端气温事件',
                        '降水异常指数',
                        '风速风向',
                        '日照时数',
                        '相对湿度变化'
                    ],
                    'priority': 'MEDIUM'
                },
                'geographical_features': {
                    'description': '地理环境特征',
                    'features': [
                        '海拔高度',
                        '地形复杂度',
                        '距海岸线距离',
                        '水系分布',
                        '森林覆盖率'
                    ],
                    'priority': 'MEDIUM'
                }
            },
            'implementation_steps': [
                '1. 收集山东省完整县级行政区划数据',
                '2. 识别并标记健康县（无发病记录）',
                '3. 收集历史气象数据扩展时间序列',
                '4. 整合多源遥感数据',
                '5. 构建统一的特征数据库',
                '6. 数据质量检查和清洗',
                '7. 特征工程和数据预处理',
                '8. 重新训练和验证模型'
            ]
        }

        # 保存增强方案
        import os
        os.makedirs('results/data_enhancement', exist_ok=True)

        with open('results/data_enhancement/enhancement_plan.json', 'w', encoding='utf-8') as f:
            json.dump(enhancement_plan, f, indent=2, ensure_ascii=False)

        # 打印方案摘要
        print("数据增强方案:")
        print(f"  当前覆盖率: {enhancement_plan['current_status']['coverage_rate']:.1%}")
        print(f"  需要补充县数: {enhancement_plan['current_status']['missing_counties']}")
        print(f"  预计新增样本: {enhancement_plan['enhancement_tasks']['healthy_county_data']['estimated_samples']}")

        print(f"\n主要任务:")
        for task_name, task_info in enhancement_plan['enhancement_tasks'].items():
            print(f"  - {task_info['description']} (优先级: {task_info['priority']})")

        print(f"\n实施步骤:")
        for step in enhancement_plan['implementation_steps']:
            print(f"  {step}")

        return enhancement_plan

def main():
    """主函数"""
    analyzer = DataEnhancementAnalyzer()

    # 分析当前数据
    analyzer.analyze_missing_counties()

    # 规划数据增强
    plan = analyzer.plan_data_enhancement()

    print(f"\n=== 数据增强分析完成 ===")
    print(f"分析报告保存到: results/data_enhancement/")

    return analyzer, plan

if __name__ == "__main__":
    analyzer, plan = main()