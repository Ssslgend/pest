#!/usr/bin/env python3
"""
山东美国白蛾发病数据筛选器
从所有发病区域中筛选出山东的县区
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple
import json

class ShandongPestDataFilter:
    """山东美国白蛾数据筛选器"""
    def __init__(self, data_dir: str = "F:\project\gitprojects\\vscode\zsl\lxy\pestBIstm"):
        self.data_dir = data_dir
        self.all_occurrence_data = None
        self.shandong_counties = []
        self.shandong_occurrence_data = None
        
    def load_shandong_counties(self) -> List[str]:
        """
        加载山东省所有县区列表
        Returns:
            山东县区列表
        """
        # 山东省所有县区列表
        shandong_counties = [
            # 济南市
            '历下区', '市中区', '槐荫区', '天桥区', '历城区', '长清区', '章丘区', '平阴县', '济阳县', '商河县',
            
            # 青岛市
            '市南区', '市北区', '黄岛区', '崂山区', '李沧区', '城阳区', '即墨区', '胶州市', '平度市', '莱西市',
            
            # 淄博市
            '淄川区', '张店区', '博山区', '临淄区', '周村区', '桓台县', '高青县', '沂源县',
            
            # 枣庄市
            '市中区', '薛城区', '峄城区', '台儿庄区', '山亭区', '滕州市',
            
            # 东营市
            '东营区', '河口区', '垦利区', '利津县', '广饶县',
            
            # 烟台市
            '芝罘区', '福山区', '牟平区', '莱山区', '长岛县', '龙口市', '莱阳市', '莱州市', '蓬莱市', 
            '招远市', '栖霞市', '海阳市',
            
            # 潍坊市
            '潍城区', '寒亭区', '坊子区', '奎文区', '临朐县', '昌乐县', '青州市', '诸城市', '寿光市', '安丘市', 
            '高密市', '昌邑市',
            
            # 济宁市
            '任城区', '兖州区', '微山县', '鱼台县', '金乡县', '嘉祥县', '汶上县', '泗水县', '梁山县', '曲阜市', '邹城市',
            
            # 泰安市
            '泰山区', '岱岳区', '宁阳县', '东平县', '新泰市', '肥城市',
            
            # 威海市
            '环翠区', '文登区', '荣成市', '乳山市',
            
            # 日照市
            '东港区', '岚山区', '五莲县', '莒县',
            
            # 莱芜市
            '莱城区', '钢城区',
            
            # 临沂市
            '兰山区', '罗庄区', '河东区', '沂南县', '郯城县', '兰陵县', '莒南县', '沂水县', '蒙阴县', '平邑县', 
            '费县', '临沭县',
            
            # 德州市
            '德城区', '陵城区', '宁津县', '庆云县', '临邑县', '齐河县', '平原县', '夏津县', '武城县', '乐陵市', '禹城市',
            
            # 聊城市
            '东昌府区', '茌平区', '阳谷县', '莘县', '东阿县', '冠县', '高唐县', '临清市',
            
            # 滨州市
            '滨城区', '沾化区', '惠民县', '阳信县', '无棣县', '博兴县', '邹平市',
            
            # 菏泽市
            '牡丹区', '定陶区', '曹县', '单县', '成武县', '巨野县', '郓城县', '鄄城县', '东明县'
        ]
        
        self.shandong_counties = shandong_counties
        print(f"加载山东省县区列表: {len(shandong_counties)}个")
        return shandong_counties
    
    def load_all_occurrence_data(self) -> pd.DataFrame:
        """
        加载所有年份的美国白蛾发生数据
        Returns:
            合并后的发生数据
        """
        try:
            all_data = []
            
            # 加载2019-2023年的数据
            for year in range(2019, 2024):
                file_path = os.path.join(self.data_dir, f"fall_webworm_occurrences_{year}_geocoded.csv")
                if os.path.exists(file_path):
                    data = pd.read_csv(file_path)
                    data['year'] = year
                    data['data_source'] = f"occurrences_{year}_geocoded.csv"
                    all_data.append(data)
                    print(f"成功加载{year}年数据: {len(data)}条记录")
                else:
                    print(f"未找到{year}年数据文件: {file_path}")
            
            if all_data:
                self.all_occurrence_data = pd.concat(all_data, ignore_index=True)
                print(f"总计加载发生数据: {len(self.all_occurrence_data)}条记录")
                
                # 显示数据基本信息
                print(f"数据年份范围: {self.all_occurrence_data['year'].min()}-{self.all_occurrence_data['year'].max()}")
                print(f"涉及县区数: {self.all_occurrence_data['原始行政区名称'].nunique()}")
                
            else:
                print("未找到任何发生数据")
                
            return self.all_occurrence_data
            
        except Exception as e:
            print(f"加载发生数据失败: {e}")
            return None
    
    def identify_shandong_occurrences(self) -> pd.DataFrame:
        """
        识别山东的发生数据
        Returns:
            山东发生数据
        """
        if self.all_occurrence_data is None:
            print("请先加载所有发生数据")
            return None
        
        if not self.shandong_counties:
            self.load_shandong_counties()
        
        print("开始识别山东的发生数据...")
        
        # 筛选山东的发生数据
        shandong_data = self.all_occurrence_data[
            self.all_occurrence_data['原始行政区名称'].isin(self.shandong_counties)
        ].copy()
        
        if len(shandong_data) > 0:
            print(f"发现山东发生数据: {len(shandong_data)}条记录")
            print(f"涉及山东县区: {shandong_data['原始行政区名称'].nunique()}个")
            
            # 按年份统计
            year_stats = shandong_data.groupby('year').size()
            print(f"各年份发生记录数:")
            for year, count in year_stats.items():
                print(f"  {year}年: {count}条")
            
            # 按县区统计
            county_stats = shandong_data.groupby('原始行政区名称').size().sort_values(ascending=False)
            print(f"发生记录最多的10个县区:")
            for county, count in county_stats.head(10).items():
                print(f"  {county}: {count}条")
            
            self.shandong_occurrence_data = shandong_data
        else:
            print("未发现山东的发生数据")
            self.shandong_occurrence_data = pd.DataFrame()
        
        return self.shandong_occurrence_data
    
    def analyze_shandong_occurrence_patterns(self) -> Dict:
        """
        分析山东发病模式
        Returns:
            分析结果字典
        """
        if self.shandong_occurrence_data is None or len(self.shandong_occurrence_data) == 0:
            print("没有山东发生数据可分析")
            return {}
        
        print("分析山东美国白蛾发生模式...")
        
        analysis_result = {
            'total_occurrences': len(self.shandong_occurrence_data),
            'years_covered': sorted(self.shandong_occurrence_data['year'].unique().tolist()),
            'affected_counties': self.shandong_occurrence_data['原始行政区名称'].nunique(),
            'county_statistics': {},
            'yearly_statistics': {},
            'geographic_distribution': {}
        }
        
        # 县区统计
        county_stats = self.shandong_occurrence_data.groupby('原始行政区名称').agg({
            'year': ['count', 'min', 'max'],
            '发生样点经度': 'mean',
            '发生样点纬度': 'mean'
        }).round(6)
        
        for county in county_stats.index:
            analysis_result['county_statistics'][county] = {
                'total_occurrences': int(county_stats.loc[county, ('year', 'count')]),
                'first_occurrence': int(county_stats.loc[county, ('year', 'min')]),
                'last_occurrence': int(county_stats.loc[county, ('year', 'max')]),
                'mean_longitude': float(county_stats.loc[county, ('发生样点经度', 'mean')]),
                'mean_latitude': float(county_stats.loc[county, ('发生样点纬度', 'mean')])
            }
        
        # 年份统计
        year_stats = self.shandong_occurrence_data.groupby('year').agg({
            '原始行政区名称': 'nunique',
            '发生样点经度': 'count'
        })
        
        for year in year_stats.index:
            analysis_result['yearly_statistics'][int(year)] = {
                'affected_counties': int(year_stats.loc[year, '原始行政区名称']),
                'total_occurrences': int(year_stats.loc[year, '发生样点经度'])
            }
        
        # 地理分布分析
        analysis_result['geographic_distribution'] = {
            'longitude_range': {
                'min': float(self.shandong_occurrence_data['发生样点经度'].min()),
                'max': float(self.shandong_occurrence_data['发生样点经度'].max()),
                'mean': float(self.shandong_occurrence_data['发生样点经度'].mean())
            },
            'latitude_range': {
                'min': float(self.shandong_occurrence_data['发生样点纬度'].min()),
                'max': float(self.shandong_occurrence_data['发生样点纬度'].max()),
                'mean': float(self.shandong_occurrence_data['发生样点纬度'].mean())
            }
        }
        
        return analysis_result
    
    def save_shandong_data(self, output_format: str = 'csv'):
        """
        保存山东发生数据
        Args:
            output_format: 输出格式 ('csv', 'json', 'both')
        """
        if self.shandong_occurrence_data is None or len(self.shandong_occurrence_data) == 0:
            print("没有山东发生数据可保存")
            return
        
        try:
            output_dir = os.path.join(self.data_dir, "datas", "shandong_pest_data")
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            
            if output_format in ['csv', 'both']:
                # 保存CSV
                csv_path = os.path.join(output_dir, f"shandong_fall_webworm_occurrences_{timestamp}.csv")
                self.shandong_occurrence_data.to_csv(csv_path, index=False, encoding='utf-8')
                print(f"山东发生数据CSV已保存至: {csv_path}")
            
            if output_format in ['json', 'both']:
                # 保存JSON
                json_path = os.path.join(output_dir, f"shandong_fall_webworm_occurrences_{timestamp}.json")
                self.shandong_occurrence_data.to_json(json_path, orient='records', force_ascii=False, indent=2)
                print(f"山东发生数据JSON已保存至: {json_path}")
            
            # 保存县区列表
            counties_path = os.path.join(output_dir, f"shandong_affected_counties_{timestamp}.txt")
            affected_counties = sorted(self.shandong_occurrence_data['原始行政区名称'].unique())
            with open(counties_path, 'w', encoding='utf-8') as f:
                f.write("山东美国白蛾发生县区列表\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"总县区数: {len(affected_counties)}\n")
                f.write(f"数据年份: {self.shandong_occurrence_data['year'].min()}-{self.shandong_occurrence_data['year'].max()}\n")
                f.write(f"总发生记录: {len(self.shandong_occurrence_data)}\n\n")
                f.write("县区列表:\n")
                f.write("-" * 30 + "\n")
                for county in affected_counties:
                    occurrence_count = len(self.shandong_occurrence_data[
                        self.shandong_occurrence_data['原始行政区名称'] == county
                    ])
                    f.write(f"{county} ({occurrence_count}条记录)\n")
            
            print(f"山东发生县区列表已保存至: {counties_path}")
            
        except Exception as e:
            print(f"保存山东发生数据失败: {e}")
    
    def run_analysis(self) -> Dict:
        """
        运行完整的山东数据筛选分析
        Returns:
            分析结果
        """
        print("开始山东美国白蛾数据筛选分析...")
        
        # 1. 加载山东县区列表
        self.load_shandong_counties()
        
        # 2. 加载所有发生数据
        self.load_all_occurrence_data()
        
        # 3. 识别山东发生数据
        self.identify_shandong_occurrences()
        
        # 4. 分析发生模式
        analysis_result = self.analyze_shandong_occurrence_patterns()
        
        # 5. 保存结果
        self.save_shandong_data('both')
        
        print("\n山东美国白蛾数据筛选分析完成!")
        
        if analysis_result:
            print(f"\n关键发现:")
            print(f"- 总发生记录: {analysis_result['total_occurrences']}条")
            print(f"- 涉及县区: {analysis_result['affected_counties']}个")
            print(f"- 数据年份: {analysis_result['years_covered']}")
            print(f"- 地理范围: 经度{analysis_result['geographic_distribution']['longitude_range']['min']:.2f}~{analysis_result['geographic_distribution']['longitude_range']['max']:.2f}")
            print(f"- 地理范围: 纬度{analysis_result['geographic_distribution']['latitude_range']['min']:.2f}~{analysis_result['geographic_distribution']['latitude_range']['max']:.2f}")
        
        return analysis_result

def main():
    """主函数"""
    analyzer = ShandongPestDataFilter()
    
    # 运行分析
    result = analyzer.run_analysis()
    
    if result:
        print(f"\n详细分析结果已保存到: pestBIstm/datas/shandong_pest_data/")

if __name__ == "__main__":
    main()