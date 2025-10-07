#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强数据集分析系统
分析增强后的数据质量、特征分布、类别平衡性等
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from enhanced_county_config import EnhancedCountyLevelConfig
import json
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class EnhancedDataAnalyzer:
    """增强数据集分析器"""

    def __init__(self):
        self.config = EnhancedCountyLevelConfig()
        self.load_data()

    def load_data(self):
        """加载增强数据"""
        print("=== 加载增强数据集 ===")

        self.data = pd.read_csv(self.config.ENHANCED_COMPLETE_DATA_PATH)
        print(f"数据集大小: {self.data.shape}")
        print(f"覆盖县数: {self.data['County'].nunique()}")
        print(f"年份范围: {self.data['Year'].min()}-{self.data['Year'].max()}")

        # 统计发病程度分布
        print(f"\n发病程度分布:")
        severity_dist = self.data['Severity_Level'].value_counts().sort_index()
        for level, count in severity_dist.items():
            percentage = count / len(self.data) * 100
            print(f"  {level}级: {count} 样本 ({percentage:.1f}%)")

        self.severity_dist = severity_dist

    def comprehensive_data_quality_analysis(self):
        """综合数据质量分析"""
        print("\n=== 综合数据质量分析 ===")

        # 1. 缺失值分析
        self.analyze_missing_values()

        # 2. 异常值分析
        self.analyze_outliers()

        # 3. 数据分布分析
        self.analyze_data_distributions()

        # 4. 特征相关性分析
        self.analyze_feature_correlations()

        # 5. 时间序列一致性分析
        self.analyze_temporal_consistency()

        # 6. 地理覆盖分析
        self.analyze_geographic_coverage()

    def analyze_missing_values(self):
        """缺失值分析"""
        print("\n--- 缺失值分析 ---")

        missing_values = self.data.isnull().sum()
        total_missing = missing_values.sum()

        if total_missing > 0:
            missing_features = missing_values[missing_values > 0]
            print(f"发现 {len(missing_features)} 个特征存在缺失值:")
            for feature, count in missing_features.items():
                percentage = count / len(self.data) * 100
                print(f"  {feature}: {count} 个缺失值 ({percentage:.1f}%)")
        else:
            print("✓ 无缺失值")

        # 创建缺失值可视化
        if total_missing > 0:
            self.create_missing_value_heatmap(missing_values)

    def analyze_outliers(self):
        """异常值分析"""
        print("\n--- 异常值分析 ---")

        numeric_features = self.data.select_dtypes(include=[np.number]).columns
        outlier_stats = {}

        for feature in numeric_features:
            if feature not in ['Year', 'Severity_Level']:
                Q1 = self.data[feature].quantile(0.25)
                Q3 = self.data[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = self.data[(self.data[feature] < lower_bound) |
                                   (self.data[feature] > upper_bound)]

                if len(outliers) > 0:
                    outlier_stats[feature] = {
                        'count': len(outliers),
                        'percentage': len(outliers) / len(self.data) * 100,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    }

        if outlier_stats:
            print(f"发现 {len(outlier_stats)} 个特征存在异常值:")
            for feature, stats in outlier_stats.items():
                print(f"  {feature}: {stats['count']} 个异常值 ({stats['percentage']:.1f}%)")
        else:
            print("✓ 无明显异常值")

        # 保存异常值统计
        self.outlier_stats = outlier_stats

    def analyze_data_distributions(self):
        """数据分布分析"""
        print("\n--- 数据分布分析 ---")

        # 按发病程度分组分析特征分布
        features_to_analyze = [
            'Spring_Temp', 'Summer_Temp', 'Annual_Precip', 'Annual_Humidity',
            'Spring_NDVI', 'Summer_NDVI', 'Forest_Cover_Percent', 'Coastal_Distance'
        ]

        # 过滤实际存在的特征
        available_features = [f for f in features_to_analyze if f in self.data.columns]

        print("关键特征统计摘要:")
        for feature in available_features:
            print(f"\n{feature}:")
            for level in sorted(self.data['Severity_Level'].unique()):
                subset = self.data[self.data['Severity_Level'] == level][feature]
                print(f"  {level}级: 均值={subset.mean():.2f}, 标准差={subset.std():.2f}")

        # 创建分布可视化
        self.create_distribution_plots(available_features)

    def analyze_feature_correlations(self):
        """特征相关性分析"""
        print("\n--- 特征相关性分析 ---")

        # 选择数值特征
        numeric_features = self.data.select_dtypes(include=[np.number]).columns
        exclude_cols = ['Year', 'Severity_Level']
        correlation_features = [f for f in numeric_features if f not in exclude_cols]

        if len(correlation_features) > 0:
            # 计算相关性矩阵
            corr_matrix = self.data[correlation_features].corr()

            # 找出高相关性特征对
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.8:  # 高相关性阈值
                        high_corr_pairs.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': corr_value
                        })

            if high_corr_pairs:
                print(f"发现 {len(high_corr_pairs)} 对高相关性特征 (|r| > 0.8):")
                for pair in high_corr_pairs:
                    print(f"  {pair['feature1']} - {pair['feature2']}: {pair['correlation']:.3f}")
            else:
                print("✓ 无明显高相关性特征对")

            # 创建相关性热力图
            self.create_correlation_heatmap(corr_matrix, high_corr_pairs)

    def analyze_temporal_consistency(self):
        """时间序列一致性分析"""
        print("\n--- 时间序列一致性分析 ---")

        # 检查每个县的时间序列完整性
        county_coverage = {}
        for county in self.data['County'].unique():
            county_data = self.data[self.data['County'] == county]
            years = sorted(county_data['Year'].unique())
            county_coverage[county] = {
                'years': years,
                'completeness': len(years) / len(self.config.TRAIN_YEARS + self.config.VAL_YEARS + self.config.TEST_YEARS),
                'severity_levels': sorted(county_data['Severity_Level'].unique())
            }

        # 统计完整性
        complete_counties = [c for c, info in county_coverage.items() if info['completeness'] == 1.0]
        incomplete_counties = [c for c, info in county_coverage.items() if info['completeness'] < 1.0]

        print(f"时间序列完整的县: {len(complete_counties)} 个")
        print(f"时间序列不完整的县: {len(incomplete_counties)} 个")

        if incomplete_counties:
            print(f"不完整的县: {incomplete_counties[:10]}...")  # 只显示前10个

        # 创建时间序列覆盖图
        self.create_temporal_coverage_plot(county_coverage)

    def analyze_geographic_coverage(self):
        """地理覆盖分析"""
        print("\n--- 地理覆盖分析 ---")

        # 按地区分析覆盖情况
        regions = {
            '济南': self.config.get_county_by_region('济南'),
            '青岛': self.config.get_county_by_region('青岛'),
            '烟台': self.config.get_county_by_region('烟台'),
            '潍坊': self.config.get_county_by_region('潍坊'),
            '临沂': self.config.get_county_by_region('临沂'),
            '其他': []  # 其他地区
        }

        # 统计各地区覆盖情况
        region_stats = {}
        for region, counties in regions.items():
            if region == '其他':
                # 其他地区包括所有未明确分类的县
                all_counties = set(self.data['County'].unique())
                classified_counties = set()
                for r, c in regions.items():
                    if r != '其他':
                        classified_counties.update(c)
                counties = list(all_counties - classified_counties)

            covered_counties = [c for c in counties if c in self.data['County'].unique()]
            region_stats[region] = {
                'total_counties': len(counties),
                'covered_counties': len(covered_counties),
                'coverage_rate': len(covered_counties) / len(counties) if len(counties) > 0 else 0,
                'samples': len(self.data[self.data['County'].isin(covered_counties)])
            }

        print("各地区覆盖情况:")
        for region, stats in region_stats.items():
            print(f"  {region}: {stats['covered_counties']}/{stats['total_counties']} 县 "
                  f"({stats['coverage_rate']:.1%}), {stats['samples']} 样本")

        # 创建地理覆盖可视化
        self.create_geographic_coverage_plot(region_stats)

    def create_missing_value_heatmap(self, missing_values):
        """创建缺失值热力图"""
        missing_features = missing_values[missing_values > 0]
        if len(missing_features) > 0:
            plt.figure(figsize=(10, max(4, len(missing_features) * 0.3)))
            missing_df = pd.DataFrame({
                'Feature': missing_features.index,
                'Missing_Count': missing_features.values,
                'Missing_Percentage': missing_features.values / len(self.data) * 100
            }).set_index('Feature')

            sns.heatmap(missing_df.T, annot=True, fmt='.1f', cmap='Reds', cbar_kws={'label': '缺失率(%)'})
            plt.title('特征缺失值分布')
            plt.tight_layout()
            plt.savefig('results/enhanced_visualizations/missing_values_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()

    def create_distribution_plots(self, features):
        """创建特征分布图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()

        for i, feature in enumerate(features[:4]):  # 只显示前4个特征
            if i < len(axes):
                for level in sorted(self.data['Severity_Level'].unique()):
                    subset = self.data[self.data['Severity_Level'] == level][feature]
                    axes[i].hist(subset, alpha=0.6, label=f'{level}级', bins=20,
                               density=True, color=self.config.CLASS_COLORS[level])

                axes[i].set_title(f'{feature} 分布')
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('密度')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/enhanced_visualizations/feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_correlation_heatmap(self, corr_matrix, high_corr_pairs):
        """创建相关性热力图"""
        # 只显示部分特征以避免图像过于复杂
        n_features = min(20, len(corr_matrix.columns))
        selected_features = corr_matrix.columns[:n_features]
        selected_corr = corr_matrix.loc[selected_features, selected_features]

        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(selected_corr, dtype=bool))
        sns.heatmap(selected_corr, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={'label': '相关系数'})
        plt.title('特征相关性热力图 (前20个特征)')
        plt.tight_layout()
        plt.savefig('results/enhanced_visualizations/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_temporal_coverage_plot(self, county_coverage):
        """创建时间序列覆盖图"""
        # 创建年份覆盖矩阵
        years = self.config.TRAIN_YEARS + self.config.VAL_YEARS + self.config.TEST_YEARS
        counties = list(county_coverage.keys())
        coverage_matrix = pd.DataFrame(0, index=counties[:20], columns=years)  # 只显示前20个县

        for county, info in county_coverage.items():
            if county in coverage_matrix.index:
                for year in info['years']:
                    coverage_matrix.loc[county, year] = 1

        plt.figure(figsize=(10, 8))
        sns.heatmap(coverage_matrix, annot=True, cmap='Greens', cbar_kws={'label': '数据覆盖'})
        plt.title('县级行政区时间序列数据覆盖情况 (前20个县)')
        plt.xlabel('年份')
        plt.ylabel('县级行政区')
        plt.tight_layout()
        plt.savefig('results/enhanced_visualizations/temporal_coverage.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_geographic_coverage_plot(self, region_stats):
        """创建地理覆盖图"""
        regions = list(region_stats.keys())
        coverage_rates = [stats['coverage_rate'] for stats in region_stats.values()]
        sample_counts = [stats['samples'] for stats in region_stats.values()]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 覆盖率
        bars1 = ax1.bar(regions, coverage_rates, color='skyblue')
        ax1.set_title('各地区数据覆盖率')
        ax1.set_ylabel('覆盖率')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        for bar, rate in zip(bars1, coverage_rates):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{rate:.1%}', ha='center', va='bottom')

        # 样本数量
        bars2 = ax2.bar(regions, sample_counts, color='lightgreen')
        ax2.set_title('各地区样本数量')
        ax2.set_ylabel('样本数')
        ax2.tick_params(axis='x', rotation=45)
        for bar, count in zip(bars2, sample_counts):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(sample_counts)*0.01,
                    f'{count}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('results/enhanced_visualizations/geographic_coverage.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_comprehensive_report(self):
        """生成综合分析报告"""
        print("\n=== 生成综合分析报告 ===")

        report = {
            'analysis_date': pd.Timestamp.now().isoformat(),
            'dataset_info': {
                'total_samples': len(self.data),
                'total_counties': self.data['County'].nunique(),
                'year_range': [int(self.data['Year'].min()), int(self.data['Year'].max())],
                'feature_count': self.config.NUM_FEATURES,
                'class_count': self.config.NUM_CLASSES
            },
            'data_quality': {
                'missing_values_count': self.data.isnull().sum().sum(),
                'outlier_features_count': len(getattr(self, 'outlier_stats', {})),
                'data_completeness': 1.0 - (self.data.isnull().sum().sum() / (len(self.data) * len(self.data.columns)))
            },
            'class_distribution': {
                str(level): {
                    'count': int(count),
                    'percentage': float(count / len(self.data) * 100)
                } for level, count in self.severity_dist.items()
            },
            'feature_categories': {
                category: len(features) for category, features in self.config.get_feature_categories().items()
            },
            'data_enhancement_summary': {
                'original_counties': 110,
                'enhanced_counties': self.data['County'].nunique(),
                'new_healthy_counties': self.data['County'].nunique() - 110,
                'healthy_samples': int(len(self.data[self.data['Severity_Level'] == 0])),
                'remote_sensing_features': len([f for f in self.config.ALL_FEATURES
                                              if any(x in f for x in ['NDVI', 'EVI', 'LST', 'Land', 'TRMM', 'Soil'])]),
                'geographical_features': len([f for f in self.config.ALL_FEATURES
                                            if any(x in f for x in ['Coastal', 'County', 'Forest', 'Influence'])])
            }
        }

        # 保存报告
        os.makedirs('results/enhanced_visualizations', exist_ok=True)
        with open('results/enhanced_visualizations/comprehensive_analysis_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 打印摘要
        self.print_analysis_summary(report)

        print(f"\n综合分析报告已保存到: results/enhanced_visualizations/comprehensive_analysis_report.json")

    def print_analysis_summary(self, report):
        """打印分析摘要"""
        print("\n" + "="*80)
        print("增强数据集综合分析摘要")
        print("="*80)

        print(f"\n数据集基本信息:")
        print(f"  总样本数: {report['dataset_info']['total_samples']}")
        print(f"  覆盖县数: {report['dataset_info']['total_counties']}")
        print(f"  年份范围: {report['dataset_info']['year_range'][0]}-{report['dataset_info']['year_range'][1]}")
        print(f"  特征数量: {report['dataset_info']['feature_count']}")
        print(f"  类别数量: {report['dataset_info']['class_count']}")

        print(f"\n数据质量:")
        print(f"  数据完整性: {report['data_quality']['data_completeness']:.1%}")
        print(f"  缺失值数量: {report['data_quality']['missing_values_count']}")
        print(f"  异常特征数量: {report['data_quality']['outlier_features_count']}")

        print(f"\n类别分布:")
        for level, info in report['class_distribution'].items():
            print(f"  {level}级: {info['count']} 样本 ({info['percentage']:.1f}%)")

        print(f"\n数据增强成果:")
        print(f"  原覆盖县数: {report['data_enhancement_summary']['original_counties']}")
        print(f"  增强后县数: {report['data_enhancement_summary']['enhanced_counties']}")
        print(f"  新增健康县: {report['data_enhancement_summary']['new_healthy_counties']}")
        print(f"  健康县样本: {report['data_enhancement_summary']['healthy_samples']}")
        print(f"  遥感特征数: {report['data_enhancement_summary']['remote_sensing_features']}")
        print(f"  地理特征数: {report['data_enhancement_summary']['geographical_features']}")

        print(f"\n特征分类:")
        for category, count in report['feature_categories'].items():
            print(f"  {category}: {count}个")

        print("\n" + "="*80)

def main():
    """主函数"""
    print("=== 增强数据集综合分析系统 ===")

    # 创建分析器
    analyzer = EnhancedDataAnalyzer()

    # 执行综合分析
    analyzer.comprehensive_data_quality_analysis()

    # 生成报告
    analyzer.generate_comprehensive_report()

    print(f"\n=== 分析完成 ===")
    print("所有分析结果和可视化图表已保存到 results/enhanced_visualizations/ 目录")

    return analyzer

if __name__ == "__main__":
    import os
    analyzer = main()