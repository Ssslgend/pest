import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像负号'-'显示为方块的问题

# 设置输出目录
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "data_analysis")
os.makedirs(output_dir, exist_ok=True)

def load_data():
    """加载并检查数据质量"""
    # 数据路径 - 根据项目结构设置
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datas', 'train.csv')
    
    print(f"加载数据: {csv_path}")
    try:
        # 尝试不同的编码
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='gbk')
        
        print(f"数据形状: {df.shape}")
        print(f"列数量: {len(df.columns)}")
        
        # 查找标签列
        label_col = 'label'
        if label_col not in df.columns:
            for possible_label in ['Label', 'TARGET', 'Target', 'y', 'Y', 'class', 'Class']:
                if possible_label in df.columns:
                    label_col = possible_label
                    print(f"使用 '{label_col}' 作为标签列")
                    break
            else:
                print("警告: 未找到标签列，请手动指定")
        
        # 排除非特征列
        excluded_cols = [label_col]
        non_feature_cols = ['id', 'ID', '发生样点纬度', '发生样点经度', 'year', 'Year']
        for col in non_feature_cols:
            if col in df.columns:
                excluded_cols.append(col)
        
        feature_cols = [col for col in df.columns if col not in excluded_cols]
        
        print(f"\n特征列数量: {len(feature_cols)}")
        print(f"标签列: {label_col}")
        
        return df, feature_cols, label_col
        
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None, [], ''

def analyze_data_distribution(df, label_col):
    """分析标签分布情况"""
    # 标签分布
    label_counts = df[label_col].value_counts()
    print("\n标签分布:")
    print(label_counts)
    
    # 计算类别比例
    label_ratio = label_counts / len(df)
    print("\n标签比例:")
    print(label_ratio)
    
    # 判断是否存在严重不平衡
    min_ratio = label_ratio.min()
    max_ratio = label_ratio.max()
    imbalance_ratio = max_ratio / min_ratio if min_ratio > 0 else float('inf')
    print(f"\n不平衡比例: {imbalance_ratio:.2f}")
    
    if imbalance_ratio > 10:
        print("警告: 数据存在严重不平衡问题!")
    elif imbalance_ratio > 3:
        print("注意: 数据存在不平衡问题")
    else:
        print("数据平衡性良好")
    
    # 绘制标签分布图
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x=df[label_col])
    plt.title('标签分布情况')
    plt.xlabel('标签值')
    plt.ylabel('样本数量')
    
    # 添加数值标签
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'bottom', 
                   fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'label_distribution.png'))
    plt.close()

def check_missing_values(df, feature_cols):
    """检查缺失值"""
    missing_values = df[feature_cols].isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    
    missing_data = pd.DataFrame({
        '缺失值数量': missing_values,
        '缺失百分比': missing_percent
    })
    
    missing_data = missing_data[missing_data['缺失值数量'] > 0].sort_values('缺失百分比', ascending=False)
    
    if len(missing_data) > 0:
        print("\n存在缺失值的特征:")
        print(missing_data)
        
        # 绘制缺失值热图
        plt.figure(figsize=(12, 8))
        sns.heatmap(df[feature_cols].isnull(), cbar=False, yticklabels=False, cmap='viridis')
        plt.title('缺失值热图')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'missing_values_heatmap.png'))
        plt.close()
        
        # 绘制缺失值百分比
        if len(missing_data) < 20:  # 如果缺失值特征不太多，绘制条形图
            plt.figure(figsize=(12, 6))
            missing_data['缺失百分比'].plot(kind='bar')
            plt.title('特征缺失值百分比')
            plt.ylabel('缺失百分比 (%)')
            plt.xlabel('特征')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'missing_values_percent.png'))
            plt.close()
    else:
        print("\n数据完整，无缺失值")

def analyze_feature_distribution(df, feature_cols):
    """分析特征分布情况"""
    # 特征统计描述
    desc = df[feature_cols].describe()
    print("\n特征统计描述:")
    print(desc)
    
    # 检查异常值和极端值
    Q1 = df[feature_cols].quantile(0.25)
    Q3 = df[feature_cols].quantile(0.75)
    IQR = Q3 - Q1
    
    outliers = ((df[feature_cols] < (Q1 - 1.5 * IQR)) | (df[feature_cols] > (Q3 + 1.5 * IQR))).sum()
    
    outlier_percent = (outliers / len(df)) * 100
    
    outlier_data = pd.DataFrame({
        '异常值数量': outliers,
        '异常值百分比': outlier_percent
    })
    
    outlier_data = outlier_data[outlier_data['异常值数量'] > 0].sort_values('异常值百分比', ascending=False)
    
    if len(outlier_data) > 0:
        print("\n存在异常值的特征:")
        print(outlier_data)
        
        # 绘制异常值百分比
        if len(outlier_data) < 20:  # 如果异常值特征不太多，绘制条形图
            plt.figure(figsize=(12, 6))
            outlier_data['异常值百分比'].plot(kind='bar')
            plt.title('特征异常值百分比')
            plt.ylabel('异常值百分比 (%)')
            plt.xlabel('特征')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'outlier_percent.png'))
            plt.close()
    else:
        print("\n数据中未检测到异常值")
    
    # 选择部分特征绘制分布图
    sample_features = feature_cols[:min(5, len(feature_cols))]
    
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(sample_features, 1):
        plt.subplot(2, 3, i)
        sns.histplot(df[feature], kde=True)
        plt.title(f'{feature} 分布')
        plt.xlabel(feature)
        plt.ylabel('频数')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_distributions.png'))
    plt.close()

def check_correlation(df, feature_cols, label_col):
    """检查特征间相关性以及与标签的相关性"""
    # 计算相关系数矩阵
    corr_matrix = df[feature_cols + [label_col]].corr()
    
    # 与标签相关性
    label_correlation = corr_matrix[label_col].drop(label_col).sort_values(ascending=False)
    
    print("\n与标签相关性最高的特征:")
    print(label_correlation.head(10))
    
    print("\n与标签相关性最低的特征:")
    print(label_correlation.tail(10))
    
    # 绘制与标签相关性图
    plt.figure(figsize=(12, 8))
    label_correlation.plot(kind='bar')
    plt.title(f'特征与{label_col}的相关性')
    plt.ylabel('相关系数')
    plt.xlabel('特征')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'label_correlation.png'))
    plt.close()
    
    # 绘制特征相关性热图
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=False)
    plt.title('特征相关性热图')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_correlation_heatmap.png'))
    plt.close()
    
    # 高度相关特征对（检测多重共线性）
    high_corr_threshold = 0.8
    high_corr_pairs = []
    
    # 只检查上三角矩阵，避免重复
    for i in range(len(feature_cols)):
        for j in range(i+1, len(feature_cols)):
            if abs(corr_matrix.iloc[i, j]) >= high_corr_threshold:
                high_corr_pairs.append((corr_matrix.index[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    
    if high_corr_pairs:
        print("\n高度相关的特征对 (|r| >= 0.8):")
        for feat1, feat2, corr in high_corr_pairs:
            print(f"{feat1} <-> {feat2}: {corr:.4f}")
        print("\n注意: 高度相关的特征可能导致多重共线性问题")
    else:
        print("\n未发现高度相关的特征对")

def analyze_data_split(df, feature_cols, label_col):
    """分析数据拆分后的平衡性"""
    X = df[feature_cols].values
    y = df[label_col].values
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 数据分割
    try:
        # 按照8:1:1的比例划分训练、验证和测试集
        X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
        
        # 检查各集合的标签分布
        train_label_dist = pd.Series(y_train).value_counts(normalize=True)
        val_label_dist = pd.Series(y_val).value_counts(normalize=True)
        test_label_dist = pd.Series(y_test).value_counts(normalize=True)
        
        print("\n数据拆分后的标签分布:")
        print(f"训练集 (n={len(y_train)}):")
        print(train_label_dist)
        print(f"\n验证集 (n={len(y_val)}):")
        print(val_label_dist)
        print(f"\n测试集 (n={len(y_test)}):")
        print(test_label_dist)
        
        # 绘制数据拆分后的标签分布
        plt.figure(figsize=(12, 8))
        
        datasets = ['训练集', '验证集', '测试集']
        class_0_ratio = [
            (y_train == 0).sum() / len(y_train),
            (y_val == 0).sum() / len(y_val),
            (y_test == 0).sum() / len(y_test)
        ]
        class_1_ratio = [
            (y_train == 1).sum() / len(y_train),
            (y_val == 1).sum() / len(y_val),
            (y_test == 1).sum() / len(y_test)
        ]
        
        x = np.arange(len(datasets))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, class_0_ratio, width, label='类别 0')
        ax.bar(x + width/2, class_1_ratio, width, label='类别 1')
        
        ax.set_ylabel('比例')
        ax.set_title('数据集拆分后的标签分布')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.legend()
        
        # 添加数值标签
        for i, v in enumerate(class_0_ratio):
            ax.text(i - width/2, v + 0.01, f'{v:.2f}', ha='center', va='bottom')
        for i, v in enumerate(class_1_ratio):
            ax.text(i + width/2, v + 0.01, f'{v:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'data_split_distribution.png'))
        plt.close()
        
        # 检查特征分布 - 绘制训练集与测试集的特征分布比较
        sampled_features = np.random.choice(feature_cols, min(3, len(feature_cols)), replace=False)
        
        plt.figure(figsize=(15, 5 * len(sampled_features)))
        for i, feature in enumerate(sampled_features):
            plt.subplot(len(sampled_features), 1, i+1)
            sns.kdeplot(df[df.index.isin(y_train.index)][feature], label='训练集')
            sns.kdeplot(df[df.index.isin(y_test.index)][feature], label='测试集')
            plt.title(f'{feature} 在训练集和测试集中的分布')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'train_test_feature_distribution.png'))
        plt.close()
        
    except Exception as e:
        print(f"数据拆分分析失败: {e}")

def generate_data_report(df, feature_cols, label_col):
    """生成数据报告文件"""
    report_path = os.path.join(output_dir, 'data_quality_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=======================================\n")
        f.write("       数据质量与分布分析报告          \n")
        f.write("=======================================\n\n")
        
        f.write(f"数据形状: {df.shape[0]} 行 x {df.shape[1]} 列\n")
        f.write(f"特征数量: {len(feature_cols)}\n")
        f.write(f"标签列: {label_col}\n\n")
        
        # 标签分布
        label_counts = df[label_col].value_counts()
        f.write("标签分布:\n")
        for label, count in label_counts.items():
            f.write(f"  类别 {label}: {count} 样本 ({count/len(df)*100:.2f}%)\n")
        
        # 不平衡分析
        min_ratio = (label_counts / len(df)).min()
        max_ratio = (label_counts / len(df)).max()
        imbalance_ratio = max_ratio / min_ratio if min_ratio > 0 else float('inf')
        f.write(f"\n类别不平衡比例: {imbalance_ratio:.2f}\n")
        
        if imbalance_ratio > 10:
            f.write("警告: 数据存在严重不平衡问题!\n")
        elif imbalance_ratio > 3:
            f.write("注意: 数据存在不平衡问题\n")
        else:
            f.write("数据平衡性良好\n")
        
        # 缺失值分析
        missing_values = df[feature_cols].isnull().sum()
        missing_percent = (missing_values / len(df)) * 100
        missing_data = pd.DataFrame({
            '缺失值数量': missing_values,
            '缺失百分比': missing_percent
        })
        missing_data = missing_data[missing_data['缺失值数量'] > 0].sort_values('缺失百分比', ascending=False)
        
        if len(missing_data) > 0:
            f.write("\n缺失值分析:\n")
            f.write(missing_data.to_string())
            f.write("\n\n建议: 考虑对缺失值进行填充或删除高缺失率的特征\n")
        else:
            f.write("\n数据完整，无缺失值\n")
        
        # 特征与标签相关性
        try:
            corr_matrix = df[feature_cols + [label_col]].corr()
            label_correlation = corr_matrix[label_col].drop(label_col).sort_values(ascending=False)
            
            f.write("\n与标签相关性最高的前10个特征:\n")
            for feat, corr in label_correlation.head(10).items():
                f.write(f"  {feat}: {corr:.4f}\n")
            
            # 高度相关特征对
            high_corr_threshold = 0.8
            high_corr_pairs = []
            
            for i in range(len(feature_cols)):
                for j in range(i+1, len(feature_cols)):
                    if abs(corr_matrix.iloc[i, j]) >= high_corr_threshold:
                        high_corr_pairs.append((corr_matrix.index[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
            
            if high_corr_pairs:
                f.write("\n高度相关的特征对 (|r| >= 0.8):\n")
                for feat1, feat2, corr in high_corr_pairs[:10]:  # 只显示前10对
                    f.write(f"  {feat1} <-> {feat2}: {corr:.4f}\n")
                
                if len(high_corr_pairs) > 10:
                    f.write(f"  ... 以及其他 {len(high_corr_pairs) - 10} 对\n")
                
                f.write("\n建议: 考虑移除高度相关的特征以减少多重共线性\n")
            else:
                f.write("\n未发现高度相关的特征对\n")
                
        except Exception as e:
            f.write(f"\n相关性分析失败: {e}\n")
        
        # 总结与建议
        f.write("\n=======================================\n")
        f.write("            总结与建议                 \n")
        f.write("=======================================\n\n")
        
        if imbalance_ratio > 3:
            f.write("1. 数据不平衡: 考虑使用以下技术:\n")
            f.write("   - 过采样少数类 (SMOTE, ADASYN)\n")
            f.write("   - 欠采样多数类\n")
            f.write("   - 类别权重调整\n")
            f.write("   - Focal Loss\n")
        
        if len(missing_data) > 0:
            f.write("\n2. 缺失值处理:\n")
            f.write("   - 均值/中位数/众数填充\n")
            f.write("   - 使用模型预测填充\n")
            f.write("   - 考虑删除缺失率高的特征\n")
        
        if high_corr_pairs and len(high_corr_pairs) > 0:
            f.write("\n3. 多重共线性:\n")
            f.write("   - 使用特征选择技术\n")
            f.write("   - 考虑主成分分析 (PCA)\n")
            f.write("   - 使用正则化方法 (L1, L2)\n")
        
        f.write("\n4. 模型选择建议:\n")
        if imbalance_ratio > 10:
            f.write("   - 推荐使用对不平衡数据鲁棒的模型\n")
            f.write("   - 集成方法如随机森林、梯度提升\n")
            f.write("   - 调整损失函数以关注少数类\n")
        else:
            f.write("   - 标准机器学习模型应该表现良好\n")
            f.write("   - 考虑使用验证技术确保泛化能力\n")
        
        f.write("\n=======================================\n")
        f.write("          报告生成完毕                 \n")
        f.write("=======================================\n")
    
    print(f"\n数据质量报告已生成: {report_path}")

def main():
    print("开始数据质量分析...")
    
    # 加载数据
    df, feature_cols, label_col = load_data()
    
    if df is None:
        print("数据加载失败，终止分析")
        return
    
    # 分析标签分布
    analyze_data_distribution(df, label_col)
    
    # 检查缺失值
    check_missing_values(df, feature_cols)
    
    # 分析特征分布
    analyze_feature_distribution(df, feature_cols)
    
    # 检查相关性
    check_correlation(df, feature_cols, label_col)
    
    # 分析数据拆分
    analyze_data_split(df, feature_cols, label_col)
    
    # 生成数据报告
    generate_data_report(df, feature_cols, label_col)
    
    print(f"\n所有分析结果已保存到: {output_dir}")
    print("数据质量分析完成!")

if __name__ == "__main__":
    main() 