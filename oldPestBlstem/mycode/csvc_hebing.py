import os
import pandas as pd
from dbfread import DBF

# 配置参数
dbf_folder = r'H:\data_new2025\baie\china_bianliang\china_weifenbu'  # 替换为您的DBF文件所在文件夹路径
output_csv = r'H:\data_new2025\baie\china_bianliang\china_19wei.csv'  # 输出文件名

def process_dbf_files(folder):
    # 获取所有DBF文件
    dbf_files = [f for f in os.listdir(folder) if f.lower().endswith('.dbf')]
    
    # 存储各文件数据的字典（key为文件名）
    data_dict = {}
    base_columns = None
    
    for file in dbf_files:
        # 读取DBF文件
        file_path = os.path.join(folder, file)
        dbf = DBF(file_path)
        
        # 获取文件名（不带扩展名）
        filename = os.path.splitext(file)[0]
        
        # 转换为DataFrame
        df = pd.DataFrame(iter(dbf))
        
        # 验证列数
        if len(df.columns) < 4:
            print(f"文件 {file} 列数不足，已跳过")
            continue
            
        # 提取前三列和第四列
        cols = df.columns.tolist()
        keep_cols = cols[:3] + [cols[3]]
        df = df[keep_cols]
        
        # 重命名第四列为文件名
        df.rename(columns={cols[3]: filename}, inplace=True)
        
        # 标准化前三列名称（假设所有文件前三列结构一致）
        if base_columns is None:
            base_columns = cols[:3]
            df.columns = ['col1', 'col2', 'col3', filename]
        else:
            df.columns = ['col1', 'col2', 'col3', filename]
        
        # 添加到字典
        data_dict[filename] = df

    # 合并所有数据（基于前三列）
    merged_df = None
    for name, df in data_dict.items():
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(
                merged_df, 
                df[['col1', 'col2', 'col3', name]],
                on=['col1', 'col2', 'col3'],
                how='outer'
            )
    
    return merged_df

# 执行处理
final_df = process_dbf_files(dbf_folder)

# 保存CSV文件
if final_df is not None:
    final_df.to_csv(output_csv, index=False)
    print(f"合并完成！已保存到 {output_csv}")
else:
    print("没有找到有效的DBF文件")