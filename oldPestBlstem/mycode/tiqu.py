import pandas as pd
from dbfread import DBF
import os

def process_data(input_csv, dbf_folder, output_csv):
    # 读取原始CSV文件
    try:
        df = pd.read_csv(input_csv, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(input_csv, encoding='gbk')
        except UnicodeDecodeError:
            df = pd.read_csv(input_csv, encoding='gb18030')
    
    print(f"CSV文件中的X,Y范围：")
    print(f"X范围：{df['X'].min()} - {df['X'].max()}")
    print(f"Y范围：{df['Y'].min()} - {df['Y'].max()}")
    
    # 遍历处理每个DBF文件
    for filename in os.listdir(dbf_folder):
        if filename.lower().endswith('.dbf'):
            filepath = os.path.join(dbf_folder, filename)
            
            try:
                # 读取DBF文件
                table = DBF(filepath)
                table_records = list(table)
                if len(table.fields) < 4:
                    print(f"跳过 {filename}：字段数不足4个")
                    continue
                
                # 获取第四个字段名
                target_col = table.fields[3].name
                print(f"当前处理文件: {filename}")
                print(f"新列名: {os.path.splitext(filename)[0]}")
                print(f"第四字段名: {target_col}")
                print(f"第四字段前5个值: {[record[target_col] for record in table_records[:5]]}")
                
                # 创建数据字典和去重
                data_dict = {}
                dbf_x_values = []
                dbf_y_values = []
                for record in table_records:
                    try:
                        x_val = float(record['X'])
                        y_val = float(record['Y'])
                        dbf_x_values.append(x_val)
                        dbf_y_values.append(y_val)
                        key = (x_val, y_val)
                        data_dict[key] = record[target_col]
                    except (KeyError, ValueError) as e:
                        print(f"跳过记录：{e}")
                        continue
                
                if dbf_x_values:
                    print(f"DBF文件中的X,Y范围：")
                    print(f"X范围：{min(dbf_x_values)} - {max(dbf_x_values)}")
                    print(f"Y范围：{min(dbf_y_values)} - {max(dbf_y_values)}")
                
                # 生成新列名（使用文件名不带扩展名）
                new_column = os.path.splitext(filename)[0]
                
                # 添加新列到DataFrame
                df[new_column] = df.apply(lambda row: data_dict.get(
                    (float(row['X']), float(row['Y'])), None), axis=1)
                
                # 打印匹配统计
                matched_count = df[new_column].notna().sum()
                print(f"匹配到的记录数：{matched_count}/{len(df)}")
                
            except Exception as e:
                print(f"处理 {filename} 时出错：{str(e)}")
                continue
    
    # 保存结果
    df.to_csv(output_csv, index=False)
    print(f"\n处理完成，结果已保存至 {output_csv}")

# 使用示例
process_data('E:/code/0424/pestBIstm/pestBIstm/datas/china_19.csv', 'H:/data_new2025/fpr/newdata', 'E:/code/0424/pestBIstm/pestBIstm/datas/china_19_1.csv')