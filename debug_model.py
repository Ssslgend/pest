# debug_model.py
"""
调试BiLSTM模型的输出维度问题
"""

import torch
import pandas as pd
import joblib
from model.bilstm import BiLSTMModel

def debug_model():
    """调试模型输出维度"""
    print("调试BiLSTM模型输出维度...")
    
    # 加载数据检查标签
    data = pd.read_csv('datas/shandong_pest_data/spatial_train_data.csv')
    
    print(f"Value_Class 唯一值: {sorted(data['Value_Class'].unique())}")
    print(f"Value_Class 分布:")
    print(data['Value_Class'].value_counts().sort_index())
    
    # 特征列
    feature_columns = [
        'Temperature', 'Humidity', 'Rainfall', 'WS', 'WD', 'Pressure', 
        'Sunshine', 'Visibility', 'Temperature_MA', 'Humidity_MA', 
        'Rainfall_MA', 'Pressure_MA', 'Temp_7day_MA', 'Humidity_7day_MA', 
        'Rainfall_7day_MA', 'Temp_Change', 'Cumulative_Rainfall_7day', 
        'Temp_Humidity_Index'
    ]
    
    input_size = len(feature_columns)
    print(f"输入特征维度: {input_size}")
    
    # 创建模型配置
    model_config = {
        'input_size': input_size,
        'hidden_size': 128,
        'num_layers': 2,
        'num_classes': 3,  # 3个类别：1,2,3 -> 0,1,2
        'dropout': 0.3
    }
    
    # 创建模型
    print("\n创建BiLSTM模型...")
    model = BiLSTMModel(model_config)
    
    # 测试模型
    print("\n测试模型前向传播...")
    batch_size = 2
    seq_len = 8
    test_input = torch.randn(batch_size, seq_len, input_size)
    
    print(f"测试输入形状: {test_input.shape}")
    
    try:
        test_output = model(test_input)
        print(f"模型输出形状: {test_output.shape}")
        print(f"输出值范围: [{test_output.min():.4f}, {test_output.max():.4f}]")
        
        # 测试损失函数
        print("\n测试损失函数...")
        criterion = torch.nn.CrossEntropyLoss()
        
        # 模拟标签 (0, 1, 2)
        test_labels = torch.LongTensor([0, 1])
        print(f"测试标签: {test_labels}")
        print(f"标签范围: [{test_labels.min()}, {test_labels.max()}]")
        
        # 计算损失
        loss = criterion(test_output, test_labels)
        print(f"损失值: {loss.item():.4f}")
        
        # 测试预测
        _, predicted = torch.max(test_output, 1)
        print(f"预测结果: {predicted}")
        print(f"预测概率: {torch.softmax(test_output, dim=1)}")
        
        print("\n✅ 模型测试成功！")
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_model()