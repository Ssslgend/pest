# 模型配置
MODEL_CONFIG = {
    'input_size': None,  # 将在运行时根据数据维度设置
    'hidden_size': 128,  # BiLSTM隐藏层大小
    'num_layers': 2,     # BiLSTM层数
    'num_classes': 4,    # 4个风险等级
    'dropout': 0.5,
    'use_attention': True,   # 是否使用注意力机制
    'use_residual': True     # 是否使用残差块
}

# 训练配置
TRAIN_CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 100,
    'patience': 10,      # 早停耐心值
    'test_size': 0.2,    # 测试集比例
    'val_size': 0.2,     # 验证集比例
    'random_state': 42
}

# 数据配置
DATA_CONFIG = {
    'data_path': 'datas/pest_rice_with_features_2_classified.csv',
    'label_column': 'Value_Class',
    'sequence_length': 30  # 时间序列长度
}