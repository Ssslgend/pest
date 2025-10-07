# config/params.py
MODEL_CONFIG = {
    "input_size": 12,       # 特征维度：8个原始特征 + 4个复合特征
    "hidden_size": 256,     # LSTM隐藏单元数
    "num_layers": 4,        # LSTM层数
    "num_classes": 4,       # 分类类别数
    "dropout": 0.3,         # Dropout率
    "bidirectional": True,  # 保持双向
    "use_attention": False,   # 是否使用注意力机制
    "use_residual": True     # 是否使用残差块
}

TRAIN_CONFIG = {
    "batch_size": 16,       # batch size
    "num_epochs": 300,      # 训练轮数
    "learning_rate": 0.0003, # 学习率
    "seq_length": 8         # 时间步长
}