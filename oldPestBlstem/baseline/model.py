import torch
import torch.nn as nn

class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # BiLSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全连接层
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2是因为双向LSTM
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x的形状: (batch_size, seq_len, input_size)
        
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # *2是因为双向LSTM
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))  # out的形状: (batch_size, seq_len, hidden_size*2)
        
        # 只取最后一个时间步的输出
        out = out[:, -1, :]  # 形状: (batch_size, hidden_size*2)
        
        # Dropout
        out = self.dropout(out)
        
        # 全连接层
        out = self.fc(out)  # 形状: (batch_size, num_classes)
        
        return out 