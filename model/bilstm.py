# model/bilstm.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AttentionLayer(nn.Module):
    """Attention layer for capturing important information in sequences"""
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        # Corrected: Ensure Linear layer input matches hidden_size
        self.attention_fc1 = nn.Linear(hidden_size, hidden_size)
        self.attention_tanh = nn.Tanh()
        self.attention_fc2 = nn.Linear(hidden_size, 1) # Output one attention score per timestep

    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_size)
        # Pass through first linear layer and tanh
        attention_hidden = self.attention_tanh(self.attention_fc1(lstm_output)) # (batch_size, seq_len, hidden_size)
        # Pass through second linear layer to get scores
        attention_scores = self.attention_fc2(attention_hidden)  # (batch_size, seq_len, 1)

        # Apply softmax to get weights
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_len, 1)

        # Apply attention weights to get context vector
        # Weighted sum of lstm_output based on attention_weights
        context = torch.sum(attention_weights * lstm_output, dim=1)  # (batch_size, hidden_size)
        return context, attention_weights


class ResidualBlock(nn.Module):
    """Residual block for alleviating gradient vanishing problem"""
    def __init__(self, hidden_size, dropout=0.2):
        super(ResidualBlock, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout # Renamed for clarity

        # Layer normalization typically placed after input or residual connection
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size * 2)  # Increase intermediate layer width
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)  # Return to original width
        self.layer_norm2 = nn.LayerNorm(hidden_size) # Second LayerNorm before final add

        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, lstm_out):
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        identity = lstm_out

        # Pre-normalization style (or apply norm before add)
        out = self.layer_norm1(lstm_out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        # out = self.dropout(out) # Optional dropout here too

        # Add residual connection and apply final normalization
        out = self.layer_norm2(out + identity)

        return out

# New probability distribution calibration layer
class ProbabilityCalibrationLayer(nn.Module):
    """Probability calibration layer to make output probability distribution more uniform"""
    def __init__(self, input_dim, bins=10):
        super(ProbabilityCalibrationLayer, self).__init__()
        self.bins = bins
        # Create learnable parameters to map diverse probabilities to more uniform distribution
        self.calibration_net = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(input_dim * 2, input_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(input_dim * 2, 1),
        )
        # Using pre-activation function transformer, not for final output sigmoid
        # but to provide stronger non-linear transformation capability
        self.transform = nn.Sequential(
            nn.Linear(1, bins),
            nn.Softplus(),
            nn.Linear(bins, 1)
        )
        
    def forward(self, x):
        # x is a single output value
        # Perform initial non-linear transformation
        x = self.calibration_net(x)
        # Apply piecewise non-linear transformation to enhance diversity
        x = self.transform(x)
        # Ensure values are in 0-1 range, but avoid using saturated regions of sigmoid
        # x = torch.clamp(x, 0.01, 0.99)
        return x

class BiLSTMModel(nn.Module):
    """BiLSTM模型 - 基于原始框架的简化版本"""
    def __init__(self, config):
        super(BiLSTMModel, self).__init__()
        
        # 从配置中获取参数
        self.input_size = config["input_size"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.num_classes = config["num_classes"]
        self.dropout = config["dropout"]
        
        # BiLSTM层 - 与原始框架一致
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )
        
        # 全连接层 - 与原始框架一致
        self.fc = nn.Linear(self.hidden_size * 2, self.num_classes)  # *2是因为双向LSTM
        
        # Dropout层 - 与原始框架一致
        self.dropout = nn.Dropout(self.dropout)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'lstm' in name and 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'lstm' in name and 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'lstm' in name and 'bias' in name:
                    # LSTM biases: forget gate bias initialization
                    n = param.size(0)
                    start, end = n // 4, n // 2
                    nn.init.constant_(param.data[start:end], 1.) # Forget gate bias to 1
                    nn.init.zeros_(param.data[:start])          # Input gate bias
                    nn.init.zeros_(param.data[end:n*3//4])     # Cell gate bias
                    nn.init.zeros_(param.data[n*3//4:])        # Output gate bias
                elif 'fc' in name and 'weight' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'fc' in name and 'bias' in name:
                    nn.init.zeros_(param.data)
        
    def forward(self, x):
        # x形状: (batch_size, seq_len, input_size)
        
        # 初始化隐藏状态 - 与原始框架一致
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))
        
        # 只取最后一个时间步的输出 - 与原始框架一致
        out = out[:, -1, :]
        
        # Dropout
        out = self.dropout(out)
        
        # 全连接层
        out = self.fc(out)
        
        return out
