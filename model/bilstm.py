# model/bilstm.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    """注意力层，用于捕获序列中的重要信息"""
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
    """残差块，用于缓解梯度消失问题"""
    def __init__(self, hidden_size, dropout=0.2):
        super(ResidualBlock, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout # Renamed for clarity

        # 层归一化通常放在输入或残差连接之后
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size) # Second LayerNorm before final add

        # 初始化权重
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

class BiLSTMModel(nn.Module):
    """改进的双向LSTM模型，带有注意力机制和残差连接 (Adapted for Configurable Output)"""
    def __init__(self, config, output_size=None): # Allow overriding output size
        super(BiLSTMModel, self).__init__()

        # 模型参数
        self.input_size = config["input_size"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.dropout = config["dropout"]
        # Default num_classes from config if output_size not given, else prioritize output_size
        default_num_classes = config.get("num_classes", 1) # Default to 1 if not in config
        self.final_output_dim = output_size if output_size is not None else default_num_classes

        # BiLSTM层
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            # Apply dropout only between LSTM layers if num_layers > 1
            dropout=self.dropout if self.num_layers > 1 else 0
        )

        lstm_output_size = self.hidden_size * 2 # Because bidirectional=True

        # 注意力层
        self.attention = AttentionLayer(lstm_output_size)

        # 使用LayerNorm替代BatchNorm，更适合小批量和序列数据
        self.layer_norm = nn.LayerNorm(lstm_output_size)

        # 减少残差块数量，避免过拟合
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(lstm_output_size, self.dropout)
            for _ in range(min(1, self.num_layers)) # 使用1个残差块，进一步减少复杂度
        ])

        # 输出层 - 添加额外的隐藏层以增强表达能力
        self.fc1 = nn.Linear(lstm_output_size, lstm_output_size // 2)
        self.fc1_activation = nn.ReLU()
        self.fc1_dropout = nn.Dropout(self.dropout)
        self.fc2 = nn.Linear(lstm_output_size // 2, self.final_output_dim)

        # 输出激活函数 - 对于二分类问题使用Sigmoid
        self.output_activation = nn.Sigmoid() if self.final_output_dim == 1 else None

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
                     # LSTM biases: forget gate bias initialization (often set to 1 or small positive)
                     n = param.size(0)
                     start, end = n // 4, n // 2
                     nn.init.constant_(param.data[start:end], 1.) # Forget gate bias to 1
                     nn.init.zeros_(param.data[:start])          # Input gate bias
                     nn.init.zeros_(param.data[end:n*3//4])     # Cell gate bias
                     nn.init.zeros_(param.data[n*3//4:])        # Output gate bias
                elif 'fc' in name and 'weight' in name:
                    # Linear layer weights
                    nn.init.xavier_uniform_(param.data)
                elif 'fc' in name and 'bias' in name:
                    # Linear layer biases
                    nn.init.zeros_(param.data)
                # Add initialization for Attention layers if needed
                elif 'attention' in name and 'weight' in name:
                     nn.init.xavier_uniform_(param.data)
                elif 'attention' in name and 'bias' in name:
                     nn.init.zeros_(param.data)


    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        if x.dim() != 3:
             raise ValueError(f"Expected 3D input (batch_size, seq_len, input_size), but got {x.dim()}D")
        if x.shape[2] != self.input_size:
             raise ValueError(f"Expected input_size {self.input_size} but got {x.shape[2]}")

        # BiLSTM前向传播
        # Initialize hidden and cell states if needed (LSTM handles zero init by default)
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch_size, seq_len, hidden_size * 2)

        # 应用残差连接 - Applying to each timestep
        processed_lstm_out = lstm_out
        for residual_block in self.residual_blocks:
            processed_lstm_out = residual_block(processed_lstm_out)

        # 应用注意力机制 using the processed output
        context, _ = self.attention(processed_lstm_out)
        # context shape: (batch_size, hidden_size * 2)

        # 使用LayerNorm替代BatchNorm，避免小批量问题
        context = self.layer_norm(context)

        # 多层输出网络
        x = self.fc1(context)
        x = self.fc1_activation(x)
        x = self.fc1_dropout(x)
        output = self.fc2(x)

        # 应用输出激活函数（对于二分类问题）
        if self.output_activation:
            output = self.output_activation(output)

        # output shape: (batch_size, final_output_dim)
        return output
