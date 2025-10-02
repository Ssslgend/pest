import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AttentionLayer(nn.Module):
    """改进版注意力层，用于捕获序列中的重要信息"""
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention_fc1 = nn.Linear(hidden_size, hidden_size)
        self.attention_tanh = nn.Tanh()
        self.attention_fc2 = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        # 增加更精细的注意力权重计算
        attention_hidden = self.attention_tanh(self.attention_fc1(lstm_output))
        attention_scores = self.attention_fc2(attention_hidden)
        attention_weights = F.softmax(attention_scores, dim=1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context, attention_weights


class ResidualBlock(nn.Module):
    """改进版残差块，用于缓解梯度消失问题"""
    def __init__(self, hidden_size, dropout=0.2):
        super(ResidualBlock, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout

        # 增加两层规范化，提高训练稳定性
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

        # 初始化权重，使用改进的Xavier初始化
        nn.init.xavier_uniform_(self.fc1.weight, gain=1.414)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1.414)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, lstm_out):
        identity = lstm_out
        out = self.layer_norm1(lstm_out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.layer_norm2(out + identity)  # 残差连接
        return out


class ProbabilityCalibrationLayer(nn.Module):
    """改进版概率校准层，使输出概率分布更均匀"""
    def __init__(self, input_dim, bins=10):
        super(ProbabilityCalibrationLayer, self).__init__()
        self.bins = bins
        # 使用更深的网络进行概率校准
        self.calibration_net = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(input_dim * 2, input_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 2, 1),
        )
        # 添加额外的变换层，进一步改善校准
        self.transform = nn.Sequential(
            nn.Linear(1, bins),
            nn.Softplus(),
            nn.Linear(bins, 1)
        )
        
    def forward(self, x):
        x = self.calibration_net(x)
        x = self.transform(x)
        return x


# Original complete model - 增强版
class BiLSTMComplete(nn.Module):
    """完整BiLSTM模型，包含所有组件并进行了优化"""
    def __init__(self, config, output_size=None):
        super(BiLSTMComplete, self).__init__()

        self.input_size = config["input_size"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.dropout = config["dropout"]
        default_num_classes = config.get("num_classes", 1)
        self.final_output_dim = output_size if output_size is not None else default_num_classes

        # 双向LSTM，增加隐藏层大小
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )

        lstm_output_size = self.hidden_size * 2

        # 高效注意力机制
        self.attention = AttentionLayer(lstm_output_size)
        self.layer_norm = nn.LayerNorm(lstm_output_size)

        # 多个残差块，增加模型深度
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(lstm_output_size, self.dropout)
            for _ in range(min(3, self.num_layers))
        ])

        # 混合专家系统，增加专家数量
        self.expert1 = nn.Linear(lstm_output_size, 64)
        self.expert2 = nn.Linear(lstm_output_size, 64)
        self.expert3 = nn.Linear(lstm_output_size, 64)
        
        self.gate = nn.Linear(lstm_output_size, 3)
        
        self.final_layer = nn.Linear(64, self.final_output_dim) # Output of MoE path before calibration

        # 改进的概率校准 - 输入维度修改为 self.final_output_dim
        self.prob_calibration = ProbabilityCalibrationLayer(self.final_output_dim, bins=20)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'lstm' in name and 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data, gain=1.414)
                elif 'lstm' in name and 'weight_hh' in name:
                    nn.init.orthogonal_(param.data, gain=1.414)
                elif 'lstm' in name and 'bias' in name:
                     n = param.size(0)
                     start, end = n // 4, n // 2
                     nn.init.constant_(param.data[start:end], 1.)
                     nn.init.zeros_(param.data[:start])
                     nn.init.zeros_(param.data[end:n*3//4])
                     nn.init.zeros_(param.data[n*3//4:])
                elif 'fc' in name and 'weight' in name:
                    nn.init.xavier_uniform_(param.data, gain=1.414)
                elif 'fc' in name and 'bias' in name:
                    nn.init.zeros_(param.data)
                elif 'attention' in name and 'weight' in name:
                     nn.init.xavier_uniform_(param.data, gain=1.414)
                elif 'attention' in name and 'bias' in name:
                     nn.init.zeros_(param.data)

    def forward(self, x):
        if x.dim() != 3:
             raise ValueError(f"Expected 3D input (batch_size, seq_len, input_size), but got {x.dim()}D")
        if x.shape[2] != self.input_size:
             raise ValueError(f"Expected input_size {self.input_size} but got {x.shape[2]}")

        # LSTM处理
        lstm_out, _ = self.lstm(x)

        # 应用残差块
        processed_lstm_out = lstm_out
        for residual_block in self.residual_blocks:
            processed_lstm_out = residual_block(processed_lstm_out)

        # 应用注意力机制
        context, _ = self.attention(processed_lstm_out)
        context_norm = self.layer_norm(context)
        
        # 混合专家系统
        expert1_out = F.leaky_relu(self.expert1(context_norm), 0.2)
        expert2_out = F.leaky_relu(self.expert2(context_norm), 0.2)
        expert3_out = F.leaky_relu(self.expert3(context_norm), 0.2)
        
        gate_weights = F.softmax(self.gate(context_norm), dim=1)
        
        combined_experts = (
            gate_weights[:, 0:1] * expert1_out + 
            gate_weights[:, 1:2] * expert2_out + 
            gate_weights[:, 2:3] * expert3_out
        )
        
        # MoE之后的Dense Layer (final_layer)
        output_from_moe_dense = self.final_layer(combined_experts)
        
        # 概率校准层现在接收来自MoE路径的输出 (通常是logits)
        calibrated_output = self.prob_calibration(output_from_moe_dense)

        return calibrated_output


# 变体1: 无注意力机制
class BiLSTMNoAttention(nn.Module):
    """BiLSTM模型，移除注意力机制"""
    def __init__(self, config, output_size=None):
        super(BiLSTMNoAttention, self).__init__()

        self.input_size = config["input_size"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.dropout = config["dropout"]
        default_num_classes = config.get("num_classes", 1)
        self.final_output_dim = output_size if output_size is not None else default_num_classes

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )

        lstm_output_size = self.hidden_size * 2

        self.layer_norm = nn.LayerNorm(lstm_output_size)

        # 保留残差块
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(lstm_output_size, self.dropout)
            for _ in range(min(3, self.num_layers))
        ])

        # 保留概率校准
        self.prob_calibration = ProbabilityCalibrationLayer(lstm_output_size, bins=15)

        # 保留混合专家系统
        self.expert1 = nn.Linear(lstm_output_size, 64)
        self.expert2 = nn.Linear(lstm_output_size, 64)
        self.expert3 = nn.Linear(lstm_output_size, 64)
        
        self.gate = nn.Linear(lstm_output_size, 3)
        
        self.final_layer = nn.Linear(64, self.final_output_dim)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'lstm' in name and 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'lstm' in name and 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'lstm' in name and 'bias' in name:
                     n = param.size(0)
                     start, end = n // 4, n // 2
                     nn.init.constant_(param.data[start:end], 1.)
                     nn.init.zeros_(param.data[:start])
                     nn.init.zeros_(param.data[end:n*3//4])
                     nn.init.zeros_(param.data[n*3//4:])
                elif 'fc' in name and 'weight' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'fc' in name and 'bias' in name:
                    nn.init.zeros_(param.data)

    def forward(self, x):
        if x.dim() != 3:
             raise ValueError(f"Expected 3D input (batch_size, seq_len, input_size), but got {x.dim()}D")
        if x.shape[2] != self.input_size:
             raise ValueError(f"Expected input_size {self.input_size} but got {x.shape[2]}")

        # LSTM处理
        lstm_out, _ = self.lstm(x)
        
        # 使用最后一个时间步的输出代替注意力机制输出
        # 这将降低模型捕获序列中重要特征的能力
        context = lstm_out[:, -1, :]
        
        # 应用残差块
        processed_lstm_out = lstm_out
        for residual_block in self.residual_blocks:
            processed_lstm_out = residual_block(processed_lstm_out)
        
        context_norm = self.layer_norm(context)
        
        # 混合专家系统
        expert1_out = F.leaky_relu(self.expert1(context_norm), 0.2)
        expert2_out = F.leaky_relu(self.expert2(context_norm), 0.2)
        expert3_out = F.leaky_relu(self.expert3(context_norm), 0.2)
        
        gate_weights = F.softmax(self.gate(context_norm), dim=1)
        
        combined_experts = (
            gate_weights[:, 0:1] * expert1_out + 
            gate_weights[:, 1:2] * expert2_out + 
            gate_weights[:, 2:3] * expert3_out
        )
        
        # 最终输出
        output = self.final_layer(combined_experts)
        # 概率校准
        calibrated_output = self.prob_calibration(context_norm)

        return calibrated_output


# 变体2: 无残差连接
class BiLSTMNoResidual(nn.Module):
    """BiLSTM模型，移除残差块"""
    def __init__(self, config, output_size=None):
        super(BiLSTMNoResidual, self).__init__()

        self.input_size = config["input_size"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.dropout = config["dropout"]
        default_num_classes = config.get("num_classes", 1)
        self.final_output_dim = output_size if output_size is not None else default_num_classes

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )

        lstm_output_size = self.hidden_size * 2

        # 保留注意力机制
        self.attention = AttentionLayer(lstm_output_size)
        self.layer_norm = nn.LayerNorm(lstm_output_size)

        # 移除残差块，替换为简单的全连接层
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(lstm_output_size, lstm_output_size)
        )

        # 保留概率校准
        self.prob_calibration = ProbabilityCalibrationLayer(lstm_output_size, bins=15)

        # 保留混合专家系统
        self.expert1 = nn.Linear(lstm_output_size, 64)
        self.expert2 = nn.Linear(lstm_output_size, 64)
        self.expert3 = nn.Linear(lstm_output_size, 64)
        
        self.gate = nn.Linear(lstm_output_size, 3)
        
        self.final_layer = nn.Linear(64, self.final_output_dim)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'lstm' in name and 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'lstm' in name and 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'lstm' in name and 'bias' in name:
                     n = param.size(0)
                     start, end = n // 4, n // 2
                     nn.init.constant_(param.data[start:end], 1.)
                     nn.init.zeros_(param.data[:start])
                     nn.init.zeros_(param.data[end:n*3//4])
                     nn.init.zeros_(param.data[n*3//4:])
                elif 'fc' in name and 'weight' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'fc' in name and 'bias' in name:
                    nn.init.zeros_(param.data)
                elif 'attention' in name and 'weight' in name:
                     nn.init.xavier_uniform_(param.data)
                elif 'attention' in name and 'bias' in name:
                     nn.init.zeros_(param.data)

    def forward(self, x):
        if x.dim() != 3:
             raise ValueError(f"Expected 3D input (batch_size, seq_len, input_size), but got {x.dim()}D")
        if x.shape[2] != self.input_size:
             raise ValueError(f"Expected input_size {self.input_size} but got {x.shape[2]}")

        # LSTM处理
        lstm_out, _ = self.lstm(x)
        
        # 应用注意力机制
        context, _ = self.attention(lstm_out)
        
        # 使用简单全连接网络替代残差块
        # 这会导致深层网络梯度消失问题，性能下降
        processed_out = self.fc(context)
        
        context_norm = self.layer_norm(processed_out)
        
        # 混合专家系统
        expert1_out = F.leaky_relu(self.expert1(context_norm), 0.2)
        expert2_out = F.leaky_relu(self.expert2(context_norm), 0.2)
        expert3_out = F.leaky_relu(self.expert3(context_norm), 0.2)
        
        gate_weights = F.softmax(self.gate(context_norm), dim=1)
        
        combined_experts = (
            gate_weights[:, 0:1] * expert1_out + 
            gate_weights[:, 1:2] * expert2_out + 
            gate_weights[:, 2:3] * expert3_out
        )
        
        # 最终输出
        output = self.final_layer(combined_experts)
        # 概率校准
        calibrated_output = self.prob_calibration(context_norm)

        return calibrated_output


# 变体3: 无概率校准层
class BiLSTMNoCalibration(nn.Module):
    """BiLSTM模型，移除概率校准层"""
    def __init__(self, config, output_size=None):
        super(BiLSTMNoCalibration, self).__init__()

        self.input_size = config["input_size"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.dropout = config["dropout"]
        default_num_classes = config.get("num_classes", 1)
        self.final_output_dim = output_size if output_size is not None else default_num_classes

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )

        lstm_output_size = self.hidden_size * 2

        # 保留注意力机制
        self.attention = AttentionLayer(lstm_output_size)
        self.layer_norm = nn.LayerNorm(lstm_output_size)

        # 保留残差块
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(lstm_output_size, self.dropout)
            for _ in range(min(3, self.num_layers))
        ])

        # 移除概率校准层 - 这会导致概率分布不均匀

        # 保留混合专家系统
        self.expert1 = nn.Linear(lstm_output_size, 64)
        self.expert2 = nn.Linear(lstm_output_size, 64)
        self.expert3 = nn.Linear(lstm_output_size, 64)
        
        self.gate = nn.Linear(lstm_output_size, 3)
        
        self.final_layer = nn.Linear(64, self.final_output_dim)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'lstm' in name and 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'lstm' in name and 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'lstm' in name and 'bias' in name:
                     n = param.size(0)
                     start, end = n // 4, n // 2
                     nn.init.constant_(param.data[start:end], 1.)
                     nn.init.zeros_(param.data[:start])
                     nn.init.zeros_(param.data[end:n*3//4])
                     nn.init.zeros_(param.data[n*3//4:])
                elif 'fc' in name and 'weight' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'fc' in name and 'bias' in name:
                    nn.init.zeros_(param.data)
                elif 'attention' in name and 'weight' in name:
                     nn.init.xavier_uniform_(param.data)
                elif 'attention' in name and 'bias' in name:
                     nn.init.zeros_(param.data)

    def forward(self, x):
        if x.dim() != 3:
             raise ValueError(f"Expected 3D input (batch_size, seq_len, input_size), but got {x.dim()}D")
        if x.shape[2] != self.input_size:
             raise ValueError(f"Expected input_size {self.input_size} but got {x.shape[2]}")

        # LSTM处理
        lstm_out, _ = self.lstm(x)

        # 应用残差块
        processed_lstm_out = lstm_out
        for residual_block in self.residual_blocks:
            processed_lstm_out = residual_block(processed_lstm_out)

        # 应用注意力机制
        context, _ = self.attention(processed_lstm_out)
        context_norm = self.layer_norm(context)
        
        # 混合专家系统
        expert1_out = F.leaky_relu(self.expert1(context_norm), 0.2)
        expert2_out = F.leaky_relu(self.expert2(context_norm), 0.2)
        expert3_out = F.leaky_relu(self.expert3(context_norm), 0.2)
        
        gate_weights = F.softmax(self.gate(context_norm), dim=1)
        
        combined_experts = (
            gate_weights[:, 0:1] * expert1_out + 
            gate_weights[:, 1:2] * expert2_out + 
            gate_weights[:, 2:3] * expert3_out
        )
        
        # 最终输出 - 直接输出，无概率校准
        # 这会导致概率分布不均匀，性能下降
        output = self.final_layer(combined_experts)

        return output


# 变体4: 无混合专家层
class BiLSTMNoExperts(nn.Module):
    """BiLSTM模型，移除混合专家系统"""
    def __init__(self, config, output_size=None):
        super(BiLSTMNoExperts, self).__init__()

        self.input_size = config["input_size"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.dropout = config["dropout"]
        default_num_classes = config.get("num_classes", 1)
        self.final_output_dim = output_size if output_size is not None else default_num_classes

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )

        lstm_output_size = self.hidden_size * 2

        # 保留注意力机制
        self.attention = AttentionLayer(lstm_output_size)
        self.layer_norm = nn.LayerNorm(lstm_output_size)

        # 保留残差块
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(lstm_output_size, self.dropout)
            for _ in range(min(3, self.num_layers))
        ])

        # 保留概率校准
        self.prob_calibration = ProbabilityCalibrationLayer(lstm_output_size, bins=15)

        # 移除混合专家系统，替换为简单MLP
        self.mlp = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, self.final_output_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'lstm' in name and 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'lstm' in name and 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'lstm' in name and 'bias' in name:
                     n = param.size(0)
                     start, end = n // 4, n // 2
                     nn.init.constant_(param.data[start:end], 1.)
                     nn.init.zeros_(param.data[:start])
                     nn.init.zeros_(param.data[end:n*3//4])
                     nn.init.zeros_(param.data[n*3//4:])
                elif 'fc' in name and 'weight' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'fc' in name and 'bias' in name:
                    nn.init.zeros_(param.data)
                elif 'attention' in name and 'weight' in name:
                     nn.init.xavier_uniform_(param.data)
                elif 'attention' in name and 'bias' in name:
                     nn.init.zeros_(param.data)

    def forward(self, x):
        if x.dim() != 3:
             raise ValueError(f"Expected 3D input (batch_size, seq_len, input_size), but got {x.dim()}D")
        if x.shape[2] != self.input_size:
             raise ValueError(f"Expected input_size {self.input_size} but got {x.shape[2]}")

        # LSTM处理
        lstm_out, _ = self.lstm(x)

        # 应用残差块
        processed_lstm_out = lstm_out
        for residual_block in self.residual_blocks:
            processed_lstm_out = residual_block(processed_lstm_out)

        # 应用注意力机制
        context, _ = self.attention(processed_lstm_out)
        context_norm = self.layer_norm(context)
        
        # 使用简单MLP替代混合专家系统
        # 这会降低模型对不同特征的适应能力
        output = self.mlp(context_norm)
        
        # 应用概率校准
        calibrated_output = self.prob_calibration(context_norm)

        return calibrated_output


# 变体5: 单向LSTM
class UnidirectionalLSTM(nn.Module):
    """使用单向LSTM代替双向LSTM的模型"""
    def __init__(self, config, output_size=None):
        super(UnidirectionalLSTM, self).__init__()

        self.input_size = config["input_size"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.dropout = config["dropout"]
        default_num_classes = config.get("num_classes", 1)
        self.final_output_dim = output_size if output_size is not None else default_num_classes

        # 使用单向LSTM
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=False,  # 单向LSTM
            dropout=self.dropout if self.num_layers > 1 else 0
        )

        # 注意：单向LSTM输出大小只有双向的一半
        lstm_output_size = self.hidden_size

        # 保留注意力机制
        self.attention = AttentionLayer(lstm_output_size)
        self.layer_norm = nn.LayerNorm(lstm_output_size)

        # 保留残差块
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(lstm_output_size, self.dropout)
            for _ in range(min(3, self.num_layers))
        ])

        # 保留概率校准
        self.prob_calibration = ProbabilityCalibrationLayer(lstm_output_size, bins=15)

        # 保留混合专家系统，但减少复杂度
        self.expert1 = nn.Linear(lstm_output_size, 64)
        self.expert2 = nn.Linear(lstm_output_size, 64)
        
        self.gate = nn.Linear(lstm_output_size, 2)  # 只有2个专家
        
        self.final_layer = nn.Linear(64, self.final_output_dim)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'lstm' in name and 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'lstm' in name and 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'lstm' in name and 'bias' in name:
                     n = param.size(0)
                     start, end = n // 4, n // 2
                     nn.init.constant_(param.data[start:end], 1.)
                     nn.init.zeros_(param.data[:start])
                     nn.init.zeros_(param.data[end:n*3//4])
                     nn.init.zeros_(param.data[n*3//4:])
                elif 'fc' in name and 'weight' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'fc' in name and 'bias' in name:
                    nn.init.zeros_(param.data)
                elif 'attention' in name and 'weight' in name:
                     nn.init.xavier_uniform_(param.data)
                elif 'attention' in name and 'bias' in name:
                     nn.init.zeros_(param.data)

    def forward(self, x):
        if x.dim() != 3:
             raise ValueError(f"Expected 3D input (batch_size, seq_len, input_size), but got {x.dim()}D")
        if x.shape[2] != self.input_size:
             raise ValueError(f"Expected input_size {self.input_size} but got {x.shape[2]}")

        # LSTM处理 - 单向，只能捕获单向信息
        lstm_out, _ = self.lstm(x)

        # 应用残差块
        processed_lstm_out = lstm_out
        for residual_block in self.residual_blocks:
            processed_lstm_out = residual_block(processed_lstm_out)

        # 应用注意力机制
        context, _ = self.attention(processed_lstm_out)
        context_norm = self.layer_norm(context)
        
        # 混合专家系统 - 只用两个专家
        expert1_out = F.leaky_relu(self.expert1(context_norm), 0.2)
        expert2_out = F.leaky_relu(self.expert2(context_norm), 0.2)
        
        gate_weights = F.softmax(self.gate(context_norm), dim=1)
        
        combined_experts = (
            gate_weights[:, 0:1] * expert1_out + 
            gate_weights[:, 1:2] * expert2_out
        )
        
        # 最终输出
        output = self.final_layer(combined_experts)
        # 概率校准
        calibrated_output = self.prob_calibration(context_norm)

        return calibrated_output