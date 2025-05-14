# pestLSTM/model/lstm.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    """Unidirectional LSTM Model for Raster Prediction"""
    def __init__(self, config, output_size=None):
        super(LSTMModel, self).__init__()

        self.input_size = config["input_size"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.dropout = config["dropout"]
        default_num_classes = config.get("num_classes", 1)
        self.final_output_dim = output_size if output_size is not None else default_num_classes

        # LSTM layer (Unidirectional)
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=False, # Set to False for Unidirectional
            dropout=self.dropout if self.num_layers > 1 else 0
        )

        lstm_output_size = self.hidden_size # Output size is just hidden_size

        # Optional: Add BatchNorm or LayerNorm if needed
        # self.norm = nn.BatchNorm1d(lstm_output_size) # Example BatchNorm
        self.norm = nn.LayerNorm(lstm_output_size) # Example LayerNorm

        # Output layer(s)
        # Simplified output layer for demonstration
        self.fc1 = nn.Linear(lstm_output_size, lstm_output_size // 2)
        self.fc1_activation = nn.ReLU()
        self.fc1_dropout = nn.Dropout(self.dropout)
        self.fc2 = nn.Linear(lstm_output_size // 2, self.final_output_dim)

        self.output_activation = nn.Sigmoid() if self.final_output_dim == 1 else None

        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'lstm' in name and 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'lstm' in name and 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'lstm' in name and 'bias' in name:
                     nn.init.zeros_(param.data)
                elif 'fc' in name and 'weight' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'fc' in name and 'bias' in name:
                    nn.init.zeros_(param.data)
                elif 'norm' in name and hasattr(param, 'weight'): # For LayerNorm/BatchNorm weights
                    nn.init.ones_(param.data)
                elif 'norm' in name and hasattr(param, 'bias'):   # For LayerNorm/BatchNorm biases
                    nn.init.zeros_(param.data)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        if x.dim() != 3:
             raise ValueError(f"Expected 3D input (batch_size, seq_len, input_size), but got {x.dim()}D")
        if x.shape[2] != self.input_size:
             raise ValueError(f"Expected input_size {self.input_size} but got {x.shape[2]}")

        lstm_out, (hn, cn) = self.lstm(x)
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        # hn shape: (num_layers, batch_size, hidden_size)

        # Use the output of the last time step from the last layer
        # For seq_len=1, lstm_out[:, -1, :] is equivalent to lstm_out[:, 0, :]
        last_step_output = lstm_out[:, -1, :] # Shape: (batch_size, hidden_size)

        # Apply normalization
        norm_output = self.norm(last_step_output)

        # Pass through fully connected layers
        x = self.fc1(norm_output)
        x = self.fc1_activation(x)
        x = self.fc1_dropout(x)
        output = self.fc2(x)

        if self.output_activation:
            output = self.output_activation(output)

        # output shape: (batch_size, final_output_dim)
        return output 