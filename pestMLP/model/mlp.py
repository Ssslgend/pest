# pestMLP/model/mlp.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPModel(nn.Module):
    """Simple Multi-Layer Perceptron for Tabular/Raster Point Data"""
    def __init__(self, config, output_size=None):
        super(MLPModel, self).__init__()

        self.input_size = config["input_size"] # Number of input features
        self.hidden_layers = config.get("hidden_layers", [64, 32]) # List of hidden layer sizes
        self.dropout_rate = config.get("dropout", 0.2)
        default_num_classes = config.get("num_classes", 1)
        self.final_output_dim = output_size if output_size is not None else default_num_classes

        layers = []
        current_dim = self.input_size

        # Dynamically create hidden layers
        for h_dim in self.hidden_layers:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU()) # Activation function
            # Optionally add BatchNorm or LayerNorm
            # layers.append(nn.BatchNorm1d(h_dim))
            # layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.Dropout(self.dropout_rate))
            current_dim = h_dim

        # Final output layer
        layers.append(nn.Linear(current_dim, self.final_output_dim))

        # Combine layers into a Sequential module
        self.network = nn.Sequential(*layers)

        # Output activation (for binary classification)
        self.output_activation = nn.Sigmoid() if self.final_output_dim == 1 else None

        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            # Add initialization for BatchNorm/LayerNorm if used
            # elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
            #     if m.weight is not None: nn.init.ones_(m.weight)
            #     if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        # x shape: (batch_size, input_size)
        # Ensure input is 2D
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1) # Remove sequence dimension if present and size 1
        elif x.dim() != 2:
             raise ValueError(f"Expected 2D input (batch_size, input_size), but got {x.dim()}D shape {x.shape}")
        if x.shape[1] != self.input_size:
             raise ValueError(f"Expected input_size {self.input_size} but got {x.shape[1]}")

        # Pass through the network
        output = self.network(x)

        # Apply final activation if defined
        if self.output_activation:
            output = self.output_activation(output)

        # output shape: (batch_size, final_output_dim)
        return output 