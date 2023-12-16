import torch.nn as nn


class RegressionNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(6, 24),
            nn.ReLU(),
            nn.Linear(24, 12),
            nn.ReLU(),
            nn.Linear(12, 6),
            nn.ReLU(),
            nn.Linear(6, 1)  # u output on screen
        )

    def forward(self, x):
        """
        Forward pass
        """
        model_pred = self.linear_relu_stack(x)
        # return model_pred[:, 0], model_pred[:, 1]
        return model_pred
