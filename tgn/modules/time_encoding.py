import torch
import torch.nn as nn
import numpy as np

class TimeEmbedding(nn.Module):
    """
    Fourier Time Embedding (Time Encoder) mapping time differences to vectors.
    """
    def __init__(self, dimension: int):
        super(TimeEmbedding, self).__init__()
        self.dimension = dimension
        self.w = nn.Linear(1, dimension)
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension))).float().unsqueeze(1))
        self.w.bias = nn.Parameter(torch.zeros(dimension).float())

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t shape: [N] or [N, 1]
        if t.dim() == 1:
            t = t.unsqueeze(1)
        # output shape: [N, dimension]
        output = torch.cos(self.w(t.float()))
        return output
