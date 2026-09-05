import torch
import torch.nn as nn
import numpy as np

class LinkPredictor(nn.Module):
    """
    MLP Link Predictor for scoring interaction edge probabilities between source and destination nodes.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int = 64):
        super(LinkPredictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, src_embeddings: torch.Tensor, dst_embeddings: torch.Tensor) -> torch.Tensor:
        """
        src_embeddings: [B, embedding_dim]
        dst_embeddings: [B, embedding_dim]
        Returns predicted logits [B]
        """
        combined = torch.cat([src_embeddings, dst_embeddings], dim=-1)
        scores = self.mlp(combined).squeeze(-1)
        return scores

class NegativeEdgeSampler:
    """
    Uniform Negative Edge Sampler for temporal graph links.
    """
    def __init__(self, min_node_id: int, max_node_id: int, seed: int = 42):
        self.min_node_id = min_node_id
        self.max_node_id = max_node_id
        self.rng = np.random.RandomState(seed)

    def sample(self, num_samples: int) -> np.ndarray:
        return self.rng.randint(self.min_node_id, self.max_node_id + 1, size=num_samples)
