import torch
import torch.nn as nn
from typing import Optional
from tgn.modules.time_encoding import TimeEmbedding

class TemporalEmbedding(nn.Module):
    """
    Computes node embeddings by combining node memory with temporal projection.
    Supports standard Identity/Linear projection module and temporal attention aggregation.
    """
    def __init__(
        self,
        node_memory_dim: int,
        edge_feat_dim: int,
        time_dim: int,
        output_dim: int,
        time_encoder: TimeEmbedding
    ):
        super(TemporalEmbedding, self).__init__()
        self.node_memory_dim = node_memory_dim
        self.edge_feat_dim = edge_feat_dim
        self.time_dim = time_dim
        self.output_dim = output_dim
        self.time_encoder = time_encoder

        self.projection = nn.Sequential(
            nn.Linear(node_memory_dim + time_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(
        self,
        node_ids: torch.Tensor,
        node_memories: torch.Tensor,
        last_update_timestamps: torch.Tensor,
        current_timestamps: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes node embeddings at current_timestamps.
        """
        time_diffs = current_timestamps - last_update_timestamps
        time_embeddings = self.time_encoder(time_diffs)

        combined_features = torch.cat([node_memories, time_embeddings], dim=-1)
        node_embeddings = self.projection(combined_features)

        return node_embeddings
