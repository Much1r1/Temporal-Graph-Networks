import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from tgn.modules.time_encoding import TimeEmbedding

class MessageFunction(nn.Module):
    """
    Computes interaction messages for source and destination nodes.
    Message format: [memory_src, memory_dst, time_delta_encoding, edge_features]
    """
    def __init__(self, memory_dim: int, edge_feat_dim: int, time_dim: int, message_dim: int):
        super(MessageFunction, self).__init__()
        self.raw_message_dim = 2 * memory_dim + time_dim + edge_feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.raw_message_dim, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim)
        )

    def forward(
        self,
        src_mem: torch.Tensor,
        dst_mem: torch.Tensor,
        time_enc: torch.Tensor,
        edge_feats: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Concatenate raw messages for src and dst
        raw_msg_src = torch.cat([src_mem, dst_mem, time_enc, edge_feats], dim=-1)
        raw_msg_dst = torch.cat([dst_mem, src_mem, time_enc, edge_feats], dim=-1)

        msg_src = self.mlp(raw_msg_src)
        msg_dst = self.mlp(raw_msg_dst)
        return msg_src, msg_dst

class LastMessageAggregator(nn.Module):
    """
    Aggregates messages for each node by keeping the message corresponding to the maximum timestamp.
    """
    def __init__(self):
        super(LastMessageAggregator, self).__init__()

    def forward(
        self,
        nodes: torch.Tensor,
        messages: torch.Tensor,
        timestamps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        nodes: [M]
        messages: [M, message_dim]
        timestamps: [M]
        Returns unique nodes and their latest aggregated messages & timestamps.
        """
        unique_nodes, inv_indices = torch.unique(nodes, return_inverse=True)
        aggregated_messages = torch.zeros((len(unique_nodes), messages.size(1)), device=messages.device, dtype=messages.dtype)
        aggregated_timestamps = torch.full((len(unique_nodes),), -float('inf'), device=timestamps.device, dtype=timestamps.dtype)

        # Iterate through messages to keep the update with the max timestamp for each unique node
        for i in range(len(nodes)):
            node_idx = inv_indices[i]
            ts = timestamps[i]
            if ts > aggregated_timestamps[node_idx]:
                aggregated_messages[node_idx] = messages[i]
                aggregated_timestamps[node_idx] = ts

        return unique_nodes, aggregated_messages, aggregated_timestamps

class NodeMemory(nn.Module):
    """
    Maintains per-node memory vector states and last update timestamps.
    Provides memory update logic using GRUCell based on aggregated messages.
    """
    def __init__(self, num_nodes: int, memory_dim: int, message_dim: int, device: torch.device):
        super(NodeMemory, self).__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.message_dim = message_dim
        self.device = device

        self.memory = nn.Parameter(torch.zeros(num_nodes, memory_dim, device=device), requires_grad=False)
        self.last_update = nn.Parameter(torch.zeros(num_nodes, device=device), requires_grad=False)
        self.gru = nn.GRUCell(message_dim, memory_dim)

    def reset(self):
        """Resets all node memories and timestamps to zero."""
        self.memory.data.zero_()
        self.last_update.data.zero_()

    def get_memory(self, node_ids: torch.Tensor) -> torch.Tensor:
        """Returns memory state for given node IDs."""
        return self.memory[node_ids]

    def set_memory(self, node_ids: torch.Tensor, new_mem: torch.Tensor):
        """Sets memory state for given node IDs."""
        self.memory[node_ids] = new_mem

    def get_last_update(self, node_ids: torch.Tensor) -> torch.Tensor:
        """Returns last update timestamp for given node IDs."""
        return self.last_update[node_ids]

    def update_memory(
        self,
        unique_nodes: torch.Tensor,
        aggregated_messages: torch.Tensor,
        timestamps: torch.Tensor
    ):
        """Updates node memory states and last update timestamps using GRU."""
        if len(unique_nodes) == 0:
            return

        current_mem = self.memory[unique_nodes]
        updated_mem = self.gru(aggregated_messages, current_mem)

        self.memory[unique_nodes] = updated_mem
        self.last_update[unique_nodes] = timestamps
