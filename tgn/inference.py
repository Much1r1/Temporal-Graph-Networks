import argparse
import torch
import numpy as np
from typing import List, Tuple, Dict, Any
from tgn.train import TGN

class TGNInferenceService:
    """
    Inference service for TGN link probability prediction and dynamic state queries.
    """
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint["config"]
        num_nodes = checkpoint["num_nodes"]
        edge_feat_dim = checkpoint["edge_feat_dim"]

        self.model = TGN(
            num_nodes=num_nodes,
            edge_feat_dim=edge_feat_dim,
            memory_dim=config["model"]["memory_dim"],
            message_dim=config["model"]["message_dim"],
            time_dim=config["model"]["time_dim"],
            embedding_dim=config["model"]["embedding_dim"],
            predictor_hidden_dim=config["model"]["predictor_hidden_dim"],
            device=self.device
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict_link_probabilities(
        self,
        src_ids: List[int],
        dst_ids: List[int],
        timestamps: List[float]
    ) -> List[float]:
        """
        Predicts link interaction probabilities for pairs of (src_id, dst_id) at timestamps.
        """
        with torch.no_grad():
            src_tensor = torch.tensor(src_ids, dtype=torch.long, device=self.device)
            dst_tensor = torch.tensor(dst_ids, dtype=torch.long, device=self.device)
            ts_tensor = torch.tensor(timestamps, dtype=torch.float, device=self.device)

            src_mem = self.model.memory.get_memory(src_tensor)
            dst_mem = self.model.memory.get_memory(dst_tensor)

            src_last_ts = self.model.memory.get_last_update(src_tensor)
            dst_last_ts = self.model.memory.get_last_update(dst_tensor)

            src_emb = self.model.embedding(src_tensor, src_mem, src_last_ts, ts_tensor)
            dst_emb = self.model.embedding(dst_tensor, dst_mem, dst_last_ts, ts_tensor)

            logits = self.model.predictor(src_emb, dst_emb)
            probs = torch.sigmoid(logits).cpu().numpy().tolist()

            if isinstance(probs, float):
                probs = [probs]

            return probs

    def get_node_memory_vector(self, node_id: int) -> List[float]:
        """
        Returns memory vector for a specific node ID.
        """
        with torch.no_grad():
            node_tensor = torch.tensor([node_id], dtype=torch.long, device=self.device)
            mem = self.model.memory.get_memory(node_tensor)
            return mem.squeeze(0).cpu().numpy().tolist()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TGN Inference Runner")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to checkpoint")
    args = parser.parse_args()

    service = TGNInferenceService(args.checkpoint)
    probs = service.predict_link_probabilities([1, 2], [3, 4], [100.0, 105.0])
    print(f"Predicted Link Probabilities: {probs}")
    mem = service.get_node_memory_vector(1)
    print(f"Node 1 memory dimension: {len(mem)}")
