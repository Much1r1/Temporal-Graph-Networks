import argparse
import os
import yaml
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional
from sklearn.metrics import roc_auc_score, average_precision_score

from tgn.data.loader import load_jodie_data, TemporalGraphData
from tgn.modules.time_encoding import TimeEmbedding
from tgn.modules.memory import NodeMemory, MessageFunction, LastMessageAggregator
from tgn.modules.embedding import TemporalEmbedding
from tgn.modules.predictor import LinkPredictor, NegativeEdgeSampler

class TGN(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        edge_feat_dim: int,
        memory_dim: int,
        message_dim: int,
        time_dim: int,
        embedding_dim: int,
        predictor_hidden_dim: int,
        device: torch.device
    ):
        super(TGN, self).__init__()
        self.num_nodes = num_nodes
        self.device = device

        self.time_encoder = TimeEmbedding(dimension=time_dim)
        self.memory = NodeMemory(num_nodes=num_nodes, memory_dim=memory_dim, message_dim=message_dim, device=device)
        self.msg_function = MessageFunction(memory_dim=memory_dim, edge_feat_dim=edge_feat_dim, time_dim=time_dim, message_dim=message_dim)
        self.aggregator = LastMessageAggregator()
        self.embedding = TemporalEmbedding(
            node_memory_dim=memory_dim,
            edge_feat_dim=edge_feat_dim,
            time_dim=time_dim,
            output_dim=embedding_dim,
            time_encoder=self.time_encoder
        )
        self.predictor = LinkPredictor(embedding_dim=embedding_dim, hidden_dim=predictor_hidden_dim)

    def reset_memory(self):
        self.memory.reset()

def eval_epoch(
    model: TGN,
    data: TemporalGraphData,
    batch_size: int,
    sampler: NegativeEdgeSampler,
    device: torch.device
) -> Tuple[float, float, float]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    num_samples = len(data.sources)
    num_batches = int(np.ceil(num_samples / batch_size))

    with torch.no_grad():
        for b in range(num_batches):
            start_idx = b * batch_size
            end_idx = min((b + 1) * batch_size, num_samples)

            src_ids = data.sources[start_idx:end_idx]
            dst_ids = data.destinations[start_idx:end_idx]
            ts = data.timestamps[start_idx:end_idx]
            edge_feats = data.edge_features[start_idx:end_idx]

            neg_dst_ids = sampler.sample(len(src_ids))

            src_tensor = torch.from_numpy(src_ids).to(device)
            dst_tensor = torch.from_numpy(dst_ids).to(device)
            neg_dst_tensor = torch.from_numpy(neg_dst_ids).to(device)
            ts_tensor = torch.from_numpy(ts).float().to(device)
            edge_feats_tensor = torch.from_numpy(edge_feats).float().to(device)

            # Get memories and last update timestamps before update
            src_mem = model.memory.get_memory(src_tensor)
            dst_mem = model.memory.get_memory(dst_tensor)
            neg_dst_mem = model.memory.get_memory(neg_dst_tensor)

            src_last_ts = model.memory.get_last_update(src_tensor)
            dst_last_ts = model.memory.get_last_update(dst_tensor)
            neg_dst_last_ts = model.memory.get_last_update(neg_dst_tensor)

            # Embeddings
            src_emb = model.embedding(src_tensor, src_mem, src_last_ts, ts_tensor)
            dst_emb = model.embedding(dst_tensor, dst_mem, dst_last_ts, ts_tensor)
            neg_dst_emb = model.embedding(neg_dst_tensor, neg_dst_mem, neg_dst_last_ts, ts_tensor)

            pos_scores = model.predictor(src_emb, dst_emb)
            neg_scores = model.predictor(src_emb, neg_dst_emb)

            scores = torch.cat([pos_scores, neg_scores], dim=0)
            labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=0)

            loss = criterion(scores, labels)
            total_loss += loss.item() * (end_idx - start_idx)

            probs = torch.sigmoid(scores).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())

            # Update memory
            time_diffs = ts_tensor - src_last_ts
            t_enc = model.time_encoder(time_diffs)
            msg_src, msg_dst = model.msg_function(src_mem, dst_mem, t_enc, edge_feats_tensor)

            all_nodes = torch.cat([src_tensor, dst_tensor], dim=0)
            all_msgs = torch.cat([msg_src, msg_dst], dim=0)
            all_ts = torch.cat([ts_tensor, ts_tensor], dim=0)

            u_nodes, u_msgs, u_ts = model.aggregator(all_nodes, all_msgs, all_ts)
            model.memory.update_memory(u_nodes, u_msgs, u_ts)

    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
    auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5
    ap = average_precision_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5

    return avg_loss, auc, ap

def train_tgn(config_path: str, data_path: Optional[str] = None):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_name = config["data"]["dataset_name"]
    csv_file = data_path if data_path else f"data/{dataset_name}.csv"

    print(f"Loading data from {csv_file}...")
    full_data, train_data, val_data, test_data = load_jodie_data(
        csv_file,
        val_ratio=config["data"]["val_ratio"],
        test_ratio=config["data"]["test_ratio"]
    )

    num_nodes = full_data.num_nodes
    edge_feat_dim = full_data.edge_features.shape[1]

    print(f"Total Interactions: {full_data.num_interactions}, Num Nodes: {num_nodes}, Edge Feat Dim: {edge_feat_dim}")
    print(f"Train: {train_data.num_interactions}, Val: {val_data.num_interactions}, Test: {test_data.num_interactions}")

    model = TGN(
        num_nodes=num_nodes,
        edge_feat_dim=edge_feat_dim,
        memory_dim=config["model"]["memory_dim"],
        message_dim=config["model"]["message_dim"],
        time_dim=config["model"]["time_dim"],
        embedding_dim=config["model"]["embedding_dim"],
        predictor_hidden_dim=config["model"]["predictor_hidden_dim"],
        device=device
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"]
    )
    criterion = nn.BCEWithLogitsLoss()

    sampler = NegativeEdgeSampler(min_node_id=0, max_node_id=num_nodes - 1)

    best_val_ap = -1.0
    patience_count = 0
    checkpoint_path = config["train"]["checkpoint_path"]
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    batch_size = config["train"]["batch_size"]
    epochs = config["train"]["epochs"]

    for epoch in range(1, epochs + 1):
        model.train()
        model.reset_memory()
        total_loss = 0.0

        num_samples = train_data.num_interactions
        num_batches = int(np.ceil(num_samples / batch_size))

        for b in range(num_batches):
            start_idx = b * batch_size
            end_idx = min((b + 1) * batch_size, num_samples)

            src_ids = train_data.sources[start_idx:end_idx]
            dst_ids = train_data.destinations[start_idx:end_idx]
            ts = train_data.timestamps[start_idx:end_idx]
            edge_feats = train_data.edge_features[start_idx:end_idx]

            neg_dst_ids = sampler.sample(len(src_ids))

            src_tensor = torch.from_numpy(src_ids).to(device)
            dst_tensor = torch.from_numpy(dst_ids).to(device)
            neg_dst_tensor = torch.from_numpy(neg_dst_ids).to(device)
            ts_tensor = torch.from_numpy(ts).float().to(device)
            edge_feats_tensor = torch.from_numpy(edge_feats).float().to(device)

            optimizer.zero_grad()

            # Get memories and timestamps
            src_mem = model.memory.get_memory(src_tensor)
            dst_mem = model.memory.get_memory(dst_tensor)
            neg_dst_mem = model.memory.get_memory(neg_dst_tensor)

            src_last_ts = model.memory.get_last_update(src_tensor)
            dst_last_ts = model.memory.get_last_update(dst_tensor)
            neg_dst_last_ts = model.memory.get_last_update(neg_dst_tensor)

            # Compute embeddings
            src_emb = model.embedding(src_tensor, src_mem, src_last_ts, ts_tensor)
            dst_emb = model.embedding(dst_tensor, dst_mem, dst_last_ts, ts_tensor)
            neg_dst_emb = model.embedding(neg_dst_tensor, neg_dst_mem, neg_dst_last_ts, ts_tensor)

            pos_scores = model.predictor(src_emb, dst_emb)
            neg_scores = model.predictor(src_emb, neg_dst_emb)

            scores = torch.cat([pos_scores, neg_scores], dim=0)
            labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=0)

            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * (end_idx - start_idx)

            # Update memory state after gradient step
            with torch.no_grad():
                time_diffs = ts_tensor - src_last_ts
                t_enc = model.time_encoder(time_diffs)
                msg_src, msg_dst = model.msg_function(src_mem, dst_mem, t_enc, edge_feats_tensor)

                all_nodes = torch.cat([src_tensor, dst_tensor], dim=0)
                all_msgs = torch.cat([msg_src, msg_dst], dim=0)
                all_ts = torch.cat([ts_tensor, ts_tensor], dim=0)

                u_nodes, u_msgs, u_ts = model.aggregator(all_nodes, all_msgs, all_ts)
                model.memory.update_memory(u_nodes, u_msgs, u_ts)

        train_loss = total_loss / num_samples

        # Evaluate on validation set
        val_loss, val_auc, val_ap = eval_epoch(model, val_data, batch_size, sampler, device)

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}")

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            patience_count = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "num_nodes": num_nodes,
                "edge_feat_dim": edge_feat_dim
            }, checkpoint_path)
            print(f"Saved best model checkpoint to {checkpoint_path}")
        else:
            patience_count += 1
            if patience_count >= config["train"]["patience"]:
                print("Early stopping triggered.")
                break

    # Evaluate on test set using best saved model
    print("\nEvaluating test set with best checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_auc, test_ap = eval_epoch(model, test_data, batch_size, sampler, device)
    print(f"Test Loss: {test_loss:.4f} | Test AUC: {test_auc:.4f} | Test AP: {test_ap:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Temporal Graph Networks (TGN)")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config file")
    parser.add_argument("--data", type=str, default=None, help="Path to input data CSV file")
    args = parser.parse_args()

    train_tgn(args.config, args.data)
