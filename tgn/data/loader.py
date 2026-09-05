import numpy as np
import pandas as pd
import torch
from typing import Tuple, Optional, Dict, Any

class TemporalGraphData:
    """
    Data container for temporal graph interaction datasets (e.g. JODIE Wikipedia / Reddit).
    """
    def __init__(
        self,
        sources: np.ndarray,
        destinations: np.ndarray,
        timestamps: np.ndarray,
        edge_idxs: np.ndarray,
        labels: np.ndarray,
        edge_features: np.ndarray,
        node_features: Optional[np.ndarray] = None
    ):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.edge_features = edge_features
        self.node_features = node_features

        self.num_interactions = len(sources)
        self.unique_nodes = np.unique(np.concatenate([sources, destinations]))
        self.num_nodes = int(self.unique_nodes.max() + 1) if len(self.unique_nodes) > 0 else 0

def load_jodie_data(
    filepath: str,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    randomize_features: bool = False
) -> Tuple[TemporalGraphData, TemporalGraphData, TemporalGraphData, TemporalGraphData]:
    """
    Loads and parses JODIE CSV formatted dataset into Train, Val, Test split TemporalGraphData containers.
    JODIE CSV format: user_id, item_id, timestamp, state_label, comma_separated_features...
    """
    df = pd.read_csv(filepath, header=None)

    sources = df.iloc[:, 0].values.astype(np.int64)
    # Ensure destination node IDs do not overlap with source IDs by remapping item IDs if needed
    dest_raw = df.iloc[:, 1].values.astype(np.int64)
    max_src = sources.max()
    destinations = dest_raw + max_src + 1

    timestamps = df.iloc[:, 2].values.astype(np.float64)
    labels = df.iloc[:, 3].values.astype(np.float64)

    # Remaining columns are edge features
    if df.shape[1] > 4:
        edge_features = df.iloc[:, 4:].values.astype(np.float34 if np.float32 == np.float64 else np.float32)
    else:
        edge_features = np.zeros((len(sources), 1), dtype=np.float32)

    edge_idxs = np.arange(len(sources), dtype=np.int64)

    num_total = len(sources)
    num_val = int(num_total * val_ratio)
    num_test = int(num_total * test_ratio)
    num_train = num_total - num_val - num_test

    full_data = TemporalGraphData(
        sources=sources,
        destinations=destinations,
        timestamps=timestamps,
        edge_idxs=edge_idxs,
        labels=labels,
        edge_features=edge_features
    )

    train_data = TemporalGraphData(
        sources=sources[:num_train],
        destinations=destinations[:num_train],
        timestamps=timestamps[:num_train],
        edge_idxs=edge_idxs[:num_train],
        labels=labels[:num_train],
        edge_features=edge_features[:num_train]
    )

    val_data = TemporalGraphData(
        sources=sources[num_train:num_train + num_val],
        destinations=destinations[num_train:num_train + num_val],
        timestamps=timestamps[num_train:num_train + num_val],
        edge_idxs=edge_idxs[num_train:num_train + num_val],
        labels=labels[num_train:num_train + num_val],
        edge_features=edge_features[num_train:num_train + num_val]
    )

    test_data = TemporalGraphData(
        sources=sources[num_train + num_val:],
        destinations=destinations[num_train + num_val:],
        timestamps=timestamps[num_train + num_val:],
        edge_idxs=edge_idxs[num_train + num_val:],
        labels=labels[num_train + num_val:],
        edge_features=edge_features[num_train + num_val:]
    )

    return full_data, train_data, val_data, test_data
