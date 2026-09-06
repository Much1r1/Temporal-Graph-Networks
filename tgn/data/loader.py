import numpy as np
import pandas as pd
import torch
from typing import Tuple, Optional, Dict, Any


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False

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
    # JODIE-format CSVs (wikipedia.csv, reddit.csv from snap.stanford.edu) have a
    # single descriptive header line, e.g.:
    #   user_id,item_id,timestamp,state_label,comma_separated_features
    # followed by real data rows with the features expanded into N columns
    # (172 for Wikipedia/Reddit). That header row has a different column count
    # than the data rows, so it must be skipped rather than parsed as data --
    # feeding it to pandas with header=None caused it to infer the wrong
    # column count from that first line and crash on line 2.
    #
    # Test fixtures and other synthetic CSVs may be headerless, so detect
    # the header rather than assuming it's always present: if the first
    # field of the first line isn't numeric, treat that line as a header.
    with open(filepath, "r") as f:
        first_line = f.readline().strip()
    first_field = first_line.split(",")[0]
    has_header = not _is_number(first_field)

    df = pd.read_csv(filepath, header=None, skiprows=1 if has_header else 0)

    sources = df.iloc[:, 0].values.astype(np.int64)
    # Ensure destination node IDs do not overlap with source IDs by remapping item IDs if needed
    dest_raw = df.iloc[:, 1].values.astype(np.int64)
    max_src = sources.max()
    destinations = dest_raw + max_src + 1

    timestamps = df.iloc[:, 2].values.astype(np.float64)
    labels = df.iloc[:, 3].values.astype(np.float64)

    # Remaining columns are edge features
    if df.shape[1] > 4:
        edge_features = df.iloc[:, 4:].values.astype(np.float32)
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