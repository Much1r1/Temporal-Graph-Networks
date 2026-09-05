# Temporal Graph Networks (TGN)

Productionized reproduction of Temporal Graph Networks (Rossi et al. 2020) with config-driven training, modular architecture, tested components, and serving inference interface.

## Repository Structure

```
├── configs/
│   └── default.yaml         # Training and model hyperparameter configurations
├── data/                    # Dataset directory
├── scripts/
│   └── download_data.sh     # Data retrieval script for Wikipedia/Reddit datasets
├── tgn/
│   ├── data/
│   │   └── loader.py        # Dataset parser & train/val/test data splitting
│   ├── modules/
│   │   ├── embedding.py     # Temporal Graph Embedding module
│   │   ├── memory.py        # Node memory, Message function, Aggregator, GRU memory updater
│   │   ├── predictor.py     # MLP Link Predictor & Uniform Negative Edge Sampler
│   │   └── time_encoding.py # Fourier Time Encoder
│   ├── train.py             # Config-driven training loop and metric evaluation (AUC/AP)
│   └── inference.py         # Serving inference interface & memory state query service
└── tests/
    └── test_tgn.py          # Pytest unit and integration test suite
```

## Quick Start

### 1. Installation

Install dependencies using `pip`:

```bash
pip install torch numpy pandas scikit-learn pyyaml pytest
```

### 2. Download Data

Download official Wikipedia / Reddit temporal graph datasets from Stanford SNAP:

```bash
bash scripts/download_data.sh data
```

### 3. Training

Run training with default YAML configuration:

```bash
PYTHONPATH=. python3 tgn/train.py --config configs/default.yaml
```

To train on custom CSV data:

```bash
PYTHONPATH=. python3 tgn/train.py --config configs/default.yaml --data data/your_dataset.csv
```

### 4. Running Tests

Run the full pytest suite:

```bash
PYTHONPATH=. pytest -v tests/
```

### 5. Serving Inference

Use `TGNInferenceService` to predict edge probabilities and inspect dynamic node memory states:

```python
from tgn.inference import TGNInferenceService

service = TGNInferenceService("checkpoints/best_model.pt")

# Predict probability of interaction between node pairs at specified timestamps
probs = service.predict_link_probabilities(src_ids=[1, 2], dst_ids=[3, 4], timestamps=[100.0, 105.0])
print("Link Probabilities:", probs)

# Fetch node memory vector state
node_memory = service.get_node_memory_vector(node_id=1)
print("Node 1 Memory Vector:", node_memory[:5])
```
