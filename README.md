# Temporal Graph Networks (TGN)

Productionized reproduction of Temporal Graph Networks (Rossi et al. 2020) with config-driven training, modular architecture, tested components, FastAPI serving inference interface, CI workflow, and Docker support.

## Repository Structure

```
├── .github/
│   └── workflows/
│       └── ci.yml           # GitHub Actions CI workflow (pytest)
├── configs/
│   └── default.yaml         # Training and model hyperparameter configurations
├── Dockerfile               # Container build definition for training and serving
├── scripts/
│   └── download_data.sh     # Data retrieval script for Wikipedia/Reddit datasets
├── serving/
│   ├── __init__.py
│   └── app.py               # FastAPI application with /health and /predict endpoints
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
    ├── test_serving.py      # FastAPI serving integration test
    └── test_tgn.py          # Pytest unit and integration test suite
```

## Quick Start

### 1. Installation

Install dependencies using `pip`:

```bash
pip install torch numpy pandas scikit-learn pyyaml pytest fastapi httpx uvicorn
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
PYTHONPATH=. python3 -m pytest -v tests/
```

### 5. Serving Inference API

Start the FastAPI inference web app using Uvicorn:

```bash
uvicorn serving.app:app --host 0.0.0.0 --port 8000
```

#### Endpoints:
- `GET /health` - Service health status
- `POST /predict` - Predict link interaction probability for `(src_node_id, dst_node_id, timestamp)`

Example `POST /predict` request body:
```json
{
  "src_node_id": 1,
  "dst_node_id": 2,
  "timestamp": 100.0
}
```

### 6. Docker & CI

- **Continuous Integration**: `.github/workflows/ci.yml` runs `pytest -v` automatically on pushes and pull requests to `main`.
- **Docker Build & Run**:
  ```bash
  docker build -t tgn:latest .
  docker run -p 8000:8000 tgn:latest
  ```
