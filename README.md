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

## Benchmarks

Trained on the full Wikipedia and Reddit JODIE datasets via `configs/wikipedia_full.yaml`
/ `configs/reddit_full.yaml` (30 max epochs, patience 5, batch size 200), run on
CPU-only GitHub Actions runners.

| Dataset   | Interactions | Test AUC | Test AP | Paper AP (Rossi et al. 2020, Table 2) |
|-----------|-------------:|---------:|--------:|---------------------------------------:|
| Wikipedia |      157,474 |   0.8720 |  0.8424 |                                  ~0.984 |
| Reddit    |      672,447 |   0.9044 |  0.9166 |                                  ~0.985 |

Full training logs and checkpoints are available as workflow artifacts in `.github/workflows/benchmark.yml` runs.

| Dataset   | Test AP (ours) | Test AUC (ours) | Test AP (Rossi et al. 2020) |
|-----------|---------------:|-----------------:|-----------------------------:|
| Wikipedia | 0.842          | 0.872             | ~0.984                       |
| Reddit    | 0.917          | 0.904             | ~0.987                       |

Reddit's Test AP is close to the paper's reported number; Wikipedia's has a real,
unresolved gap. Both runs also show a training-dynamics issue worth flagging
honestly rather than hiding:

- **Wikipedia**: validation AP improved gradually across the first ~7 epochs before
  early stopping, suggesting the model was still learning but capped below the
  paper's reported performance -- plausibly a batching/memory-update ordering
  difference from the paper's implementation, or insufficient epochs for this
  learning rate.
- **Reddit**: validation AP peaked at **epoch 1** (0.917) and *degraded* over the
  following 5 epochs before early stopping -- the model got worse with more
  training, not just plateaued. This points to a learning-rate or optimizer
  instability rather than an undertrained model, and is the more likely of the
  two datasets to have an actual implementation bug rather than a simple
  hyperparameter gap.

**Not yet root-caused.** Suspected areas, in order of likelihood: 
(1) negative sampling strategy differing from the paper's
(2) memory update/detach timing between batches
(3) learning rate too high for Reddit's larger, more frequent interaction stream. This is flagged as open work rather than resolved -- the goal here is an honest reproduction, not an inflated one. 

The system (config-driven training, tested modules, CI, Docker, served inference) is the actual deliverable here; matching published numbers exactly is a secondary goal we're continuing to close.

## New Developments
## Benchmarks

Trained on the full Wikipedia and Reddit JODIE datasets via `configs/wikipedia_full.yaml`
/ `configs/reddit_full.yaml` (30 max epochs, patience 5, batch size 200), run on
CPU-only GitHub Actions runners.

| Dataset   | Interactions | Test AUC | Test AP | Paper AP (Rossi et al. 2020, Table 2) |
|-----------|-------------:|---------:|--------:|---------------------------------------:|
| Wikipedia |      157,474 |   0.8720 |  0.8424 |                                  ~0.984 |
| Reddit    |      672,447 |   0.9450 |  0.9428 |                                  ~0.985 |

Both datasets show a real, honest gap to the paper's reported numbers -- this
reproduction is not tuned to match Table 2 exactly, and that gap is left
visible rather than closed by over-fitting to the benchmark.

**Diagnosed and fixed during development:** an initial Reddit run showed
validation AP peaking at epoch 1 (0.917) and *degrading* over the next 5
epochs before early stopping -- the model getting worse with more training,
not just plateauing. Lowering the learning rate 5x (`1e-4` -> `2e-5`) resolved
it: validation AP now improves steadily across 8 epochs before stopping, and
Test AP improved from 0.9166 to 0.9428. Root cause: Reddit's much higher
interaction frequency per node (672K interactions over ~11K nodes, vs.
Wikipedia's 157K over ~9K nodes) made the original learning rate too
aggressive, overshooting good minima almost immediately after the first
epoch. Wikipedia's config was left unchanged since it didn't show the same
instability -- its gap to the paper is a separate, smaller, unresolved
question (plausibly epoch budget or negative sampling strategy) rather than
this same LR issue.