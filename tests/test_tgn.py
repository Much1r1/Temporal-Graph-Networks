import os
import pytest
import numpy as np
import torch
import pandas as pd

from tgn.data.loader import load_jodie_data, TemporalGraphData
from tgn.modules.time_encoding import TimeEmbedding
from tgn.modules.memory import NodeMemory, MessageFunction, LastMessageAggregator
from tgn.modules.embedding import TemporalEmbedding
from tgn.modules.predictor import LinkPredictor, NegativeEdgeSampler
from tgn.train import TGN, eval_epoch
from tgn.inference import TGNInferenceService

@pytest.fixture
def dummy_csv_path(tmp_path):
    csv_file = tmp_path / "dummy_data.csv"
    data = []
    for i in range(50):
        src = i % 5
        dst = (i + 1) % 5
        ts = float(i * 10)
        label = float(i % 2)
        feats = [0.1 * j for j in range(4)]
        data.append([src, dst, ts, label] + feats)
    pd.DataFrame(data).to_csv(csv_file, header=False, index=False)
    return str(csv_file)

def test_data_loader(dummy_csv_path):
    full, train, val, test = load_jodie_data(dummy_csv_path, val_ratio=0.2, test_ratio=0.2)
    assert full.num_interactions == 50
    assert train.num_interactions == 30
    assert val.num_interactions == 10
    assert test.num_interactions == 10
    assert full.num_nodes > 0

def test_time_embedding():
    time_enc = TimeEmbedding(dimension=16)
    t = torch.tensor([0.0, 10.0, 100.0])
    emb = time_enc(t)
    assert emb.shape == (3, 16)

def test_memory_module():
    device = torch.device("cpu")
    mem = NodeMemory(num_nodes=10, memory_dim=16, message_dim=16, device=device)
    assert mem.get_memory(torch.tensor([0, 1])).shape == (2, 16)

    msg_func = MessageFunction(memory_dim=16, edge_feat_dim=4, time_dim=8, message_dim=16)
    agg = LastMessageAggregator()

    src_mem = mem.get_memory(torch.tensor([0, 1]))
    dst_mem = mem.get_memory(torch.tensor([2, 3]))
    t_enc = torch.randn(2, 8)
    edge_feats = torch.randn(2, 4)

    msg_src, msg_dst = msg_func(src_mem, dst_mem, t_enc, edge_feats)
    assert msg_src.shape == (2, 16)
    assert msg_dst.shape == (2, 16)

    nodes = torch.tensor([0, 1, 0])
    msgs = torch.randn(3, 16)
    ts = torch.tensor([1.0, 2.0, 3.0])

    u_nodes, u_msgs, u_ts = agg(nodes, msgs, ts)
    assert len(u_nodes) == 2
    mem.update_memory(u_nodes, u_msgs, u_ts)
    assert mem.get_last_update(torch.tensor([0])).item() == 3.0

def test_last_message_aggregator_out_of_order():
    agg = LastMessageAggregator()
    nodes = torch.tensor([1, 1, 1])
    # 3 messages for node 1 with out-of-order timestamps
    # Entry 0: ts 10.0
    # Entry 1: ts 30.0 (MAX timestamp)
    # Entry 2: ts 20.0 (Last array entry, but lower timestamp)
    msg0 = torch.ones(16) * 1.0
    msg1 = torch.ones(16) * 3.0  # expected selected message
    msg2 = torch.ones(16) * 2.0
    msgs = torch.stack([msg0, msg1, msg2], dim=0)
    timestamps = torch.tensor([10.0, 30.0, 20.0])

    u_nodes, u_msgs, u_ts = agg(nodes, msgs, timestamps)
    assert len(u_nodes) == 1
    assert u_nodes[0].item() == 1
    assert u_ts[0].item() == 30.0
    assert torch.allclose(u_msgs[0], msg1)

def test_memory_update_isolation():
    device = torch.device("cpu")
    mem = NodeMemory(num_nodes=5, memory_dim=16, message_dim=16, device=device)

    # Record initial memory states for all nodes
    initial_memories = mem.memory.clone()

    # Update only node 2
    nodes_to_update = torch.tensor([2])
    aggregated_msgs = torch.randn(1, 16)
    timestamps = torch.tensor([50.0])

    mem.update_memory(nodes_to_update, aggregated_msgs, timestamps)

    # Confirm node 2 memory changed
    assert not torch.allclose(mem.get_memory(torch.tensor([2])), initial_memories[2])
    assert mem.get_last_update(torch.tensor([2])).item() == 50.0

    # Confirm other nodes' memory remain unchanged
    other_node_ids = torch.tensor([0, 1, 3, 4])
    assert torch.equal(mem.get_memory(other_node_ids), initial_memories[other_node_ids])
    assert torch.equal(mem.get_last_update(other_node_ids), torch.zeros(4))

def test_embedding_and_predictor():
    time_enc = TimeEmbedding(dimension=8)
    emb_mod = TemporalEmbedding(node_memory_dim=16, edge_feat_dim=4, time_dim=8, output_dim=16, time_encoder=time_enc)
    predictor = LinkPredictor(embedding_dim=16, hidden_dim=32)

    nodes = torch.tensor([0, 1])
    memories = torch.randn(2, 16)
    last_ts = torch.tensor([0.0, 0.0])
    curr_ts = torch.tensor([5.0, 5.0])

    embs = emb_mod(nodes, memories, last_ts, curr_ts)
    assert embs.shape == (2, 16)

    scores = predictor(embs[:1], embs[1:])
    assert scores.shape == (1,)

def test_negative_sampler():
    sampler = NegativeEdgeSampler(min_node_id=0, max_node_id=10)
    samples = sampler.sample(20)
    assert len(samples) == 20
    assert (samples >= 0).all() and (samples <= 10).all()

def test_full_tgn_training_and_inference(dummy_csv_path, tmp_path):
    device = torch.device("cpu")
    full, train, val, test = load_jodie_data(dummy_csv_path, val_ratio=0.2, test_ratio=0.2)

    model = TGN(
        num_nodes=full.num_nodes,
        edge_feat_dim=full.edge_features.shape[1],
        memory_dim=16,
        message_dim=16,
        time_dim=8,
        embedding_dim=16,
        predictor_hidden_dim=16,
        device=device
    )

    sampler = NegativeEdgeSampler(min_node_id=0, max_node_id=full.num_nodes - 1)
    loss, auc, ap = eval_epoch(model, val, batch_size=10, sampler=sampler, device=device)
    assert isinstance(loss, float)
    assert 0.0 <= auc <= 1.0
    assert 0.0 <= ap <= 1.0

    ckpt_path = tmp_path / "model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "model": {
                "memory_dim": 16,
                "message_dim": 16,
                "time_dim": 8,
                "embedding_dim": 16,
                "predictor_hidden_dim": 16
            }
        },
        "num_nodes": full.num_nodes,
        "edge_feat_dim": full.edge_features.shape[1]
    }, str(ckpt_path))

    service = TGNInferenceService(str(ckpt_path), device="cpu")
    probs = service.predict_link_probabilities([0, 1], [2, 3], [10.0, 12.0])
    assert len(probs) == 2
    mem_vec = service.get_node_memory_vector(0)
    assert len(mem_vec) == 16
