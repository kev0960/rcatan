"""Checks for train/dense_conv.py. These fail with NotImplementedError until
the layer is written; the equivalence test is the one that proves it matches
PyG's TransformerConv."""

import pytest
import torch
from catan_engine import Engine
from torch_geometric.nn import TransformerConv

from train.dense_conv import DenseTransformerConv, build_neighbor_table

HEADS = 4
CHANNELS = 32
IN_CHANNELS = 128
EDGE_DIM = 4


@pytest.fixture
def endpoints() -> torch.Tensor:
    return torch.from_numpy(Engine.road_endpoints())


def test_neighbor_table_covers_every_road_both_ways(endpoints):
    neighbors, road_of, valid = build_neighbor_table(endpoints)

    assert set(valid.sum(dim=1).tolist()) <= {2, 3}

    got = {
        (i, int(neighbors[i, s]), int(road_of[i, s]))
        for i, s in valid.nonzero().tolist()
    }
    want = set()
    for r, (a, b) in enumerate(endpoints.tolist()):
        want.add((a, b, r))
        want.add((b, a, r))
    assert got == want


def test_output_shape(endpoints):
    """TODO: random x (B, 54, IN_CHANNELS) and roads (B, 72, EDGE_DIM) give
    (B, 54, HEADS * CHANNELS)."""
    B = 16
    x = torch.randn(B, 54, IN_CHANNELS)
    roads = torch.randn(B, 72, EDGE_DIM)
    dense_conv = DenseTransformerConv(
        IN_CHANNELS, CHANNELS, heads=HEADS, edge_dim=EDGE_DIM, road_endpoints=endpoints
    )
    assert dense_conv(x, roads).shape == (B, 54, HEADS * CHANNELS)


def test_matches_pyg_transformer_conv(endpoints):
    """The real check: same weights in, same outputs out."""
    torch.manual_seed(0)
    pyg_conv = TransformerConv(
        IN_CHANNELS, CHANNELS, heads=HEADS, concat=True, edge_dim=EDGE_DIM, beta=True
    )
    dense_conv = DenseTransformerConv(
        IN_CHANNELS, CHANNELS, heads=HEADS, edge_dim=EDGE_DIM, road_endpoints=endpoints
    )
    with torch.no_grad():
        for name in ["query", "key", "value", "edge", "skip", "beta"]:
            src = getattr(pyg_conv, f"lin_{name}")
            dst = getattr(dense_conv, name)
            dst.weight.copy_(src.weight)
            assert (src.bias is None) == (dst.bias is None), name
            if src.bias is not None:
                dst.bias.copy_(src.bias)

    B = 3
    x = torch.randn(B, 54, IN_CHANNELS)
    roads = torch.randn(B, 72, EDGE_DIM)

    # Same layout as GraphEncoder: one big graph of B * 54 nodes.
    edge_index = torch.cat([endpoints.T, endpoints.T.flip(0)], dim=1)
    offsets = torch.arange(B) * 54
    batched_edge_index = (edge_index[:, None, :] + offsets[None, :, None]).reshape(2, -1)
    edge_attr = torch.cat([roads, roads], dim=1).reshape(-1, EDGE_DIM)

    with torch.no_grad():
        expected = pyg_conv(
            x.reshape(B * 54, IN_CHANNELS), batched_edge_index, edge_attr
        ).view(B, 54, -1)
        actual = dense_conv(x, roads)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)


def test_padding_slots_are_ignored(endpoints):
    """For a node with only 2 neighbors, changing what its padding slot points
    at must not change any output."""
    torch.manual_seed(0)
    dense_conv = DenseTransformerConv(
        IN_CHANNELS, CHANNELS, heads=HEADS, edge_dim=EDGE_DIM, road_endpoints=endpoints
    ).eval()
    x = torch.randn(2, 54, IN_CHANNELS)
    roads = torch.randn(2, 72, EDGE_DIM)

    padding = (~dense_conv.valid).nonzero().tolist()
    assert padding, "expected some coastal nodes with a padding slot"

    with torch.no_grad():
        before = dense_conv(x, roads)
        for i, s in padding:
            # Point every padding slot at a real, different node and road.
            dense_conv.neighbors[i, s] = (i + 1) % 54
            dense_conv.road_of[i, s] = 71
        after = dense_conv(x, roads)

    torch.testing.assert_close(after, before, atol=0, rtol=0)
