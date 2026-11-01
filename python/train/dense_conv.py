"""Dense TransformerConv for Catan's fixed board graph.

Computes the same thing as `torch_geometric.nn.TransformerConv(
    in_channels, out_channels, heads=H, concat=True, edge_dim=edge_dim, beta=True)`
but with a fixed neighbor table instead of an edge list, so every step is a
regular dense tensor op (gather, mask, softmax over slots, sum) with no
scatter.

Shapes used below:
    B = batch size, N = 54 nodes, S = 3 neighbor slots per node,
    R = 72 roads, H = heads, C = channels per head (H * C = hidden size).
"""

import torch
from torch import nn

NUM_NODES = 54
MAX_NEIGHBORS = 3


def build_neighbor_table(
    road_endpoints: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Turn the road list into a fixed per-node neighbor table.

    Args:
        road_endpoints: (R, 2) long. Road r joins nodes road_endpoints[r, 0]
            and road_endpoints[r, 1] (from `Engine.road_endpoints()`).

    Returns:
        neighbors: (N, S) long. neighbors[i, s] is the node in slot s of node
            i's neighbor list.
        road_of: (N, S) long. road_of[i, s] is the road joining node i to
            neighbors[i, s].
        valid: (N, S) bool. False for padding slots (nodes with only 2
            neighbors). Padding slots may hold any in-range index, e.g. 0.
    """

    adjacency = torch.full((NUM_NODES, NUM_NODES), -1, dtype=torch.int8)
    for index, (left, right) in enumerate(road_endpoints):
        adjacency[left, right] = index
        adjacency[right, left] = index

    neighbors = torch.zeros((NUM_NODES, MAX_NEIGHBORS), dtype=torch.long)
    road_of = torch.zeros((NUM_NODES, MAX_NEIGHBORS), dtype=torch.long)
    valid = torch.zeros((NUM_NODES, MAX_NEIGHBORS), dtype=torch.bool)

    for i in range(NUM_NODES):
        neighbors_i = (adjacency[i] != -1).nonzero(as_tuple=False).flatten()

        neighbors[i, : len(neighbors_i)] = neighbors_i
        road_of[i, : len(neighbors_i)] = adjacency[i, neighbors_i]
        valid[i, : len(neighbors_i)] = True

    return neighbors, road_of, valid


class DenseTransformerConv(nn.Module):
    """One graph attention layer over the fixed board graph.

    For each node i and head h, over its valid neighbor slots s (node j,
    road r):
        score[i, s] = query_i . (key_j + edge_r) / sqrt(C)
        alpha[i, :] = softmax over s of score[i, :]   (padding gets weight 0)
        msg_i       = sum_s alpha[i, s] * (value_j + edge_r)
    Heads are concatenated, then blended with a skip connection:
        skip_i = W_skip(x_i)
        beta_i = sigmoid(W_beta([msg_i, skip_i, msg_i - skip_i]))
        out_i  = beta_i * skip_i + (1 - beta_i) * msg_i
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int,
        edge_dim: int,
        road_endpoints: torch.Tensor,
    ):
        """
        Args:
            in_channels: size of each input node vector.
            out_channels: C, channels per head. Output size is heads * C.
            heads: H, number of attention heads.
            edge_dim: size of each road's feature vector.
            road_endpoints: (R, 2) long, passed to `build_neighbor_table`.
        """
        super().__init__()

        self.query = nn.Linear(in_channels, heads * out_channels)
        self.key = nn.Linear(in_channels, heads * out_channels)
        self.value = nn.Linear(in_channels, heads * out_channels)
        self.edge = nn.Linear(edge_dim, heads * out_channels, bias=False)

        self.skip = nn.Linear(in_channels, heads * out_channels)
        self.beta = nn.Linear(3 * heads * out_channels, 1, bias=False)

        neighbors, road_of, valid = build_neighbor_table(road_endpoints)

        self.register_buffer("neighbors", neighbors, persistent=False)
        self.register_buffer("road_of", road_of, persistent=False)
        self.register_buffer("valid", valid, persistent=False)

        self.heads = heads
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, roads: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, in_channels) node vectors.
            roads: (B, R, edge_dim) road features (one per road, not per
                direction; both directions share them).

        Returns:
            (B, N, H * C) updated node vectors.

            1. q, k, v from x                          -> (B, N, H, C) each
               e from roads                            -> (B, R, H, C)
            2. Gather each node's neighbors:
               k, v by the neighbor table              -> (B, N, S, H, C)
               e by the road table                     -> (B, N, S, H, C)
            3. score = query . (key + edge) / sqrt(C)  -> (B, N, S, H)
            4. Mask padding slots to -inf, softmax over the slot dim
                                                       -> (B, N, S, H)
            5. msg = sum over slots of alpha * (value + edge)
                                                       -> (B, N, H, C)
               then concatenate heads                  -> (B, N, H * C)
            6. Beta gate with the skip connection      -> (B, N, H * C)
        """

        query = self.query(x).unflatten(
            dim=-1, sizes=(self.heads, self.out_channels)
        )
        key = self.key(x).unflatten(
            dim=-1, sizes=(self.heads, self.out_channels)
        )
        value = self.value(x).unflatten(
            dim=-1, sizes=(self.heads, self.out_channels)
        )
        edge = self.edge(roads).unflatten(
            dim=-1, sizes=(self.heads, self.out_channels)
        )

        # For a given node i, we need to get key of neighboring node and edge values.
        # For each batch, "pick" the neighbor for each node
        # (B, N, S, H, C)
        neighbor_keys = key[:, self.neighbors]
        neighbor_values = value[:, self.neighbors]

        # For each batch, "pick" the road for each node
        road_edges = edge[:, self.road_of]

        # query: (B, N, H, C)
        # unsqueezed query: (B, N, 1, H, C)
        # neighbor_keys: (B, N, S, H, C)
        # score: (B, N, S, H)
        score = (query.unsqueeze(2) * (neighbor_keys + road_edges)).sum(
            dim=-1
        ) / (self.out_channels ** 0.5)

        # valid: (N, S) --> (1, N, S, 1)
        masked_score = score.masked_fill(~self.valid.unsqueeze(0).unsqueeze(-1), float('-inf'))

        # alpha: (B, N, S, H, 1)
        alpha = masked_score.softmax(dim=2).unsqueeze(-1)

        # message: (B, N, H, C). Summed over S (neighbors)
        message = (alpha * (neighbor_values + road_edges)).sum(dim=2)

        # (B, N, H, C) --> (B, N, H * C)
        message = message.flatten(start_dim=2)

        # Information about the node itself. (B, N, H * C)
        skip = self.skip(x)

        # Concat: (B, N, 3 * H * C)
        # Beta gate: (B, N, 1)
        beta = self.beta(torch.cat([message, skip, message - skip], dim=-1)).sigmoid()

        return beta * skip + (1 - beta) * message