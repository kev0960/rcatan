"""Graph actor and critic over Catan's board graph.

The board is a fixed graph: 54 intersections (nodes) joined by 72 roads
(edges). Both networks take the same flat state vector as the feed-forward
ones, split it into per-node, per-road and global features, and run
`DenseTransformerConv` layers along the roads. Weights are shared across
nodes, so what is learned at one spot carries over to every other spot.

The slices below follow the layout documented in `state_to_vec`
(crates/engine_py/src/lib.rs). Owners are already rotated so index 0 is the
current player.
"""

import torch
from torch import nn

from train.dense_conv import DenseTransformerConv

NUM_NODES = 54
NUM_ROADS = 72
STATE_LEN = 2031
# The state before the card estimates were appended (2026-11-07). Checkpoints
# trained on it (230211, 115058, 221927, ...) still load: their networks read
# only the first 2016 floats.
LEGACY_STATE_LEN = 2016

ROADS = slice(0, 288)  # 72 * owner one-hot[4]
SETTLEMENTS = slice(288, 504)  # 54 * owner one-hot[4]
CITIES = slice(504, 720)  # 54 * owner one-hot[4]
NODE_FEATURES = slice(732, 1920)  # 54 * [22]: production, robber, port
# Per-game values: 12, then everything from 1920 to the end of the state
# (96, plus 15 card estimates on the current state).
GLOBALS_HEAD = slice(720, 732)
GLOBALS_TAIL_START = 1920

ROAD_DIM = 4
NODE_DIM = 4 + 4 + 22


def global_dim(state_len: int = STATE_LEN) -> int:
    """Global features per game: 123 on the current state, 108 on the
    legacy one."""
    return GLOBALS_HEAD.stop - GLOBALS_HEAD.start + state_len - GLOBALS_TAIL_START


GLOBAL_DIM = global_dim()

# Action space layout (see `action_to_index` in crates/engine_py/src/lib.rs).
NUM_OTHER_ACTIONS = 120  # [180, 300): everything that isn't a road or building
# Inside those: [180, 185) discard, then [185, 204) move the robber to tile
# 0-18, then [204, 300) the rest (steal, trades, end turn, dev cards, ...).
NUM_TILES = 19
NUM_DISCARD_ACTIONS = 5

HIDDEN_SIZE = 128
NUM_LAYERS = 3
NUM_HEADS = 4


def split_state(state: torch.Tensor):
    """Split a (B, STATE_LEN) state into node (B, 54, 30), road (B, 72, 4) and
    global (B, GLOBAL_DIM) features. A legacy (B, 2016) state gives 108
    globals."""
    b = state.shape[0]
    roads = state[:, ROADS].view(b, NUM_ROADS, ROAD_DIM)
    nodes = torch.cat(
        [
            state[:, SETTLEMENTS].view(b, NUM_NODES, 4),
            state[:, CITIES].view(b, NUM_NODES, 4),
            state[:, NODE_FEATURES].view(b, NUM_NODES, 22),
        ],
        dim=-1,
    )
    globals_ = torch.cat(
        [state[:, GLOBALS_HEAD], state[:, GLOBALS_TAIL_START:]], dim=-1
    )
    return nodes, roads, globals_


class GraphEncoder(nn.Module):
    """Embeds every node of every board in the batch.

    Every board has the same graph, so each layer looks up neighbors in a
    fixed (54, 3) table and the batch stays a plain (B, 54, H) tensor.
    """

    def __init__(
        self,
        road_endpoints: torch.Tensor,
        num_layers: int = NUM_LAYERS,
        state_len: int = STATE_LEN,
    ):
        super().__init__()
        # The encoder reads only the first `state_len` floats, so a network
        # built for the legacy state still runs on the current one.
        self.state_len = state_len
        self.global_dim = global_dim(state_len)

        endpoints = torch.as_tensor(road_endpoints, dtype=torch.long)
        assert endpoints.shape == (NUM_ROADS, 2)
        # Not persistent: it is board data, not a learned weight.
        self.register_buffer("road_endpoints", endpoints, persistent=False)

        # Each node also sees the global features, so the player's hand,
        # score and phase shape the attention at every layer.
        self.input = nn.Linear(NODE_DIM + self.global_dim, HIDDEN_SIZE)
        self.convs = nn.ModuleList(
            DenseTransformerConv(
                HIDDEN_SIZE,
                HIDDEN_SIZE // NUM_HEADS,
                heads=NUM_HEADS,
                edge_dim=ROAD_DIM,
                road_endpoints=endpoints,
            )
            for _ in range(num_layers)
        )
        self.norms = nn.ModuleList(
            nn.LayerNorm(HIDDEN_SIZE) for _ in range(num_layers)
        )
        self.out_norm = nn.LayerNorm(HIDDEN_SIZE)

    def forward(self, state: torch.Tensor):
        """Returns node embeddings (B, 54, H), road features (B, 72, 4) and
        global features (B, global_dim)."""
        nodes, roads, globals_ = split_state(state[:, : self.state_len])
        b = state.shape[0]

        x = torch.cat(
            [nodes, globals_[:, None, :].expand(b, NUM_NODES, self.global_dim)],
            dim=-1,
        )
        h = self.input(x)

        # Pre-norm residual blocks: each layer passes messages one road further.
        for conv, norm in zip(self.convs, self.norms):
            h = h + torch.relu(conv(norm(h), roads))

        return self.out_norm(h), roads, globals_


def _pool(h: torch.Tensor, globals_: torch.Tensor) -> torch.Tensor:
    """Summarize the board as mean + max over nodes, plus the globals."""
    return torch.cat([h.mean(dim=1), h.amax(dim=1), globals_], dim=-1)


def pooled_dim(state_len: int = STATE_LEN) -> int:
    return 2 * HIDDEN_SIZE + global_dim(state_len)


class GraphActor(nn.Module):
    """Policy over the 300 actions. Each action is scored by the part of the
    board it acts on: a road by its two end nodes, a settlement or city by its
    node, and everything else by a pooled summary of the whole board.

    With `tile_nodes` (the "entity-robber" head), the 19 robber moves are also
    scored by the part of the board they act on: each tile from its six corner
    nodes. Without it (the "entity" head, every run up to 2026-11-06), they
    come from the pooled summary like the other 120 actions. That summary is
    the same for every tile, so the pooled head had to learn, separately for
    each of the 19 outputs, which spot on the board its tile is; and what it
    learned about blocking one tile did not carry over to any other.
    """

    def __init__(
        self,
        road_endpoints: torch.Tensor,
        num_layers: int = NUM_LAYERS,
        tile_nodes: torch.Tensor | None = None,
        state_len: int = STATE_LEN,
    ):
        super().__init__()
        self.encoder = GraphEncoder(road_endpoints, num_layers, state_len)
        g = self.encoder.global_dim

        # Symmetric in the two end nodes, so a road scores the same from either
        # end: [h_u + h_v, h_u * h_v, road features].
        self.road_head = nn.Sequential(
            nn.Linear(2 * HIDDEN_SIZE + ROAD_DIM, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1),
        )
        # Per node: [settlement, city].
        self.node_head = nn.Linear(HIDDEN_SIZE, 2)

        num_pooled = NUM_OTHER_ACTIONS
        self.robber_head = None
        if tile_nodes is not None:
            corners = torch.as_tensor(tile_nodes, dtype=torch.long)
            assert corners.shape == (NUM_TILES, 6)
            # Not persistent: board data, like road_endpoints.
            self.register_buffer("tile_nodes", corners, persistent=False)
            # One function shared by all 19 tiles, like node_head for the 54
            # nodes. Its input is [mean, max] over the tile's six corners:
            # order-free, since a tile's corners have no natural order. The
            # mean says how much is built around the tile, the max catches a
            # single standout corner (e.g. the leader's city). The globals add
            # the scores and hands, i.e. who is worth blocking. Corners already
            # see the dice numbers and the robber's current tile in their
            # node features.
            self.robber_head = nn.Sequential(
                nn.Linear(2 * HIDDEN_SIZE + g, HIDDEN_SIZE),
                nn.ReLU(),
                nn.Linear(HIDDEN_SIZE, 1),
            )
            num_pooled -= NUM_TILES

        self.pooled_head = nn.Sequential(
            nn.Linear(pooled_dim(state_len), HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, num_pooled),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        h, roads, globals_ = self.encoder(state)

        endpoints = self.encoder.road_endpoints
        h_u, h_v = h[:, endpoints[:, 0]], h[:, endpoints[:, 1]]
        road_logits = self.road_head(
            torch.cat([h_u + h_v, h_u * h_v, roads], dim=-1)
        ).squeeze(-1)

        node_logits = self.node_head(h)
        other_logits = self.pooled_head(_pool(h, globals_))

        if self.robber_head is not None:
            # (B, 19, 6, H): the six corner embeddings of each tile.
            corners = h[:, self.tile_nodes]
            b = state.shape[0]
            tile_input = torch.cat(
                [
                    corners.mean(dim=2),
                    corners.amax(dim=2),
                    globals_[:, None, :].expand(b, NUM_TILES, globals_.shape[-1]),
                ],
                dim=-1,
            )
            robber_logits = self.robber_head(tile_input).squeeze(-1)
            # Put the robber moves back in their place, after the discards.
            other_logits = torch.cat(
                [
                    other_logits[:, :NUM_DISCARD_ACTIONS],
                    robber_logits,
                    other_logits[:, NUM_DISCARD_ACTIONS:],
                ],
                dim=-1,
            )

        # Concatenated in action-index order:
        #   [0, 72) roads, [72, 126) settlements, [126, 180) cities, [180, 300).
        return torch.cat(
            [road_logits, node_logits[..., 0], node_logits[..., 1], other_logits],
            dim=-1,
        )


class GraphCritic(nn.Module):
    """State value from the current player's perspective. Has its own encoder,
    separate from the actor's."""

    def __init__(self, road_endpoints: torch.Tensor, state_len: int = STATE_LEN):
        super().__init__()
        self.encoder = GraphEncoder(road_endpoints, state_len=state_len)
        self.value_head = nn.Sequential(
            nn.Linear(pooled_dim(state_len), HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        h, _, globals_ = self.encoder(state)
        return self.value_head(_pool(h, globals_))
