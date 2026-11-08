"""Transformer actor: learned action tokens attend over the board.

The graph encoder (the same one `GraphActor` uses) turns the state into one
embedding per node. Together with the roads and the globals, those become 127
board tokens. Then:

- a board transformer (encoder) lets the board tokens attend to each other;
- an action reader runs 300 learned action tokens, one per action index.
  Each layer, every action token reads the board (cross-attention) and then
  goes through a feed-forward network. Action tokens never attend to each
  other, and the board never reads the actions.

Each action's logit is read from its own action token.

Nothing tells an action which part of the board it is about: action 89
("settlement at node 17") starts as a random vector and learns through
attention where to look.

The "transformer-wired" head (`tile_nodes` given) changes that: each action
token starts from the board tokens of what it acts on, plus its learned
embedding (see `wired_action_sources`):

    road r            mean of road token r and its two end nodes' tokens
    settlement/city n node token n
    robber to tile t  mean of the tile's six corner node tokens
    everything else   the global token

The cross-attention layers still let every action read the whole board, so
the network still learns what matters around each action, but no longer has
to find its own spot first. This is what the entity heads get from their
per-road and per-node readouts.

    state (B, 2031)
      -> GraphEncoder                    h (B, 54, H), roads (B, 72, 4), globals (B, 123)
      -> board tokens                    (B, 127, H)  + pos_emb
      -> TransformerEncoder (2 layers)   (B, 127, H)  board <-> board
      -> action tokens                   (B, 300, H)  action_emb
                                                      (+ own board tokens if wired)
      -> CrossAttentionLayer x 2         (B, 300, H)  action -> board only
      -> readout                         (B, 300) logits

Speed and memory (see the notes on each piece below):

- cross-attention only: no 300 x 300 action self-attention;
- FFN_SIZE = 2 x HIDDEN_SIZE instead of the usual 4x;
- on the GPU, the two transformers run in bfloat16 (autocast). The graph
  encoder and the readout stay in float32.

Checkpoints from before these changes (runs up to 2026-11-05, e.g.
20261104-230947) have a self-attention block and a 512-wide FFN in the action
decoder, so they no longer load into this class.

Train with `python -m train --actor-head transformer` (or
`transformer-wired`).
"""

import torch
from torch import nn

from train.network_graph import (
    HIDDEN_SIZE,
    NUM_HEADS,
    NUM_LAYERS,
    NUM_NODES,
    NUM_OTHER_ACTIONS,
    NUM_ROADS,
    NUM_TILES,
    ROAD_DIM,
    STATE_LEN,
    GraphEncoder,
)

# Number of actions.
NUM_ACTIONS = NUM_ROADS + 2 * NUM_NODES + NUM_OTHER_ACTIONS
# 127: one token per node, one per road, and one for the globals.
NUM_BOARD_TOKENS = NUM_NODES + NUM_ROADS + 1
# Layers in the board transformer (encoder) and the action reader.
BOARD_LAYERS = 2
ACTION_LAYERS = 2
# Width of the feed-forward network inside every transformer layer:
# Linear(HIDDEN_SIZE -> FFN_SIZE) -> ReLU -> Linear(FFN_SIZE -> HIDDEN_SIZE),
# applied to each token on its own. The original transformer used 4x
# HIDDEN_SIZE. Here it was the largest cost, because it runs on all 300 action
# tokens in every action layer: 2 x 128 x 512 = 131k multiply-adds per token
# per layer at 4x. 2x halves that compute and that activation memory. The
# first transformer run (20261104-230947, 4x) used 512.
FFN_SIZE = 2 * HIDDEN_SIZE
# The usual transformer init for learned embeddings (used for action_emb).
EMBED_INIT_STD = 0.02
# Init std for pos_emb, the per-board-token position embedding. It is added to
# the board tokens, and the node tokens come out of GraphEncoder's final
# LayerNorm with about std 1 per element. At 0.02 (run 20261105-222120) the
# position was about 2% of each token, so tokens with similar content (empty
# nodes, empty roads) looked almost the same to the attention, and an action
# token had little to tell "node 17" from "node 18" with. That makes learning
# where to look slow. At 1.0 the position is as large as the content from the
# start, so each board token is distinct, while still learned, not hand-wired.
POS_EMB_INIT_STD = 1.0
# What the two transformers compute in on the GPU. bfloat16 keeps float32's
# exponent range (so no overflow or loss scaling, unlike float16) with an
# 8-bit mantissa: about 2-3 significant digits. That's fine inside the
# network, where every LayerNorm and softmax is still computed in float32 by
# autocast, but too coarse for the logits themselves: PPO compares
# log-probabilities from collection with ones from training, and rounding
# noise there would look like policy change. So the readout runs in float32.
# On the CPU (tests, the server) everything stays float32.
AUTOCAST_DTYPE = torch.bfloat16
# Init std for action_emb in the wired head. There, a settlement and a city on
# the same node start from the same board token; only action_emb tells them
# apart. At EMBED_INIT_STD it would be about 2% of the token, so the two would
# start with almost the same logit and gradient. At 1.0 it is as large as the
# board part, like pos_emb.
WIRED_ACTION_EMB_INIT_STD = 1.0

# Board token layout: nodes, then roads, then the global token.
ROAD_TOKENS_START = NUM_NODES
GLOBAL_TOKEN = NUM_NODES + NUM_ROADS
# Action index layout (see `action_to_index` in crates/engine_py/src/lib.rs).
SETTLEMENT_ACTIONS_START = NUM_ROADS
CITY_ACTIONS_START = NUM_ROADS + NUM_NODES
OTHER_ACTIONS_START = NUM_ROADS + 2 * NUM_NODES
ROBBER_ACTIONS_START = OTHER_ACTIONS_START + 5  # after the 5 discards


def wired_action_sources(
    road_endpoints: torch.Tensor, tile_nodes: torch.Tensor
) -> torch.Tensor:
    """(300, 127) matrix: row a holds the weights of the board tokens action a
    starts from, summing to 1. The wired head's action tokens are
    `action_emb + sources @ board`."""
    endpoints = torch.as_tensor(road_endpoints, dtype=torch.long)
    corners = torch.as_tensor(tile_nodes, dtype=torch.long)
    assert endpoints.shape == (NUM_ROADS, 2)
    assert corners.shape == (NUM_TILES, 6)

    sources = torch.zeros(NUM_ACTIONS, NUM_BOARD_TOKENS)
    for r, (u, v) in enumerate(endpoints.tolist()):
        for token in (ROAD_TOKENS_START + r, u, v):
            sources[r, token] += 1 / 3
    for n in range(NUM_NODES):
        sources[SETTLEMENT_ACTIONS_START + n, n] = 1.0
        sources[CITY_ACTIONS_START + n, n] = 1.0
    robber = range(ROBBER_ACTIONS_START, ROBBER_ACTIONS_START + NUM_TILES)
    for a in range(OTHER_ACTIONS_START, NUM_ACTIONS):
        if a in robber:
            for node in corners[a - ROBBER_ACTIONS_START].tolist():
                sources[a, node] += 1 / 6
        else:
            sources[a, GLOBAL_TOKEN] = 1.0
    return sources


class CrossAttentionLayer(nn.Module):
    """One action-reader layer: each action token reads the board, then goes
    through a feed-forward network. Pre-norm, like the board transformer:

        x = x + cross_attn(norm1(x), board, board)
        x = x + ffn(norm2(x))

    This is `nn.TransformerDecoderLayer` minus its first block, the
    self-attention among the target (action) tokens. Why drop it:

    - It adds no information. Every action token already reads the whole
      board, which is everything the policy can know. Self-attention would only
      let an action reuse what another action token computed ("the best
      settlement spot is taken, so...").
    - The comparison between actions happens anyway, in the softmax over the
      300 logits.
    - It was about 30% of the action decoder's compute, and its 300 x 300
      attention scores (per head, per sample) were a big part of its memory.

    The cost: each action is judged independently from the board. If action
    tokens turn out to need each other, put the self-attention back.
    """

    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(HIDDEN_SIZE)
        # Queries come from the action tokens, keys and values from the board.
        self.cross_attn = nn.MultiheadAttention(
            HIDDEN_SIZE,
            NUM_HEADS,
            # PPO's old/new log-prob ratio breaks with dropout.
            dropout=0.0,
            # Our tensors are (B, tokens, H).
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(HIDDEN_SIZE)
        # Same activation (ReLU) as nn.TransformerEncoderLayer's default.
        self.ffn = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, FFN_SIZE),
            nn.ReLU(),
            nn.Linear(FFN_SIZE, HIDDEN_SIZE),
        )

    def forward(self, x: torch.Tensor, board: torch.Tensor) -> torch.Tensor:
        """x: (B, 300, H) action tokens; board: (B, 127, H). Returns (B, 300, H)."""
        q = self.norm1(x)
        # need_weights=False lets PyTorch use its fused attention kernels
        # (scaled_dot_product_attention), which never store the full
        # (B, heads, 300, 127) attention matrix. With True, it computes and
        # returns the head-averaged weights, which we never use.
        attended, _ = self.cross_attn(q, key=board, value=board, need_weights=False)
        x = x + attended
        return x + self.ffn(self.norm2(x))


class TransformerActor(nn.Module):
    """Policy over the 300 actions, scored from learned action tokens that
    attend over the node, road and global tokens.

    With `tile_nodes` (the "transformer-wired" head), each action token also
    starts from the board tokens it acts on; see `wired_action_sources`."""

    def __init__(
        self,
        road_endpoints: torch.Tensor,
        num_layers: int = NUM_LAYERS,
        state_len: int = STATE_LEN,
        tile_nodes: torch.Tensor | None = None,
    ):
        super().__init__()
        # Same encoder as GraphActor: h is (B, 54, HIDDEN_SIZE).
        self.encoder = GraphEncoder(road_endpoints, num_layers, state_len)

        self.road_embed = nn.Linear(ROAD_DIM, HIDDEN_SIZE)
        self.global_embed = nn.Linear(self.encoder.global_dim, HIDDEN_SIZE)

        # Specific embedding to indicate position in the board and action
        # index.
        self.pos_emb = nn.Embedding(NUM_BOARD_TOKENS, HIDDEN_SIZE)
        self.action_emb = nn.Embedding(NUM_ACTIONS, HIDDEN_SIZE)

        board_transformer_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_SIZE,
            nhead=NUM_HEADS,
            dim_feedforward=FFN_SIZE,
            # PPO's old/new log-prob ratio breaks with dropout.
            dropout=0.0,
            # Our tensors are (B, tokens, H).
            batch_first=True,
            # Pre-norm is more stable than post-norm.
            norm_first=True,
        )

        self.board_transformer = nn.TransformerEncoder(
            board_transformer_layer,
            num_layers=BOARD_LAYERS,
            # Pre-norm needs a final norm.
            norm=nn.LayerNorm(HIDDEN_SIZE),
            enable_nested_tensor=False,
        )

        self.action_layers = nn.ModuleList(
            CrossAttentionLayer() for _ in range(ACTION_LAYERS)
        )
        # Pre-norm needs a final norm here too (nn.TransformerDecoder's `norm`).
        self.action_norm = nn.LayerNorm(HIDDEN_SIZE)

        self.readout = nn.Linear(HIDDEN_SIZE, 1)

        self.wired = tile_nodes is not None
        if self.wired:
            # Not persistent: board data, like the encoder's road_endpoints.
            self.register_buffer(
                "action_sources",
                wired_action_sources(road_endpoints, tile_nodes),
                persistent=False,
            )

        nn.init.normal_(self.pos_emb.weight, std=POS_EMB_INIT_STD)
        nn.init.normal_(
            self.action_emb.weight,
            std=WIRED_ACTION_EMB_INIT_STD if self.wired else EMBED_INIT_STD,
        )
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """(B, 2031) states -> (B, 300) logits, in action-index order."""
        # The graph encoder stays in float32: it is a small share of the
        # actor's compute, and this keeps it identical to GraphActor's.
        h, roads, globals_ = self.encoder(state)
        b = state.shape[0]

        # Autocast runs matmuls (Linear, attention) in AUTOCAST_DTYPE and keeps
        # precision-sensitive ops (LayerNorm, softmax) in float32. The weights
        # stay float32: they are cast per op, and the optimizer updates the
        # float32 copies, so no separate master weights are needed. Off on the
        # CPU, where bfloat16 matmuls are slow and the tests compare in float32.
        on_gpu = state.device.type == "cuda"
        with torch.autocast(
            device_type=state.device.type, dtype=AUTOCAST_DTYPE, enabled=on_gpu
        ):
            # (B, 127, H)
            board_tokens = torch.cat(
                [
                    h,  # (B, 54, H)
                    self.road_embed(roads),  # (B, 72, H)
                    self.global_embed(globals_).unsqueeze(1),  # (B, 1, H)
                ],
                dim=1,
            )

            board_tokens = board_tokens + self.pos_emb.weight[None, :, :]

            # (B, 127, H)
            board = self.board_transformer(board_tokens)

            # (300, H) -> (B, 300, H)
            x = self.action_emb.weight.expand(b, NUM_ACTIONS, HIDDEN_SIZE)
            if self.wired:
                # Each action's own board tokens, after the board transformer.
                x = x + torch.einsum("at,bth->bah", self.action_sources, board)

            # (B, 300, H)
            for layer in self.action_layers:
                x = layer(x, board)
            out = self.action_norm(x)

        # Outside autocast, in float32 (see AUTOCAST_DTYPE). (B, 300, 1)
        logits = self.readout(out.float())

        # (B, 300)
        logits = logits.squeeze(-1)

        return logits
