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

    state (B, 2016)
      -> GraphEncoder                    h (B, 54, H), roads (B, 72, 4), globals (B, 108)
      -> board tokens                    (B, 127, H)  + pos_emb
      -> TransformerEncoder (2 layers)   (B, 127, H)  board <-> board
      -> action tokens                   (B, 300, H)  action_emb
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

Train with `python -m train --actor-head transformer`.
"""

import torch
from torch import nn

from train.network_graph import (
    GLOBAL_DIM,
    HIDDEN_SIZE,
    NUM_HEADS,
    NUM_LAYERS,
    NUM_NODES,
    NUM_OTHER_ACTIONS,
    NUM_ROADS,
    ROAD_DIM,
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
# The usual transformer init for learned embeddings.
EMBED_INIT_STD = 0.02
# What the two transformers compute in on the GPU. bfloat16 keeps float32's
# exponent range (so no overflow or loss scaling, unlike float16) with an
# 8-bit mantissa: about 2-3 significant digits. That's fine inside the
# network, where every LayerNorm and softmax is still computed in float32 by
# autocast, but too coarse for the logits themselves: PPO compares
# log-probabilities from collection with ones from training, and rounding
# noise there would look like policy change. So the readout runs in float32.
# On the CPU (tests, the server) everything stays float32.
AUTOCAST_DTYPE = torch.bfloat16


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
    attend over the node, road and global tokens."""

    def __init__(
        self,
        road_endpoints: torch.Tensor,
        num_layers: int = NUM_LAYERS,
    ):
        super().__init__()
        # Same encoder as GraphActor: h is (B, 54, HIDDEN_SIZE).
        self.encoder = GraphEncoder(road_endpoints, num_layers)

        self.road_embed = nn.Linear(ROAD_DIM, HIDDEN_SIZE)
        self.global_embed = nn.Linear(GLOBAL_DIM, HIDDEN_SIZE)

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

        nn.init.normal_(self.pos_emb.weight, std=EMBED_INIT_STD)
        nn.init.normal_(self.action_emb.weight, std=EMBED_INIT_STD)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """(B, 2016) states -> (B, 300) logits, in action-index order."""
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

            # (B, 300, H)
            for layer in self.action_layers:
                x = layer(x, board)
            out = self.action_norm(x)

        # Outside autocast, in float32 (see AUTOCAST_DTYPE). (B, 300, 1)
        logits = self.readout(out.float())

        # (B, 300)
        logits = logits.squeeze(-1)

        return logits
