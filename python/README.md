# PPO

## Commands

All commands should run the current directory (python)

Running the train.

```sh
uv run python -m train
```

To prevent building the rust library every time, use the `--no-sync` flag.

Pick the actor and critic network with `--network` (default `graph`):

- `graph`: graph attention over the board's 54 intersections and 72 roads
  (`train/network_graph.py`)
- `ff`: feed-forward over the flat state vector (`train/network_ff.py`)

```sh
uv run python -m train --network ff
```

Checkpoints record which network they were trained with, so the server and
validation load either kind.

`--actor-layers` sets how many attention layers the graph actor has
(default 3; the critic always has 3). Each convolution passes local messages
one road further across the board.

```sh
uv run python -m train --actor-layers 6
```

Global fusion (pooling the board into every node after each convolution,
2026-11-03) was removed on 2026-11-05: it didn't beat the plain encoder.
Checkpoints trained with it no longer load; the others do.

`--actor-head` picks how the graph actor turns the encoder's output into the
300 action logits (default `entity-robber`):

- `entity-robber`: a road is scored from its two end nodes, a settlement or
  city from its node, a robber move from the six corner nodes of its tile,
  and everything else from a pooled summary of the board (`GraphActor` with
  `Engine.tile_nodes()`).
- `entity`: the same, except the robber moves also come from the pooled
  summary. Every graph run up to 2026-11-06 (e.g. 20261102-230211,
  20261103-115058) used it, and checkpoints that don't record a head load as
  `entity`.
- `transformer`: a 2-layer transformer lets the 54 node, 72 road and one
  global token attend to each other. Then 300 learned action tokens go
  through 2 cross-attention layers: they read the board tokens but not each
  other. Each logit is read from its action token, so each action learns
  where on the board to look (`train/network_transformer.py`). On the GPU
  the transformers run in bfloat16. It's several times slower to train than
  `entity`. Checkpoints from before 2026-11-05 (with action self-attention)
  no longer load.

```sh
uv run python -m train --actor-head transformer \
    --mini-batch-size 4096 --micro-batch-size 1024
```

`--mini-batch-size` is the samples per optimizer step (default 1024).
`--micro-batch-size` splits each minibatch into smaller forward/backward
passes whose gradients are added up before the step (gradient
accumulation), so a big minibatch fits in GPU memory. Both are recorded in
`config.json`.

`--max-grad-norm` (default 2.0; 0.5 in 20261105-222120, where it clipped
every step) scales the actor's and the critic's
gradients down to that total norm before each optimizer step when they are
above it, so no single update can be huge; 0 turns it off. Runs from before
2026-11-05 had none. The train log records the norms before clipping
(`actor_grad_norm`, `actor_grad_norm_max`, `critic_grad_norm`,
`critic_grad_norm_max`) and the share of actor steps clipped
(`actor_grad_clipped`).

Checkpoints and `config.json` record `actor_head`; older checkpoints are
`entity`. Resuming keeps the saved head. The critic is always `GraphCritic`.

The actor and critic run through `torch.compile` by default, which roughly
halves the time per iteration for the graph network. The first iteration is
slower while it compiles. Turn it off with `--no-compile`, for example when
debugging or stepping through the model:

```sh
uv run python -m train --no-compile
```

Choose how advantages and the critic's targets are computed with
`--advantage` (default `gae`):

- `gae`: generalized advantage estimation. `--gae-lambda` (default `0.95`)
  sets how far it looks ahead with real rewards before trusting the critic:
  `0` is one-step TD, `1` is Monte Carlo. A game the buffer cuts off
  bootstraps from the critic instead of treating the rest as worth 0.
- `mc`: Monte Carlo discounted returns to the end of the game (or the buffer).

```sh
uv run python -m train --advantage gae --gae-lambda 0.9
uv run python -m train --advantage mc
```

The training log has two explained-variance columns (1 = the critic predicts
perfectly, 0 = no better than always predicting the average, below 0 = worse):

- `explained_var`: against the critic's own targets. With GAE those are built
  partly from the critic's values, so this flatters the critic. Compare it
  only between runs that use the same setting.
- `explained_var_mc`: against the Monte Carlo returns, which the critic can't
  influence. Use this one to compare runs. In `mc` mode the two are equal.

Continue an earlier run with `--resume`, giving a run name under
`checkpoints/` or a run directory. `--iterations` sets how many more
iterations to train. Without it, a resume finishes what the run had planned
(for example, the last 50 of a 300-iteration run that stopped at 250):

```sh
uv run python -m train --resume 20261101-125508 --iterations 100
```

The run keeps its checkpoint directory, so the 50%/75% self-play checks
still find its earlier checkpoints, and iteration numbers carry on from where
it stopped. Logs go to new files ending in `-from<iter>`. Every validation
(and the end of training) saves `train_latest.pt` with the actor, critic and
optimizer state, and resuming restores all of it. Runs from before
`train_latest.pt` existed only restore the actor from their newest
`actor_iter_N.pt`. Their critic starts from scratch, so the first iterations
learn from noisy advantages. The network comes from the checkpoint, and
`--network` must match it if given. `--advantage` and `--gae-lambda` can
change between invocations.

The actor's learning rate is 1e-4 by default and stays there. The critic's
is always `--critic-lr-scale` times the actor's (default 1, the same rate), and
follows the same schedule. With `--lr-decay linear` it falls linearly over
`--iterations` to `--actor-lr-min` (default 3e-5), so late updates make
smaller, less noisy changes to a nearly trained policy:

```sh
uv run python -m train --lr-decay linear --iterations 300
```

The training log's `actor_lr` and `critic_lr` columns show the rates used each
iteration. `config.json` records the schedule: `actor_lr_start`, and for a
linear decay `actor_lr_end` and the iterations it spans (`lr_decay_from` to
`lr_decay_to`).

A resume reads the run's `config.json` and keeps its `--lr-decay`,
`--actor-lr-min` and `--critic-lr-scale` unless you pass them (runs from
before `--critic-lr-scale` used 1). The rate is computed from the schedule
for the iteration it resumes at, so an interrupted run continues exactly as
planned. Iterations past the schedule's end stay at `--actor-lr-min`, and a
new `--actor-lr-min` moves the end of the decay. Switching to linear on a
resume starts a new decay from where the run stopped. To continue a finished
run at a fixed 3e-5:

```sh
uv run python -m train --resume 20261102-230211 --iterations 200 \
  --actor-lr-min 3e-5
```

Each iteration collects `--batch-size` games in parallel (default 64) for
3000 steps, then trains the actor for `--actor-passes` passes over that data
(default 4) and the critic for `--critic-passes` (default 2 for graph, 4 for
ff, at most `--actor-passes`). Collecting more games with fewer passes keeps
the number of updates per iteration the same but uses each sample fewer
times, which leaves the critic less room to memorize the buffer's games:

```sh
uv run python -m train --batch-size 128 --actor-passes 2 --critic-passes 1
```

The buffer lives on the GPU: 64 games use about 6 GB for the graph network,
128 about twice that.

Each run's settings are in `checkpoints/<timestamp>/config.json`: network,
layers, advantage, passes, batch size, learning rate and decay, baseline, the
git commit (and whether it had uncommitted changes), and total iterations.
Logs are just `logs/train-<timestamp>.csv` and
`logs/validation-<timestamp>.csv`. Resuming updates `config.json` to the new
settings and adds an entry to its `resumes` list. Every checkpoint also
carries the config under `"config"`:

```python
import torch
torch.load("checkpoints/<timestamp>/actor_latest.pt", weights_only=True)["config"]
```

To also play a checkpoint from another run during validation (for example,
the feed-forward baseline while training the graph network), set
`CATAN_BASELINE_CHECKPOINT`:

```sh
CATAN_BASELINE_CHECKPOINT=checkpoints/<timestamp>/actor_latest.pt \
  uv run python -m train --network graph
```

Compare two trained actors head to head. Each argument is a run name under
`checkpoints/`, a run directory, or an actor `.pt` file (a run uses its
`actor_latest.pt`). Without arguments it plays the newest graph run against
the newest feed-forward run.

```sh
uv run python -m train.compare 20261101-125508 20261101-123034
uv run python -m train.compare 20261101-125508 checkpoints/20261101-123034/actor_iter_100.pt --matches 1000
```

It plays 1 A vs 3 B and 1 B vs 3 A (parity 25%), then 2 A vs 2 B (parity
50%), and prints each win rate with a 95% confidence interval.

Running the tests.

```sh
uv run python -m pytest
```

Running the profiler

```sh
samply record uv run --no-sync python -m train
```

Running the game server (API on port 8000, on all interfaces). Override with
`HOST` / `PORT`; `HOST=127.0.0.1` keeps it to this machine.

```sh
uv run python -m server
```

Then, in another terminal, run the web UI from `web/` and open
http://localhost:5173, or `http://<tailscale-ip-or-name>:5173` from another
device. Vite listens on all interfaces and proxies `/games`, `/checkpoints`
and `/training` to port 8000 on this machine.

```sh
cd ../web
pnpm dev
```

The server loads bots from `checkpoints/<timestamp>/actor_latest.pt`
(written by `python -m train`), relative to the directory it starts in. Each
run shows up in the UI's "bot:" selector; "latest" picks the newest one. If no
checkpoint loads, for example one trained on an older state-vector size, it
falls back to a random bot.

- `CATAN_CHECKPOINT_DIR`: checkpoint root (default `checkpoints`)
- `CATAN_ACTOR_CHECKPOINT`: pin a specific checkpoint file as the default bot

