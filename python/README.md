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
(default 3; the critic always has 3). Each layer lets a node see one road
further across the board.

```sh
uv run python -m train --actor-layers 6
```

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
is always `--critic-lr-scale` times the actor's (default 10, so 1e-3), and
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
run at a fixed 3e-5 with a critic at 10x:

```sh
uv run python -m train --resume 20261102-230211 --iterations 200 \
  --actor-lr-min 3e-5 --critic-lr-scale 10
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

Running the game server (API on port 8000). Override with `HOST` / `PORT`.

```sh
uv run python -m server
```

Then, in another terminal, run the web UI from `web/` and open
http://localhost:5173. Vite proxies `/games` and `/checkpoints` to port 8000.

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

