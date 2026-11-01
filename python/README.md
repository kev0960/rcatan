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

The actor and critic run through `torch.compile` by default, which roughly
halves the time per iteration for the graph network. The first iteration is
slower while it compiles. Turn it off with `--no-compile`, for example when
debugging or stepping through the model:

```sh
uv run python -m train --no-compile
```

To also play a checkpoint from another run during validation (for example,
the feed-forward baseline while training the graph network), set
`CATAN_BASELINE_CHECKPOINT`:

```sh
CATAN_BASELINE_CHECKPOINT=checkpoints/<timestamp>/actor_latest.pt \
  uv run python -m train --network graph
```

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

