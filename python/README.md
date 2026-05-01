# PPO

## Commands

All commands should run the current directory (python)

Running the train.

```sh
uv run python -m train
```

To prevent building the rust library every time, use the `--no-sync` flag.

Running the tests.

```sh
uv run python -m pytest
```

Running the profiler

```sh
samply record uv run --no-sync python -m train
```

