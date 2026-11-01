"""Head-to-head win rate between the latest graph and feed-forward actors.

Usage: python -m train.compare [--graph PATH] [--ff PATH] [--matches N]

Without paths, picks the newest `checkpoints/<run>/actor_latest.pt` of each
network that still matches the engine's state size. Plays 1 graph vs 3 ff
and 1 ff vs 3 graph (parity 25%), then 2 graph vs 2 ff (parity 50%).
"""

import argparse
import math
from pathlib import Path

import torch

from train.ppo import ACTOR_INPUT_SIZE, DEVICE, play_matches

CHECKPOINT_ROOT = Path("checkpoints")


def checkpoint_network(path: Path) -> tuple[str, int]:
    """(network, state_len) recorded in a checkpoint; legacy files are ff."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in ckpt:
        return ckpt["network"], ckpt["state_len"]
    return "ff", ckpt["fc1.weight"].shape[1]


def latest_checkpoint(network: str, root: Path = CHECKPOINT_ROOT) -> Path:
    """Newest run (run dirs are timestamps) whose latest actor is `network`
    and was trained on the current state size."""
    for path in sorted(root.glob("*/actor_latest.pt"), reverse=True):
        net, state_len = checkpoint_network(path)
        if net == network and state_len == ACTOR_INPUT_SIZE:
            return path
    raise FileNotFoundError(
        f"no {network} checkpoint with a {ACTOR_INPUT_SIZE}-float state in {root}"
    )


def ci95(p: float, n: int) -> float:
    return 1.96 * math.sqrt(p * (1 - p) / n) if n > 0 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m train.compare")
    parser.add_argument("--graph", type=Path, help="graph actor checkpoint")
    parser.add_argument("--ff", type=Path, help="feed-forward actor checkpoint")
    parser.add_argument(
        "--matches", type=int, default=500, help="games per matchup"
    )
    args = parser.parse_args()

    # Imported here so --help works without building the actors.
    from train.ppo import load_actor

    paths = {
        "graph": args.graph or latest_checkpoint("graph"),
        "ff": args.ff or latest_checkpoint("ff"),
    }
    actors = {name: load_actor(path, DEVICE) for name, path in paths.items()}
    for name, path in paths.items():
        print(f"{name:>5}: {path}")
    print(f"{args.matches} games per matchup\n")

    for me, opp, n in (("graph", "ff", 1), ("ff", "graph", 1), ("graph", "ff", 2)):
        label = f"{n} {me} vs {4 - n} {opp}"
        m = play_matches(
            actors[me], actors[opp], args.matches, desc=label, num_seats=n
        )
        wr = m["win_rate"]
        print(
            f"{label:<16}: win {wr:6.1%} ± {ci95(wr, args.matches):.1%}"
            f" (parity {n / 4:.0%})"
            f"  | vp/seat {m['avg_vp']:.2f} vs {m['avg_opp_vp']:.2f}"
            f"  | steps {m['avg_steps']:.0f}"
        )


if __name__ == "__main__":
    main()
