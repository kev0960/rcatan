"""Head-to-head win rate between two actor checkpoints.

Usage: python -m train.compare [A] [B] [--matches N]

A and B are each a run name under `checkpoints/` (e.g. 20261101-125508), a
run directory, or an actor `.pt` file; a run uses its `actor_latest.pt`.
Without them, A is the newest graph run and B the newest feed-forward run
that still match the engine's state size. Plays 1 A vs 3 B and 1 B vs 3 A
(parity 25%), then 2 A vs 2 B (parity 50%).
"""

import argparse
import math
from pathlib import Path

import torch

from train.ppo import DEVICE, LOADABLE_STATE_LENS, play_matches

CHECKPOINT_ROOT = Path("checkpoints")
LATEST = "actor_latest.pt"


def checkpoint_network(path: Path) -> tuple[str, int]:
    """(network, state_len) recorded in a checkpoint; legacy files are ff."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in ckpt:
        return ckpt["network"], ckpt["state_len"]
    return "ff", ckpt["fc1.weight"].shape[1]


def latest_checkpoint(network: str, root: Path = CHECKPOINT_ROOT) -> Path:
    """Newest run (run dirs are timestamps) whose latest actor is `network`
    and was trained on a state size that still loads."""
    for path in sorted(root.glob(f"*/{LATEST}"), reverse=True):
        net, state_len = checkpoint_network(path)
        if net == network and state_len in LOADABLE_STATE_LENS:
            return path
    raise FileNotFoundError(
        f"no {network} checkpoint with a loadable state size in {root}"
    )


def resolve_checkpoint(arg: str, root: Path = CHECKPOINT_ROOT) -> Path:
    """Actor file for a run name, run directory or `.pt` path."""
    path = Path(arg)
    if path.is_file():
        return path
    for run_dir in (path, root / arg):
        if (run_dir / LATEST).is_file():
            return run_dir / LATEST
    raise FileNotFoundError(f"no {LATEST} for {arg!r} (looked in {path}, {root / arg})")


def describe(path: Path) -> str:
    """Short label, e.g. "20261101-125508 (graph)" or "actor_iter_100.pt (ff)"."""
    network, _ = checkpoint_network(path)
    name = path.parent.name if path.name == LATEST else path.name
    return f"{name} ({network})"


def ci95(p: float, n: int) -> float:
    return 1.96 * math.sqrt(p * (1 - p) / n) if n > 0 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m train.compare")
    parser.add_argument(
        "a",
        nargs="?",
        help="run name, run directory or actor .pt (default: newest graph run)",
    )
    parser.add_argument(
        "b",
        nargs="?",
        help="run name, run directory or actor .pt (default: newest ff run)",
    )
    parser.add_argument(
        "--matches", type=int, default=500, help="games per matchup"
    )
    args = parser.parse_args()

    # Imported here so --help works without building the actors.
    from train.ppo import load_actor

    paths = {
        "A": resolve_checkpoint(args.a) if args.a else latest_checkpoint("graph"),
        "B": resolve_checkpoint(args.b) if args.b else latest_checkpoint("ff"),
    }
    actors = {name: load_actor(path, DEVICE) for name, path in paths.items()}
    for name, path in paths.items():
        print(f"{name}: {describe(path)}  [{path}]")
    print(f"{args.matches} games per matchup\n")

    for me, opp, n in (("A", "B", 1), ("B", "A", 1), ("A", "B", 2)):
        label = f"{n} {me} vs {4 - n} {opp}"
        m = play_matches(
            actors[me], actors[opp], args.matches, desc=label, num_seats=n
        )
        wr = m["win_rate"]
        print(
            f"{label:<10}: win {wr:6.1%} ± {ci95(wr, args.matches):.1%}"
            f" (parity {n / 4:.0%})"
            f"  | vp/seat {m['avg_vp']:.2f} vs {m['avg_opp_vp']:.2f}"
            f"  | steps {m['avg_steps']:.0f}"
        )


if __name__ == "__main__":
    main()
