"""Benchmark one of our trained actors against catanatron's bots.

    uv run --no-sync python -m bench.catanatron RUN [--opponent F] [--games 200]
        [--seats 1] [--workers 8] [--seed 0]

RUN is a run name under checkpoints/, a run directory or an actor `.pt` file
(as in `train.compare`). `--opponent` is a catanatron player spec, filling
every seat the bot doesn't take:

    R       random
    W       weighted random (prefers building)
    VP      greedy on victory points
    F       value function: greedy on a hand-tuned board evaluation
    AB:2    alpha-beta search, depth 2, on the same evaluation (the strongest
            built-in bot, and the slowest)
    M:100   Monte Carlo tree search, 100 simulations

`--seats` is how many of the four seats our bot plays (1-3). It wins a game
if any of its seats wins; parity is seats / 4 (25% for 1 vs 3, 50% for 2 vs
2). Seat order is shuffled every game. Games use catanatron's rules and
boards (10 VP, discard over 7, base map, ports shuffled), and stop at its
turn limit; those count as no winner.
"""

import argparse
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor

from bench.catanatron.games import play_games


def main():
    parser = argparse.ArgumentParser(
        prog="python -m bench.catanatron",
        description="Play a trained actor against catanatron's bots.",
    )
    parser.add_argument("run", help="run name, run directory or actor .pt file")
    parser.add_argument(
        "--opponent", default="F", help="catanatron player spec (default: F)"
    )
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument(
        "--seats", type=int, default=1, choices=[1, 2, 3],
        help="seats our bot plays (default: 1, i.e. 1 vs 3)",
    )
    parser.add_argument(
        "--workers", type=int, default=min(8, os.cpu_count() or 1),
        help="parallel game processes (CPU only)",
    )
    parser.add_argument("--seed", type=int, default=0, help="first game's seed")
    args = parser.parse_args()

    from tqdm import tqdm

    from train.compare import describe, resolve_checkpoint

    checkpoint = resolve_checkpoint(args.run)
    print(
        f"{describe(checkpoint)} x{args.seats} vs {args.opponent} "
        f"x{4 - args.seats}, {args.games} games, {args.workers} workers"
    )

    seeds = list(range(args.seed, args.seed + args.games))
    start = time.time()
    results = []
    workers = max(1, min(args.workers, args.games))
    with ProcessPoolExecutor(workers) as pool, tqdm(total=args.games) as bar:
        # One game per task so the bar moves; each worker loads the actor once
        # per task, which is cheap next to a game.
        futures = [
            pool.submit(play_games, str(checkpoint), args.opponent, args.seats, [s])
            for s in seeds
        ]
        for f in futures:
            results.extend(f.result())
            bar.update(1)

    finished = [r for r in results if r[0] is not None]
    wins = sum(r[0] for r in finished)
    n = len(results)
    rate = wins / n
    stderr = math.sqrt(rate * (1 - rate) / n) if n else 0.0
    parity = args.seats / 4
    our_vp = sum(sum(r[1]) for r in results) / (n * args.seats)
    their_vp = sum(sum(r[2]) for r in results) / (n * (4 - args.seats))
    turns = sum(r[3] for r in results) / n
    print(
        f"win {rate:.1%} ± {1.96 * stderr:.1%} (parity {parity:.0%}) | "
        f"vp/seat {our_vp:.2f} vs {their_vp:.2f} | turns {turns:.0f} | "
        f"no winner {n - len(finished)} | {time.time() - start:.0f}s"
    )


if __name__ == "__main__":
    main()
