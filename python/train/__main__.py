import argparse
import csv
import os
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from train.ppo import (
    ADVANTAGES,
    DEFAULT_ADVANTAGE,
    DEFAULT_NETWORK,
    GAE_LAMBDA,
    NETWORKS,
    PPO,
    TRAINING_STATE,
    find_resume_checkpoint,
)

BATCH_SIZE = 64
SEED = 42
NUM_STEPS = 3000
NUM_ITERATIONS = 300
MINI_BATCH_SIZE = 1024
VALIDATE_EVERY = 10
VALIDATE_MATCHES_RANDOM = 100
VALIDATE_MATCHES_CHECKPOINT = 200
LOG_DIR = Path("logs")
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
CHECKPOINT_ROOT = Path("checkpoints")
# Actor checkpoint from another run to play head-to-head during validation,
# e.g. the feed-forward baseline when training the graph network.
BASELINE_CHECKPOINT = os.environ.get("CATAN_BASELINE_CHECKPOINT")


def _open_csv(path: Path, header: list[str]):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    f = path.open("w", newline="", buffering=1)
    writer = csv.writer(f)
    writer.writerow(header)
    f.flush()
    return f, writer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m train")
    parser.add_argument(
        "--network",
        choices=NETWORKS,
        help="actor and critic architecture: graph (graph attention over the "
        f"board) or ff (feed-forward) (default: {DEFAULT_NETWORK}, or the "
        "resumed run's)",
    )
    parser.add_argument(
        "--resume",
        metavar="RUN",
        help="continue a run: a run name under checkpoints/ or a run "
        "directory. Restores the actor, critic and optimizers from its "
        f"{TRAINING_STATE}; older runs without one restore only the actor "
        "from their newest actor_iter_N.pt",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=NUM_ITERATIONS,
        help="iterations to train in this invocation, on top of any resumed "
        "ones (default: %(default)s)",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run the actor and critic through torch.compile for faster "
        "training; the first iterations are slower while it compiles "
        "(default: on)",
    )
    parser.add_argument(
        "--advantage",
        choices=ADVANTAGES,
        default=DEFAULT_ADVANTAGE,
        help="how advantages and critic targets are computed: gae "
        "(generalized advantage estimation) or mc (Monte Carlo discounted "
        "returns) (default: %(default)s)",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=GAE_LAMBDA,
        help="GAE lambda: 0 = one-step TD, 1 = Monte Carlo; only used with "
        "--advantage gae (default: %(default)s)",
    )
    args = parser.parse_args()
    if not 0 <= args.gae_lambda <= 1:
        parser.error("--gae-lambda must be between 0 and 1")
    return args


def log_suffix(args: argparse.Namespace) -> str:
    """Run settings for log file names, e.g. "graph-gae0.95" or "ff-mc"."""
    advantage = (
        f"gae{args.gae_lambda:g}" if args.advantage == "gae" else args.advantage
    )
    return f"{args.network}-{advantage}"


def main():
    args = parse_args()

    resume_path = None
    if args.resume:
        checkpoint_dir = Path(args.resume)
        if not checkpoint_dir.is_dir():
            checkpoint_dir = CHECKPOINT_ROOT / args.resume
        resume_path, resumed_network = find_resume_checkpoint(checkpoint_dir)
        if args.network is not None and args.network != resumed_network:
            raise SystemExit(
                f"--network {args.network} doesn't match {resume_path} "
                f"({resumed_network})"
            )
        args.network = resumed_network
    else:
        checkpoint_dir = CHECKPOINT_ROOT / RUN_TIMESTAMP
        args.network = args.network or DEFAULT_NETWORK
    latest_checkpoint = checkpoint_dir / "actor_latest.pt"
    training_state = checkpoint_dir / TRAINING_STATE

    ppo = PPO(
        batch_size=BATCH_SIZE,
        network=args.network,
        compile=args.compile,
        advantage=args.advantage,
        gae_lambda=args.gae_lambda,
    )
    start_iter = 0
    if resume_path is not None:
        start_iter, critic_restored = ppo.load_checkpoint(resume_path)
        tqdm.write(
            f"resumed {resume_path} after iteration {start_iter}"
            + ("" if critic_restored else " (actor only; critic starts fresh)")
        )

    # A resumed run keeps its checkpoint dir's name; "-fromN" keeps each
    # invocation's logs separate. Iteration numbers continue across them.
    run_name = f"{checkpoint_dir.name}-{log_suffix(args)}"
    if start_iter:
        run_name += f"-from{start_iter}"
    train_log = LOG_DIR / f"train-{run_name}.csv"
    validation_log = LOG_DIR / f"validation-{run_name}.csv"
    train_file, train_writer = _open_csv(
        train_log,
        [
            "iter",
            "loss",
            "actor_loss",
            "critic_loss",
            "entropy",
            "return_mean",
            "return_std",
            "value_mean",
            "value_std",
            "explained_var",
            "approx_kl",
            "clip_frac",
            "explained_var_mc",
        ],
    )
    val_file, val_writer = _open_csv(
        validation_log,
        [
            "iter",
            "win_rate_vs_random",
            "avg_steps_vs_random",
            "avg_vp_vs_random",
            "avg_opp_vp_vs_random",
            "win_rate_vs_50",
            "avg_steps_vs_50",
            "avg_vp_vs_50",
            "avg_opp_vp_vs_50",
            "win_rate_vs_75",
            "avg_steps_vs_75",
            "avg_vp_vs_75",
            "avg_opp_vp_vs_75",
            "win_rate_vs_baseline",
            "avg_steps_vs_baseline",
            "avg_vp_vs_baseline",
            "avg_opp_vp_vs_baseline",
        ],
    )

    try:
        tqdm.write(
            f"network: {args.network}, compile: {args.compile}, "
            f"advantage: {args.advantage}"
            + (f" (lambda {args.gae_lambda})" if args.advantage == "gae" else "")
            + f", baseline: {BASELINE_CHECKPOINT}"
        )
        completed = start_iter
        pbar = tqdm(
            range(start_iter, start_iter + args.iterations),
            desc="iter",
            initial=start_iter,
            total=start_iter + args.iterations,
        )
        for i in pbar:
            ppo.observe(NUM_STEPS)
            metrics = ppo.train(mini_batch_size=MINI_BATCH_SIZE)
            pbar.set_postfix(
                loss=f"{metrics['loss']:.3f}",
                actor=f"{metrics['actor_loss']:.3f}",
                critic=f"{metrics['critic_loss']:.3f}",
                entropy=f"{metrics['entropy']:.3f}",
                ev=f"{metrics['explained_var']:.2f}",
                ev_mc=f"{metrics['explained_var_mc']:.2f}",
                kl=f"{metrics['approx_kl']:.4f}",
            )

            train_writer.writerow(
                [
                    i + 1,
                    f"{metrics['loss']:.6f}",
                    f"{metrics['actor_loss']:.6f}",
                    f"{metrics['critic_loss']:.6f}",
                    f"{metrics['entropy']:.6f}",
                    f"{metrics['return_mean']:.6f}",
                    f"{metrics['return_std']:.6f}",
                    f"{metrics['value_mean']:.6f}",
                    f"{metrics['value_std']:.6f}",
                    f"{metrics['explained_var']:.6f}",
                    f"{metrics['approx_kl']:.6f}",
                    f"{metrics['clip_frac']:.6f}",
                    f"{metrics['explained_var_mc']:.6f}",
                ]
            )
            train_file.flush()
            completed = i + 1

            if (i + 1) % VALIDATE_EVERY == 0:
                val_metrics = ppo.validate(
                    current_iter=i + 1,
                    num_matches_random=VALIDATE_MATCHES_RANDOM,
                    num_matches_checkpoint=VALIDATE_MATCHES_CHECKPOINT,
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_step=VALIDATE_EVERY,
                    baseline_checkpoint=BASELINE_CHECKPOINT,
                )
                tqdm.write(
                    f"[iter {i + 1}] "
                    f"vs random: win {val_metrics['win_rate_vs_random']:.2%} "
                    f"vp {val_metrics['avg_vp_vs_random']:.2f}/{val_metrics['avg_opp_vp_vs_random']:.2f} | "
                    f"vs 50%: win {val_metrics['win_rate_vs_50']:.2%} "
                    f"vp {val_metrics['avg_vp_vs_50']:.2f}/{val_metrics['avg_opp_vp_vs_50']:.2f} | "
                    f"vs 75%: win {val_metrics['win_rate_vs_75']:.2%} "
                    f"vp {val_metrics['avg_vp_vs_75']:.2f}/{val_metrics['avg_opp_vp_vs_75']:.2f} | "
                    f"vs baseline: win {val_metrics['win_rate_vs_baseline']:.2%} "
                    f"vp {val_metrics['avg_vp_vs_baseline']:.2f}/{val_metrics['avg_opp_vp_vs_baseline']:.2f} "
                    f"(random: {VALIDATE_MATCHES_RANDOM}, ckpt: {VALIDATE_MATCHES_CHECKPOINT} matches)"
                )
                val_writer.writerow(
                    [
                        i + 1,
                        f"{val_metrics['win_rate_vs_random']:.6f}",
                        f"{val_metrics['avg_steps_vs_random']:.2f}",
                        f"{val_metrics['avg_vp_vs_random']:.4f}",
                        f"{val_metrics['avg_opp_vp_vs_random']:.4f}",
                        f"{val_metrics['win_rate_vs_50']:.6f}",
                        f"{val_metrics['avg_steps_vs_50']:.2f}",
                        f"{val_metrics['avg_vp_vs_50']:.4f}",
                        f"{val_metrics['avg_opp_vp_vs_50']:.4f}",
                        f"{val_metrics['win_rate_vs_75']:.6f}",
                        f"{val_metrics['avg_steps_vs_75']:.2f}",
                        f"{val_metrics['avg_vp_vs_75']:.4f}",
                        f"{val_metrics['avg_opp_vp_vs_75']:.4f}",
                        f"{val_metrics['win_rate_vs_baseline']:.6f}",
                        f"{val_metrics['avg_steps_vs_baseline']:.2f}",
                        f"{val_metrics['avg_vp_vs_baseline']:.4f}",
                        f"{val_metrics['avg_opp_vp_vs_baseline']:.4f}",
                    ]
                )
                val_file.flush()

                ppo.save_actor(checkpoint_dir / f"actor_iter_{i + 1}.pt")
                ppo.save_actor(latest_checkpoint)
                ppo.save_training_state(training_state, completed)

    finally:
        train_file.close()
        val_file.close()
        ppo.save_actor(latest_checkpoint)
        ppo.save_training_state(training_state, completed)


if __name__ == "__main__":
    main()
