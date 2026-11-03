import argparse
import csv
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from train.network_graph import NUM_LAYERS
from train.ppo import (
    ACTOR_PASSES,
    CRITIC_PASSES,
    DISCOUNT_FACTOR,
    ADVANTAGES,
    DEFAULT_ADVANTAGE,
    DEFAULT_NETWORK,
    GAE_LAMBDA,
    LEARNING_RATE,
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
LR_DECAYS = ("constant", "linear")
# With --lr-decay linear, the learning rate ends at this fraction of
# LEARNING_RATE, so the last iterations still learn and a resumed run can
# continue from there.
LR_FLOOR = 0.1
VALIDATE_EVERY = 10
VALIDATE_MATCHES_RANDOM = 100
VALIDATE_MATCHES_CHECKPOINT = 200
LOG_DIR = Path("logs")
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
CHECKPOINT_ROOT = Path("checkpoints")
# The run's settings, in its checkpoint dir.
RUN_CONFIG = "config.json"
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
        "--actor-layers",
        type=int,
        help="attention layers in the graph actor; the critic keeps "
        f"{NUM_LAYERS} (default: {NUM_LAYERS}, or the resumed run's)",
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
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="games played in parallel while collecting; each iteration "
        f"collects --batch-size x {NUM_STEPS} decisions (default: %(default)s)",
    )
    parser.add_argument(
        "--actor-passes",
        type=int,
        default=ACTOR_PASSES,
        help="passes over each iteration's buffer for the actor "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--critic-passes",
        type=int,
        help="passes for the critic, at most --actor-passes (default: "
        + ", ".join(f"{n} {p}" for n, p in CRITIC_PASSES.items())
        + ", capped at --actor-passes)",
    )
    parser.add_argument(
        "--lr-decay",
        choices=LR_DECAYS,
        default="constant",
        help=f"constant: keep the learning rate where it starts (the resumed "
        f"run's rate, or {LEARNING_RATE:g}). linear: "
        "lower it linearly over this invocation's --iterations, from where it "
        f"starts (the resumed run's rate, or {LEARNING_RATE:g}) to "
        f"{LR_FLOOR:g} x {LEARNING_RATE:g} (default: %(default)s)",
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
    if args.actor_layers is not None and args.actor_layers < 1:
        parser.error("--actor-layers must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.actor_passes < 1:
        parser.error("--actor-passes must be at least 1")
    if args.critic_passes is not None and not (
        1 <= args.critic_passes <= args.actor_passes
    ):
        parser.error("--critic-passes must be between 1 and --actor-passes")
    return args


def git_revision() -> dict:
    """Commit the code ran from, and whether it had uncommitted changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status.strip())}


def write_config(checkpoint_dir: Path, config: dict, start_iter: int) -> dict:
    """Write the run's `config.json` and return it. A resumed run keeps its
    `started_at`, takes this invocation's settings, and adds an entry to
    `resumes` with the settings that changed ({name: [old, new]})."""
    path = checkpoint_dir / RUN_CONFIG
    now = datetime.now().isoformat(timespec="seconds")
    previous = json.loads(path.read_text()) if path.is_file() else None
    resumes = previous["resumes"] if previous else []
    if start_iter:
        changed = {
            key: [previous.get(key), value]
            for key, value in config.items()
            if previous is not None
            and key not in ("iterations", "git")
            and previous.get(key) != value
        }
        resumes = resumes + [{"at": now, "from_iter": start_iter, "changed": changed}]
    config = {
        "run": checkpoint_dir.name,
        "started_at": previous["started_at"] if previous else now,
        **config,
        "resumes": resumes,
    }
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n")
    return config


def main():
    args = parse_args()

    resume_path = None
    if args.resume:
        checkpoint_dir = Path(args.resume)
        if not checkpoint_dir.is_dir():
            checkpoint_dir = CHECKPOINT_ROOT / args.resume
        resume_path, resumed_network, resumed_layers = find_resume_checkpoint(
            checkpoint_dir
        )
        if args.network is not None and args.network != resumed_network:
            raise SystemExit(
                f"--network {args.network} doesn't match {resume_path} "
                f"({resumed_network})"
            )
        if args.actor_layers is not None and args.actor_layers != resumed_layers:
            raise SystemExit(
                f"--actor-layers {args.actor_layers} doesn't match {resume_path} "
                f"({resumed_layers})"
            )
        args.network = resumed_network
        args.actor_layers = resumed_layers
    else:
        checkpoint_dir = CHECKPOINT_ROOT / RUN_TIMESTAMP
        args.network = args.network or DEFAULT_NETWORK
        if args.network == "ff" and args.actor_layers is not None:
            raise SystemExit("--actor-layers only applies to --network graph")
        args.actor_layers = args.actor_layers or NUM_LAYERS
    latest_checkpoint = checkpoint_dir / "actor_latest.pt"
    training_state = checkpoint_dir / TRAINING_STATE

    ppo = PPO(
        batch_size=args.batch_size,
        network=args.network,
        compile=args.compile,
        advantage=args.advantage,
        gae_lambda=args.gae_lambda,
        actor_layers=args.actor_layers,
        actor_passes=args.actor_passes,
        critic_passes=args.critic_passes,
    )
    start_iter = 0
    if resume_path is not None:
        start_iter, critic_restored = ppo.load_checkpoint(resume_path)
        tqdm.write(
            f"resumed {resume_path} after iteration {start_iter}"
            + ("" if critic_restored else " (actor only; critic starts fresh)")
        )

    # Logs are named after the run; its settings are in config.json. A resumed
    # run's "-fromN" keeps each invocation's logs separate. Iteration numbers
    # continue across them.
    run_name = checkpoint_dir.name
    if start_iter:
        run_name += f"-from{start_iter}"
    train_log = LOG_DIR / f"train-{run_name}.csv"
    validation_log = LOG_DIR / f"validation-{run_name}.csv"

    config = {
        # Total iterations once this invocation finishes.
        "iterations": start_iter + args.iterations,
        "network": args.network,
        "actor_layers": args.actor_layers,
        "advantage": args.advantage,
        "gae_lambda": args.gae_lambda,
        "discount_factor": DISCOUNT_FACTOR,
        "batch_size": args.batch_size,
        "num_steps": NUM_STEPS,
        "mini_batch_size": MINI_BATCH_SIZE,
        "actor_passes": ppo.actor_passes,
        "critic_passes": ppo.critic_passes,
        "lr_decay": args.lr_decay,
        "lr_start": ppo.lr,
        "lr_end": LR_FLOOR * LEARNING_RATE if args.lr_decay == "linear" else ppo.lr,
        "compile": args.compile,
        "baseline_checkpoint": BASELINE_CHECKPOINT,
        "git": git_revision(),
    }
    # Every checkpoint saved from here on carries the run's config.
    ppo.config = write_config(checkpoint_dir, config, start_iter)
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
            "lr",
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
            f"network: {args.network}"
            + (f" (actor layers {args.actor_layers})" if args.network == "graph" else "")
            + f", compile: {args.compile}, "
            f"advantage: {args.advantage}"
            + (f" (lambda {args.gae_lambda})" if args.advantage == "gae" else "")
            + f", lr: {ppo.lr:g} ({args.lr_decay})"
            + f", games: {args.batch_size} x {NUM_STEPS} steps"
            + f", passes: actor {ppo.actor_passes} / critic {ppo.critic_passes}"
            + f", baseline: {BASELINE_CHECKPOINT}"
        )
        completed = start_iter
        pbar = tqdm(
            range(start_iter, start_iter + args.iterations),
            desc="iter",
            initial=start_iter,
            total=start_iter + args.iterations,
        )
        # A resumed run restores its learning rate with its optimizers, so a
        # linear decay picks up from there instead of jumping back up.
        lr_start = ppo.lr
        lr_end = LR_FLOOR * LEARNING_RATE
        for i in pbar:
            if args.lr_decay == "linear":
                progress = (i - start_iter) / max(args.iterations - 1, 1)
                ppo.set_lr(lr_start + (lr_end - lr_start) * progress)
            ppo.observe(NUM_STEPS)
            metrics = ppo.train(mini_batch_size=MINI_BATCH_SIZE)
            pbar.set_postfix(
                loss=f"{metrics['loss']:.3f}",
                actor=f"{metrics['actor_loss']:.3f}",
                critic=f"{metrics['critic_loss']:.3f}",
                entropy=f"{metrics['entropy']:.3f}",
                ev=f"{metrics['explained_var']:.2f}",
                ev_mc=f"{metrics['explained_var_mc']:.2f}",
                lr=f"{ppo.lr:.2g}",
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
                    f"{ppo.lr:.3g}",
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
