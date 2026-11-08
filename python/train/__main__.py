import argparse
import csv
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

from train.network_graph import NUM_LAYERS
from train.ppo import (
    ACTOR_HEADS,
    ACTOR_PASSES,
    ADVANTAGES,
    CRITIC_LR_SCALE,
    CRITIC_PASSES,
    DEFAULT_ACTOR_HEAD,
    DEFAULT_ADVANTAGE,
    DEFAULT_NETWORK,
    DISCOUNT_FACTOR,
    GAE_LAMBDA,
    LEARNING_RATE,
    MAX_GRAD_NORM,
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
# The actor's learning rate never goes below --actor-lr-min: a linear decay ends
# there, and a resumed run whose rate is lower is raised to it. The critic's is
# always --critic-lr-scale times the actor's.
ACTOR_LR_MIN = 3e-5
# Runs from before --critic-lr-scale trained the critic at the actor's rate.
LEGACY_CRITIC_LR_SCALE = 1.0
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
        "--actor-head",
        choices=ACTOR_HEADS,
        help="how the graph actor scores actions. entity-robber: a road from "
        "its two end nodes, a settlement or city from its node, a robber move "
        "from its tile's six corners, everything else from a pooled board "
        "summary. entity: the same, but robber moves come from the pooled "
        "summary too (graph runs up to 2026-11-06). transformer: a board transformer over the node, "
        "road and global tokens, then learned action tokens that read the board "
        "through cross-attention; each logit is read from its action token. "
        "transformer-wired: the same, but each action token starts from the "
        "board tokens it acts on (its road and end nodes, its node, its tile's "
        "corners, or the global token). Both transformer heads are slower and "
        "run in bfloat16 on the GPU "
        f"(default: {DEFAULT_ACTOR_HEAD}, or the resumed run's)",
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
        "--mini-batch-size",
        type=int,
        help="samples per optimizer step. Bigger means fewer, less noisy "
        "updates per iteration, so the policy drifts less from the one that "
        "collected the data. Doesn't need to fit in GPU memory: see "
        f"--micro-batch-size (default: {MINI_BATCH_SIZE})",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        help="samples per forward/backward pass, at most --mini-batch-size. "
        "A smaller value splits each minibatch into micro-batches and adds up "
        "their gradients before one optimizer step (gradient accumulation): "
        "the same update with less GPU memory. E.g. --mini-batch-size 4096 "
        "--micro-batch-size 1024 updates on 4096 samples with the memory of "
        "1024 (default: --mini-batch-size, i.e. no accumulation)",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=MAX_GRAD_NORM,
        help="before each optimizer step, scale the actor's and the critic's "
        "gradients (separately) down to this total norm if they are above it; "
        "caps the size of any single update. 0 turns it off. Runs from before "
        "2026-11-05 had none (default: %(default)s)",
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
        help=f"constant: keep the learning rate where it starts ({LEARNING_RATE:g}, "
        "or where a resumed run stopped). linear: lower it linearly from "
        f"{LEARNING_RATE:g} to --actor-lr-min over --iterations. A resumed run "
        "continues its own schedule, and stays at --actor-lr-min past its end "
        "(default: constant, or the resumed run's)",
    )
    parser.add_argument(
        "--actor-lr-min",
        type=float,
        help="the actor's lowest learning rate: where a linear decay ends, and what a "
        f"resumed run with a lower rate is raised to (default: {ACTOR_LR_MIN:g}, "
        "or the resumed run's)",
    )
    parser.add_argument(
        "--critic-lr-scale",
        type=float,
        help="the critic's learning rate as a multiple of the actor's; it "
        f"follows the same decay (default: {CRITIC_LR_SCALE:g}, or the resumed "
        "run's)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        help="iterations to train in this invocation, on top of any resumed "
        f"ones (default: {NUM_ITERATIONS}, or what a resumed run has left of "
        "its planned total)",
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
    if args.critic_lr_scale is not None and args.critic_lr_scale <= 0:
        parser.error("--critic-lr-scale must be positive")
    if args.actor_lr_min is not None and args.actor_lr_min <= 0:
        parser.error("--actor-lr-min must be positive")
    if args.iterations is not None and args.iterations < 1:
        parser.error("--iterations must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    # Resolved here, not as argparse defaults, so tests can patch
    # MINI_BATCH_SIZE.
    if args.mini_batch_size is None:
        args.mini_batch_size = MINI_BATCH_SIZE
    if args.micro_batch_size is None:
        args.micro_batch_size = args.mini_batch_size
    if args.mini_batch_size < 1:
        parser.error("--mini-batch-size must be at least 1")
    if not 1 <= args.micro_batch_size <= args.mini_batch_size:
        parser.error("--micro-batch-size must be between 1 and --mini-batch-size")
    if args.max_grad_norm < 0:
        parser.error("--max-grad-norm must be 0 (off) or positive")
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


def read_config(checkpoint_dir: Path) -> dict | None:
    path = checkpoint_dir / RUN_CONFIG
    return json.loads(path.read_text()) if path.is_file() else None


def resolve_lr_settings(
    args: argparse.Namespace, previous: dict | None, start_iter: int
) -> None:
    """Fill in the learning-rate flags and --iterations that weren't given:
    from the resumed run's config.json, else the defaults."""
    prev = previous or {}
    if args.lr_decay is None:
        args.lr_decay = prev.get("lr_decay", "constant")
    if args.actor_lr_min is None:
        # Runs from before --actor-lr-min recorded only where the decay ended.
        legacy_end = prev.get("lr_end") if prev.get("lr_decay") == "linear" else None
        args.actor_lr_min = prev.get("actor_lr_min", legacy_end or ACTOR_LR_MIN)
    if args.critic_lr_scale is None:
        args.critic_lr_scale = (
            prev.get("critic_lr_scale", LEGACY_CRITIC_LR_SCALE)
            if previous
            else CRITIC_LR_SCALE
        )
    if args.iterations is None:
        if not start_iter:
            args.iterations = NUM_ITERATIONS
        else:
            remaining = prev.get("iterations", 0) - start_iter
            if remaining < 1:
                raise SystemExit(
                    f"the run already finished its {start_iter} iterations; "
                    "pass --iterations to train more"
                )
            args.iterations = remaining


def lr_schedule(
    args: argparse.Namespace, previous: dict | None, start_iter: int, current_lr: float
) -> dict:
    """The actor's learning-rate schedule as config fields. A resumed linear
    run keeps its start rate and iteration range (only the end follows
    --actor-lr-min), so an interrupted run carries on as planned."""
    if args.lr_decay == "constant":
        return {"actor_lr_start": max(current_lr, args.actor_lr_min)}
    prev = previous or {}
    if prev.get("lr_decay") == "linear":
        start = prev.get("actor_lr_start", prev.get("lr_start"))
        # Older configs only know the total; that is the range as long as the
        # run was never resumed.
        legacy = not prev.get("resumes")
        decay_from = prev.get("lr_decay_from", 0 if legacy else None)
        decay_to = prev.get("lr_decay_to", prev.get("iterations") if legacy else None)
        if None not in (start, decay_from, decay_to):
            return {
                "actor_lr_start": start,
                "actor_lr_end": args.actor_lr_min,
                "lr_decay_from": decay_from,
                "lr_decay_to": decay_to,
            }
    # A new decay: from here (a fresh run's 1e-4, or where a resumed run
    # stopped) down to --actor-lr-min over this invocation.
    return {
        "actor_lr_start": max(current_lr, args.actor_lr_min),
        "actor_lr_end": args.actor_lr_min,
        "lr_decay_from": start_iter,
        "lr_decay_to": start_iter + args.iterations,
    }


def scheduled_actor_lr(schedule: dict, iteration: int, lr_min: float) -> float:
    """The actor's learning rate for 0-based `iteration`."""
    start = schedule["actor_lr_start"]
    if "lr_decay_to" not in schedule:
        return max(start, lr_min)
    span = max(schedule["lr_decay_to"] - schedule["lr_decay_from"] - 1, 1)
    progress = min(max((iteration - schedule["lr_decay_from"]) / span, 0.0), 1.0)
    return max(start + (schedule["actor_lr_end"] - start) * progress, lr_min)


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
    previous_config = None
    if args.resume:
        checkpoint_dir = Path(args.resume)
        if not checkpoint_dir.is_dir():
            checkpoint_dir = CHECKPOINT_ROOT / args.resume
        (
            resume_path,
            resumed_network,
            resumed_layers,
            resumed_head,
        ) = find_resume_checkpoint(checkpoint_dir)
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
        if args.actor_head is not None and args.actor_head != resumed_head:
            raise SystemExit(
                f"--actor-head {args.actor_head} doesn't match {resume_path} "
                f"({resumed_head}); start a new run to change it"
            )
        args.network = resumed_network
        args.actor_layers = resumed_layers
        args.actor_head = resumed_head
        previous_config = read_config(checkpoint_dir)
    else:
        checkpoint_dir = CHECKPOINT_ROOT / RUN_TIMESTAMP
        args.network = args.network or DEFAULT_NETWORK
        if args.network == "ff" and args.actor_layers is not None:
            raise SystemExit("--actor-layers only applies to --network graph")
        if args.network == "ff" and args.actor_head is not None:
            raise SystemExit("--actor-head only applies to --network graph")
        args.actor_head = args.actor_head or DEFAULT_ACTOR_HEAD
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
        actor_head=args.actor_head,
        max_grad_norm=args.max_grad_norm or None,
    )
    start_iter = 0
    if resume_path is not None:
        start_iter, critic_restored = ppo.load_checkpoint(resume_path)
        tqdm.write(
            f"resumed {resume_path} after iteration {start_iter}"
            + ("" if critic_restored else " (actor only; critic starts fresh)")
        )
    resolve_lr_settings(args, previous_config, start_iter)
    ppo.critic_lr_scale = args.critic_lr_scale
    schedule = lr_schedule(args, previous_config, start_iter, ppo.actor_lr)
    # Sets the critic's rate from the actor's too, whatever the checkpoint had.
    ppo.set_actor_lr(scheduled_actor_lr(schedule, start_iter, args.actor_lr_min))

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
        "actor_head": ppo.actor_head,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "advantage": args.advantage,
        "gae_lambda": args.gae_lambda,
        "discount_factor": DISCOUNT_FACTOR,
        "batch_size": args.batch_size,
        "num_steps": NUM_STEPS,
        "mini_batch_size": args.mini_batch_size,
        # Samples per forward/backward; below mini_batch_size means gradient
        # accumulation. Runs from before 2026-11-05 had no accumulation.
        "micro_batch_size": args.micro_batch_size,
        "actor_passes": ppo.actor_passes,
        "critic_passes": ppo.critic_passes,
        # null: no gradient clipping, as in runs from before 2026-11-05.
        "max_grad_norm": ppo.max_grad_norm,
        "lr_decay": args.lr_decay,
        "actor_lr_min": args.actor_lr_min,
        # actor_lr_start, and for linear actor_lr_end and the 0-based
        # iterations it decays over (lr_decay_from to lr_decay_to).
        **schedule,
        "critic_lr_scale": args.critic_lr_scale,
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
            "actor_lr",
            "critic_lr",
            # Gradient norms before clipping: mean and max over the
            # iteration's steps, and the share of actor steps clipped.
            "actor_grad_norm",
            "actor_grad_norm_max",
            "actor_grad_clipped",
            "critic_grad_norm",
            "critic_grad_norm_max",
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
            "win_rate_vs_baseline_2v2",
            "avg_steps_vs_baseline_2v2",
            "avg_vp_vs_baseline_2v2",
            "avg_opp_vp_vs_baseline_2v2",
        ],
    )

    try:
        tqdm.write(
            f"network: {args.network}"
            + (
                f" (actor layers {args.actor_layers}, "
                f"actor head: {ppo.actor_head})"
                if args.network == "graph" else ""
            )
            + f", compile: {args.compile}, "
            f"advantage: {args.advantage}"
            + (f" (lambda {args.gae_lambda})" if args.advantage == "gae" else "")
            + f", lr: actor {ppo.actor_lr:g} / critic {ppo.critic_lr:g} ({args.lr_decay})"
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
        for i in pbar:
            ppo.set_actor_lr(scheduled_actor_lr(schedule, i, args.actor_lr_min))
            ppo.observe(NUM_STEPS)
            metrics = ppo.train(
                mini_batch_size=args.mini_batch_size,
                micro_batch_size=args.micro_batch_size,
            )
            pbar.set_postfix(
                loss=f"{metrics['loss']:.3f}",
                actor=f"{metrics['actor_loss']:.3f}",
                critic=f"{metrics['critic_loss']:.3f}",
                entropy=f"{metrics['entropy']:.3f}",
                ev=f"{metrics['explained_var']:.2f}",
                ev_mc=f"{metrics['explained_var_mc']:.2f}",
                actor_lr=f"{ppo.actor_lr:.2g}",
                kl=f"{metrics['approx_kl']:.4f}",
                gnorm=f"{metrics['actor_grad_norm']:.2f}",
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
                    f"{ppo.actor_lr:.3g}",
                    f"{ppo.critic_lr:.3g}",
                    f"{metrics['actor_grad_norm']:.6f}",
                    f"{metrics['actor_grad_norm_max']:.6f}",
                    f"{metrics['actor_grad_clipped']:.6f}",
                    f"{metrics['critic_grad_norm']:.6f}",
                    f"{metrics['critic_grad_norm_max']:.6f}",
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
                    f"2 vs 2 baseline: win {val_metrics['win_rate_vs_baseline_2v2']:.2%} "
                    f"vp {val_metrics['avg_vp_vs_baseline_2v2']:.2f}/{val_metrics['avg_opp_vp_vs_baseline_2v2']:.2f} "
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
                        f"{val_metrics['win_rate_vs_baseline_2v2']:.6f}",
                        f"{val_metrics['avg_steps_vs_baseline_2v2']:.2f}",
                        f"{val_metrics['avg_vp_vs_baseline_2v2']:.4f}",
                        f"{val_metrics['avg_opp_vp_vs_baseline_2v2']:.4f}",
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
