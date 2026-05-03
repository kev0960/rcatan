import csv
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from train.ppo import PPO

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
TRAIN_LOG = LOG_DIR / f"train-{RUN_TIMESTAMP}.csv"
VALIDATION_LOG = LOG_DIR / f"validation-{RUN_TIMESTAMP}.csv"
CHECKPOINT_DIR = Path("checkpoints")
LATEST_CHECKPOINT = CHECKPOINT_DIR / "actor_latest.pt"


def _open_csv(path: Path, header: list[str]):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    f = path.open("w", newline="", buffering=1)
    writer = csv.writer(f)
    writer.writerow(header)
    f.flush()
    return f, writer


def main():
    train_file, train_writer = _open_csv(
        TRAIN_LOG,
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
        ],
    )
    val_file, val_writer = _open_csv(
        VALIDATION_LOG,
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
            "win_rate_vs_90",
            "avg_steps_vs_90",
            "avg_vp_vs_90",
            "avg_opp_vp_vs_90",
        ],
    )

    try:
        ppo = PPO(batch_size=BATCH_SIZE)
        pbar = tqdm(range(NUM_ITERATIONS), desc="iter")
        for i in pbar:
            ppo.observe(NUM_STEPS)
            metrics = ppo.train(mini_batch_size=MINI_BATCH_SIZE)
            pbar.set_postfix(
                loss=f"{metrics['loss']:.3f}",
                actor=f"{metrics['actor_loss']:.3f}",
                critic=f"{metrics['critic_loss']:.3f}",
                entropy=f"{metrics['entropy']:.3f}",
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
                ]
            )
            train_file.flush()

            if (i + 1) % VALIDATE_EVERY == 0:
                val_metrics = ppo.validate(
                    current_iter=i + 1,
                    num_matches_random=VALIDATE_MATCHES_RANDOM,
                    num_matches_checkpoint=VALIDATE_MATCHES_CHECKPOINT,
                    checkpoint_dir=CHECKPOINT_DIR,
                    checkpoint_step=VALIDATE_EVERY,
                )
                tqdm.write(
                    f"[iter {i + 1}] "
                    f"vs random: win {val_metrics['win_rate_vs_random']:.2%} "
                    f"vp {val_metrics['avg_vp_vs_random']:.2f}/{val_metrics['avg_opp_vp_vs_random']:.2f} | "
                    f"vs 50%: win {val_metrics['win_rate_vs_50']:.2%} "
                    f"vp {val_metrics['avg_vp_vs_50']:.2f}/{val_metrics['avg_opp_vp_vs_50']:.2f} | "
                    f"vs 75%: win {val_metrics['win_rate_vs_75']:.2%} "
                    f"vp {val_metrics['avg_vp_vs_75']:.2f}/{val_metrics['avg_opp_vp_vs_75']:.2f} | "
                    f"vs 90%: win {val_metrics['win_rate_vs_90']:.2%} "
                    f"vp {val_metrics['avg_vp_vs_90']:.2f}/{val_metrics['avg_opp_vp_vs_90']:.2f} "
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
                        f"{val_metrics['win_rate_vs_90']:.6f}",
                        f"{val_metrics['avg_steps_vs_90']:.2f}",
                        f"{val_metrics['avg_vp_vs_90']:.4f}",
                        f"{val_metrics['avg_opp_vp_vs_90']:.4f}",
                    ]
                )
                val_file.flush()

                ppo.save_actor(CHECKPOINT_DIR / f"actor_iter_{i + 1}.pt")
                ppo.save_actor(LATEST_CHECKPOINT)

    finally:
        train_file.close()
        val_file.close()
        ppo.save_actor(LATEST_CHECKPOINT)


if __name__ == "__main__":
    main()
