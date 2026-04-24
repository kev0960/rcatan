import csv
from pathlib import Path
from tqdm import tqdm
from train.ppo import PPO

BATCH_SIZE = 32
SEED = 42
NUM_STEPS = 1000
NUM_ITERATIONS = 300
MINI_BATCH_SIZE = 1024
VALIDATE_EVERY = 10
VALIDATE_MATCHES = 200
LOG_DIR = Path("logs")
TRAIN_LOG = LOG_DIR / "train.csv"
VALIDATION_LOG = LOG_DIR / "validation.csv"


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
        VALIDATION_LOG, ["iter", "win_rate", "avg_steps"]
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
                val_metrics = ppo.validate(num_matches=VALIDATE_MATCHES)
                tqdm.write(
                    f"[iter {i + 1}] win rate vs 3 random: "
                    f"{val_metrics['win_rate']:.2%} "
                    f"avg steps: {val_metrics['avg_steps']:.1f} "
                    f"({VALIDATE_MATCHES} matches)"
                )
                val_writer.writerow(
                    [
                        i + 1,
                        f"{val_metrics['win_rate']:.6f}",
                        f"{val_metrics['avg_steps']:.2f}",
                    ]
                )
                val_file.flush()

    finally:
        train_file.close()
        val_file.close()


if __name__ == "__main__":
    main()
