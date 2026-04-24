import numpy as np
from catan_engine import Engine
from train.ppo import PPO

BATCH_SIZE = 32
SEED = 42
NUM_STEPS = 1000


def main():
    ppo = PPO(batch_size=BATCH_SIZE)
    ppo.observe(NUM_STEPS)
    ppo.train()

    print("done")


if __name__ == "__main__":
    main()
