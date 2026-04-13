import numpy as np
from catan_engine import Engine

BATCH_SIZE = 32
SEED = 42
NUM_STEPS = 1000


def main():
    engine = Engine(batch_size=BATCH_SIZE, seed=SEED)

    for step in range(NUM_STEPS):
        masks = np.array(engine.get_action_mask())  # (BATCH_SIZE, 284) bool

        # Pick a random valid action per game.
        actions = [int(np.random.choice(np.where(m)[0])) for m in masks]

        engine.step(actions)

        if step % 100 == 0:
            avg_actions = masks.sum(axis=1).mean()
            print(f"step {step:>4d}  avg valid actions: {avg_actions:.1f}")

    print("done")


if __name__ == "__main__":
    main()
