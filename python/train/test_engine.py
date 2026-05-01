import random

import numpy as np
from catan_engine import Engine


def test_engine_runs_one_game_until_end():
    random.seed(42)
    batch_size = 32
    engine = Engine(batch_size=batch_size, seed=42)

    for _ in range(1000):
        mask = engine.get_action_mask()
        chosen = [int(random.choice(np.flatnonzero(mask[i]))) for i in range(batch_size)]
        _, ended, final_vps = engine.step(chosen)
        if ended.any():
            assert final_vps[ended].max() >= 10
            return

    raise AssertionError("no game ended within 1000 iterations")
