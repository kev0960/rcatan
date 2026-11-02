import numpy as np
import pytest

from train.ppo import DISCOUNT_FACTOR, PPO

G = DISCOUNT_FACTOR


def _make_ppo(players, rewards, ended):
    # Skip __init__ so we don't construct Engine/networks.
    p = PPO.__new__(PPO)
    p.buf_states = [None] * len(players)
    p.buf_players = players
    p.buf_rewards = rewards
    p.buf_game_ended_mask = ended
    return p


def _random_buffer(T, B, seed, end_prob=0.08):
    rng = np.random.default_rng(seed)
    players = [rng.integers(0, 4, size=B).tolist() for _ in range(T)]
    rewards = [rng.normal(size=(B, 4)).astype(np.float32) for _ in range(T)]
    ended = [(rng.random(B) < end_prob).tolist() for _ in range(T)]
    values = rng.normal(size=(T, B)).astype(np.float32)
    return players, rewards, ended, values


def _reference_gae(ppo, values, lam):
    """Straightforward per-game, per-player GAE to check the vectorized one."""
    rewards = ppo.compute_rewards()
    players = np.asarray(ppo.buf_players)
    ended = np.asarray(ppo.buf_game_ended_mask, dtype=bool)
    T, B = values.shape
    adv = np.zeros((T, B))

    for b in range(B):
        # Split this slot's rows into games; the last one may be unfinished.
        games, start = [], 0
        for i in range(T):
            if ended[i, b]:
                games.append((range(start, i + 1), True))
                start = i + 1
        if start < T:
            games.append((range(start, T), False))

        for rows, finished in games:
            for p in range(4):
                mine = [i for i in rows if players[i, b] == p]
                deltas = []
                for k, i in enumerate(mine):
                    if k + 1 < len(mine):
                        v_next = values[mine[k + 1], b]
                    else:
                        v_next = 0.0 if finished else values[i, b]
                    deltas.append(rewards[i, b] + G * v_next - values[i, b])
                for k, i in enumerate(mine):
                    adv[i, b] = sum(
                        (G * lam) ** j * d for j, d in enumerate(deltas[k:])
                    )
    return adv


@pytest.mark.parametrize("lam", [0.0, 0.5, 0.95, 1.0])
def test_matches_reference(lam):
    players, rewards, ended, values = _random_buffer(T=120, B=5, seed=0)
    ppo = _make_ppo(players, rewards, ended)

    advantages, targets = ppo.compute_gae(values, lam)

    np.testing.assert_allclose(
        advantages, _reference_gae(ppo, values, lam), rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(targets, advantages + values, atol=1e-6)


def test_lambda_one_without_truncation_is_monte_carlo():
    """With every game finished inside the buffer, lambda = 1 gives exactly
    the discounted returns, whatever the critic says."""
    players, rewards, ended, values = _random_buffer(T=80, B=4, seed=1)
    for b in range(4):
        ended[-1][b] = True
    ppo = _make_ppo(players, rewards, ended)

    _, targets = ppo.compute_gae(values, 1.0)

    np.testing.assert_allclose(
        targets, ppo.compute_discounted_rewards(), rtol=1e-4, atol=1e-4
    )


def test_lambda_zero_is_one_step_td():
    # Single game, players 0 1 0 1, ends at the last step.
    players = [[0], [1], [0], [1]]
    rewards = [np.array([[1.0, 2.0, 0.0, 0.0]], dtype=np.float32)] * 4
    ended = [[False], [False], [False], [True]]
    values = np.array([[0.1], [0.2], [0.3], [0.4]], dtype=np.float32)
    ppo = _make_ppo(players, rewards, ended)
    R = ppo.compute_rewards()[:, 0]

    advantages, _ = ppo.compute_gae(values, 0.0)

    v = values[:, 0]
    expected = [
        R[0] + G * v[2] - v[0],  # player 0's next decision is step 2
        R[1] + G * v[3] - v[1],  # player 1's next decision is step 3
        R[2] + G * 0.0 - v[2],  # game over: the future is worth 0
        R[3] + G * 0.0 - v[3],
    ]
    np.testing.assert_allclose(advantages[:, 0], expected, rtol=1e-5)


def test_unfinished_game_bootstraps_from_own_last_value():
    players = [[0], [1]]
    rewards = [np.array([[1.0, 2.0, 0.0, 0.0]], dtype=np.float32)] * 2
    values = np.array([[0.5], [-0.7]], dtype=np.float32)
    ppo = _make_ppo(players, rewards, [[False], [False]])
    R = ppo.compute_rewards()[:, 0]

    advantages, _ = ppo.compute_gae(values, 0.95)

    # No later decision in the buffer: V_next stands in as the player's own V.
    expected = [R[0] + G * 0.5 - 0.5, R[1] + G * -0.7 - -0.7]
    np.testing.assert_allclose(advantages[:, 0], expected, rtol=1e-5)


def test_next_game_does_not_leak_into_previous_one():
    players, rewards, ended, values = _random_buffer(T=60, B=1, seed=2, end_prob=0)
    ended[29][0] = True  # first game ends at step 29
    ppo = _make_ppo(players, rewards, ended)
    before, _ = ppo.compute_gae(values, 0.95)

    values2 = values.copy()
    values2[30:] += 10.0
    for t in range(30, 60):
        rewards[t] = rewards[t] + 10.0
    after, _ = _make_ppo(players, rewards, ended).compute_gae(values2, 0.95)

    np.testing.assert_allclose(after[:30], before[:30], atol=1e-6)
