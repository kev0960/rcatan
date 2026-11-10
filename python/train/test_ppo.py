import pytest
import numpy as np

from train.ppo import PPO


def _make_ppo(buf_players, buf_rewards, buf_game_ended_mask):
    # Skip __init__ so we don't construct Engine/networks.
    p = PPO.__new__(PPO)
    p.buf_states = [None] * len(buf_players)
    p.buf_players = buf_players
    p.buf_rewards = buf_rewards
    p.buf_game_ended_mask = buf_game_ended_mask
    return p


# Reproduce the example in compute_rewards's docstring (single game, B=1):
#
#   step  player   rewards at this step: r_p_t = 100*p + t
#   0     0        (  0, 100, 200, 300)
#   1     1        (  1, 101, 201, 301)
#   2     1        (  2, 102, 202, 302)
#   3     1        (  3, 103, 203, 303)
#   4     2        (  4, 104, 204, 304)
#   5     3        (  5, 105, 205, 305)
#   6     0        (  6, 106, 206, 306)  <-- last step, truncation finalizes
def test_compute_rewards_matches_docstring_example():
    players = [[0], [1], [1], [1], [2], [3], [0]]
    rewards = [
        np.array([[100 * p + t for p in range(4)]], dtype=np.float32) for t in range(7)
    ]
    ended = [[False]] * 7

    ppo = _make_ppo(players, rewards, ended)
    result = ppo.compute_rewards()

    expected = np.array(
        [
            [0 + 1 + 2 + 3 + 4 + 5],  # t=0, player 0: turns 0..5
            [100 + 101],  # t=1, player 1: finalized at step 2
            [102],  # t=2, player 1: finalized at step 3
            [103 + 104 + 105 + 106],  # t=3, player 1: finalized at truncation
            [
                200 + 201 + 202 + 203 + 204 + 205 + 206
            ],  # t=4, player 2: from step 0 through truncation
            [
                300 + 301 + 302 + 303 + 304 + 305 + 306
            ],  # t=5, player 3: from step 0 through truncation
            [6],  # t=6, player 0: finalized at truncation
        ],
        dtype=np.float32,
    )

    assert result.shape == (7, 1)
    np.testing.assert_allclose(result, expected, atol=1e-6)

    discounted_result = ppo.compute_discounted_rewards()
    expected = np.array(
        [
            [expected[0][0] + expected[6][0] * 0.99],
            [expected[1][0] + expected[2][0] * 0.99 + expected[3][0] * 0.99**2],
            [expected[2][0] + expected[3][0] * 0.99],
            [expected[3][0]],
            [expected[4][0]],
            [expected[5][0]],
            [expected[6][0]],
        ]
    )

    np.testing.assert_allclose(discounted_result, expected, atol=1e-6)


def test_compute_rewards_complex_case():
    players = [[0, 0], [1, 0], [1, 1], [1, 2], [2, 3], [3, 0], [0, 1]]
    rewards = [
        np.array(
            [[100 * p + t for p in range(4)], [100 * p + t for p in range(4)]],
            dtype=np.float32,
        )
        for t in range(7)
    ]
    ended = [[False, False]] * 7

    ppo = _make_ppo(players, rewards, ended)
    result = ppo.compute_rewards()

    expected = np.array(
        [
            [0 + 1 + 2 + 3 + 4 + 5, 0],
            [100 + 101, 1 + 2 + 3 + 4],
            [102, 100 + 101 + 102 + 103 + 104 + 105],
            [103 + 104 + 105 + 106, 200 + 201 + 202 + 203 + 204 + 205 + 206],
            [
                200 + 201 + 202 + 203 + 204 + 205 + 206,
                300 + 301 + 302 + 303 + 304 + 305 + 306,
            ],
            [300 + 301 + 302 + 303 + 304 + 305 + 306, 5 + 6],
            [6, 106],
        ],
        dtype=np.float32,
    )

    assert result.shape == (7, 2)
    np.testing.assert_allclose(result, expected, atol=1e-5)

    discounted_result = ppo.compute_discounted_rewards()
    expected = np.array(
        [
            [
                expected[0][0] + expected[6][0] * 0.99,
                expected[0][1] + expected[1][1] * 0.99 + expected[5][1] * 0.99**2,
            ],
            [
                expected[1][0] + expected[2][0] * 0.99 + expected[3][0] * 0.99**2,
                expected[1][1] + expected[5][1] * 0.99,
            ],
            [
                expected[2][0] + expected[3][0] * 0.99,
                expected[2][1] + expected[6][1] * 0.99,
            ],
            [expected[3][0], expected[3][1]],
            [expected[4][0], expected[4][1]],
            [expected[5][0], expected[5][1]],
            [expected[6][0], expected[6][1]],
        ]
    )

    np.testing.assert_allclose(discounted_result, expected, atol=1e-6)


def test_compute_rewards_finalizes_on_game_end():
    players = [[0], [1], [0], [1]]
    rewards = [
        np.array([[1.0, 10.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[2.0, 20.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[4.0, 40.0, 0.0, 0.0]], dtype=np.float32),  # game ends here
        np.array([[8.0, 80.0, 0.0, 0.0]], dtype=np.float32),
    ]
    ended = [[False], [False], [True], [False]]

    ppo = _make_ppo(players, rewards, ended)
    result = ppo.compute_rewards()

    expected = np.array(
        [
            [3.0],  # t=0, p0: r_0_0 + r_0_1
            [70.0],  # t=1, p1: finalized at game end = r_1_0 + r_1_1 + r_1_2
            [4.0],  # t=2, p0: finalized at game end = r_0_2
            [80.0],  # t=3, p1: finalized at truncation = r_1_3
        ],
        dtype=np.float32,
    )

    assert result.shape == (4, 1)
    np.testing.assert_allclose(result, expected, atol=1e-6)

    discounted_result = ppo.compute_discounted_rewards()
    expected = np.array(
        [
            [expected[0][0] + expected[2][0] * 0.99],
            [expected[1][0]],
            [expected[2][0]],
            [expected[3][0]],
        ]
    )
    np.testing.assert_allclose(discounted_result, expected, atol=1e-6)


def test_compute_rewards_handles_batch_independence():
    players = [
        [0, 1],
        [1, 0],
        [0, 1],
    ]
    rewards = [
        np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32),
        np.array([[2, 0, 0, 0], [0, 2, 0, 0]], dtype=np.float32),
        np.array([[4, 0, 0, 0], [0, 4, 0, 0]], dtype=np.float32),
    ]
    ended = [[False, False], [False, False], [False, False]]

    ppo = _make_ppo(players, rewards, ended)
    result = ppo.compute_rewards()

    # Columns are (batch=0, batch=1); both batches behave symmetrically.
    expected = np.array(
        [
            [3.0, 3.0],  # t=0
            [0.0, 0.0],  # t=1
            [4.0, 4.0],  # t=2
        ],
        dtype=np.float32,
    )

    assert result.shape == (3, 2)
    np.testing.assert_allclose(result, expected, atol=1e-6)

    discounted_result = ppo.compute_discounted_rewards()
    expected = np.array(
        [
            [
                expected[0][0] + expected[2][0] * 0.99,
                expected[0][1] + expected[2][1] * 0.99,
            ],
            [expected[1][0], expected[1][1]],
            [expected[2][0], expected[2][1]],
        ]
    )
    np.testing.assert_allclose(discounted_result, expected, atol=1e-6)


def test_vp_reward_scale_fades_linearly_between_start_and_end():
    from train.ppo import vp_reward_scale

    assert vp_reward_scale(None, 500) == 1.0
    fade = (150, 450)
    assert vp_reward_scale(fade, 0) == 1.0
    assert vp_reward_scale(fade, 149) == 1.0
    assert vp_reward_scale(fade, 150) == 1.0
    assert vp_reward_scale(fade, 300) == 0.5
    assert vp_reward_scale(fade, 449) == pytest.approx(1 / 300)
    assert vp_reward_scale(fade, 450) == 0.0
    assert vp_reward_scale(fade, 599) == 0.0


def test_scale_vp_rewards_keeps_the_win_reward():
    from train.ppo import scale_vp_rewards

    rewards = np.array(
        [[0.1, 0.0, -0.1, 0.2], [1.0, -1.0, -1.0, -1.0]], dtype=np.float32
    )
    ended = np.array([False, True])
    scaled = scale_vp_rewards(rewards, ended, 0.25)
    assert scaled.dtype == np.float32
    np.testing.assert_allclose(scaled[0], [0.025, 0.0, -0.025, 0.05])
    np.testing.assert_array_equal(scaled[1], rewards[1])
    assert scale_vp_rewards(rewards, ended, 1.0) is rewards
