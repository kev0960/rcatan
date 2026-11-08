"""`RcatanPlayer` in real catanatron games.

    uv run --no-sync python -m pytest -q bench/catanatron
"""

from collections import Counter

import catanatron.players  # noqa: F401  (registers the built-in bots)
import numpy as np
import pytest
import torch
from catanatron import ActionType, Color, Game
from catanatron.registry import REGISTRY

from bench.catanatron.player import RcatanPlayer
from train.ppo import build_actor

COLORS = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]


class CheckedPlayer(RcatanPlayer):
    """Checks every snapshot against the game it came from, and counts the
    action types it plays."""

    played = Counter()

    def _snapshot(self, game):
        snap = super()._snapshot(game)
        state = game.state
        me = state.color_to_index[self.color]
        for k in range(1, 4):
            seat = (me + k) % 4
            size = sum(
                state.player_state[f"P{seat}_{r}_IN_HAND"]
                for r in ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
            )
            # Every hand change was fed to the estimator, so each estimate
            # adds up to the opponent's (public) hand size.
            assert snap.estimates[k - 1].sum() == pytest.approx(size, abs=1e-3)
        assert sum(snap.hands[me]) == sum(
            state.player_state[f"P{me}_{r}_IN_HAND"]
            for r in ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
        )
        return snap

    def decide(self, game, playable_actions):
        action = super().decide(game, playable_actions)
        CheckedPlayer.played[action.action_type] += 1
        return action


@pytest.fixture(scope="module")
def actor():
    torch.manual_seed(0)
    return build_actor("graph").eval()


@pytest.mark.parametrize("seed", range(4))
def test_four_bots_play_whole_games(actor, seed):
    """All four seats ours, so every move type goes through the translation
    (catanatron raises on any illegal action)."""
    game = Game([CheckedPlayer(c, actor) for c in COLORS], seed=seed)
    game.play()
    assert game.state.num_turns > 0


def test_one_bot_against_catanatron(actor):
    players = [CheckedPlayer(COLORS[0], actor)] + [
        REGISTRY.build("W", c) for c in COLORS[1:]
    ]
    game = Game(players, seed=7)
    game.play()


def test_every_kind_of_move_was_played():
    """Run after the games above: the untrained actor tried each move type
    our engine and catanatron share at least once."""
    expected = {
        ActionType.BUILD_SETTLEMENT,
        ActionType.BUILD_ROAD,
        ActionType.END_TURN,
        ActionType.MOVE_ROBBER,
        ActionType.DISCARD_RESOURCE,
        ActionType.MARITIME_TRADE,
    }
    missing = expected - set(CheckedPlayer.played)
    assert not missing, f"never played: {missing}; played {CheckedPlayer.played}"
    np.testing.assert_array_less(0, [CheckedPlayer.played[t] for t in expected])
