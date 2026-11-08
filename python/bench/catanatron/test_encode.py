"""The Python encoder must match the engine's `state_to_vec` exactly.

Plays random games in our engine, builds a `Snapshot` from each state's
`full_state`, and compares `encode` with `Engine.state()`. `full_state` has
no breakdown of the current player's dev cards and no card estimates, so
those floats ([1982, 1985), [1986, 1997), [2016, 2031)) are left out; the
rest of the vector is compared at every step.

    uv run --no-sync python -m pytest -q bench/catanatron
"""

import numpy as np
import pytest
from catan_engine import Engine

from bench.catanatron.encode import (
    PHASE_CONTINUE,
    PHASE_DISCARD,
    PHASE_INITIAL_ROAD,
    PHASE_INITIAL_SETTLEMENT,
    PHASE_MOVE_ROBBER,
    PHASE_STEAL,
    STATE_LEN,
    Board,
    Snapshot,
    encode,
)

RESOURCE_INDEX = {"brick": 0, "lumber": 1, "ore": 2, "grain": 3, "wool": 4}
PHASES = {
    "initial_settlement_placement": PHASE_INITIAL_SETTLEMENT,
    "initial_road_placement": PHASE_INITIAL_ROAD,
    "discard": PHASE_DISCARD,
    "move_robber": PHASE_MOVE_ROBBER,
    "steal_resource": PHASE_STEAL,
    "continue_action": PHASE_CONTINUE,
    "game_ended": None,
}
# Floats `full_state` can't fill in (see the module docstring).
UNCHECKED = np.zeros(STATE_LEN, dtype=bool)
UNCHECKED[1982:1985] = True
UNCHECKED[1986:1997] = True
UNCHECKED[2016:2031] = True


def snapshot_from_engine(fs: dict) -> Snapshot:
    tiles = [None] * 19
    robber = None
    for t in fs["tiles"]:
        if t["resource"] != "desert":
            tiles[t["tile_id"]] = (RESOURCE_INDEX[t["resource"]], t["dice"])
        if t["has_robber"]:
            robber = t["tile_id"]
    board = Board(
        tiles=tiles,
        tile_nodes=Engine.tile_nodes(),
        node_ports=Engine.node_ports().tolist(),
    )
    cur = fs["current_player"]
    players = fs["players"]
    hands = [
        [p["cards"][name] for name in RESOURCE_INDEX] for p in players
    ]
    to_discard = 0
    if fs["phase"] == "discard":
        target = fs["phase_data"]["target_counts"][cur]
        to_discard = max(sum(hands[cur]) - target, 0)
    return Snapshot(
        board=board,
        current=cur,
        robber_tile=robber,
        roads={r["road_id"]: r["player"] for r in fs["roads"]},
        buildings={
            s["node_id"]: (s["player"], s["building"] == "city")
            for s in fs["structures"]
        },
        hands=hands,
        longest_road=fs["longest_road"]["player"] if fs["longest_road"] else None,
        largest_army=fs["largest_army"]["player"] if fs["largest_army"] else None,
        phase=PHASES[fs["phase"]],
        pending_dev=None,
        to_discard=to_discard,
        dev_playable=[0] * 5,
        dev_bought=[0] * 5,
        used_dev_this_turn=False,
        vps=[
            p["vp"] if i == cur else p["public_vp"] for i, p in enumerate(players)
        ],
        knights=[p["knights_played"] for p in players],
        road_lengths=[p["road_length"] for p in players],
        dev_counts=[p["dev_cards"] for p in players],
        estimates=np.zeros((3, 5), dtype=np.float32),
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_python_encoder_matches_the_engine(seed):
    batch = 8
    engine = Engine(batch, seed)
    rng = np.random.default_rng(seed)
    checked = 0
    for _ in range(600):
        states, _ = engine.state()
        for b in range(batch):
            fs = engine.full_state(b)
            ours = encode(snapshot_from_engine(fs))
            diff = np.flatnonzero(
                ~UNCHECKED & ~np.isclose(ours, states[b], atol=1e-6)
            )
            assert diff.size == 0, (
                f"game {b}, phase {fs['phase']}: floats {diff[:10].tolist()} "
                f"differ: python {ours[diff[:10]]} vs engine {states[b][diff[:10]]}"
            )
            checked += 1
        masks = engine.get_action_mask()
        actions = [int(rng.choice(np.flatnonzero(m))) for m in masks]
        engine.step(actions)
    assert checked == 600 * batch
