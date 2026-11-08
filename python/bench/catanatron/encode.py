"""Our engine's state vector, built from a game snapshot in Python.

A copy of `state_to_vec` in crates/engine_py/src/lib.rs, so that our actors can
play games run by another engine (catanatron). `Snapshot` holds everything
the encoder reads, in our ids (nodes, roads, tiles, Resource::ALL order),
with players as seats 0-3. `test_encode.py` checks this file against the Rust
encoder on real games, so keep the two in step.
"""

from dataclasses import dataclass, field, replace

import numpy as np

STATE_LEN = 2031

# Resource::ALL order.
NUM_RESOURCES = 5
# Phase one-hot slots.
PHASE_INITIAL_SETTLEMENT = 0
PHASE_INITIAL_ROAD = 1
PHASE_DISCARD = 2
PHASE_MOVE_ROBBER = 3
PHASE_STEAL = 4
PHASE_CONTINUE = 5
# Dev card being resolved.
PENDING_ROAD_BUILDING = 0
PENDING_YEAR_OF_PLENTY = 1
PENDING_MONOPOLY = 2

# Pips per dice number 2..12.
PIPS = [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]


def dice_slot(dice: int) -> int:
    """Slot of a dice number in the 10-wide per-number features (no 7)."""
    return dice - 2 if dice < 7 else dice - 3


@dataclass
class Board:
    """Static board data, in our ids."""

    # Per tile id: (resource index, dice number), or None for the desert.
    tiles: list[tuple[int, int] | None]
    # (19, 6) corner node ids per tile, and the reverse: tiles per node.
    tile_nodes: np.ndarray
    node_tiles: list[list[int]] = field(init=False)
    # Port per node: 0-4 a 2:1 port (Resource::ALL order), 5 a 3:1, -1 none.
    node_ports: list[int] = field(default_factory=lambda: [-1] * 54)

    def __post_init__(self):
        self.node_tiles = [[] for _ in range(54)]
        for tile, corners in enumerate(self.tile_nodes.tolist()):
            for node in corners:
                self.node_tiles[node].append(tile)


@dataclass
class Snapshot:
    """Everything the encoder needs at one decision, for player `current`."""

    board: Board
    current: int
    robber_tile: int
    # Road id -> seat, node id -> (seat, is_city).
    roads: dict[int, int]
    buildings: dict[int, tuple[int, bool]]
    # Resource cards per seat, Resource::ALL order. Only `current`'s breakdown
    # reaches the encoding; the others count as totals.
    hands: list[list[int]]
    longest_road: int | None
    largest_army: int | None
    phase: int | None  # PHASE_*, None once the game has ended
    pending_dev: int | None  # PENDING_*
    to_discard: int
    # Current player's dev cards in DevCard::ALL order (knight, VP, road
    # building, year of plenty, monopoly): playable, and bought this turn.
    dev_playable: list[int]
    dev_bought: list[int]
    used_dev_this_turn: bool
    # Per seat. `vps[current]` includes its VP cards; the others are public.
    vps: list[int]
    knights: list[int]
    road_lengths: list[int]
    dev_counts: list[int]  # held + bought this turn
    # (3, 5): expected cards of the others, next player first.
    estimates: np.ndarray

    def with_changes(self, **changes) -> "Snapshot":
        return replace(self, **changes)


def encode(s: Snapshot) -> np.ndarray:
    """(2031,) float32 state vector, as the engine's `state_to_vec`."""
    v = np.zeros(STATE_LEN, dtype=np.float32)
    cur = s.current

    def rot(seat: int) -> int:
        return (seat - cur) % 4

    # [0, 288) roads, [288, 504) settlements, [504, 720) cities.
    for road, seat in s.roads.items():
        v[road * 4 + rot(seat)] = 1.0
    for node, (seat, is_city) in s.buildings.items():
        v[(504 if is_city else 288) + node * 4 + rot(seat)] = 1.0

    # [720, 725) own hand, [725, 728) others' hand sizes.
    for r in range(NUM_RESOURCES):
        v[720 + r] = min(s.hands[cur][r] / 10, 1.0)
    for k in range(1, 4):
        v[725 + k - 1] = min(sum(s.hands[(cur + k) % 4]) / 20, 1.0)

    # [728, 732) longest road holder.
    if s.longest_road is not None:
        v[728 + rot(s.longest_road)] = 1.0

    # [732, 1920) node features: 54 x 22.
    for node in range(54):
        base = 732 + node * 22
        port = s.board.node_ports[node]
        if port >= 0:
            v[base + 16 + port] = 1.0
        for tile in s.board.node_tiles[node]:
            info = s.board.tiles[tile]
            if info is None:  # the desert produces nothing
                continue
            resource, dice = info
            v[base + resource] += PIPS[dice - 2] / 15
            v[base + 5 + dice_slot(dice)] += 0.5
            if tile == s.robber_tile:
                v[base + 15] = 1.0

    # [1920, 1970) cards the current player gets per (resource, dice).
    received = np.zeros(50, dtype=np.float32)
    for tile, info in enumerate(s.board.tiles):
        if info is None or tile == s.robber_tile:
            continue
        resource, dice = info
        for node in s.board.tile_nodes[tile].tolist():
            building = s.buildings.get(node)
            if building is not None and building[0] == cur:
                received[resource * 10 + dice_slot(dice)] += (
                    2 if building[1] else 1
                ) / 4
    v[1920:1970] = np.minimum(received, 1.0)

    # [1970, 1976) ports the current player owns.
    for node, (seat, _) in s.buildings.items():
        port = s.board.node_ports[node]
        if seat == cur and port >= 0:
            v[1970 + port] = 1.0

    # [1976, 1982) phase, [1982, 1985) dev card being resolved.
    if s.phase is not None:
        v[1976 + s.phase] = 1.0
    if s.pending_dev is not None:
        v[1982 + s.pending_dev] = 1.0

    # 1985 cards still to discard.
    v[1985] = min(s.to_discard / 10, 1.0)

    # [1986, 1991) playable dev cards, [1991, 1996) bought this turn, 1996.
    for i in range(5):
        v[1986 + i] = min(s.dev_playable[i] / 5, 1.0)
        v[1991 + i] = min(s.dev_bought[i] / 2, 1.0)
    v[1996] = 1.0 if s.used_dev_this_turn else 0.0

    # [1997, 2013) VPs, knights, largest army, road lengths (rotated).
    for k in range(4):
        seat = (cur + k) % 4
        v[1997 + k] = s.vps[seat] / 10
        v[2001 + k] = min(s.knights[seat] / 7, 1.0)
        v[2009 + k] = s.road_lengths[seat] / 15
    if s.largest_army is not None:
        v[2005 + rot(s.largest_army)] = 1.0

    # [2013, 2016) others' dev card counts.
    for k in range(1, 4):
        v[2013 + k - 1] = min(s.dev_counts[(cur + k) % 4] / 5, 1.0)

    # [2016, 2031) others' estimated hands.
    v[2016:2031] = np.minimum(np.asarray(s.estimates).reshape(15) / 10, 1.0)
    return v
