"""A catanatron player that plays with one of our trained actors.

Each decision: build our engine's state vector from the catanatron game
(`encode.py`, with the board matched by `board_map.py`), run the actor, and
play the legal catanatron action whose action index scores highest (greedy,
as in our validation games).

The two engines split some moves differently:

- Rolling: our engine rolls on its own, so the bot always rolls when it can.
  It never plays a dev card before rolling (catanatron allows that; our
  engine doesn't).
- Robber: catanatron moves it and picks the victim in one action; our engine
  takes two. The bot picks the tile, then scores the victims on the state
  our engine would show next (robber moved, "steal" phase).
- Year of plenty and monopoly: one catanatron action, but several in our
  engine (play the card, then pick resources one at a time). The bot picks
  the card first, then each resource on the matching follow-up state.
- Trades between players don't exist in our engine: the bot rejects every
  offer.

The card estimates in the state come from our engine's estimator
(`CardEstimator`), fed with every hand change it could see at a real table:
production, building, buying, trading, discards and monopolies are public;
for a steal, only the thief and the victim learn the card.
"""

import numpy as np
import torch
from catan_engine import CardEstimator, Engine
from catanatron import ActionType, Color, Player
from catanatron.models.enums import ActionPrompt

from bench.catanatron.board_map import RESOURCE_INDEX, build_board_map
from bench.catanatron.encode import (
    PENDING_MONOPOLY,
    PENDING_ROAD_BUILDING,
    PENDING_YEAR_OF_PLENTY,
    PHASE_CONTINUE,
    PHASE_DISCARD,
    PHASE_INITIAL_ROAD,
    PHASE_INITIAL_SETTLEMENT,
    PHASE_MOVE_ROBBER,
    PHASE_STEAL,
    Board,
    Snapshot,
    encode,
)

# Catanatron resource names in Resource::ALL order.
RESOURCES = sorted(RESOURCE_INDEX, key=RESOURCE_INDEX.get)
# Catanatron dev card names in DevCard::ALL order.
DEV_CARDS = ["KNIGHT", "VICTORY_POINT", "ROAD_BUILDING", "YEAR_OF_PLENTY", "MONOPOLY"]

# Action indices (see `action_to_index` in crates/engine_py/src/lib.rs).
SETTLEMENT, CITY, DISCARD, ROBBER, STEAL, TRADE = 72, 126, 180, 185, 204, 208
END_TURN, BUY_DEV, USE_DEV, YEAR_OF_PLENTY, MONOPOLY = 283, 284, 285, 290, 295
USE_KNIGHT = USE_DEV + DEV_CARDS.index("KNIGHT")
USE_ROAD_BUILDING = USE_DEV + DEV_CARDS.index("ROAD_BUILDING")
USE_YEAR_OF_PLENTY = USE_DEV + DEV_CARDS.index("YEAR_OF_PLENTY")
USE_MONOPOLY = USE_DEV + DEV_CARDS.index("MONOPOLY")


class RcatanPlayer(Player):
    """Plays a catanatron seat with `actor` (from `train.ppo.load_actor`)."""

    LABEL = "rcatan"

    def __init__(self, color: Color, actor: torch.nn.Module, device="cpu"):
        super().__init__(color)
        self.actor = actor.eval()
        self.device = device

    # ---- Game lifecycle --------------------------------------------------

    def before(self, game):
        """New game: match the board and start a fresh card estimate."""
        board_map = build_board_map(game.state.board.map)
        self.map = board_map
        self.board = Board(
            tiles=board_map.tiles,
            tile_nodes=Engine.tile_nodes(),
            node_ports=board_map.node_ports,
        )
        self.estimator = CardEstimator()
        # The action stepped last and the hands before it; its effect is read
        # at the next step() or decide(), once catanatron has applied it.
        self.pending = None
        # Our dev cards bought this turn (DevCard::ALL order): they can't be
        # played yet, and catanatron doesn't count them separately.
        self.bought = [0] * 5
        self.bought_turn = None

    def step(self, game, action):
        """Called before every action anyone takes."""
        self._catch_up(game)
        self.pending = (action, _hands(game.state), self._my_devs(game.state))

    def _catch_up(self, game):
        """Feed the effect of the last stepped action to the estimator."""
        if self.pending is None:
            return
        action, before, devs_before = self.pending
        self.pending = None
        state = game.state
        after = _hands(state)
        seat = state.color_to_index

        if action.action_type == ActionType.MOVE_ROBBER and action.value[1]:
            thief, victim = seat[action.color], seat[action.value[1]]
            for r in range(5):
                if after[thief][r] - before[thief][r] == 1 and (
                    before[victim][r] - after[victim][r] == 1
                ):
                    self.estimator.observe_steal(thief, victim, r)
                    break
            return

        # Everything else is public. Spend first, then gain: within one action
        # (a trade, say) the cards paid were in the hand before.
        for p in range(4):
            for r in range(5):
                if after[p][r] < before[p][r]:
                    self.estimator.observe_spend(p, r, before[p][r] - after[p][r])
        for p in range(4):
            for r in range(5):
                if after[p][r] > before[p][r]:
                    self.estimator.observe_gain(p, r, after[p][r] - before[p][r])

        if (
            action.color == self.color
            and action.action_type == ActionType.BUY_DEVELOPMENT_CARD
        ):
            if self.bought_turn != state.num_turns:
                self.bought, self.bought_turn = [0] * 5, state.num_turns
            devs = self._my_devs(state)
            for i in range(5):
                self.bought[i] += max(devs[i] - devs_before[i], 0)

    def _my_devs(self, state) -> list[int]:
        key = f"P{state.color_to_index[self.color]}"
        return [state.player_state[f"{key}_{d}_IN_HAND"] for d in DEV_CARDS]

    # ---- Decisions -------------------------------------------------------

    def decide(self, game, playable_actions):
        self._catch_up(game)
        actions = list(playable_actions)
        if len(actions) == 1:
            return actions[0]
        by_type = {}
        for a in actions:
            by_type.setdefault(a.action_type, []).append(a)
        # Moves our engine doesn't have.
        if ActionType.ROLL in by_type:
            return by_type[ActionType.ROLL][0]
        for answer in (ActionType.REJECT_TRADE, ActionType.CANCEL_TRADE):
            if answer in by_type:
                return by_type[answer][0]

        snap = self._snapshot(game)
        logits = self._logits(snap)
        groups = {}
        for a in actions:
            index = self._index(a)
            if index is not None:
                groups.setdefault(index, []).append(a)
        if not groups:  # only moves we can't score (e.g. trade offers)
            return actions[0]
        best = max(groups, key=lambda i: logits[i])
        options = groups[best]
        if len(options) == 1:
            return options[0]

        kind = options[0].action_type
        if kind == ActionType.MOVE_ROBBER:
            return self._pick_victim(snap, best - ROBBER, options)
        if kind == ActionType.PLAY_YEAR_OF_PLENTY:
            return self._pick_year_of_plenty(snap, options)
        if kind == ActionType.PLAY_MONOPOLY:
            return self._pick_monopoly(snap, options)
        return options[0]

    def _logits(self, snap: Snapshot) -> np.ndarray:
        state = torch.from_numpy(encode(snap)).to(self.device)[None]
        with torch.no_grad():
            return self.actor(state)[0].float().cpu().numpy()

    def _index(self, a) -> int | None:
        """Our action index for a catanatron action (the first of our steps
        for moves our engine splits in several)."""
        t, v = a.action_type, a.value
        if t == ActionType.BUILD_ROAD:
            return self.map.edge(*v)
        if t == ActionType.BUILD_SETTLEMENT:
            return SETTLEMENT + self.map.node[v]
        if t == ActionType.BUILD_CITY:
            return CITY + self.map.node[v]
        if t == ActionType.DISCARD_RESOURCE:
            return DISCARD + RESOURCE_INDEX[v]
        if t == ActionType.MOVE_ROBBER:
            return ROBBER + self.map.tile[v[0]]
        if t == ActionType.MARITIME_TRADE:
            give = RESOURCE_INDEX[v[0]]
            amount = sum(x is not None for x in v[:4])
            return TRADE + give * 15 + (amount - 2) * 5 + RESOURCE_INDEX[v[4]]
        return {
            ActionType.END_TURN: END_TURN,
            ActionType.BUY_DEVELOPMENT_CARD: BUY_DEV,
            ActionType.PLAY_KNIGHT_CARD: USE_KNIGHT,
            ActionType.PLAY_ROAD_BUILDING: USE_ROAD_BUILDING,
            ActionType.PLAY_YEAR_OF_PLENTY: USE_YEAR_OF_PLENTY,
            ActionType.PLAY_MONOPOLY: USE_MONOPOLY,
        }.get(t)

    def _pick_victim(self, snap, tile, options):
        """Robber moved to `tile`: score the victims on the steal state."""
        victims = [a for a in options if a.value[1] is not None]
        if not victims:
            return options[0]
        seat = self._seat_of
        logits = self._logits(
            snap.with_changes(robber_tile=tile, phase=PHASE_STEAL)
        )
        return max(
            victims,
            key=lambda a: logits[STEAL + (seat(a.value[1]) - snap.current) % 4],
        )

    def _after_dev_card(self, snap, card, pending):
        """The state after playing `card`, while it resolves."""
        playable = list(snap.dev_playable)
        playable[DEV_CARDS.index(card)] -= 1
        return snap.with_changes(
            phase=PHASE_CONTINUE,
            pending_dev=pending,
            dev_playable=playable,
            used_dev_this_turn=True,
        )

    def _pick_year_of_plenty(self, snap, options):
        """One resource at a time, as our engine asks."""
        state = self._after_dev_card(snap, "YEAR_OF_PLENTY", PENDING_YEAR_OF_PLENTY)
        logits = self._logits(state)
        first = max(
            {RESOURCE_INDEX[a.value[0]] for a in options}
            | {RESOURCE_INDEX[a.value[-1]] for a in options},
            key=lambda r: logits[YEAR_OF_PLENTY + r],
        )
        with_first = [
            a for a in options if first in [RESOURCE_INDEX[x] for x in a.value]
        ]
        if len(with_first) == 1:
            return with_first[0]
        hands = [list(h) for h in state.hands]
        hands[state.current][first] += 1
        logits = self._logits(state.with_changes(hands=hands))

        def second(a):
            rest = [RESOURCE_INDEX[x] for x in a.value]
            rest.remove(first)
            return logits[YEAR_OF_PLENTY + rest[0]] if rest else -np.inf

        return max(with_first, key=second)

    def _pick_monopoly(self, snap, options):
        logits = self._logits(
            self._after_dev_card(snap, "MONOPOLY", PENDING_MONOPOLY)
        )
        return max(options, key=lambda a: logits[MONOPOLY + RESOURCE_INDEX[a.value]])

    # ---- State ------------------------------------------------------------

    def _seat_of(self, color) -> int:
        return self._game.state.color_to_index[color]

    def _snapshot(self, game) -> Snapshot:
        self._game = game
        state = game.state
        board = state.board
        seat = state.color_to_index
        me = seat[self.color]
        ps = state.player_state

        def key(p, name):
            return ps[f"P{p}_{name}"]

        buildings = {
            self.map.node[n]: (seat[c], kind == "CITY")
            for n, (c, kind) in board.buildings.items()
        }
        roads = {}
        for edge, c in board.roads.items():
            roads[self.map.edge(*edge)] = seat[c]

        prompt = state.current_prompt
        phase = {
            ActionPrompt.BUILD_INITIAL_SETTLEMENT: PHASE_INITIAL_SETTLEMENT,
            ActionPrompt.BUILD_INITIAL_ROAD: PHASE_INITIAL_ROAD,
            ActionPrompt.DISCARD: PHASE_DISCARD,
            ActionPrompt.MOVE_ROBBER: PHASE_MOVE_ROBBER,
        }.get(prompt, PHASE_CONTINUE)

        devs = self._my_devs(state)
        if self.bought_turn != state.num_turns:
            self.bought, self.bought_turn = [0] * 5, state.num_turns
        holder = lambda flag: next(  # noqa: E731
            (p for p in range(4) if key(p, flag)), None
        )
        return Snapshot(
            board=self.board,
            current=me,
            robber_tile=self.map.tile[board.robber_coordinate],
            roads=roads,
            buildings=buildings,
            hands=_hands(state),
            longest_road=holder("HAS_ROAD"),
            largest_army=holder("HAS_ARMY"),
            phase=phase,
            pending_dev=PENDING_ROAD_BUILDING if state.is_road_building else None,
            to_discard=state.discard_counts[me] if state.is_discarding else 0,
            dev_playable=[devs[i] - self.bought[i] for i in range(5)],
            dev_bought=list(self.bought),
            used_dev_this_turn=key(me, "HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"),
            vps=[
                key(p, "ACTUAL_VICTORY_POINTS" if p == me else "VICTORY_POINTS")
                for p in range(4)
            ],
            knights=[key(p, "PLAYED_KNIGHT") for p in range(4)],
            road_lengths=[key(p, "LONGEST_ROAD_LENGTH") for p in range(4)],
            dev_counts=[
                sum(key(p, f"{d}_IN_HAND") for d in DEV_CARDS) for p in range(4)
            ],
            estimates=self.estimator.expected_cards(me),
        )


def _hands(state) -> list[list[int]]:
    """Resource cards per seat, in Resource::ALL order."""
    return [
        [state.player_state[f"P{p}_{r}_IN_HAND"] for r in RESOURCES]
        for p in range(len(state.colors))
    ]
