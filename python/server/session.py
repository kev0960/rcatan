"""Per-game session state and engine driver.

Each `GameSession` owns a single-game `Engine(batch_size=1)`. The session
exposes a `view()` snapshot for the client and an `apply_human_action()` that
also auto-runs AI turns until either the human is on the clock again or the
game ends.

Concurrency: each session has its own `threading.Lock`. FastAPI's default
threadpool calls sync handlers off the event loop, so a per-session lock is
sufficient — simultaneous actions on the same game are serialized.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

from catan_engine import Engine

from .actions import decode_action
from .bot import Bot, RandomBot

AnyBot = Bot | RandomBot


@dataclass
class GameSession:
    game_id: str
    human_seat: int
    engine: Engine
    bot: AnyBot
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_ai_actions: list[dict[str, Any]] = field(default_factory=list)
    # Every action taken in this game (human and AI), oldest first.
    history: list[dict[str, Any]] = field(default_factory=list)
    finished: bool = False
    winner: int | None = None
    # Snapshot of the finished game. The engine resets to a fresh game as soon
    # as one ends, so the live `full_state` no longer shows this game.
    final_state: dict[str, Any] | None = None

    @classmethod
    def new(
        cls,
        bot: AnyBot,
        human_seat: int,
        seed: int,
    ) -> "GameSession":
        engine = Engine(batch_size=1, seed=seed)
        session = cls(
            game_id=str(uuid.uuid4()),
            human_seat=human_seat,
            engine=engine,
            bot=bot,
        )
        session._advance_until_human_or_end()
        return session

    def _current_player(self) -> int:
        _, players = self.engine.state()
        return int(players[0])

    def _state_and_mask(self) -> tuple[list[float], list[bool]]:
        states, _ = self.engine.state()
        mask = self.engine.get_action_mask()[0]
        return list(states[0]), list(mask)

    def _advance_until_human_or_end(self) -> None:
        """Run AI turns until it is the human's turn (or the game ends).
        Records each AI action in `last_ai_actions` for the next client view."""
        self.last_ai_actions = []
        while not self.finished:
            if self._current_player() == self.human_seat:
                return
            state, mask = self._state_and_mask()
            action_index = self.bot.pick_argmax(state, mask)
            entry, ended = self._step(action_index)
            self.last_ai_actions.append(entry)
            if ended:
                self._finalize()
                return

    def _step(self, action_index: int) -> tuple[dict[str, Any], bool]:
        """Apply one action and append it to `history`. The seat is captured
        *before* stepping, since the engine may advance current_player (e.g.
        on EndTurn)."""
        seat = self._current_player()
        was_setup = self._in_setup()
        _, ended, _ = self.engine.step([action_index])
        decoded = decode_action(action_index, seat)
        entry: dict[str, Any] = {
            "seq": len(self.history),
            "seat": seat,
            "action": decoded,
        }
        self.history.append(entry)
        # The engine rolls dice automatically for the next player when a turn
        # ends (and for the first player once setup finishes); there is no
        # roll action, so log the roll as its own entry. Skip on game end:
        # the engine auto-resets, so its state no longer reflects this game.
        if not ended[0]:
            fs = self.engine.full_state(0)
            rolled = decoded["kind"] == "end_turn" or was_setup
            dice = fs.get("last_dice_roll")
            if rolled and dice is not None and not self._in_setup(fs):
                self.history.append(
                    {
                        "seq": len(self.history),
                        "seat": int(fs["current_player"]),
                        "action": {
                            "index": -1,
                            "kind": "roll_dice",
                            "label": f"Rolled {dice}",
                        },
                        "dice": dice,
                    }
                )
        return entry, bool(ended[0])

    def _in_setup(self, fs: dict[str, Any] | None = None) -> bool:
        fs = fs if fs is not None else self.engine.full_state(0)
        return str(fs.get("phase", "")).startswith("initial")

    def _finalize(self) -> None:
        self.final_state = self.engine.final_state(0)
        self.finished = True
        self.winner = self.final_state.get("winner") if self.final_state else None

    def apply_human_action(self, action_index: int) -> None:
        if self.finished:
            raise RuntimeError("game already ended")
        if self._current_player() != self.human_seat:
            raise RuntimeError(
                f"not human's turn (current_player={self._current_player()})"
            )
        _, mask = self._state_and_mask()
        if not mask[action_index]:
            raise ValueError(f"action {action_index} is not legal in current state")
        _, ended = self._step(action_index)
        if ended:
            self._finalize()
            return
        self._advance_until_human_or_end()

    def _mask_player(self, p: dict[str, Any]) -> dict[str, Any]:
        """Hide opponents' VP cards while the game is running: they only see
        public VPs (buildings, longest road, largest army)."""
        if self.finished or p["id"] == self.human_seat:
            return p
        return {**p, "vp": p["public_vp"]}

    def view(self) -> dict[str, Any]:
        """Snapshot the session for the client."""
        if self.finished and self.final_state is not None:
            full_state = dict(self.final_state)
        else:
            full_state = self.engine.full_state(0)
        full_state["players"] = [
            self._mask_player(p) for p in full_state["players"]
        ]
        is_human_turn = (not self.finished) and self._current_player() == self.human_seat
        legal_actions: list[dict[str, Any]] = []
        top_ai_moves: list[dict[str, Any]] = []
        if is_human_turn:
            state, mask = self._state_and_mask()
            legal_actions = [
                decode_action(i, self.human_seat) for i, m in enumerate(mask) if m
            ]
            top_ai_moves = [
                {**decode_action(idx, self.human_seat), "logit": logit}
                for idx, logit in self.bot.top_k(state, mask, k=3)
            ]
        return {
            "game_id": self.game_id,
            "human_seat": self.human_seat,
            "is_human_turn": is_human_turn,
            "finished": self.finished,
            "winner": self.winner,
            "state": full_state,
            "legal_actions": legal_actions,
            "top_ai_moves": top_ai_moves,
            "last_ai_actions": self.last_ai_actions,
            "history": self.history,
        }
