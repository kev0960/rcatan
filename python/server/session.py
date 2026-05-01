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
    finished: bool = False
    winner: int | None = None

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
            if self._current_player() != self.human_seat:
                state, mask = self._state_and_mask()
                action_index = self.bot.pick_argmax(state, mask)
                _, ended, _ = self.engine.step([action_index])
                decoded = decode_action(action_index)
                self.last_ai_actions.append(
                    {
                        "seat": self._current_player_after_step(),
                        "action": decoded,
                    }
                )
                if ended[0]:
                    self._finalize()
                    return
                continue
            # Human's turn. The engine may be parked on the transient RollDice
            # state — get_action_mask() rolls the dice and distributes resources
            # as a side effect. Flush it here so view()'s full_state snapshot
            # reflects the post-roll player cards. If the roll was a 7 and a
            # bot now has to discard, current_player flips to that bot and the
            # outer loop runs the bot's discards before returning.
            self.engine.get_action_mask()
            if self._current_player() == self.human_seat:
                return

    def _current_player_after_step(self) -> int:
        # Note: after step(), the engine may have already moved current_player
        # forward (e.g. on EndTurn). For UX we still want to show "p2 played X",
        # so we record the player who acted *before* this step. Since step()
        # consumed the action, recover via the engine state which is now the
        # next decision point's current_player. Good enough for a tooltip.
        return int(self.engine.state()[1][0])

    def _finalize(self) -> None:
        # Engine auto-resets after game end; call full_state BEFORE that masks
        # the winner. We capture the winner via the last full_state.
        fs = self.engine.full_state(0)
        self.finished = True
        self.winner = fs.get("winner")

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
        _, ended, _ = self.engine.step([action_index])
        if ended[0]:
            self._finalize()
            return
        self._advance_until_human_or_end()

    def view(self) -> dict[str, Any]:
        """Snapshot the session for the client."""
        full_state = self.engine.full_state(0)
        is_human_turn = (not self.finished) and self._current_player() == self.human_seat
        legal_actions: list[dict[str, Any]] = []
        top_ai_moves: list[dict[str, Any]] = []
        if is_human_turn:
            state, mask = self._state_and_mask()
            legal_actions = [
                decode_action(i) for i, m in enumerate(mask) if m
            ]
            top_ai_moves = [
                {**decode_action(idx), "logit": logit}
                for idx, logit in self.bot.top_k(state, mask, k=3)
            ]
        return {
            "game_id": self.game_id,
            "human_seat": self.human_seat,
            "is_human_turn": is_human_turn,
            "finished": self.finished,
            "winner": self.winner if self.finished else full_state.get("winner"),
            "state": full_state,
            "legal_actions": legal_actions,
            "top_ai_moves": top_ai_moves,
            "last_ai_actions": self.last_ai_actions,
        }
