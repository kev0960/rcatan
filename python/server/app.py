"""Starlette app exposing endpoints for human-vs-bot play.

Uses Starlette directly rather than FastAPI to avoid a pydantic/Python 3.14
incompatibility (pydantic 2.13.3 fails to evaluate `EmailStr | None` on 3.14).

Endpoints:
  POST /games                  - start a new game (body: {"human_seat"?, "seed"?})
  GET  /games/{game_id}        - current snapshot (no state mutation)
  POST /games/{game_id}/action - apply a human action (body: {"action_index"})
  DELETE /games/{game_id}      - drop a session
"""

from __future__ import annotations

import json
import os
import random
import threading
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .bot import Bot, RandomBot
from .session import GameSession

NUM_ACTIONS = 300
CHECKPOINT_PATH = Path(
    os.environ.get("CATAN_ACTOR_CHECKPOINT", "checkpoints/actor_latest.pt")
)


def _load_bot() -> Bot | RandomBot:
    if CHECKPOINT_PATH.exists():
        try:
            print(f"[server] loading actor from {CHECKPOINT_PATH}")
            return Bot.from_checkpoint(CHECKPOINT_PATH)
        except Exception as e:  # noqa: BLE001
            print(f"[server] failed to load checkpoint ({e}); falling back to random bot")
            return RandomBot()
    print(
        f"[server] no checkpoint at {CHECKPOINT_PATH}; using RandomBot. "
        "Train one via `uv run -m train` to produce checkpoints/actor_latest.pt."
    )
    return RandomBot()


async def _read_json(request: Request) -> dict[str, Any]:
    raw = await request.body()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise _http_error(400, f"invalid json: {e}") from None
    if not isinstance(data, dict):
        raise _http_error(400, "expected json object")
    return data


def _http_error(status: int, detail: str) -> Exception:
    return _HTTPError(status, detail)


class _HTTPError(Exception):
    def __init__(self, status: int, detail: str):
        self.status = status
        self.detail = detail
        super().__init__(detail)


def _err(status: int, detail: str) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=status)


def _validate_seat(value: Any) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or not 0 <= value <= 3:
        raise _http_error(400, "human_seat must be an int in [0, 3]")
    return value


def _validate_action_index(value: Any) -> int:
    if not isinstance(value, int) or not 0 <= value < NUM_ACTIONS:
        raise _http_error(400, f"action_index must be an int in [0, {NUM_ACTIONS})")
    return value


def create_app() -> Starlette:
    bot = _load_bot()
    sessions: dict[str, GameSession] = {}
    sessions_lock = threading.Lock()
    rng = random.Random()

    def _get(game_id: str) -> GameSession:
        with sessions_lock:
            session = sessions.get(game_id)
        if session is None:
            raise _http_error(404, "game not found")
        return session

    async def new_game(request: Request) -> JSONResponse:
        try:
            body = await _read_json(request)
            seat = _validate_seat(body.get("human_seat"))
            seed = body.get("seed")
            if seed is not None and not isinstance(seed, int):
                raise _http_error(400, "seed must be an int")
        except _HTTPError as e:
            return _err(e.status, e.detail)
        if seat is None:
            seat = rng.randint(0, 3)
        if seed is None:
            seed = rng.randint(0, 2**31 - 1)
        session = GameSession.new(bot=bot, human_seat=seat, seed=seed)
        with sessions_lock:
            sessions[session.game_id] = session
        with session.lock:
            return JSONResponse(session.view())

    async def get_game(request: Request) -> JSONResponse:
        try:
            session = _get(request.path_params["game_id"])
        except _HTTPError as e:
            return _err(e.status, e.detail)
        with session.lock:
            return JSONResponse(session.view())

    async def post_action(request: Request) -> JSONResponse:
        try:
            session = _get(request.path_params["game_id"])
            body = await _read_json(request)
            action_index = _validate_action_index(body.get("action_index"))
        except _HTTPError as e:
            return _err(e.status, e.detail)
        with session.lock:
            try:
                session.apply_human_action(action_index)
            except (RuntimeError, ValueError) as e:
                return _err(400, str(e))
            return JSONResponse(session.view())

    async def delete_game(request: Request) -> JSONResponse:
        with sessions_lock:
            sessions.pop(request.path_params["game_id"], None)
        return JSONResponse({"ok": True})

    routes = [
        Route("/games", new_game, methods=["POST"]),
        Route("/games/{game_id}", get_game, methods=["GET"]),
        Route("/games/{game_id}/action", post_action, methods=["POST"]),
        Route("/games/{game_id}", delete_game, methods=["DELETE"]),
    ]

    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        ),
    ]

    return Starlette(routes=routes, middleware=middleware)


app = create_app()
