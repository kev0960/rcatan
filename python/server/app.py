"""Starlette app exposing endpoints for human-vs-bot play.

Uses Starlette directly rather than FastAPI to avoid a pydantic/Python 3.14
incompatibility (pydantic 2.13.3 fails to evaluate `EmailStr | None` on 3.14).

Endpoints:
  GET  /checkpoints            - list available checkpoint runs (newest first)
  POST /games                  - start a new game
                                 (body: {"human_seat"?, "seed"?, "checkpoint"?})
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

# Checkpoints are organized as `checkpoints/<run-timestamp>/actor_latest.pt`,
# where the run timestamp is `%Y%m%d-%H%M%S` (so lexical order == chronological
# order). `CATAN_CHECKPOINT_DIR` overrides the root; `CATAN_ACTOR_CHECKPOINT`
# pins an explicit checkpoint file, taking precedence over run discovery.
CHECKPOINT_ROOT = Path(os.environ.get("CATAN_CHECKPOINT_DIR", "checkpoints"))
CHECKPOINT_OVERRIDE = os.environ.get("CATAN_ACTOR_CHECKPOINT")
LATEST_FILENAME = "actor_latest.pt"


def list_runs(root: Path | None = None) -> list[str]:
    """Run directories under `root` (default `CHECKPOINT_ROOT`) that hold an
    `actor_latest.pt`, newest first. Names are timestamps, so reverse-sorting
    is newest-first."""
    root = root if root is not None else CHECKPOINT_ROOT
    if not root.is_dir():
        return []
    runs = [
        p.name
        for p in root.iterdir()
        if p.is_dir() and (p / LATEST_FILENAME).exists()
    ]
    return sorted(runs, reverse=True)


def resolve_checkpoint(run: str | None) -> Path | None:
    """Resolve which checkpoint file to load.

    - explicit `run` -> `<root>/<run>/actor_latest.pt`
    - else `CATAN_ACTOR_CHECKPOINT` if set
    - else the latest run's `actor_latest.pt`
    - else the legacy flat `<root>/actor_latest.pt`

    Returns None if the resolved path does not exist.
    """
    if run is not None:
        candidate = CHECKPOINT_ROOT / run / LATEST_FILENAME
        return candidate if candidate.exists() else None
    if CHECKPOINT_OVERRIDE:
        override = Path(CHECKPOINT_OVERRIDE)
        return override if override.exists() else None
    runs = list_runs()
    if runs:
        return CHECKPOINT_ROOT / runs[0] / LATEST_FILENAME
    legacy = CHECKPOINT_ROOT / LATEST_FILENAME
    return legacy if legacy.exists() else None


def _load_bot(run: str | None = None) -> Bot | RandomBot:
    path = resolve_checkpoint(run)
    if path is not None:
        try:
            print(f"[server] loading actor from {path}")
            return Bot.from_checkpoint(path)
        except Exception as e:  # noqa: BLE001
            print(f"[server] failed to load checkpoint ({e}); falling back to random bot")
            return RandomBot()
    where = f"run {run!r}" if run else f"under {CHECKPOINT_ROOT}"
    print(
        f"[server] no checkpoint for {where}; using RandomBot. "
        "Train one via `uv run -m train` to produce "
        "checkpoints/<timestamp>/actor_latest.pt."
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


def _validate_run(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise _http_error(400, "checkpoint must be a string run name")
    if value not in list_runs():
        raise _http_error(404, f"unknown checkpoint run: {value}")
    return value


def create_app() -> Starlette:
    sessions: dict[str, GameSession] = {}
    sessions_lock = threading.Lock()
    rng = random.Random()

    # Lazily loaded bots keyed by run name (None -> server default). Loading a
    # checkpoint is expensive, so each run is only loaded once and reused across
    # games.
    bots: dict[str | None, Bot | RandomBot] = {}
    bots_lock = threading.Lock()

    def get_bot(run: str | None) -> Bot | RandomBot:
        with bots_lock:
            if run in bots:
                return bots[run]
        bot = _load_bot(run)
        with bots_lock:
            return bots.setdefault(run, bot)

    # Warm the default bot at startup so the first game doesn't pay the load.
    get_bot(None)

    def _get(game_id: str) -> GameSession:
        with sessions_lock:
            session = sessions.get(game_id)
        if session is None:
            raise _http_error(404, "game not found")
        return session

    async def list_checkpoints(request: Request) -> JSONResponse:
        runs = list_runs()
        return JSONResponse({"runs": runs, "default": runs[0] if runs else None})

    async def new_game(request: Request) -> JSONResponse:
        try:
            body = await _read_json(request)
            seat = _validate_seat(body.get("human_seat"))
            run = _validate_run(body.get("checkpoint"))
            seed = body.get("seed")
            if seed is not None and not isinstance(seed, int):
                raise _http_error(400, "seed must be an int")
        except _HTTPError as e:
            return _err(e.status, e.detail)
        if seat is None:
            seat = rng.randint(0, 3)
        if seed is None:
            seed = rng.randint(0, 2**31 - 1)
        session = GameSession.new(bot=get_bot(run), human_seat=seat, seed=seed)
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
        Route("/checkpoints", list_checkpoints, methods=["GET"]),
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
