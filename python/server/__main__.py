"""Run with: `uv run -m server` (defaults to 0.0.0.0:8000)."""

import os

import uvicorn


def main() -> None:
    # 0.0.0.0 listens on every interface (LAN, Tailscale), not just this
    # machine. HOST=127.0.0.1 keeps it local. There is no authentication.
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("server.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
