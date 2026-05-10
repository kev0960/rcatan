import type { GameView } from "./types";

async function unwrap<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export async function newGame(opts: {
  human_seat?: number;
  seed?: number;
}): Promise<GameView> {
  const res = await fetch("/games", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(opts),
  });
  return unwrap<GameView>(res);
}

export async function getGame(gameId: string): Promise<GameView> {
  const res = await fetch(`/games/${gameId}`);
  return unwrap<GameView>(res);
}

export async function postAction(
  gameId: string,
  actionIndex: number,
): Promise<GameView> {
  const res = await fetch(`/games/${gameId}/action`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ action_index: actionIndex }),
  });
  return unwrap<GameView>(res);
}
