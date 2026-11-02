import type { GameView } from "./types";

async function unwrap<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export interface CheckpointList {
  runs: string[];
  default: string | null;
}

export async function listCheckpoints(): Promise<CheckpointList> {
  const res = await fetch("/checkpoints");
  return unwrap<CheckpointList>(res);
}

export async function newGame(opts: {
  human_seat?: number;
  seed?: number;
  checkpoint?: string;
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

export interface TrainingRunSummary {
  name: string;
  timestamp: string;
  tag: string | null;
  has_train: boolean;
  modified: number;
}

export interface MetricsTable {
  columns: string[];
  rows: Record<string, number | null>[];
}

export interface TrainingRun {
  name: string;
  train: MetricsTable;
  validation: MetricsTable;
}

export async function listTrainingRuns(): Promise<TrainingRunSummary[]> {
  const res = await fetch("/training");
  return (await unwrap<{ runs: TrainingRunSummary[] }>(res)).runs;
}

export async function getTrainingRun(name: string): Promise<TrainingRun> {
  const res = await fetch(`/training/${encodeURIComponent(name)}`);
  return unwrap<TrainingRun>(res);
}
