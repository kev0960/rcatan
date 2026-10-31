import { useEffect, useState } from "react";
import { Board } from "./board/Board";
import { PLAYER_COLORS } from "./board/layout";
import { listCheckpoints, newGame, postAction } from "./api";
import type { DecodedAction, GameView } from "./types";

const NON_BOARD_KINDS = new Set([
  "discard_resource",
  "steal_card",
  "maritime_trade",
  "end_turn",
  "buy_dev_card",
  "play_knight",
  "play_victory_point",
  "play_road_building",
  "play_year_of_plenty",
  "play_monopoly",
  "year_of_plenty_select",
  "monopoly_select",
]);

// Kinds with at most one legal instance — render as a single inline button
// instead of a labelled group with one button inside.
const SINGLE_BUTTON_KINDS = new Set([
  "roll_dice",
  "end_turn",
  "buy_dev_card",
  "play_knight",
  "play_victory_point",
  "play_road_building",
  "play_year_of_plenty",
  "play_monopoly",
]);

function isBoardAction(a: DecodedAction): boolean {
  return (
    a.kind === "build_road" ||
    a.kind === "build_settlement" ||
    a.kind === "build_city" ||
    a.kind === "move_robber"
  );
}

function PhaseLabel({ view }: { view: GameView }) {
  const { state } = view;
  const detail = state.phase_data ? ` ${JSON.stringify(state.phase_data)}` : "";
  return (
    <span>
      <strong>phase:</strong> {state.phase}
      {detail}
    </span>
  );
}

function PlayersPanel({ view }: { view: GameView }) {
  return (
    <div className="players">
      {view.state.players.map((p) => {
        const totalCards =
          p.cards.brick +
          p.cards.lumber +
          p.cards.ore +
          p.cards.grain +
          p.cards.wool;
        const isCurrent = view.state.current_player === p.id;
        const isHuman = view.human_seat === p.id;
        return (
          <div
            key={p.id}
            className={`player${isCurrent ? " current" : ""}`}
            style={{ borderLeft: `6px solid ${PLAYER_COLORS[p.id]}` }}
          >
            <div className="player-header">
              <strong>P{p.id}</strong>
              {isHuman && <span className="badge">you</span>}
              {isCurrent && <span className="badge current-badge">turn</span>}
              <span className="vp">{p.vp} VP</span>
            </div>
            <div className="cards">
              {isHuman ? (
                <>
                  <span title="brick">🧱 {p.cards.brick}</span>
                  <span title="lumber">🌲 {p.cards.lumber}</span>
                  <span title="ore">⛰️ {p.cards.ore}</span>
                  <span title="grain">🌾 {p.cards.grain}</span>
                  <span title="wool">🐑 {p.cards.wool}</span>
                </>
              ) : (
                <span>cards: {totalCards}</span>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

const ACTION_KIND_LABELS: Record<string, string> = {
  roll_dice: "Roll dice",
  build_road: "Build road",
  build_settlement: "Build settlement",
  build_city: "Build city",
  buy_dev_card: "Buy dev card",
  play_knight: "Play knight",
  play_victory_point: "Reveal victory point",
  play_road_building: "Play road building",
  play_year_of_plenty: "Play year of plenty",
  play_monopoly: "Play monopoly",
  year_of_plenty_select: "Year of plenty: pick a resource",
  monopoly_select: "Monopoly: pick a resource",
  move_robber: "Move robber",
  steal_card: "Steal card",
  discard_resource: "Discard",
  maritime_trade: "Maritime trade",
  end_turn: "End turn",
};

function LegalActionsPanel({
  view,
  onPick,
}: {
  view: GameView;
  onPick: (idx: number) => void;
}) {
  if (view.legal_actions.length === 0) return null;
  const byKind = new Map<string, DecodedAction[]>();
  for (const a of view.legal_actions) {
    const list = byKind.get(a.kind) ?? [];
    list.push(a);
    byKind.set(a.kind, list);
  }
  return (
    <div className="action-list">
      <h4>Available actions ({view.legal_actions.length})</h4>
      {Array.from(byKind.entries()).map(([kind, acts]) => {
        const label = ACTION_KIND_LABELS[kind] ?? kind;
        const onBoard = !NON_BOARD_KINDS.has(kind) && isBoardAction(acts[0]);
        if (SINGLE_BUTTON_KINDS.has(kind) && acts.length === 1) {
          return (
            <button key={kind} onClick={() => onPick(acts[0].index)}>
              {label}
            </button>
          );
        }
        if (onBoard) {
          return (
            <div key={kind} className="action-summary">
              <span>{label}</span>
              <span className="count">
                {acts.length} {acts.length === 1 ? "spot" : "spots"} on board
              </span>
            </div>
          );
        }
        return (
          <div key={kind} className="action-group">
            <div className="action-group-label">
              {label} ({acts.length})
            </div>
            {acts.map((a) => (
              <button key={a.index} onClick={() => onPick(a.index)}>
                {a.label}
              </button>
            ))}
          </div>
        );
      })}
    </div>
  );
}

function TopAiPanel({ view }: { view: GameView }) {
  if (!view.is_human_turn || view.top_ai_moves.length === 0) return null;
  return (
    <div className="ai-panel">
      <h4>Bot would pick</h4>
      <ol>
        {view.top_ai_moves.map((m, i) => {
          const onBoard = isBoardAction(m);
          return (
            <li key={i}>
              <span className={`rank rank-${i}`}>#{i + 1}</span>
              <span>{m.label}</span>
              {m.logit !== undefined && (
                <span className="logit">{m.logit.toFixed(2)}</span>
              )}
              {!onBoard && (
                <span className="hint" title="not shown on board">
                  ◇
                </span>
              )}
            </li>
          );
        })}
      </ol>
      <p className="hint">
        On-board suggestions are also outlined in yellow on the board.
      </p>
    </div>
  );
}

function AiActionLog({ view }: { view: GameView }) {
  if (view.last_ai_actions.length === 0) return null;
  return (
    <div className="ai-log">
      <h4>Since your last move</h4>
      <ul>
        {view.last_ai_actions.map((entry, i) => (
          <li key={i}>
            <span
              className="dot"
              style={{ background: PLAYER_COLORS[entry.seat] }}
            />
            P{entry.seat}: {entry.action.label}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function App() {
  const [view, setView] = useState<GameView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [runs, setRuns] = useState<string[]>([]);
  // "" means "let the server pick the latest checkpoint".
  const [checkpoint, setCheckpoint] = useState<string>("");

  async function start(seat?: number) {
    setBusy(true);
    setError(null);
    try {
      const v = await newGame({
        ...(seat !== undefined ? { human_seat: seat } : {}),
        ...(checkpoint ? { checkpoint } : {}),
      });
      setView(v);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function act(idx: number) {
    if (!view) return;
    setBusy(true);
    setError(null);
    try {
      const v = await postAction(view.game_id, idx);
      setView(v);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  // Auto-start a game on first load.
  useEffect(() => {
    if (!view) start();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Load the available checkpoint runs for the selector.
  useEffect(() => {
    listCheckpoints()
      .then((c) => setRuns(c.runs))
      .catch(() => setRuns([]));
  }, []);

  return (
    <div className="app">
      <header>
        <h1>rcatan</h1>
        <div className="controls">
          <label className="checkpoint-select">
            bot:
            <select
              value={checkpoint}
              onChange={(e) => setCheckpoint(e.target.value)}
              disabled={busy}
            >
              <option value="">latest</option>
              {runs.map((r) => (
                <option key={r} value={r}>
                  {r}
                </option>
              ))}
            </select>
          </label>
          <button onClick={() => start()} disabled={busy}>
            New game (random seat)
          </button>
          {[0, 1, 2, 3].map((s) => (
            <button key={s} onClick={() => start(s)} disabled={busy}>
              New as P{s}
            </button>
          ))}
        </div>
      </header>

      {error && <div className="error">error: {error}</div>}

      {view && (
        <div className="layout">
          <div className="board-wrap">
            <div className="status">
              {view.state.last_dice_roll !== null && (
                <span
                  className={`dice${view.state.last_dice_roll === 7 ? " seven" : ""}`}
                  title="last dice roll"
                >
                  🎲 {view.state.last_dice_roll}
                </span>
              )}
              <PhaseLabel view={view} />
              {view.is_human_turn ? (
                <span className="your-turn">— your turn</span>
              ) : (
                <span>— bot thinking…</span>
              )}
              {view.finished && (
                <span className="finished">
                  · game over — winner: P{view.winner}
                </span>
              )}
            </div>
            <Board
              state={view.state}
              legalActions={view.legal_actions}
              topAiMoves={view.top_ai_moves}
              isHumanTurn={view.is_human_turn}
              onAction={act}
            />
          </div>

          <aside className="sidebar">
            <PlayersPanel view={view} />
            <TopAiPanel view={view} />
            {view.is_human_turn && (
              <LegalActionsPanel view={view} onPick={act} />
            )}
            <AiActionLog view={view} />
          </aside>
        </div>
      )}
    </div>
  );
}
