import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  getTrainingRun,
  listTrainingRuns,
  type MetricsTable,
  type RunConfig,
  type TrainingRun,
  type TrainingRunSummary,
} from "../api";

// Categorical palette (fixed order; color follows the entity, never its rank).
const SERIES_COLORS = [
  "#2a78d6",
  "#eb6834",
  "#1baf7a",
  "#eda100",
  "#e87ba4",
  "#008300",
  "#6250d6",
  "#e34948",
];

type Row = Record<string, number | null>;

interface SeriesSpec {
  key: string;
  label: string;
  color: string;
}

interface ChartSpec {
  id: string;
  title: string;
  series: SeriesSpec[];
  percent?: boolean;
  yDomain?: [number | "auto", number | "auto"];
}

// ---------- formatting ----------

function formatTimestamp(ts: string): string {
  // "20261101-125508" -> "2026-11-01 12:55:08"
  const m = /^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})$/.exec(ts);
  if (!m) return ts;
  return `${m[1]}-${m[2]}-${m[3]} ${m[4]}:${m[5]}:${m[6]}`;
}

function formatValue(v: number, percent?: boolean): string {
  if (percent) return `${(v * 100).toFixed(1)}%`;
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 100) return v.toFixed(0);
  if (a >= 1) return v.toFixed(2);
  if (a >= 0.01) return v.toFixed(3);
  return v.toExponential(2);
}

/**
 * Settings tag from a run's config.json, in the style of the old log-name
 * tags: always network and advantage, other settings only when they differ
 * from the training defaults, e.g. "graph-gae0.95-b128-p2c1-lrlinear".
 */
function configTag(c: RunConfig): string {
  const parts: string[] = [];
  if (c.network) parts.push(c.network);
  if (c.network === "graph" && c.actor_layers !== undefined && c.actor_layers !== 3) {
    parts.push(`a${c.actor_layers}`);
  }
  if (c.advantage === "gae") parts.push(`gae${c.gae_lambda ?? ""}`);
  else if (c.advantage) parts.push(c.advantage);
  if (c.batch_size !== undefined && c.batch_size !== 64) parts.push(`b${c.batch_size}`);
  const defaultCritic = c.network === "ff" ? 4 : 2;
  if (
    (c.actor_passes !== undefined && c.actor_passes !== 4) ||
    (c.critic_passes !== undefined && c.critic_passes !== defaultCritic)
  ) {
    parts.push(`p${c.actor_passes ?? 4}c${c.critic_passes ?? defaultCritic}`);
  }
  if (c.lr_decay && c.lr_decay !== "constant") parts.push(`lr${c.lr_decay}`);
  return parts.join("-");
}

// ---------- chart specs ----------

/** Opponent name from a validation column, e.g. "win_rate_vs_75" -> "75". */
function opponentsOf(table: MetricsTable): string[] {
  const opps: string[] = [];
  for (const c of table.columns) {
    const m = /^win_rate_vs_(.+)$/.exec(c);
    if (m && table.rows.some((r) => r[c] !== null && r[c] !== undefined)) {
      opps.push(m[1]);
    }
  }
  return opps;
}

function opponentLabel(opp: string): string {
  if (/^\d+$/.test(opp)) return `vs ${opp}%`;
  return `vs ${opp}`;
}

// Stable color per opponent so the same opponent is the same color in every
// chart and every run.
const KNOWN_OPPONENTS = ["random", "50", "75", "90", "baseline"];
function opponentColor(opp: string, fallbackIndex: number): string {
  const i = KNOWN_OPPONENTS.indexOf(opp);
  return SERIES_COLORS[(i >= 0 ? i : KNOWN_OPPONENTS.length + fallbackIndex) % SERIES_COLORS.length];
}

function validationCharts(table: MetricsTable): ChartSpec[] {
  const opps = opponentsOf(table);
  const groups: { prefix: string; title: string; percent?: boolean; yDomain?: ChartSpec["yDomain"] }[] = [
    { prefix: "win_rate_vs_", title: "Win rate", percent: true, yDomain: [0, 1] },
    { prefix: "avg_vp_vs_", title: "Avg VP (agent)" },
    { prefix: "avg_opp_vp_vs_", title: "Avg VP (opponents)" },
    { prefix: "avg_steps_vs_", title: "Avg game length (steps)" },
  ];
  return groups
    .map((g) => ({
      id: `val-${g.prefix}`,
      title: g.title,
      percent: g.percent,
      yDomain: g.yDomain,
      series: opps
        .filter((o) => table.columns.includes(g.prefix + o))
        .map((o, i) => ({
          key: g.prefix + o,
          label: opponentLabel(o),
          color: opponentColor(o, i),
        })),
    }))
    .filter((c) => c.series.length > 0);
}

const TRAIN_GROUPS: { title: string; keys: string[]; yDomain?: ChartSpec["yDomain"] }[] = [
  { title: "Loss", keys: ["loss", "actor_loss", "critic_loss"] },
  { title: "Entropy", keys: ["entropy"] },
  { title: "Return vs value (mean)", keys: ["return_mean", "value_mean"] },
  { title: "Return vs value (std)", keys: ["return_std", "value_std"] },
  { title: "Explained variance", keys: ["explained_var", "explained_var_mc"], yDomain: ["auto", 1] },
  { title: "Approx KL", keys: ["approx_kl"] },
  { title: "Clip fraction", keys: ["clip_frac"] },
];

function trainCharts(table: MetricsTable): ChartSpec[] {
  const cols = new Set(table.columns);
  const used = new Set<string>(["iter"]);
  const charts: ChartSpec[] = [];
  for (const g of TRAIN_GROUPS) {
    const keys = g.keys.filter((k) => cols.has(k));
    keys.forEach((k) => used.add(k));
    if (keys.length === 0) continue;
    charts.push({
      id: `train-${g.title}`,
      title: g.title,
      yDomain: g.yDomain,
      series: keys.map((k, i) => ({ key: k, label: k, color: SERIES_COLORS[i] })),
    });
  }
  // Any metric not covered above gets its own chart, so new columns show up
  // without code changes.
  for (const c of table.columns) {
    if (used.has(c)) continue;
    charts.push({
      id: `train-${c}`,
      title: c,
      series: [{ key: c, label: c, color: SERIES_COLORS[0] }],
    });
  }
  return charts;
}

/** Exponential moving average (TensorBoard-style, debiased), skipping nulls. */
function smooth(rows: Row[], keys: string[], weight: number): Row[] {
  if (weight <= 0) return rows;
  const state: Record<string, { last: number; n: number }> = {};
  return rows.map((r) => {
    const out: Row = { ...r };
    for (const k of keys) {
      const v = r[k];
      if (v === null || v === undefined) continue;
      const s = (state[k] ??= { last: 0, n: 0 });
      s.last = s.last * weight + (1 - weight) * v;
      s.n += 1;
      out[k] = s.last / (1 - Math.pow(weight, s.n));
    }
    return out;
  });
}

// ---------- components ----------

function MetricChart({ spec, rows }: { spec: ChartSpec; rows: Row[] }) {
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const multi = spec.series.length > 1;
  const latest = useMemo(() => {
    for (let i = rows.length - 1; i >= 0; i--) {
      const v = rows[i][spec.series[0].key];
      if (v !== null && v !== undefined) return v;
    }
    return null;
  }, [rows, spec.series]);

  function toggle(key: string) {
    setHidden((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  return (
    <div className="chart-card">
      <div className="chart-title">
        <span>{spec.title}</span>
        {!multi && latest !== null && (
          <span className="chart-latest">latest {formatValue(latest, spec.percent)}</span>
        )}
      </div>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={rows} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="#e6e8ec" vertical={false} />
          <XAxis
            dataKey="iter"
            type="number"
            domain={["dataMin", "dataMax"]}
            tick={{ fontSize: 11, fill: "#6a7280" }}
            tickLine={false}
            axisLine={{ stroke: "#cfd6e0" }}
          />
          <YAxis
            domain={spec.yDomain ?? ["auto", "auto"]}
            allowDataOverflow={spec.yDomain !== undefined}
            tickFormatter={(v: number) => formatValue(v, spec.percent)}
            tick={{ fontSize: 11, fill: "#6a7280" }}
            tickLine={false}
            axisLine={false}
            width={56}
          />
          <Tooltip
            formatter={(v) => (typeof v === "number" ? formatValue(v, spec.percent) : String(v))}
            labelFormatter={(l) => `iter ${l}`}
            contentStyle={{ fontSize: 12, borderRadius: 4, borderColor: "#cfd6e0" }}
          />
          {multi && (
            <Legend
              itemSorter={null}
              wrapperStyle={{ fontSize: 12, color: "#4a5568", cursor: "pointer" }}
              onClick={(e) => {
                const key = (e as { dataKey?: unknown }).dataKey;
                if (typeof key === "string") toggle(key);
              }}
              formatter={(value: string, entry) => {
                const key = (entry as { dataKey?: unknown }).dataKey;
                const off = typeof key === "string" && hidden.has(key);
                return <span style={{ color: off ? "#b0b6c0" : "#4a5568" }}>{value}</span>;
              }}
            />
          )}
          {spec.series.map((s) => (
            <Line
              key={s.key}
              dataKey={s.key}
              name={s.label}
              stroke={s.color}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, strokeWidth: 2, stroke: "#fff" }}
              isAnimationActive={false}
              hide={hidden.has(s.key)}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function LatestValidation({ table }: { table: MetricsTable }) {
  const opps = opponentsOf(table);
  if (table.rows.length === 0 || opps.length === 0) return null;
  const last = table.rows[table.rows.length - 1];
  return (
    <div className="stat-row">
      <div className="stat">
        <div className="stat-label">validated at iter</div>
        <div className="stat-value">{last.iter ?? "—"}</div>
      </div>
      {opps.map((o) => {
        const v = last[`win_rate_vs_${o}`];
        return (
          <div className="stat" key={o}>
            <div className="stat-label">win rate {opponentLabel(o)}</div>
            <div className="stat-value">
              {v === null || v === undefined ? "—" : formatValue(v, true)}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function RunDetail({ run }: { run: TrainingRun }) {
  const [smoothing, setSmoothing] = useState(0.6);
  const valCharts = useMemo(() => validationCharts(run.validation), [run]);
  const trCharts = useMemo(() => trainCharts(run.train), [run]);
  const trainRows = useMemo(
    () =>
      smooth(
        run.train.rows,
        run.train.columns.filter((c) => c !== "iter"),
        smoothing,
      ),
    [run, smoothing],
  );

  return (
    <div className="run-detail">
      <h2>{run.name}</h2>
      <LatestValidation table={run.validation} />

      <section>
        <h3>Config</h3>
        {run.config === null ? (
          <p className="hint">No config.json for this run (trained before it was recorded).</p>
        ) : (
          <pre className="config-json">{run.config}</pre>
        )}
      </section>

      <section>
        <h3>Validation</h3>
        {valCharts.length === 0 ? (
          <p className="hint">No validation rows yet.</p>
        ) : (
          <div className="chart-grid">
            {valCharts.map((c) => (
              <MetricChart key={c.id} spec={c} rows={run.validation.rows} />
            ))}
          </div>
        )}
      </section>

      <section>
        <div className="section-head">
          <h3>Training</h3>
          {trCharts.length > 0 && (
            <label className="smoothing">
              smoothing
              <input
                type="range"
                min={0}
                max={0.99}
                step={0.01}
                value={smoothing}
                onChange={(e) => setSmoothing(Number(e.target.value))}
              />
              <span className="mono">{smoothing.toFixed(2)}</span>
            </label>
          )}
        </div>
        {trCharts.length === 0 ? (
          <p className="hint">No train-{run.name}.csv for this run.</p>
        ) : (
          <div className="chart-grid">
            {trCharts.map((c) => (
              <MetricChart key={c.id} spec={c} rows={trainRows} />
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

export default function TrainingPage() {
  const [runs, setRuns] = useState<TrainingRunSummary[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [run, setRun] = useState<TrainingRun | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  function refreshList() {
    listTrainingRuns()
      .then((r) => {
        setRuns(r);
        setSelected((cur) => cur ?? r[0]?.name ?? null);
      })
      .catch((e) => setError(String(e)));
  }

  useEffect(refreshList, []);

  useEffect(() => {
    if (!selected) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    getTrainingRun(selected)
      .then((r) => {
        if (!cancelled) setRun(r);
      })
      .catch((e) => {
        if (!cancelled) setError(String(e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selected]);

  return (
    <div className="training-layout">
      <aside className="run-list">
        <div className="run-list-head">
          <h4>Training runs ({runs.length})</h4>
          <button onClick={refreshList} title="Rescan logs">
            ↻
          </button>
        </div>
        {runs.length === 0 && <p className="hint">No logs/validation-&lt;time&gt;.csv found.</p>}
        <ul>
          {runs.map((r) => (
            <li key={r.name}>
              <button
                className={r.name === selected ? "active" : ""}
                onClick={() => setSelected(r.name)}
              >
                <span className="run-time">{formatTimestamp(r.timestamp)}</span>
                <span className="run-meta">
                  {r.config && <span className="badge">{configTag(r.config)}</span>}
                  {r.tag && <span className="badge">{r.tag}</span>}
                  {!r.has_train && <span className="badge muted">no train log</span>}
                </span>
              </button>
            </li>
          ))}
        </ul>
      </aside>
      <main className="run-main">
        {error && <div className="error">error: {error}</div>}
        {loading && !run && <p className="hint">loading…</p>}
        {run && <RunDetail key={run.name} run={run} />}
      </main>
    </div>
  );
}
