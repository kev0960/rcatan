import { useMemo } from "react";
import type { DecodedAction, FullState } from "../types";
import {
  BOARD_HEIGHT,
  BOARD_WIDTH,
  HEX_RADIUS,
  PLAYER_COLORS,
  RESOURCE_COLORS,
  nodePositions,
  portPositions,
  roadPositions,
  tilePositions,
} from "./layout";

interface Props {
  state: FullState;
  legalActions: DecodedAction[];
  topAiMoves: DecodedAction[];
  isHumanTurn: boolean;
  onAction: (index: number) => void;
}

interface ClickableSets {
  roads: Map<number, number>; // road_id -> action_index
  settlements: Map<number, number>; // node_id -> action_index
  cities: Map<number, number>;
  tiles: Map<number, number>; // tile_id -> action_index (move robber)
}

interface SuggestionSets {
  roads: Map<number, number>; // road_id -> rank (0 = best)
  settlements: Map<number, number>;
  cities: Map<number, number>;
  tiles: Map<number, number>;
}

function buildClickable(
  legalActions: DecodedAction[],
  isHumanTurn: boolean,
): ClickableSets {
  const sets: ClickableSets = {
    roads: new Map(),
    settlements: new Map(),
    cities: new Map(),
    tiles: new Map(),
  };
  if (!isHumanTurn) return sets;
  for (const a of legalActions) {
    if (a.kind === "build_road" && a.road_id !== undefined) {
      sets.roads.set(a.road_id, a.index);
    } else if (a.kind === "build_settlement" && a.node_id !== undefined) {
      sets.settlements.set(a.node_id, a.index);
    } else if (a.kind === "build_city" && a.node_id !== undefined) {
      sets.cities.set(a.node_id, a.index);
    } else if (a.kind === "move_robber" && a.tile_id !== undefined) {
      sets.tiles.set(a.tile_id, a.index);
    }
  }
  return sets;
}

function buildSuggestions(top: DecodedAction[]): SuggestionSets {
  const sets: SuggestionSets = {
    roads: new Map(),
    settlements: new Map(),
    cities: new Map(),
    tiles: new Map(),
  };
  top.forEach((a, rank) => {
    if (a.kind === "build_road" && a.road_id !== undefined) {
      sets.roads.set(a.road_id, rank);
    } else if (a.kind === "build_settlement" && a.node_id !== undefined) {
      sets.settlements.set(a.node_id, rank);
    } else if (a.kind === "build_city" && a.node_id !== undefined) {
      sets.cities.set(a.node_id, rank);
    } else if (a.kind === "move_robber" && a.tile_id !== undefined) {
      sets.tiles.set(a.tile_id, rank);
    }
  });
  return sets;
}

const SUGGESTION_STROKE = ["#f1c40f", "#f39c12", "#e67e22"];
const SUGGESTION_OPACITY = [1.0, 0.8, 0.6];

export function Board({
  state,
  legalActions,
  topAiMoves,
  isHumanTurn,
  onAction,
}: Props) {
  const tiles = useMemo(() => tilePositions(), []);
  const nodes = useMemo(() => nodePositions(), []);
  const roads = useMemo(() => roadPositions(), []);
  const ports = useMemo(() => portPositions(), []);

  const clickable = useMemo(
    () => buildClickable(legalActions, isHumanTurn),
    [legalActions, isHumanTurn],
  );
  const suggestions = useMemo(
    () => buildSuggestions(topAiMoves),
    [topAiMoves],
  );

  // Lookups for rendering current state.
  const roadOwner = useMemo(() => {
    const m = new Map<number, number>();
    for (const r of state.roads) m.set(r.road_id, r.player);
    return m;
  }, [state.roads]);

  const structureAt = useMemo(() => {
    const m = new Map<
      number,
      { building: "settlement" | "city"; player: number }
    >();
    for (const s of state.structures) {
      m.set(s.node_id, { building: s.building, player: s.player });
    }
    return m;
  }, [state.structures]);

  return (
    <svg
      width={BOARD_WIDTH}
      height={BOARD_HEIGHT}
      viewBox={`0 0 ${BOARD_WIDTH} ${BOARD_HEIGHT}`}
      style={{ background: "#1f4e79", borderRadius: 8 }}
    >
      {/* Tiles */}
      {tiles.map((t) => {
        const tileData = state.tiles[t.tileId];
        const fill = RESOURCE_COLORS[tileData?.resource ?? "desert"];
        const robberSuggestionRank = suggestions.tiles.get(t.tileId);
        const isClickableTile = clickable.tiles.has(t.tileId);
        const onClick = isClickableTile
          ? () => onAction(clickable.tiles.get(t.tileId)!)
          : undefined;
        return (
          <g key={`tile-${t.tileId}`}>
            <polygon
              points={t.polygon}
              fill={fill}
              stroke={
                isClickableTile
                  ? "#fff"
                  : robberSuggestionRank !== undefined
                    ? SUGGESTION_STROKE[robberSuggestionRank]
                    : "#163b5a"
              }
              strokeWidth={
                isClickableTile || robberSuggestionRank !== undefined ? 4 : 2
              }
              style={{ cursor: isClickableTile ? "pointer" : "default" }}
              onClick={onClick}
            />
            {tileData && tileData.resource !== "desert" && (
              <>
                <circle
                  cx={t.cx}
                  cy={t.cy}
                  r={HEX_RADIUS * 0.28}
                  fill="#f8f4e1"
                  stroke="#222"
                  strokeWidth={1}
                />
                <text
                  x={t.cx}
                  y={t.cy + 5}
                  textAnchor="middle"
                  fontSize={16}
                  fontWeight="bold"
                  fill={
                    tileData.dice === 6 || tileData.dice === 8
                      ? "#c0392b"
                      : "#222"
                  }
                >
                  {tileData.dice}
                </text>
              </>
            )}
            {tileData?.has_robber && (
              <circle
                cx={t.cx}
                cy={t.cy - HEX_RADIUS * 0.55}
                r={HEX_RADIUS * 0.18}
                fill="#222"
                stroke="#fff"
                strokeWidth={1.5}
              />
            )}
          </g>
        );
      })}

      {/* Ports */}
      {ports.map((p, i) => {
        const isAny = p.kind === "any";
        const fill = isAny ? "#ecf0f1" : RESOURCE_COLORS[p.kind];
        const darkText =
          p.kind === "grain" || p.kind === "wool" || p.kind === "any";
        return (
          <g key={`port-${i}`} pointerEvents="none">
            <line
              x1={p.ax}
              y1={p.ay}
              x2={p.cx}
              y2={p.cy}
              stroke="#8b6f47"
              strokeWidth={2}
              strokeLinecap="round"
            />
            <line
              x1={p.bx}
              y1={p.by}
              x2={p.cx}
              y2={p.cy}
              stroke="#8b6f47"
              strokeWidth={2}
              strokeLinecap="round"
            />
            <circle
              cx={p.cx}
              cy={p.cy}
              r={13}
              fill={fill}
              stroke="#fff"
              strokeWidth={2}
            />
            <text
              x={p.cx}
              y={p.cy + 4}
              textAnchor="middle"
              fontSize={11}
              fontWeight="bold"
              fill={darkText ? "#222" : "#fff"}
            >
              {isAny ? "3:1" : "2:1"}
            </text>
          </g>
        );
      })}

      {/* Roads */}
      {roads.map((r) => {
        const owner = roadOwner.get(r.roadId);
        const clickIdx = clickable.roads.get(r.roadId);
        const suggestionRank = suggestions.roads.get(r.roadId);
        const stroke =
          owner !== undefined
            ? PLAYER_COLORS[owner]
            : clickIdx !== undefined
              ? "rgba(255,255,255,0.85)"
              : suggestionRank !== undefined
                ? SUGGESTION_STROKE[suggestionRank]
                : "rgba(255,255,255,0.05)";
        const width =
          owner !== undefined
            ? 7
            : clickIdx !== undefined || suggestionRank !== undefined
              ? 6
              : 3;
        return (
          <line
            key={`road-${r.roadId}`}
            x1={r.x1}
            y1={r.y1}
            x2={r.x2}
            y2={r.y2}
            stroke={stroke}
            strokeWidth={width}
            strokeLinecap="round"
            strokeOpacity={
              suggestionRank !== undefined && owner === undefined
                ? SUGGESTION_OPACITY[suggestionRank]
                : 1
            }
            style={{
              cursor: clickIdx !== undefined ? "pointer" : "default",
            }}
            onClick={
              clickIdx !== undefined ? () => onAction(clickIdx) : undefined
            }
          />
        );
      })}

      {/* Nodes */}
      {nodes.map((n) => {
        const struct = structureAt.get(n.nodeId);
        const settlementClick = clickable.settlements.get(n.nodeId);
        const cityClick = clickable.cities.get(n.nodeId);
        const settlementSuggest = suggestions.settlements.get(n.nodeId);
        const citySuggest = suggestions.cities.get(n.nodeId);
        const click = settlementClick ?? cityClick;
        const suggestRank = settlementSuggest ?? citySuggest;

        if (struct) {
          // Existing settlement/city. If upgrading to city is a legal action
          // for this settlement, make it clickable.
          const color = PLAYER_COLORS[struct.player];
          const upgradable = cityClick !== undefined;
          const stroke = upgradable
            ? "#fff"
            : citySuggest !== undefined
              ? SUGGESTION_STROKE[citySuggest]
              : "#222";
          const strokeWidth = upgradable || citySuggest !== undefined ? 3 : 1.5;
          const cursor = upgradable ? "pointer" : "default";
          const onClick = upgradable ? () => onAction(cityClick!) : undefined;
          if (struct.building === "settlement") {
            return (
              <polygon
                key={`node-${n.nodeId}`}
                points={`${n.x},${n.y - 11} ${n.x + 9},${n.y - 2} ${n.x + 9},${n.y + 9} ${n.x - 9},${n.y + 9} ${n.x - 9},${n.y - 2}`}
                fill={color}
                stroke={stroke}
                strokeWidth={strokeWidth}
                style={{ cursor }}
                onClick={onClick}
              />
            );
          }
          return (
            <rect
              key={`node-${n.nodeId}`}
              x={n.x - 11}
              y={n.y - 11}
              width={22}
              height={22}
              fill={color}
              stroke="#222"
              strokeWidth={1.5}
            />
          );
        }

        if (click !== undefined) {
          return (
            <circle
              key={`node-${n.nodeId}`}
              cx={n.x}
              cy={n.y}
              r={9}
              fill="rgba(255,255,255,0.9)"
              stroke="#222"
              strokeWidth={1}
              style={{ cursor: "pointer" }}
              onClick={() => onAction(click)}
            />
          );
        }

        if (suggestRank !== undefined) {
          return (
            <circle
              key={`node-${n.nodeId}`}
              cx={n.x}
              cy={n.y}
              r={8}
              fill="none"
              stroke={SUGGESTION_STROKE[suggestRank]}
              strokeWidth={3}
              strokeOpacity={SUGGESTION_OPACITY[suggestRank]}
            />
          );
        }

        // Tiny dot to hint a node is here.
        return (
          <circle
            key={`node-${n.nodeId}`}
            cx={n.x}
            cy={n.y}
            r={2}
            fill="rgba(255,255,255,0.15)"
          />
        );
      })}

      {/* Node id labels (debug annotation, matches engine game_viewer.rs) */}
      {nodes.map((n) => (
        <text
          key={`node-label-${n.nodeId}`}
          x={n.x + 8}
          y={n.y - 6}
          textAnchor="start"
          fontSize={9}
          fill="rgba(255,255,255,0.55)"
          pointerEvents="none"
        >
          {n.nodeId.toString().padStart(2, "0")}
        </text>
      ))}
    </svg>
  );
}
