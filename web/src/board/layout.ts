// Board geometry. Mirrors `crates/engine/src/table.rs`:
//   - TILE_INDEX_TO_START_COORDS: each tile_id's (col, row) on a doubled grid
//   - TILE_TO_NODE: 6 corner node_ids per tile, in canonical order
//   - TILE_TO_ROAD: 6 edge road_ids per tile, in canonical order
//
// Single tile (pointy-top hex). Numbers are canonical corner/edge indices:
//
//               T(5)
//               / \
//          (4) /   \ (5)
//             /     \
//      UL(3) +       + UR(4)
//            |       |
//        (2) |       | (3)
//            |       |
//      LL(1) +       + LR(2)
//             \     /
//          (0) \   / (1)
//               \ /
//               B(0)

export const TILE_START_COORDS: Array<[number, number]> = [
  [3, 0], [5, 0], [7, 0],
  [2, 2], [4, 2], [6, 2], [8, 2],
  [1, 4], [3, 4], [5, 4], [7, 4], [9, 4],
  [2, 6], [4, 6], [6, 6], [8, 6],
  [3, 8], [5, 8], [7, 8],
];

export const TILE_TO_NODE: number[][] = [
  [0, 3, 4, 7, 8, 12],
  [1, 4, 5, 8, 9, 13],
  [2, 5, 6, 9, 10, 14],
  [7, 11, 12, 16, 17, 22],
  [8, 12, 13, 17, 18, 23],
  [9, 13, 14, 18, 19, 24],
  [10, 14, 15, 19, 20, 25],
  [16, 21, 22, 27, 28, 33],
  [17, 22, 23, 28, 29, 34],
  [18, 23, 24, 29, 30, 35],
  [19, 24, 25, 30, 31, 36],
  [20, 25, 26, 31, 32, 37],
  [28, 33, 34, 38, 39, 43],
  [29, 34, 35, 39, 40, 44],
  [30, 35, 36, 40, 41, 45],
  [31, 36, 37, 41, 42, 46],
  [39, 43, 44, 47, 48, 51],
  [40, 44, 45, 48, 49, 52],
  [41, 45, 46, 49, 50, 53],
];

export const TILE_TO_ROAD: number[][] = [
  [0, 1, 6, 7, 11, 12],
  [2, 3, 7, 8, 13, 14],
  [4, 5, 8, 9, 15, 16],
  [10, 11, 18, 19, 24, 25],
  [12, 13, 19, 20, 26, 27],
  [14, 15, 20, 21, 28, 29],
  [16, 17, 21, 22, 30, 31],
  [23, 24, 33, 34, 39, 40],
  [25, 26, 34, 35, 41, 42],
  [27, 28, 35, 36, 43, 44],
  [29, 30, 36, 37, 45, 46],
  [31, 32, 37, 38, 47, 48],
  [40, 41, 49, 50, 54, 55],
  [42, 43, 50, 51, 56, 57],
  [44, 45, 51, 52, 58, 59],
  [46, 47, 52, 53, 60, 61],
  [55, 56, 62, 63, 66, 67],
  [57, 58, 63, 64, 68, 69],
  [59, 60, 64, 65, 70, 71],
];

export const NUM_TILES = 19;
export const NUM_NODES = 54;
export const NUM_ROADS = 72;

// --- Pixel layout ---

export const HEX_RADIUS = 56; // distance from hex center to apex
const SQRT3 = Math.sqrt(3);
const STEP_X = (SQRT3 / 2) * HEX_RADIUS; // pixel per engine col
const STEP_Y = 0.75 * HEX_RADIUS; // pixel per engine row

// Canvas size — fit all tiles + margin.
const MAX_COL = 10; // engine cols 0..10
const MAX_ROW = 10; // engine rows 0..10 (tiles span up to row 10)
const MARGIN = 60;

export const BOARD_WIDTH = MAX_COL * STEP_X + 2 * MARGIN;
export const BOARD_HEIGHT = MAX_ROW * STEP_Y + 2 * MARGIN;

// Tile center in pixels. Engine (col, row) — row 0 at bottom of board.
function tileCenter(tileId: number): { x: number; y: number } {
  const [sc, sr] = TILE_START_COORDS[tileId];
  // Tile spans engine (sc..sc+2, sr..sr+2). Center at (sc+1, sr+1).
  const ec = sc + 1;
  const er = sr + 1;
  // Tile centers occupy engine cols 2..10, so corners span cols 1..11.
  // Subtract the empty leftmost column so the content sits centered in the
  // canvas (otherwise it skews right and clips the right-side ports).
  return {
    x: MARGIN + (ec - 1) * STEP_X,
    // Flip y: high engine_row -> low svg y (top of screen)
    y: MARGIN + (MAX_ROW - er) * STEP_Y,
  };
}

// Corner offsets relative to tile center, in canonical order [B, LL, LR, UL, UR, T].
// SVG y grows downward, so "below center" => positive y_offset.
const CORNER_OFFSETS: Array<{ dx: number; dy: number }> = [
  { dx: 0, dy: +HEX_RADIUS }, // B (bottom apex)
  { dx: -STEP_X, dy: +HEX_RADIUS / 2 }, // LL
  { dx: +STEP_X, dy: +HEX_RADIUS / 2 }, // LR
  { dx: -STEP_X, dy: -HEX_RADIUS / 2 }, // UL
  { dx: +STEP_X, dy: -HEX_RADIUS / 2 }, // UR
  { dx: 0, dy: -HEX_RADIUS }, // T (top apex)
];

// Edge endpoints relative to tile (using corner indices into CORNER_OFFSETS):
//   bottom-left  = LL(1) — B(0)
//   bottom-right = B(0) — LR(2)
//   left         = LL(1) — UL(3)
//   right        = LR(2) — UR(4)
//   upper-left   = UL(3) — T(5)
//   upper-right  = UR(4) — T(5)
const EDGE_ENDS: Array<[number, number]> = [
  [1, 0], // bottom-left
  [0, 2], // bottom-right
  [1, 3], // left (vertical)
  [2, 4], // right (vertical)
  [3, 5], // upper-left
  [4, 5], // upper-right
];

export interface NodePos {
  nodeId: number;
  x: number;
  y: number;
}

export interface RoadPos {
  roadId: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface TilePos {
  tileId: number;
  cx: number;
  cy: number;
  // Hex polygon points as a string suitable for SVG <polygon points=...>
  polygon: string;
}

// Compute a single position for each unique node_id by walking tiles.
// (Shared corners produce identical positions.)
export function nodePositions(): NodePos[] {
  const out: Array<NodePos | undefined> = new Array(NUM_NODES);
  for (let t = 0; t < NUM_TILES; t++) {
    const c = tileCenter(t);
    for (let i = 0; i < 6; i++) {
      const nodeId = TILE_TO_NODE[t][i];
      if (out[nodeId] !== undefined) continue;
      const off = CORNER_OFFSETS[i];
      out[nodeId] = { nodeId, x: c.x + off.dx, y: c.y + off.dy };
    }
  }
  return out as NodePos[];
}

export function roadPositions(): RoadPos[] {
  const out: Array<RoadPos | undefined> = new Array(NUM_ROADS);
  for (let t = 0; t < NUM_TILES; t++) {
    const c = tileCenter(t);
    for (let i = 0; i < 6; i++) {
      const roadId = TILE_TO_ROAD[t][i];
      if (out[roadId] !== undefined) continue;
      const [a, b] = EDGE_ENDS[i];
      const oa = CORNER_OFFSETS[a];
      const ob = CORNER_OFFSETS[b];
      out[roadId] = {
        roadId,
        x1: c.x + oa.dx,
        y1: c.y + oa.dy,
        x2: c.x + ob.dx,
        y2: c.y + ob.dy,
      };
    }
  }
  return out as RoadPos[];
}

// Perimeter traversal (counter-clockwise) into CORNER_OFFSETS:
// B -> LR -> UR -> T -> UL -> LL. Canonical order is for identification,
// not drawing — using it directly produces a self-intersecting polygon.
const POLYGON_ORDER = [0, 2, 4, 5, 3, 1];

export function tilePositions(): TilePos[] {
  return Array.from({ length: NUM_TILES }, (_, t) => {
    const c = tileCenter(t);
    const points = POLYGON_ORDER.map((i) => {
      const o = CORNER_OFFSETS[i];
      return `${c.x + o.dx},${c.y + o.dy}`;
    }).join(" ");
    return { tileId: t, cx: c.x, cy: c.y, polygon: points };
  });
}

export const PLAYER_COLORS = ["#e74c3c", "#3498db", "#27ae60", "#f1c40f"];

export const RESOURCE_COLORS: Record<string, string> = {
  brick: "#c0392b",
  lumber: "#196f3d",
  ore: "#5d6d7e",
  grain: "#f5b041",
  wool: "#abebc6",
  desert: "#d7c39d",
};

// Ports — mirrors `NODE_TO_PORTS` in `crates/engine/src/player.rs`.
// Each port sits on a coastal edge shared by two nodes: identified by
// (tileId, edgeIdx) so we can derive geometry from the tile center.
export type PortKind = "brick" | "lumber" | "ore" | "grain" | "wool" | "any";

export interface PortInfo {
  kind: PortKind;
  tileId: number;
  edgeIdx: number; // index into EDGE_ENDS
  nodeIds: [number, number];
}

export const PORTS: PortInfo[] = [
  { kind: "any", tileId: 0, edgeIdx: 0, nodeIds: [3, 0] },
  { kind: "any", tileId: 1, edgeIdx: 1, nodeIds: [1, 5] },
  { kind: "any", tileId: 11, edgeIdx: 3, nodeIds: [26, 32] },
  { kind: "any", tileId: 16, edgeIdx: 4, nodeIds: [47, 51] },
  { kind: "brick", tileId: 3, edgeIdx: 2, nodeIds: [11, 16] },
  { kind: "lumber", tileId: 12, edgeIdx: 2, nodeIds: [33, 38] },
  { kind: "ore", tileId: 15, edgeIdx: 5, nodeIds: [42, 46] },
  { kind: "wool", tileId: 6, edgeIdx: 1, nodeIds: [10, 15] },
  { kind: "grain", tileId: 17, edgeIdx: 5, nodeIds: [52, 49] },
];

export interface PortPos {
  kind: PortKind;
  cx: number; // port marker center
  cy: number;
  // Coastal node positions (dock endpoints).
  ax: number;
  ay: number;
  bx: number;
  by: number;
}

const PORT_OUTWARD_OFFSET = 30;

export function portPositions(): PortPos[] {
  return PORTS.map((p) => {
    const c = tileCenter(p.tileId);
    const [a, b] = EDGE_ENDS[p.edgeIdx];
    const oa = CORNER_OFFSETS[a];
    const ob = CORNER_OFFSETS[b];
    const ax = c.x + oa.dx;
    const ay = c.y + oa.dy;
    const bx = c.x + ob.dx;
    const by = c.y + ob.dy;
    const mx = (ax + bx) / 2;
    const my = (ay + by) / 2;
    // Outward direction from tile center through edge midpoint.
    const dx = mx - c.x;
    const dy = my - c.y;
    const len = Math.hypot(dx, dy) || 1;
    const cx = mx + (dx / len) * PORT_OUTWARD_OFFSET;
    const cy = my + (dy / len) * PORT_OUTWARD_OFFSET;
    return { kind: p.kind, cx, cy, ax, ay, bx, by };
  });
}
