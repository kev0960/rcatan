// Wire types — mirror what the Starlette backend returns.

export interface Tile {
  tile_id: number;
  resource: "brick" | "lumber" | "ore" | "grain" | "wool" | "desert";
  dice: number;
  has_robber: boolean;
}

export interface RoadOnBoard {
  road_id: number;
  player: number;
}

export interface StructureOnBoard {
  node_id: number;
  building: "settlement" | "city";
  player: number;
}

export interface PlayerInfo {
  id: number;
  vp: number;
  cards: {
    brick: number;
    lumber: number;
    ore: number;
    grain: number;
    wool: number;
  };
}

export type Phase =
  | "initial_settlement_placement"
  | "initial_road_placement"
  | "discard"
  | "move_robber"
  | "steal_resource"
  | "continue_action"
  | "game_ended";

export interface FullState {
  current_player: number;
  last_dice_roll: number | null;
  winner: number | null;
  phase: Phase;
  phase_data: Record<string, unknown> | null;
  longest_road: { player: number; length: number } | null;
  tiles: Tile[];
  roads: RoadOnBoard[];
  structures: StructureOnBoard[];
  players: PlayerInfo[];
}

export interface DecodedAction {
  index: number;
  kind: string;
  label: string;
  // Optional fields per kind
  road_id?: number;
  node_id?: number;
  tile_id?: number;
  player_id?: number;
  resource?: string;
  give_resource?: string;
  give_amount?: number;
  want_resource?: string;
  dev_card?: string;
  // Only present in top_ai_moves
  logit?: number;
}

export interface AIActionLog {
  seat: number;
  action: DecodedAction;
}

export interface GameView {
  game_id: string;
  human_seat: number;
  is_human_turn: boolean;
  finished: boolean;
  winner: number | null;
  state: FullState;
  legal_actions: DecodedAction[];
  top_ai_moves: DecodedAction[];
  last_ai_actions: AIActionLog[];
}
