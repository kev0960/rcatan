use crate::DevCard;
use crate::PlayerId;
use crate::board::Board;

pub struct State {
    pub current_player: PlayerId,
    pub board: Board,
    pub longest_road: Option<(PlayerId, u8)>,
    // Resource card counts per player: player_cards[player_id][resource_idx].
    // Only the 5 resource slots are exposed (not dev cards).
    pub player_cards: [[u8; 5]; 4],
    pub phase: Phase,
    // The multi-step dev card being resolved (RoadBuilding, YearOfPlenty,
    // Monopoly), if any. `phase` is ContinueAction while it resolves.
    pub pending_dev_card: Option<DevCard>,
    // Dev card counts per player in DevCard::ALL order: playable ones, and
    // ones bought this turn (not playable until next turn).
    pub dev_cards: [[u8; 5]; 4],
    pub dev_cards_bought: [[u8; 5]; 4],
    // Whether the current player already played a dev card this turn.
    pub used_dev_card_this_turn: bool,
    // Full VPs, including hidden VP cards. Only the current player's own
    // entry is public; use `public_vps` for everyone else.
    pub vps: [u8; 4],
    // VPs visible to all players: buildings, longest road and largest army.
    pub public_vps: [u8; 4],
    pub knights_played: [u8; 4],
    pub largest_army: Option<(PlayerId, u8)>,
    // Each player's longest road length.
    pub road_lengths: [u8; 4],
}

// A coarse phase tag that's stable to surface to external consumers (UI,
// serialization). The engine's internal `GameState` carries a few transient
// sub-states and associated data that aren't all useful outside training.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    // Snake-order placement of starting settlements + roads. The current
    // player is exposed separately on `FullState`.
    InitialSettlementPlacement,
    InitialRoadPlacement,
    // A 7 was rolled. Players over the discard threshold are still discarding.
    // `target_counts[i]` is the post-discard hand size each player must reach.
    Discard { target_counts: [u8; 4] },
    // Discards are done. The roller picks a tile to move the robber to.
    MoveRobber,
    // Robber has been moved; choose a player to steal from.
    StealResource,
    // Normal turn after dice roll: build / trade / end turn.
    ContinueAction,
    GameEnded,
}

// Rich snapshot of the game intended for rendering / serialization (UI, REST
// responses). Includes everything in `State` plus per-player VPs, the most
// recent dice roll, the current phase, and the winner if any.
pub struct FullState {
    pub current_player: PlayerId,
    pub board: Board,
    pub longest_road: Option<(PlayerId, u8)>,
    pub player_cards: [[u8; 5]; 4],
    pub vps: [u8; 4],
    pub last_dice_roll: Option<u8>,
    pub phase: Phase,
    pub winner: Option<PlayerId>,
}
