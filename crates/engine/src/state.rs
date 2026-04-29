use crate::PlayerId;
use crate::board::Board;

pub struct State {
    pub current_player: PlayerId,
    pub board: Board,
    pub longest_road: Option<(PlayerId, u8)>,
    // Resource card counts per player: player_cards[player_id][resource_idx].
    // Only the 5 resource slots are exposed (not dev cards).
    pub player_cards: [[u8; 5]; 4],
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
