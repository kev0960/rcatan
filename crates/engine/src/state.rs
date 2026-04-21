use crate::{PlayerId, board::Board};

pub struct State {
    pub current_player: PlayerId,
    pub board: Board,
    pub longest_road: Option<(PlayerId, u8)>,
    // Resource card counts per player: player_cards[player_id][resource_idx].
    // Only the 5 resource slots are exposed (not dev cards).
    pub player_cards: [[u8; 5]; 4],
}
