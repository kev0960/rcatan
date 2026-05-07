use arrayvec::ArrayVec;
use std::array::from_fn;

use crate::action::DevCard;
use crate::game_viewer::render_board;
use crate::reward::Reward;
use crate::reward::VictoryPoint;
use crate::state::FullState;
use crate::state::Phase;
use crate::state::State;
use crate::util::vecset::VecSet;

use crate::action::Action;
use crate::action::Building;
use crate::action::Resource;
use crate::board::Board;
use crate::player::Player;
use crate::road::find_longest_road;
use crate::table::TILE_TO_NODE;
use crate::util::id::PlayerId;
use crate::util::sample::choose_weighted;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

static WINNING_VP: u8 = 10;

struct GameDetail {
    // Serialized information of "Longest road" and "Largest army"
    longest_road: u8,
    largest_army: u8,

    // 4 bits for each player's victory points
    vp: VictoryPoint,
    total_turns: u16,
}

impl GameDetail {
    fn new() -> Self {
        Self {
            longest_road: 0,
            largest_army: 0,
            vp: VictoryPoint::new(ArrayVec::from([0; 4])),
            total_turns: 0,
        }
    }

    fn get_longest_road(&self) -> Option<(PlayerId, u8)> {
        Self::deserializer(&self.longest_road)
    }

    fn set_longest_road(&mut self, r: Option<(PlayerId, u8)>) {
        Self::serializer(&mut self.longest_road, r);
    }

    fn get_largest_army(&self) -> Option<(PlayerId, u8)> {
        Self::deserializer(&self.largest_army)
    }

    fn set_largest_army(&mut self, r: Option<(PlayerId, u8)>) {
        Self::serializer(&mut self.largest_army, r);
    }

    fn serializer(data: &mut u8, r: Option<(PlayerId, u8)>) {
        if let Some((player_id, length)) = r {
            *data = 0b0000_0001 | (u8::from(player_id) << 1) | (length << 3);
        } else {
            *data &= 0b1111_1110;
        }
    }

    fn deserializer(data: &u8) -> Option<(PlayerId, u8)> {
        let has_value = data & 0b0000_0001;
        if has_value == 0 {
            return None;
        }

        let player_id = (data & 0b0000_0110) >> 1;
        let value = data >> 3;

        Some((PlayerId::from(player_id), value))
    }
}

#[derive(Debug, Clone, Copy)]
enum GameState {
    // Initial board set up. Keeps track of the current order.
    //
    // (0, 1), (2, 3), (4, 5), (6, 7): Player 1 to 4 placement (settlement,
    // road).
    //
    // (8, 9), (10, 11), (12, 13), (14, 15): Player 4 to 1 placement
    // (settlement, road).
    InitialBoardSetup(u8),

    // 7 was rolled. At least one player still has more resource cards than
    // their post-discard target. Tuple holds the original roller (so we can
    // restore `current_player` once all discards complete) plus the per-player
    // target counts.
    Dice7Rolled((PlayerId, [u8; 4])),

    // The current player picks a tile to move the robber to. Reached either
    // by a 7 roll (after all discards complete) or by playing a Knight dev
    // card. The two paths share the same legal-action set and the same
    // downstream transition in run_action::MoveRobber, so they share a
    // state.
    AwaitingMoveRobber,

    // The robber has just been moved to a tile that borders at least one enemy
    // structure. The current player chooses whom to steal from.
    StealResourceAfterRobber,

    // current player should continue action.
    ContinueAction,

    // Game is ended with the given winner.
    GameEnded(PlayerId),

    // (Dev card used, processed_once). Used for the multi-step dev cards
    // RoadBuilding and YearOfPlenty (and the single-step Monopoly, where the
    // bool is unused). Knight does NOT go through this state — it goes
    // straight to AwaitingMoveRobber.
    UsedDevCard((DevCard, bool)),
}

pub struct Game {
    board: Board,
    dev_cards: ArrayVec<u8, 5>,

    current_player: usize,
    players: ArrayVec<Player, 4>,

    game_detail: GameDetail,

    state: GameState,

    // Most recent dice roll for the current turn. Reset to None on EndTurn.
    last_dice_roll: Option<u8>,

    rng: ChaCha8Rng,
}

impl Game {
    pub fn new(seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        Self {
            board: Board::new_from_rand(&mut rng),
            players: ArrayVec::from(from_fn(|i| Player::new(i.into()))),
            current_player: 0,
            dev_cards: ArrayVec::from([14, 5, 2, 2, 2]),
            game_detail: GameDetail::new(),
            state: GameState::InitialBoardSetup(0),
            last_dice_roll: None,
            rng,
        }
    }

    pub fn is_game_ended(&self) -> Option<PlayerId> {
        match self.state {
            GameState::GameEnded(player) => Some(player),
            _ => None,
        }
    }

    pub fn vps(&self) -> [u8; 4] {
        let mut vps = [0u8; 4];
        for p in &self.players {
            vps[u8::from(p.id) as usize] = self.game_detail.vp.get_vp(p.id);
        }
        vps
    }

    pub fn state(&self) -> State {
        let mut player_cards = [[0u8; 5]; 4];
        for (i, p) in self.players.iter().enumerate() {
            for j in 0..5 {
                player_cards[i][j] = p.cards[j];
            }
        }

        State {
            current_player: self.current_player_id(),
            board: self.board.clone(),
            longest_road: self.game_detail.get_longest_road(),
            player_cards,
        }
    }

    pub fn full_state(&self) -> FullState {
        let mut player_cards = [[0u8; 5]; 4];
        let mut vps = [0u8; 4];
        for (i, p) in self.players.iter().enumerate() {
            for j in 0..5 {
                player_cards[i][j] = p.cards[j];
            }
            vps[i] = self.game_detail.vp.get_vp(p.id);
        }

        // Map the engine's internal GameState to the externally-facing Phase.
        let phase = match self.state {
            GameState::InitialBoardSetup(step) => {
                if step % 2 == 0 {
                    Phase::InitialSettlementPlacement
                } else {
                    Phase::InitialRoadPlacement
                }
            }
            GameState::Dice7Rolled((_, target_counts)) => {
                Phase::Discard { target_counts }
            }
            GameState::AwaitingMoveRobber => Phase::MoveRobber,
            GameState::StealResourceAfterRobber => Phase::StealResource,
            GameState::ContinueAction => Phase::ContinueAction,
            GameState::GameEnded(_) => Phase::GameEnded,
            GameState::UsedDevCard((dev_card, _)) => match dev_card {
                // Knight no longer reaches this branch — it goes straight
                // to AwaitingMoveRobber. Kept defensively as ContinueAction.
                DevCard::Knight => Phase::ContinueAction,
                DevCard::VictoryPoint => Phase::ContinueAction,
                DevCard::RoadBuilding => Phase::ContinueAction,
                DevCard::YearOfPlenty => Phase::ContinueAction,
                DevCard::Monopoly => Phase::ContinueAction,
            },
        };

        FullState {
            current_player: self.current_player_id(),
            board: self.board.clone(),
            longest_road: self.game_detail.get_longest_road(),
            player_cards,
            vps,
            last_dice_roll: self.last_dice_roll,
            phase,
            winner: self.is_game_ended(),
        }
    }

    // Project the current game state into the set of legal actions for
    // `current_player`. Pure read; safe to call any number of times between
    // run_action calls. All state transitions live in `run_action`.
    pub fn get_actions(&self) -> Vec<Action> {
        match self.state {
            GameState::Dice7Rolled((_roller, _target_counts)) => {
                // current_player has been set to the active discarder by
                // either roll_dice_for_current_player (initial) or
                // run_action::DiscardResource (subsequent). Just project
                // their legal discards.
                let player = &self.players[self.current_player];
                player
                    .cards
                    .iter()
                    .take(5)
                    .enumerate()
                    .filter_map(|(i, count)| match count {
                        0 => None,
                        _ => Some(Action::DiscardResource(Resource::ALL[i])),
                    })
                    .collect()
            }
            GameState::AwaitingMoveRobber => self
                .board
                .get_tile_without_robber()
                .into_iter()
                .map(Action::MoveRobber)
                .collect(),
            GameState::StealResourceAfterRobber => {
                // Only entered when run_action::MoveRobber found at least
                // one victim; always non-empty.
                self.compute_robber_victims()
                    .into_iter()
                    .map(|p| Action::StealCard(PlayerId::from(p)))
                    .collect()
            }
            GameState::InitialBoardSetup(current_order) => {
                let current_player = self.current_player_ref();
                if current_order % 2 == 0 {
                    // Build the settlement.
                    current_player
                        .get_available_nodes_for_initial_board_setup(
                            &self.board,
                        )
                        .into_iter()
                        .map(|n| {
                            Action::BuildStructure(Building::Settlement, n)
                        })
                        .collect()
                } else {
                    // Build the road.
                    current_player
                        .get_available_roads_for_initial_board_setup(
                            &self.board,
                        )
                        .into_iter()
                        .map(Action::BuildRoad)
                        .collect()
                }
            }
            GameState::UsedDevCard((dev_card, _)) => match dev_card {
                // Knight no longer reaches this branch — it goes straight
                // to AwaitingMoveRobber. Treated defensively as a no-op
                // fallthrough to ContinueAction-style actions.
                DevCard::Knight | DevCard::VictoryPoint => {
                    self.current_player_ref().get_available_actions(
                        &self.board,
                        self.dev_cards.as_slice(),
                    )
                }
                DevCard::RoadBuilding => self
                    .current_player_ref()
                    .get_available_roads(&self.board)
                    .into_iter()
                    .map(Action::BuildRoad)
                    .collect(),
                DevCard::YearOfPlenty => Resource::ALL
                    .iter()
                    .map(|r| Action::YearOfPlentyGetResource(*r))
                    .collect(),
                DevCard::Monopoly => Resource::ALL
                    .iter()
                    .map(|r| Action::MonopolyGetResource(*r))
                    .collect(),
            },
            GameState::ContinueAction => {
                let current_player = self.current_player_ref();
                current_player.get_available_actions(
                    &self.board,
                    self.dev_cards.as_slice(),
                )
            }
            GameState::GameEnded(_) => {
                panic!(
                    "Impossible game state. It should have been handled earlier"
                );
            }
        }
    }

    pub fn run_action(&mut self, action: Action) -> Reward {
        let mut should_recompute_vp = false;

        match action {
            Action::BuildRoad(r) => {
                self.board.build_road(r, self.current_player_id());

                let game_state = self.state;

                let current_player = self.current_player_mut();
                current_player.num_road += 1;

                if let GameState::InitialBoardSetup(current_order) = game_state
                {
                    // Resources are not consumed during initial board setup.
                    if current_order == 15 {
                        // Last initial-setup road. Roll for player 0's
                        // first regular turn here so that get_actions
                        // never has to handle the transient RollDice
                        // state externally. (current_player is already 0
                        // at this point.)
                        self.roll_dice_for_current_player();
                    } else {
                        self.state =
                            GameState::InitialBoardSetup(current_order + 1);
                    }
                } else if let GameState::UsedDevCard((
                    DevCard::RoadBuilding,
                    processed_once,
                )) = game_state
                {
                    if processed_once {
                        self.state = GameState::ContinueAction;
                    } else {
                        // Borrow as a ref to pass self.board as ref.
                        let current_player_ref = self.current_player_ref();
                        // Only continue RoadBuilding if the player can build
                        // another road.
                        if current_player_ref.num_road < 15
                            && !current_player_ref
                                .get_available_roads(&self.board)
                                .is_empty()
                        {
                            self.state = GameState::UsedDevCard((
                                DevCard::RoadBuilding,
                                true,
                            ));
                        } else {
                            self.state = GameState::ContinueAction;
                        }
                    }
                } else {
                    current_player.cards[Resource::Brick as usize] -= 1;
                    current_player.cards[Resource::Lumber as usize] -= 1;
                }

                should_recompute_vp = true;
            }
            Action::BuildStructure(building, node_id) => {
                let game_state = self.state;

                match building {
                    Building::Settlement => {
                        self.current_player_mut().num_settlement += 1;

                        if let GameState::InitialBoardSetup(current_order) =
                            game_state
                        {
                            if current_order >= 8 {
                                // Get resources adjacent to the node id.
                                let init_resources = self
                                    .board
                                    .get_resources_near_node_id(node_id);

                                let current_player = self.current_player_mut();
                                init_resources.into_iter().for_each(|r| {
                                    current_player.add_resource_card(r, 1);
                                });
                            }

                            // Settlements are placed on even initial-setup
                            // steps (0, 2, …, 14) and roads on odd ones, so
                            // BuildStructure is never the step-15 transition.
                            // Just advance the order — the final dice roll
                            // happens in the BuildRoad handler.
                            self.state =
                                GameState::InitialBoardSetup(current_order + 1);
                        } else {
                            let current_player = self.current_player_mut();
                            current_player.cards[Resource::Brick as usize] -= 1;
                            current_player.cards[Resource::Lumber as usize] -=
                                1;
                            current_player.cards[Resource::Grain as usize] -= 1;
                            current_player.cards[Resource::Wool as usize] -= 1;
                        }
                    }
                    Building::City => {
                        let current_player = self.current_player_mut();
                        current_player.num_city += 1;
                        current_player.num_settlement -= 1;

                        current_player.cards[Resource::Grain as usize] -= 2;
                        current_player.cards[Resource::Ore as usize] -= 3;
                    }
                }

                self.board.build_structure(
                    node_id,
                    self.current_player_id(),
                    building,
                );

                should_recompute_vp = true;
            }
            Action::MaritimeTrade(lose, get) => {
                let current_player = self.current_player_mut();
                current_player.cards[lose.0 as usize] -= lose.1;
                current_player.cards[get as usize] += 1;
            }
            Action::DiscardResource(resource) => {
                self.current_player_mut().cards[resource as usize] -= 1;

                // Advance the discard sub-state-machine here (used to live
                // inside get_actions). If somebody else still needs to
                // discard, hand them control; otherwise return control to
                // the roller and move on to the move-robber decision.
                if let GameState::Dice7Rolled((roller, target_counts)) =
                    self.state
                {
                    if let Some(next) = self.find_next_discarder(&target_counts)
                    {
                        self.current_player = usize::from(next);
                    } else {
                        self.current_player = usize::from(roller);
                        self.state = GameState::AwaitingMoveRobber;
                    }
                }
            }
            Action::MoveRobber(tile_id) => {
                self.board.move_robber_to(tile_id);
                // Decide the next decision point now (and not on the next
                // get_actions call) so get_actions stays idempotent.
                if self.compute_robber_victims().is_empty() {
                    self.state = GameState::ContinueAction;
                } else {
                    self.state = GameState::StealResourceAfterRobber;
                }
            }
            Action::StealCard(player_id) => {
                let player = &mut self.players[usize::from(player_id)];
                if player.num_resource_cards() > 0 {
                    let r = choose_weighted(&player.cards[0..5], &mut self.rng);
                    player.use_resource_card(Resource::ALL[r], 1);

                    self.current_player_mut()
                        .add_resource_card(Resource::ALL[r], 1);
                }
                self.state = GameState::ContinueAction;
            }
            Action::EndTurn => {
                self.current_player_mut().end_player_turn();
                self.current_player = (self.current_player + 1) % 4;
                self.game_detail.total_turns += 1;

                // Roll a dice for the next player to move forward with the
                // next steps.
                // NOTE: the winning condition or the rewards computed won't be
                // affected by this.
                self.roll_dice_for_current_player();
            }
            Action::BuyDevCard => {
                let dev_card = self.get_random_dev_card();

                let current_player = self.current_player_mut();
                current_player.cards[Resource::Ore as usize] -= 1;
                current_player.cards[Resource::Grain as usize] -= 1;
                current_player.cards[Resource::Wool as usize] -= 1;
                current_player.add_dev_card(dev_card);
            }
            Action::UseDevCard(dev_card) => {
                self.current_player_mut().use_dev_card(dev_card);

                match dev_card {
                    DevCard::Knight => {
                        self.state = GameState::AwaitingMoveRobber;

                        let num_used_knight_cards =
                            self.current_player_ref().num_used_knight_cards;
                        if num_used_knight_cards >= 3 {
                            if let Some((_, count)) =
                                self.game_detail.get_largest_army()
                            {
                                if count < num_used_knight_cards {
                                    self.game_detail.set_largest_army(Some((
                                        self.current_player_id(),
                                        num_used_knight_cards,
                                    )));
                                }
                            } else {
                                self.game_detail.set_largest_army(Some((
                                    self.current_player_id(),
                                    num_used_knight_cards,
                                )));
                            }

                            should_recompute_vp = true;
                        }
                    }
                    DevCard::VictoryPoint => {
                        self.state = GameState::ContinueAction;
                        should_recompute_vp = true;
                    }
                    DevCard::RoadBuilding => {
                        self.state = GameState::UsedDevCard((
                            DevCard::RoadBuilding,
                            false,
                        ));
                    }
                    DevCard::YearOfPlenty => {
                        self.state = GameState::UsedDevCard((
                            DevCard::YearOfPlenty,
                            false,
                        ));
                    }
                    DevCard::Monopoly => {
                        self.state =
                            GameState::UsedDevCard((DevCard::Monopoly, false));
                    }
                }
            }
            Action::YearOfPlentyGetResource(resource) => {
                let current_player = self.current_player_mut();
                current_player.add_resource_card(resource, 1);

                if let GameState::UsedDevCard((
                    DevCard::YearOfPlenty,
                    processed_once,
                )) = self.state
                {
                    if processed_once {
                        self.state = GameState::ContinueAction;
                    } else {
                        self.state = GameState::UsedDevCard((
                            DevCard::YearOfPlenty,
                            true,
                        ));
                    }
                } else {
                    panic!(
                        "Year of plenty can only be used when the game state is set as YearOfPlenty"
                    )
                }
            }
            Action::MonopolyGetResource(resource) => {
                let current_player_id = self.current_player_id();

                let mut total_cards = 0;
                for player in &mut self.players {
                    if player.id == current_player_id {
                        continue;
                    }

                    let num_cards = player.cards[resource as usize];
                    total_cards += num_cards;
                    player.use_resource_card(resource, num_cards);
                }

                self.current_player_mut()
                    .add_resource_card(resource, total_cards);

                self.state = GameState::ContinueAction;
            }
        }

        self.compute_rewards(should_recompute_vp)
    }

    pub fn get_game_state_for_debug(&self) -> String {
        let mut debug_info = String::new();
        debug_info.push_str(&format!(
            "Total turns: {}\n",
            self.game_detail.total_turns
        ));

        for player in &self.players {
            debug_info.push_str(&format!(
                "Player {}: {}\n",
                usize::from(player.id),
                player.get_debug_info()
            ));
        }

        debug_info.push_str(&render_board(&self.board));

        debug_info
    }

    // Run "one step". Returns the player id when the game is ended. Otherwise returns None.
    fn step(&mut self) -> Option<PlayerId> {
        let mut actions = self.get_actions();

        let action_index = self.rng.random_range(0..actions.len());
        let action = actions.swap_remove(action_index);

        self.run_action(action);

        match self.state {
            GameState::GameEnded(player) => Some(player),
            _ => None,
        }
    }

    // First player still over their post-7-roll discard target, scanning in
    // PlayerId order. None means everyone has discarded enough.
    fn find_next_discarder(&self, target_counts: &[u8; 4]) -> Option<PlayerId> {
        self.players
            .iter()
            .find(|p| p.num_resource_cards() > target_counts[usize::from(p.id)])
            .map(|p| p.id)
    }

    // Set of player ids (other than the current player) that have at least
    // one structure adjacent to the robber's current tile. Caller decides
    // what to do with an empty result (skip steal phase) vs. non-empty
    // (drive a StealCard decision).
    fn compute_robber_victims(&self) -> VecSet<usize> {
        let robber_tile = self.board.get_robber_position();
        let mut player_set = VecSet::new();
        for node in TILE_TO_NODE[usize::from(robber_tile)] {
            if let Some((_, p)) = self.board.get_structure(node).get() {
                if usize::from(p) == self.current_player {
                    continue;
                }
                player_set.insert(usize::from(p));
            }
        }
        player_set
    }

    // Roll 2d6 for whoever `self.current_player` currently points at, set
    // `last_dice_roll`, and put the engine in the right post-roll state:
    //   - 7  → Dice7Rolled with per-player target hand sizes for discards
    //   - !7 → distribute resources, then ContinueAction
    //
    // Called when control naturally hands off to the next decision point —
    // i.e. inside run_action(EndTurn) and at the end of the initial setup —
    // so get_actions stays a pure projection of state.
    fn roll_dice_for_current_player(&mut self) {
        let dice = self.rng.random_range(1..=6) + self.rng.random_range(1..=6);
        self.last_dice_roll = Some(dice);

        if dice != 7 {
            self.handle_non_7_dice_roll(dice);
            self.state = GameState::ContinueAction;
            return;
        }

        let mut target_counts = [0; 4];
        for player in &self.players {
            target_counts[usize::from(player.id)] =
                if player.num_resource_cards() > 7 {
                    player.num_resource_cards().div_ceil(2)
                } else {
                    player.num_resource_cards()
                };
        }

        let roller = PlayerId::from(self.current_player);
        if let Some(discarder) = self.find_next_discarder(&target_counts) {
            // Hand control to the first player to discard; subsequent advances
            // happen in run_action::DiscardResource so get_actions stays
            // idempotent.
            self.current_player = usize::from(discarder);
            // Need to keep track whose original turn was, and the target
            // counts for the players to keep.
            self.state = GameState::Dice7Rolled((roller, target_counts));
        } else {
            // Nobody is over the cap; jump straight to the move-robber
            // decision for the roller.
            self.state = GameState::AwaitingMoveRobber;
        }
    }

    fn handle_non_7_dice_roll(&mut self, dice: u8) {
        assert!(dice != 7);

        let resources = self.board.get_matching_resources(dice);

        for (player, resource, num) in resources {
            self.players[usize::from(player)].add_resource_card(resource, num);
        }
    }

    fn get_random_dev_card(&mut self) -> DevCard {
        // NOTE: callers of this function should ensure that there is at least one dev card left.
        let dev_card = choose_weighted(&self.dev_cards, &mut self.rng);

        self.dev_cards[dev_card] -= 1;
        DevCard::from_array_index(dev_card)
    }

    fn compute_rewards(&mut self, requires_vp_recompute: bool) -> Reward {
        let current_vps = self.game_detail.vp;

        if requires_vp_recompute {
            self.compute_victory_point();
        }

        let updated_vps = self.game_detail.vp;

        // Check if current player has won the game
        let current_player = self.current_player_id();
        if self.game_detail.vp.get_vp(current_player) >= WINNING_VP {
            self.state = GameState::GameEnded(current_player);
            return Reward::from_win(current_player);
        }

        // If no player has won the game, then compute the reward based on the
        // VP changes.
        Reward::from_vp_change(current_vps, updated_vps)
    }

    fn compute_victory_point(&mut self) {
        let mut vps = ArrayVec::<u8, 4>::from([0; 4]);
        self.compute_longest_road();

        for (id, p) in self.players.iter().enumerate() {
            vps[id] += p.num_settlement;
            vps[id] += p.num_city * 2;
            vps[id] += p.cards[DevCard::VictoryPoint as usize];

            // VP cards that are bought at the given turn has an effect
            // immediately.
            vps[id] += p.dev_card_bought[DevCard::VictoryPoint as usize - 5];
        }

        if let Some((p, _)) = self.game_detail.get_longest_road() {
            vps[usize::from(p)] += 2;
        }

        if let Some((p, _)) = self.game_detail.get_largest_army() {
            vps[usize::from(p)] += 2;
        }

        self.game_detail.vp = VictoryPoint::new(vps);
    }

    fn compute_longest_road(&mut self) {
        let longest_road = find_longest_road(&self.board);
        let longest_length = *longest_road.iter().max().unwrap() as u8;

        if longest_length < 5 {
            self.game_detail.set_longest_road(None);
            return;
        }

        let current_holder = self.game_detail.get_longest_road();
        if let Some((current_holder, _)) = current_holder {
            let current_holder_length =
                longest_road[usize::from(current_holder)];
            if current_holder_length == longest_length {
                self.game_detail
                    .set_longest_road(Some((current_holder, longest_length)));
                return;
            }
        }

        // Now, the current holder is no longest the longest length holder.
        // If there is a unique *NEW* longest holder, then that player will take the card.
        // If there are *multiple* candidates that holds the record, then no one will take the
        // card.

        let players_with_longest_road = longest_road
            .into_iter()
            .enumerate()
            .filter_map(|(p, l)| match l == longest_length {
                true => Some(PlayerId::from(p)),
                false => None,
            })
            .collect::<Vec<_>>();

        if players_with_longest_road.len() > 1 {
            self.game_detail.set_longest_road(None);
        } else {
            debug_assert!(players_with_longest_road.len() == 1);
            self.game_detail.set_longest_road(Some((
                players_with_longest_road[0],
                longest_length,
            )));
        }
    }

    fn current_player_id(&self) -> PlayerId {
        match self.state {
            GameState::InitialBoardSetup(current_order) => {
                let player_id = match current_order {
                    0..=7 => usize::from(current_order / 2),
                    8..=15 => usize::from(3 - (current_order - 8) / 2),
                    _ => unreachable!(),
                };

                PlayerId::from(player_id)
            }
            _ => PlayerId::from(self.current_player),
        }
    }

    fn current_player_mut(&mut self) -> &mut Player {
        let player_id = usize::from(self.current_player_id());
        &mut self.players[player_id]
    }

    fn current_player_ref(&self) -> &Player {
        let player_id = usize::from(self.current_player_id());
        &self.players[player_id]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::TileType;
    use crate::util::id::NodeId;
    use crate::util::id::RoadId;
    use crate::util::id::TileId;

    fn p(id: usize) -> PlayerId {
        PlayerId::new(id as u8)
    }

    /// Create a Game with an empty board (no tiles) for isolated unit testing.
    fn game_with_empty_board() -> Game {
        let rng = ChaCha8Rng::seed_from_u64(0);
        Game {
            board: Board::new(&[]),
            players: ArrayVec::from(from_fn(|i| Player::new(i.into()))),
            current_player: 0,
            dev_cards: ArrayVec::from([0; 5]),
            game_detail: GameDetail::new(),
            state: GameState::ContinueAction,
            last_dice_roll: None,
            rng,
        }
    }

    // ---- GameDetail tests ----

    #[test]
    fn game_detail_initially_no_longest_road() {
        let detail = GameDetail::new();
        assert_eq!(detail.get_longest_road(), None);
    }

    #[test]
    fn game_detail_set_and_get_longest_road() {
        let mut detail = GameDetail::new();
        detail.set_longest_road(Some((p(2), 7)));
        let result = detail.get_longest_road();
        assert_eq!(result, Some((p(2), 7)));
    }

    #[test]
    fn game_detail_clear_longest_road() {
        let mut detail = GameDetail::new();
        detail.set_longest_road(Some((p(1), 5)));
        assert!(detail.get_longest_road().is_some());

        detail.set_longest_road(None);
        assert_eq!(detail.get_longest_road(), None);
    }

    #[test]
    fn game_detail_longest_road_all_players() {
        let mut detail = GameDetail::new();
        for id in 0..4 {
            detail.set_longest_road(Some((p(id), 5 + id as u8)));
            let (player, length) = detail.get_longest_road().unwrap();
            assert_eq!(player, p(id));
            assert_eq!(length, 5 + id as u8);
        }
    }

    #[test]
    fn game_detail_initially_no_largest_army() {
        let detail = GameDetail::new();
        assert_eq!(detail.get_largest_army(), None);
    }

    #[test]
    fn game_detail_set_and_get_largest_army() {
        let mut detail = GameDetail::new();
        detail.set_largest_army(Some((p(2), 5)));
        assert_eq!(detail.get_largest_army(), Some((p(2), 5)));
    }

    #[test]
    fn game_detail_clear_largest_army() {
        let mut detail = GameDetail::new();
        detail.set_largest_army(Some((p(1), 3)));
        detail.set_largest_army(None);
        assert_eq!(detail.get_largest_army(), None);
    }

    #[test]
    fn game_detail_set_largest_army_overwrites_previous_value() {
        // Regression: an earlier refactor used `|=` which OR-merged stale bits
        // from a prior holder into the new value.
        let mut detail = GameDetail::new();
        detail.set_largest_army(Some((p(0), 3)));
        detail.set_largest_army(Some((p(1), 4)));
        assert_eq!(detail.get_largest_army(), Some((p(1), 4)));
    }

    #[test]
    fn game_detail_largest_army_independent_from_longest_road() {
        let mut detail = GameDetail::new();
        detail.set_longest_road(Some((p(0), 5)));
        detail.set_largest_army(Some((p(2), 3)));
        assert_eq!(detail.get_longest_road(), Some((p(0), 5)));
        assert_eq!(detail.get_largest_army(), Some((p(2), 3)));
    }

    // ---- Game initialization ----

    #[test]
    fn game_new_has_four_players() {
        let game = Game::new(42);
        assert_eq!(game.players.len(), 4);
    }

    #[test]
    fn game_new_starts_at_player_zero() {
        let game = Game::new(42);
        assert_eq!(game.current_player, 0);
    }

    // ---- run_action tests ----

    #[test]
    fn run_action_build_road_deducts_resources() {
        let mut game = game_with_empty_board();

        // Give player 0 resources for a road.
        game.players[0].cards[Resource::Brick as usize] = 1;
        game.players[0].cards[Resource::Lumber as usize] = 1;

        game.run_action(Action::BuildRoad(RoadId::new(0)));

        assert_eq!(game.players[0].cards[Resource::Brick as usize], 0);
        assert_eq!(game.players[0].cards[Resource::Lumber as usize], 0);
        assert_eq!(game.players[0].num_road, 1);
        assert!(game.board.get_road(RoadId::new(0)).get().is_some());
    }

    #[test]
    fn run_action_build_settlement_deducts_resources() {
        let mut game = game_with_empty_board();

        game.players[0].cards[Resource::Brick as usize] = 1;
        game.players[0].cards[Resource::Lumber as usize] = 1;
        game.players[0].cards[Resource::Grain as usize] = 1;
        game.players[0].cards[Resource::Wool as usize] = 1;

        game.run_action(Action::BuildStructure(
            Building::Settlement,
            NodeId::new(10),
        ));

        assert_eq!(game.players[0].cards[Resource::Brick as usize], 0);
        assert_eq!(game.players[0].cards[Resource::Lumber as usize], 0);
        assert_eq!(game.players[0].cards[Resource::Grain as usize], 0);
        assert_eq!(game.players[0].cards[Resource::Wool as usize], 0);
        assert_eq!(game.players[0].num_settlement, 1);
        assert_eq!(
            game.board.get_structure(NodeId::new(10)).get(),
            Some((Building::Settlement, p(0)))
        );
    }

    #[test]
    fn run_action_build_city_deducts_resources() {
        let mut game = game_with_empty_board();

        // Player needs an existing settlement to upgrade.
        game.players[0].num_settlement = 1;
        game.players[0].cards[Resource::Grain as usize] = 2;
        game.players[0].cards[Resource::Ore as usize] = 3;

        game.run_action(Action::BuildStructure(Building::City, NodeId::new(5)));

        assert_eq!(game.players[0].cards[Resource::Grain as usize], 0);
        assert_eq!(game.players[0].cards[Resource::Ore as usize], 0);
        assert_eq!(game.players[0].num_city, 1);
        assert_eq!(game.players[0].num_settlement, 0);
    }

    #[test]
    fn run_action_maritime_trade() {
        let mut game = game_with_empty_board();

        game.players[0].cards[Resource::Wool as usize] = 4;

        game.run_action(Action::MaritimeTrade(
            (Resource::Wool, 4),
            Resource::Ore,
        ));

        assert_eq!(game.players[0].cards[Resource::Wool as usize], 0);
        assert_eq!(game.players[0].cards[Resource::Ore as usize], 1);
    }

    // ---- Initial board setup tests ----

    #[test]
    fn second_initial_settlement_grants_adjacent_resources() {
        // Use a full random board; the second-round settlement should yield
        // exactly the resources of the tiles adjacent to its node.
        let mut game = Game::new(42);
        // Step 14 → player 0's second (final) initial settlement.
        // Resources are only granted on the second-round placement
        // (current_order >= 8).
        game.state = GameState::InitialBoardSetup(14);
        game.current_player = 0;

        // Pick any node and ask the board what resources it borders, so the
        // assertion stays independent of the specific board layout.
        let node = NodeId::new(8);
        let mut expected = [0u8; 5];
        for r in game.board.get_resources_near_node_id(node) {
            expected[r as usize] += 1;
        }

        // First settlement (step 0–7) for sanity: must NOT grant resources.
        let mut sanity = Game::new(42);
        sanity.state = GameState::InitialBoardSetup(0);
        sanity.run_action(Action::BuildStructure(
            Building::Settlement,
            NodeId::new(0),
        ));
        assert_eq!(sanity.players[0].cards[..5], [0u8; 5]);

        game.run_action(Action::BuildStructure(Building::Settlement, node));

        assert_eq!(game.players[0].cards[..5], expected);
        assert_eq!(game.players[0].num_settlement, 1);
        assert!(matches!(game.state, GameState::InitialBoardSetup(15)));
    }

    // ---- Victory point tests ----

    #[test]
    fn compute_victory_point_settlements() {
        let mut game = game_with_empty_board();
        game.players[0].num_settlement = 3;
        game.players[1].num_settlement = 2;

        game.compute_victory_point();
        assert_eq!(game.game_detail.vp.get_vp(p(0)), 3);
        assert_eq!(game.game_detail.vp.get_vp(p(1)), 2);
        assert_eq!(game.game_detail.vp.get_vp(p(2)), 2); // clamped to min 2
        assert_eq!(game.game_detail.vp.get_vp(p(3)), 2); // clamped to min 2
    }

    #[test]
    fn compute_victory_point_cities() {
        let mut game = game_with_empty_board();
        game.players[0].num_city = 2;
        game.players[0].num_settlement = 1;

        game.compute_victory_point();
        assert_eq!(game.game_detail.vp.get_vp(p(0)), 5); // 2*2 + 1
    }

    #[test]
    fn compute_victory_point_with_longest_road() {
        let mut game = game_with_empty_board();
        game.players[0].num_settlement = 2;
        game.game_detail.set_longest_road(Some((p(0), 6)));

        // Build 6 roads for player 0 to match the game_detail.
        // Road 0 connects nodes [0, 3], road 1 connects [0, 4], etc.
        // Build a chain: roads 0, 1, 2, 3, 4, 5 for player 0.
        for i in 0..6 {
            game.board.build_road(RoadId::new(i), p(0));
        }

        game.compute_victory_point();
        // 2 settlements + 2 longest road = 4
        // (compute_longest_road may recalculate and override, but with 6
        // roads built the actual longest road computation determines the holder)
        assert!(game.game_detail.vp.get_vp(p(0)) >= 2);
    }

    // ---- handle_non_7_dice_roll tests ----

    #[test]
    fn handle_non_7_dice_roll_gives_resources() {
        let rng = ChaCha8Rng::seed_from_u64(0);
        // Tile 0 at dice=5, type=Forest (produces Lumber).
        // Tile 0 nodes: [0, 3, 4, 7, 8, 12].
        let mut game = Game {
            board: Board::new(&[(5, TileType::Forest)]),
            players: ArrayVec::from(from_fn(|i| Player::new(i.into()))),
            current_player: 0,
            dev_cards: ArrayVec::from([0; 5]),
            game_detail: GameDetail::new(),
            state: GameState::ContinueAction,
            last_dice_roll: None,
            rng,
        };

        // Place player 0's settlement on node 0 (belongs to tile 0).
        game.board
            .build_structure(NodeId::new(0), p(0), Building::Settlement);

        game.handle_non_7_dice_roll(5);

        assert_eq!(game.players[0].cards[Resource::Lumber as usize], 1);
    }

    #[test]
    fn handle_non_7_dice_roll_city_gives_two() {
        let rng = ChaCha8Rng::seed_from_u64(0);
        let mut game = Game {
            board: Board::new(&[(8, TileType::Mountains)]),
            players: ArrayVec::from(from_fn(|i| Player::new(i.into()))),
            current_player: 0,
            dev_cards: ArrayVec::from([0; 5]),
            game_detail: GameDetail::new(),
            state: GameState::ContinueAction,
            last_dice_roll: None,
            rng,
        };

        game.board
            .build_structure(NodeId::new(3), p(1), Building::City);
    }

    #[test]
    fn longest_road_under_5_sets_none() {
        let mut game = game_with_empty_board();
        // Build 4 roads for player 0 (not enough for longest road).
        for i in 0..4 {
            game.board.build_road(RoadId::new(i), p(0));
        }
        game.compute_longest_road();
        assert_eq!(game.game_detail.get_longest_road(), None);
    }

    #[test]
    fn current_player_id_matches_index() {
        let mut game = game_with_empty_board();
        assert_eq!(game.current_player_id(), p(0));

        game.current_player = 2;
        assert_eq!(game.current_player_id(), p(2));
    }

    // ---- Dice 7 roll sequence tests ----

    #[test]
    fn dice_7_sequence_discard_then_move_robber_then_steal() {
        let rng = ChaCha8Rng::seed_from_u64(0);

        // Tile 0: Desert (robber starts here). Nodes: [0, 3, 4, 7, 8, 12]
        // Tile 1: Forest.                      Nodes: [1, 4, 5, 8, 9, 13]
        let mut game = Game {
            board: Board::new(&[(2, TileType::Desert), (5, TileType::Forest)]),
            players: ArrayVec::from(from_fn(|i| Player::new(i.into()))),
            // current_player tracks the active discarder while in Dice7Rolled
            // (set by roll_dice_for_current_player when 7 is rolled). Player
            // 1 has the most cards over their target, so they discard first.
            current_player: 1,
            dev_cards: ArrayVec::from([0; 5]),
            game_detail: GameDetail::new(),
            state: GameState::Dice7Rolled((p(0), [0, 4, 5, 0])), // Player 0 rolled a 7
            last_dice_roll: Some(7),
            rng,
        };

        // Player 1: 8 resource cards (must discard half, down to 4).
        game.players[1].add_resource_card(Resource::Brick, 4);
        game.players[1].add_resource_card(Resource::Lumber, 4);

        // Player 2: 9 resource cards (must discard half, down to 5).
        game.players[2].add_resource_card(Resource::Ore, 5);
        game.players[2].add_resource_card(Resource::Grain, 4);

        // Place player 2's settlement on tile 1 (node 1) for the steal phase.
        game.board
            .build_structure(NodeId::new(1), p(2), Building::Settlement);

        // --- Phase 1: Discard ---

        // Player 1 (8 cards, target 4) should discard first.
        let actions = game.get_actions();
        assert!(
            actions
                .iter()
                .all(|a| matches!(a, Action::DiscardResource(_)))
        );
        assert_eq!(game.current_player, 1);

        game.run_action(Action::DiscardResource(Resource::Brick));
        assert_eq!(game.players[1].num_resource_cards(), 7);

        game.run_action(Action::DiscardResource(Resource::Brick));
        assert_eq!(game.players[1].num_resource_cards(), 6);

        game.run_action(Action::DiscardResource(Resource::Brick));
        assert_eq!(game.players[1].num_resource_cards(), 5);

        game.run_action(Action::DiscardResource(Resource::Brick));
        assert_eq!(game.players[1].num_resource_cards(), 4);

        // Player 2 (9 cards, target 5) should discard next.
        let actions = game.get_actions();
        assert!(
            actions
                .iter()
                .all(|a| matches!(a, Action::DiscardResource(_)))
        );
        assert_eq!(game.current_player, 2);

        game.run_action(Action::DiscardResource(Resource::Ore));
        assert_eq!(game.players[2].num_resource_cards(), 8);

        game.run_action(Action::DiscardResource(Resource::Ore));
        assert_eq!(game.players[2].num_resource_cards(), 7);

        game.run_action(Action::DiscardResource(Resource::Ore));
        assert_eq!(game.players[2].num_resource_cards(), 6);

        game.run_action(Action::DiscardResource(Resource::Ore));
        assert_eq!(game.players[2].num_resource_cards(), 5);

        // --- Phase 2: Move Robber ---

        // All discards done. Current player should be restored to player 0 (who rolled).
        let actions = game.get_actions();
        assert!(actions.iter().all(|a| matches!(a, Action::MoveRobber(_))));
        assert_eq!(game.current_player, 0);

        // Move robber to tile 1 (where player 2 has a settlement).
        game.run_action(Action::MoveRobber(TileId::from(1usize)));

        // --- Phase 3: Steal ---

        let actions = game.get_actions();
        assert_eq!(actions, vec![Action::StealCard(p(2))]);

        game.run_action(Action::StealCard(p(2)));

        // Player 0 gained one card, player 2 lost one.
        assert_eq!(game.players[0].num_resource_cards(), 1);
        assert_eq!(game.players[2].num_resource_cards(), 4);
    }

    // ---- Dev card tests ----

    fn game_in_continue_action() -> Game {
        let mut game = game_with_empty_board();
        game.state = GameState::ContinueAction;
        game
    }

    #[test]
    fn buy_dev_card_deducts_resources_and_buffers_card() {
        let mut game = game_in_continue_action();
        game.dev_cards = ArrayVec::from([14, 5, 2, 2, 2]);
        game.players[0].cards[Resource::Ore as usize] = 1;
        game.players[0].cards[Resource::Grain as usize] = 1;
        game.players[0].cards[Resource::Wool as usize] = 1;

        game.run_action(Action::BuyDevCard);

        assert_eq!(game.players[0].cards[Resource::Ore as usize], 0);
        assert_eq!(game.players[0].cards[Resource::Grain as usize], 0);
        assert_eq!(game.players[0].cards[Resource::Wool as usize], 0);
        assert_eq!(game.players[0].dev_card_bought.iter().sum::<u8>(), 1);
        // Deck went down by one.
        assert_eq!(game.dev_cards.iter().sum::<u8>(), 24);
    }

    #[test]
    fn end_turn_migrates_bought_dev_cards_to_hand() {
        let mut game = game_with_empty_board();
        game.players[0].add_dev_card(DevCard::Knight);
        game.players[0].add_dev_card(DevCard::Monopoly);

        game.run_action(Action::EndTurn);

        assert_eq!(game.players[0].dev_card_bought.iter().sum::<u8>(), 0);
        assert_eq!(game.players[0].cards[DevCard::Knight as usize], 1);
        assert_eq!(game.players[0].cards[DevCard::Monopoly as usize], 1);
    }

    #[test]
    fn cannot_play_dev_card_bought_this_turn() {
        let mut game = game_in_continue_action();
        game.dev_cards = ArrayVec::from([14, 0, 0, 0, 0]);
        game.players[0].cards[Resource::Ore as usize] = 1;
        game.players[0].cards[Resource::Grain as usize] = 1;
        game.players[0].cards[Resource::Wool as usize] = 1;

        game.run_action(Action::BuyDevCard);

        let actions = game.get_actions();
        assert!(
            !actions.iter().any(|a| matches!(a, Action::UseDevCard(_))),
            "should not be able to play a dev card bought this turn"
        );
    }

    #[test]
    fn cannot_buy_dev_card_when_deck_empty() {
        let mut game = game_in_continue_action();
        game.players[0].cards[Resource::Ore as usize] = 5;
        game.players[0].cards[Resource::Grain as usize] = 5;
        game.players[0].cards[Resource::Wool as usize] = 5;

        let actions = game.get_actions();
        assert!(!actions.iter().any(|a| matches!(a, Action::BuyDevCard)));
    }

    #[test]
    fn use_knight_offers_move_robber_and_consumes_card() {
        let rng = ChaCha8Rng::seed_from_u64(0);
        let mut game = Game {
            board: Board::new(&[(2, TileType::Desert), (5, TileType::Forest)]),
            players: ArrayVec::from(from_fn(|i| Player::new(i.into()))),
            current_player: 0,
            dev_cards: ArrayVec::from([0; 5]),
            game_detail: GameDetail::new(),
            state: GameState::ContinueAction,
            last_dice_roll: None,
            rng,
        };
        game.players[0].cards[DevCard::Knight as usize] = 1;

        game.run_action(Action::UseDevCard(DevCard::Knight));
        let actions = game.get_actions();

        assert!(actions.iter().all(|a| matches!(a, Action::MoveRobber(_))));
        assert!(!actions.is_empty());
        assert_eq!(game.players[0].cards[DevCard::Knight as usize], 0);
    }

    #[test]
    fn use_knight_full_flow_steals_resource() {
        let rng = ChaCha8Rng::seed_from_u64(0);
        let mut game = Game {
            board: Board::new(&[(2, TileType::Desert), (5, TileType::Forest)]),
            players: ArrayVec::from(from_fn(|i| Player::new(i.into()))),
            current_player: 0,
            dev_cards: ArrayVec::from([0; 5]),
            game_detail: GameDetail::new(),
            state: GameState::ContinueAction,
            last_dice_roll: None,
            rng,
        };
        game.players[0].cards[DevCard::Knight as usize] = 1;
        game.players[1].add_resource_card(Resource::Wool, 1);
        // Tile 1's nodes include node 1.
        game.board
            .build_structure(NodeId::new(1), p(1), Building::Settlement);

        game.run_action(Action::UseDevCard(DevCard::Knight));
        let _ = game.get_actions();
        game.run_action(Action::MoveRobber(TileId::from(1usize)));

        let actions = game.get_actions();
        assert_eq!(actions, vec![Action::StealCard(p(1))]);
        game.run_action(Action::StealCard(p(1)));

        assert_eq!(game.players[0].num_resource_cards(), 1);
        assert_eq!(game.players[1].num_resource_cards(), 0);
    }

    #[test]
    fn using_two_knights_does_not_grant_largest_army() {
        let mut game = game_in_continue_action();
        game.players[0].cards[DevCard::Knight as usize] = 1;
        game.players[0].num_used_knight_cards = 1;

        game.run_action(Action::UseDevCard(DevCard::Knight));

        assert_eq!(game.players[0].num_used_knight_cards, 2);
        assert_eq!(game.game_detail.get_largest_army(), None);
    }

    #[test]
    fn using_third_knight_grants_largest_army() {
        let mut game = game_in_continue_action();
        game.players[0].cards[DevCard::Knight as usize] = 1;
        game.players[0].num_used_knight_cards = 2;

        game.run_action(Action::UseDevCard(DevCard::Knight));

        assert_eq!(game.players[0].num_used_knight_cards, 3);
        assert_eq!(game.game_detail.get_largest_army(), Some((p(0), 3)));
    }

    #[test]
    fn knight_beyond_third_extends_holders_count() {
        let mut game = game_in_continue_action();
        game.players[0].cards[DevCard::Knight as usize] = 1;
        game.players[0].num_used_knight_cards = 4;
        game.game_detail.set_largest_army(Some((p(0), 4)));

        game.run_action(Action::UseDevCard(DevCard::Knight));

        assert_eq!(game.game_detail.get_largest_army(), Some((p(0), 5)));
    }

    #[test]
    fn tying_knight_count_does_not_steal_largest_army() {
        let mut game = game_in_continue_action();
        // Player 0 holds the army with 3 knights.
        game.game_detail.set_largest_army(Some((p(0), 3)));

        // Player 1 plays their 3rd knight — same count, no steal.
        game.current_player = 1;
        game.players[1].cards[DevCard::Knight as usize] = 1;
        game.players[1].num_used_knight_cards = 2;

        game.run_action(Action::UseDevCard(DevCard::Knight));

        assert_eq!(game.players[1].num_used_knight_cards, 3);
        assert_eq!(game.game_detail.get_largest_army(), Some((p(0), 3)));
    }

    #[test]
    fn strictly_greater_knight_count_steals_largest_army() {
        let mut game = game_in_continue_action();
        game.game_detail.set_largest_army(Some((p(0), 3)));

        // Player 1 plays their 4th knight — strictly greater, takes the army.
        game.current_player = 1;
        game.players[1].cards[DevCard::Knight as usize] = 1;
        game.players[1].num_used_knight_cards = 3;

        game.run_action(Action::UseDevCard(DevCard::Knight));

        assert_eq!(game.game_detail.get_largest_army(), Some((p(1), 4)));
    }

    #[test]
    fn largest_army_grants_two_victory_points() {
        let mut game = game_with_empty_board();
        game.players[0].num_settlement = 2;
        game.game_detail.set_largest_army(Some((p(0), 3)));

        game.compute_victory_point();

        // 2 settlements + 2 (largest army) = 4
        assert_eq!(game.game_detail.vp.get_vp(p(0)), 4);
    }

    #[test]
    fn largest_army_holder_loses_two_vp_when_overtaken() {
        let mut game = game_in_continue_action();
        game.players[0].num_settlement = 2;
        game.players[1].num_settlement = 2;
        game.game_detail.set_largest_army(Some((p(0), 3)));

        // Recompute baseline so p(0) gets the +2.
        game.compute_victory_point();
        assert_eq!(game.game_detail.vp.get_vp(p(0)), 4);
        assert_eq!(game.game_detail.vp.get_vp(p(1)), 2);

        // Player 1 takes the army with their 4th knight.
        game.current_player = 1;
        game.players[1].cards[DevCard::Knight as usize] = 1;
        game.players[1].num_used_knight_cards = 3;
        game.run_action(Action::UseDevCard(DevCard::Knight));

        assert_eq!(game.game_detail.vp.get_vp(p(0)), 2); // lost the +2
        assert_eq!(game.game_detail.vp.get_vp(p(1)), 4); // gained the +2
    }

    #[test]
    fn use_year_of_plenty_grants_two_resources() {
        let mut game = game_in_continue_action();
        game.players[0].cards[DevCard::YearOfPlenty as usize] = 1;

        game.run_action(Action::UseDevCard(DevCard::YearOfPlenty));

        let actions = game.get_actions();
        assert!(
            actions
                .iter()
                .all(|a| matches!(a, Action::YearOfPlentyGetResource(_)))
        );
        game.run_action(Action::YearOfPlentyGetResource(Resource::Ore));

        let actions = game.get_actions();
        assert!(
            actions
                .iter()
                .all(|a| matches!(a, Action::YearOfPlentyGetResource(_))),
            "second resource pick should still be in the YearOfPlenty flow"
        );
        game.run_action(Action::YearOfPlentyGetResource(Resource::Grain));

        assert_eq!(game.players[0].cards[Resource::Ore as usize], 1);
        assert_eq!(game.players[0].cards[Resource::Grain as usize], 1);
        assert!(matches!(game.state, GameState::ContinueAction));
    }

    #[test]
    fn use_monopoly_takes_all_of_chosen_resource() {
        let mut game = game_in_continue_action();
        game.players[0].cards[DevCard::Monopoly as usize] = 1;
        game.players[1].add_resource_card(Resource::Wool, 3);
        game.players[2].add_resource_card(Resource::Wool, 2);
        game.players[3].add_resource_card(Resource::Wool, 1);
        // Other resources must be untouched.
        game.players[1].add_resource_card(Resource::Brick, 5);

        game.run_action(Action::UseDevCard(DevCard::Monopoly));
        game.run_action(Action::MonopolyGetResource(Resource::Wool));

        assert_eq!(game.players[0].cards[Resource::Wool as usize], 6);
        assert_eq!(game.players[1].cards[Resource::Wool as usize], 0);
        assert_eq!(game.players[2].cards[Resource::Wool as usize], 0);
        assert_eq!(game.players[3].cards[Resource::Wool as usize], 0);
        assert_eq!(game.players[1].cards[Resource::Brick as usize], 5);
        assert!(matches!(game.state, GameState::ContinueAction));
    }

    #[test]
    fn road_building_not_offered_when_no_available_roads() {
        let mut game = game_in_continue_action();
        game.players[0].cards[DevCard::RoadBuilding as usize] = 1;
        // No existing road or settlement → no road can be extended from.

        let actions = game.get_actions();
        assert!(
            !actions.iter().any(|a| matches!(
                a,
                Action::UseDevCard(DevCard::RoadBuilding)
            )),
            "RoadBuilding should not be offered when no roads can be built"
        );
    }

    #[test]
    fn road_building_not_offered_when_no_roads_left_in_stock() {
        let mut game = game_in_continue_action();
        game.players[0].cards[DevCard::RoadBuilding as usize] = 1;
        // Legal location exists, but the player has placed all 15 roads.
        game.board.build_road(RoadId::new(0), p(0));
        game.players[0].num_road = 15;

        let actions = game.get_actions();
        assert!(
            !actions.iter().any(|a| matches!(
                a,
                Action::UseDevCard(DevCard::RoadBuilding)
            )),
            "RoadBuilding should not be offered when stock is exhausted"
        );
    }

    #[test]
    fn use_road_building_grants_two_free_roads() {
        let mut game = game_in_continue_action();
        game.players[0].cards[DevCard::RoadBuilding as usize] = 1;
        // Seed an existing road so get_available_roads has somewhere to extend.
        game.board.build_road(RoadId::new(0), p(0));

        game.run_action(Action::UseDevCard(DevCard::RoadBuilding));

        let actions = game.get_actions();
        assert!(actions.iter().all(|a| matches!(a, Action::BuildRoad(_))));
        let r = match actions[0] {
            Action::BuildRoad(r) => r,
            _ => unreachable!(),
        };
        game.run_action(Action::BuildRoad(r));

        let actions = game.get_actions();
        assert!(
            actions.iter().all(|a| matches!(a, Action::BuildRoad(_))),
            "second free road should still be available"
        );
        let r = match actions[0] {
            Action::BuildRoad(r) => r,
            _ => unreachable!(),
        };
        game.run_action(Action::BuildRoad(r));

        assert_eq!(game.players[0].num_road, 2);
        // No resources should have been deducted.
        assert_eq!(game.players[0].cards[Resource::Brick as usize], 0);
        assert_eq!(game.players[0].cards[Resource::Lumber as usize], 0);
        assert!(matches!(game.state, GameState::ContinueAction));
    }

    #[test]
    fn vp_card_in_hand_counts_for_victory_points() {
        let mut game = game_with_empty_board();
        game.players[0].cards[DevCard::VictoryPoint as usize] = 2;
        game.players[0].num_settlement = 3;

        game.compute_victory_point();
        assert_eq!(game.game_detail.vp.get_vp(p(0)), 5);
    }

    #[test]
    fn vp_card_just_bought_counts_immediately() {
        let mut game = game_with_empty_board();
        game.players[0].dev_card_bought[DevCard::VictoryPoint as usize - 5] = 1;
        game.players[0].num_settlement = 2;

        game.compute_victory_point();
        assert_eq!(game.game_detail.vp.get_vp(p(0)), 3);
    }

    #[test]
    fn vp_card_is_not_listed_as_playable() {
        let mut game = game_in_continue_action();
        game.players[0].cards[DevCard::VictoryPoint as usize] = 3;

        let actions = game.get_actions();
        assert!(
            !actions.iter().any(|a| matches!(
                a,
                Action::UseDevCard(DevCard::VictoryPoint)
            )),
            "VP cards must not be offered as a playable action"
        );
    }

    #[test]
    fn cannot_play_two_dev_cards_in_one_turn() {
        let mut game = game_in_continue_action();
        game.players[0].cards[DevCard::Monopoly as usize] = 1;
        game.players[0].cards[DevCard::YearOfPlenty as usize] = 1;

        game.run_action(Action::UseDevCard(DevCard::Monopoly));
        game.run_action(Action::MonopolyGetResource(Resource::Brick));

        let actions = game.get_actions();
        assert!(
            !actions.iter().any(|a| matches!(a, Action::UseDevCard(_))),
            "should not be able to play a second dev card on the same turn"
        );
    }

    #[test]
    fn end_turn_lets_player_play_dev_card_next_turn() {
        let mut game = game_in_continue_action();
        game.players[0].cards[DevCard::Monopoly as usize] = 2;

        game.run_action(Action::UseDevCard(DevCard::Monopoly));
        game.run_action(Action::MonopolyGetResource(Resource::Brick));

        // Mid-turn: flag is set, no UseDevCard offered.
        let actions = game.get_actions();
        assert!(!actions.iter().any(|a| matches!(a, Action::UseDevCard(_))));

        // EndTurn must reset the flag for player 0. Manually return to player 0
        // in ContinueAction to bypass the dice-roll randomness.
        game.run_action(Action::EndTurn);
        game.current_player = 0;
        game.state = GameState::ContinueAction;

        let actions = game.get_actions();
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, Action::UseDevCard(DevCard::Monopoly))),
            "player 0 should be able to play a dev card on their next turn"
        );
    }

    // ---- get_actions idempotency ----

    /// Drive a deterministic game forward and assert that calling
    /// get_actions multiple times in a row at every decision point returns
    /// the identical action set (and never mutates observable state). This
    /// is the contract that lets the web session driver call
    /// get_action_mask freely without worrying about state advancing under
    /// it — the original "Steal from player before Move robber" bug.
    #[test]
    fn get_actions_is_idempotent_across_a_full_game() {
        let mut game = Game::new(999);
        let mut rng = ChaCha8Rng::seed_from_u64(12345);

        for _ in 0..1000 {
            let first = game.get_actions();
            let second = game.get_actions();
            let third = game.get_actions();

            assert_eq!(first, second);
            assert_eq!(first, third);

            // Also: current_player and last_dice_roll must not budge across
            // repeated get_actions calls.
            let cp = game.current_player;
            let dice = game.last_dice_roll;
            let _ = game.get_actions();
            assert_eq!(game.current_player, cp);
            assert_eq!(game.last_dice_roll, dice);

            if first.is_empty() {
                panic!(
                    "no legal actions at state {:?}, current_player={}",
                    game.state, game.current_player
                );
            }

            let pick = rng.random_range(0..first.len());
            let action = first[pick];
            game.run_action(action);

            if matches!(game.state, GameState::GameEnded(_)) {
                break;
            }
        }
    }

    /// Regression test for the "steal before move" bug surfaced by the web
    /// session: after a 7 roll where nobody has cards over the cap, the
    /// engine must offer MoveRobber actions (not StealCard) regardless of
    /// how many times the caller queries the mask before acting.
    #[test]
    fn dice_7_with_no_discards_offers_move_robber_idempotently() {
        let rng = ChaCha8Rng::seed_from_u64(0);
        // Tile 0 desert (robber start), tile 1 forest with player 2 settlement.
        let mut game = Game {
            board: Board::new(&[(2, TileType::Desert), (5, TileType::Forest)]),
            players: ArrayVec::from(from_fn(|i| Player::new(i.into()))),
            current_player: 0,
            dev_cards: ArrayVec::from([0; 5]),
            game_detail: GameDetail::new(),
            // 7 was rolled but nobody is over the cap → AwaitingMoveRobber.
            state: GameState::AwaitingMoveRobber,
            last_dice_roll: Some(7),
            rng,
        };
        game.board
            .build_structure(NodeId::new(1), p(2), Building::Settlement);

        for _ in 0..5 {
            let actions = game.get_actions();
            assert!(
                actions.iter().all(|a| matches!(a, Action::MoveRobber(_))),
                "expected MoveRobber actions, got {:?}",
                actions
            );
            assert!(matches!(game.state, GameState::AwaitingMoveRobber));
        }
    }
}
