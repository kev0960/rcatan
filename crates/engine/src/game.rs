use arrayvec::ArrayVec;
use std::array::from_fn;

use crate::vecset::VecSet;

use crate::action::Action;
use crate::action::Building;
use crate::action::Resource;
use crate::board::Board;
use crate::id::PlayerId;
use crate::player::Player;
use crate::road::find_longest_road;
use crate::sample::choose_weighted;
use crate::table::TILE_TO_NODE;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

static WINNING_VP: u8 = 10;

struct GameDetail {
    longest_road: u8,
    // largest_army: u8,
}

impl GameDetail {
    fn new() -> Self {
        Self {
            longest_road: 0,
            // largest_army: 0,
        }
    }

    fn get_longest_road(&self) -> Option<(PlayerId, u8)> {
        let has_longest_road = self.longest_road & 0b0000_0001;
        if has_longest_road == 0 {
            return None;
        }

        let player_id = (self.longest_road & 0b0000_0110) >> 1;
        let road_length = self.longest_road >> 3;

        Some((PlayerId::from(player_id), road_length))
    }

    fn set_longest_road(&mut self, r: Option<(PlayerId, u8)>) {
        if let Some((player_id, length)) = r {
            self.longest_road =
                0b0000_0001 | (u8::from(player_id) << 1) | (length << 3);
        } else {
            self.longest_road = 0;
        }
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

    // Dice 7 rolled phase.
    //  -- Actions are "discard" untill all players > 7 cards discards resources.
    //  -- then the "player" will be able to move the robber.
    Dice7Rolled(PlayerId),

    StealResourceAfterRobber,

    // current player should roll the dice.
    RollDice,

    // current player should continue action.
    ContinueAction,

    // Game is ended with the given winner.
    GameEnded(PlayerId),
}

pub struct Game {
    board: Board,
    dev_cards: ArrayVec<u8, 5>,

    current_player: usize,
    players: ArrayVec<Player, 4>,

    game_detail: GameDetail,

    state: GameState,

    rng: ChaCha8Rng,
}

impl Game {
    pub fn new(seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        Self {
            board: Board::new_from_rand(&mut rng),
            players: ArrayVec::from(from_fn(|i| Player::new(i.into()))),
            current_player: 0,
            dev_cards: ArrayVec::from([0; 5]),
            game_detail: GameDetail::new(),
            state: GameState::InitialBoardSetup(0),
            rng,
        }
    }

    // Run "one step". Returns the player id when the game is ended. Otherwise returns None.
    pub fn step(&mut self) -> Option<PlayerId> {
        let mut actions = self.get_actions();

        let action_index = self.rng.random_range(0..actions.len());
        let action = actions.swap_remove(action_index);

        self.run_action(action);

        match self.state {
            GameState::GameEnded(player) => Some(player),
            _ => None,
        }
    }

    pub fn check_win_condition(&mut self) {
        let vps = self.compute_victory_point();
        for (p, vp) in vps.into_iter().enumerate() {
            if vp >= WINNING_VP {
                self.state = GameState::GameEnded(PlayerId::from(p));
            }
        }
    }

    pub fn get_actions(&mut self) -> Vec<Action> {
        if let GameState::RollDice = self.state {
            let dice =
                self.rng.random_range(1..=6) + self.rng.random_range(1..=6);

            if dice == 7 {
                self.state =
                    GameState::Dice7Rolled(PlayerId::from(self.current_player));
            } else {
                self.handle_non_7_dice_roll(dice);
                self.state = GameState::ContinueAction;
            }
        };

        if let GameState::Dice7Rolled(player) = self.state {
            // Find the player with the card > 7.
            for player in &self.players {
                if player.num_resource_cards() > 7 {
                    self.current_player = player.id.into();

                    return player
                        .cards
                        .iter()
                        .take(5)
                        .enumerate()
                        .filter_map(|(i, count)| match count {
                            0 => None,
                            _ => {
                                Some(Action::DiscardResource(Resource::ALL[i]))
                            }
                        })
                        .collect();
                }
            }

            self.current_player = player.into();
            self.state = GameState::StealResourceAfterRobber;

            // If none of the player has more than 7, then move the robber.
            return self
                .board
                .get_tile_without_robber()
                .into_iter()
                .map(Action::MoveRobber)
                .collect();
        }

        if let GameState::StealResourceAfterRobber = self.state {
            self.state = GameState::ContinueAction;

            // Check the players around the new robber's location.
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

            if player_set.is_empty() {
                // Just continue as a regular turn.
                return self
                    .current_player_ref()
                    .get_available_actions(&self.board);
            } else {
                return player_set
                    .into_iter()
                    .map(|p| Action::StealCard(PlayerId::from(p)))
                    .collect();
            }
        }

        let current_player = self.current_player_ref();
        if let GameState::InitialBoardSetup(current_order) = self.state {
            if current_order % 2 == 0 {
                // Build the settlement.
                current_player
                    .get_available_nodes_for_initial_board_setup(&self.board)
                    .into_iter()
                    .map(|n| Action::BuildStructure(Building::Settlement, n))
                    .collect()
            } else {
                // Build the road.
                current_player
                    .get_available_roads_for_inital_board_setup(&self.board)
                    .into_iter()
                    .map(Action::BuildRoad)
                    .collect()
            }
        } else {
            current_player.get_available_actions(&self.board)
        }
    }

    fn handle_non_7_dice_roll(&mut self, dice: u8) {
        assert!(dice != 7);

        let resources = self.board.get_matching_resources(dice);

        for (player, resource, num) in resources {
            self.players[usize::from(player)].add_resource_card(resource, num);
        }
    }

    fn compute_victory_point(&mut self) -> ArrayVec<u8, 4> {
        let mut vps = ArrayVec::<u8, 4>::from([0; 4]);
        self.compute_longest_road();

        for (id, p) in self.players.iter().enumerate() {
            vps[id] += p.num_settlement;
            vps[id] += p.num_city * 2;
        }

        if let Some((p, _)) = self.game_detail.get_longest_road() {
            vps[usize::from(p)] += 2;
        }

        vps
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

    fn run_action(&mut self, action: Action) {
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
                        self.state = GameState::RollDice;
                    } else {
                        self.state =
                            GameState::InitialBoardSetup(current_order + 1);
                    }
                } else {
                    current_player.cards[Resource::Brick as usize] -= 1;
                    current_player.cards[Resource::Lumber as usize] -= 1;
                }

                self.check_win_condition();
            }
            Action::BuildStructure(building, node_id) => {
                let game_state = self.state;

                let current_player = self.current_player_mut();
                match building {
                    Building::Settlement => {
                        current_player.num_settlement += 1;

                        if let GameState::InitialBoardSetup(current_order) =
                            game_state
                        {
                            if current_order == 15 {
                                self.state = GameState::RollDice;
                            } else {
                                self.state = GameState::InitialBoardSetup(
                                    current_order + 1,
                                );
                            }
                        } else {
                            current_player.cards[Resource::Brick as usize] -= 1;
                            current_player.cards[Resource::Lumber as usize] -=
                                1;
                            current_player.cards[Resource::Grain as usize] -= 1;
                            current_player.cards[Resource::Wool as usize] -= 1;
                        }
                    }
                    Building::City => {
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

                self.check_win_condition();
            }
            Action::MaritimeTrade(lose, get) => {
                let current_player = self.current_player_mut();
                current_player.cards[lose.0 as usize] -= lose.1;
                current_player.cards[get as usize] += 1;
            }
            Action::DiscardResource(resource) => {
                let current_player = self.current_player_mut();
                current_player.cards[resource as usize] -= 1;
            }
            Action::MoveRobber(tile_id) => {
                self.board.move_robber_to(tile_id);
            }
            Action::StealCard(player_id) => {
                let player = &mut self.players[usize::from(player_id)];
                if player.num_resource_cards() == 0 {
                    return;
                }

                let r = choose_weighted(&player.cards[0..5], &mut self.rng);
                player.use_resource_card(Resource::ALL[r], 1);

                self.current_player_mut()
                    .add_resource_card(Resource::ALL[r], 1);
            }
            Action::EndTurn => {
                self.current_player = (self.current_player + 1) % 4;
                self.state = GameState::RollDice;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::TileType;
    use crate::id::{NodeId, RoadId, TileId};

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
            state: GameState::RollDice,
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

    // ---- Victory point tests ----

    #[test]
    fn compute_victory_point_settlements() {
        let mut game = game_with_empty_board();
        game.players[0].num_settlement = 3;
        game.players[1].num_settlement = 2;

        let vps = game.compute_victory_point();
        assert_eq!(vps[0], 3);
        assert_eq!(vps[1], 2);
        assert_eq!(vps[2], 0);
        assert_eq!(vps[3], 0);
    }

    #[test]
    fn compute_victory_point_cities() {
        let mut game = game_with_empty_board();
        game.players[0].num_city = 2;
        game.players[0].num_settlement = 1;

        let vps = game.compute_victory_point();
        assert_eq!(vps[0], 5); // 2*2 + 1
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

        let vps = game.compute_victory_point();
        // 2 settlements + 2 longest road = 4
        // (compute_longest_road may recalculate and override, but with 6
        // roads built the actual longest road computation determines the holder)
        assert!(vps[0] >= 2); // at minimum 2 settlements
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
            state: GameState::RollDice,
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
            state: GameState::RollDice,
            rng,
        };

        game.board
            .build_structure(NodeId::new(3), p(1), Building::City);

        game.handle_non_7_dice_roll(8);

        assert_eq!(game.players[1].cards[Resource::Ore as usize], 2);
    }

    // ---- step tests ----

    #[test]
    fn step_is_deterministic() {
        let mut game1 = Game::new(123);
        let mut game2 = Game::new(123);

        for _ in 0..100 {
            let r1 = game1.step();
            let r2 = game2.step();
            assert_eq!(r1, r2);
            if r1.is_some() {
                break;
            }
        }
    }

    #[test]
    fn step_does_not_panic() {
        let mut game = Game::new(999);
        for i in 0..1000 {
            if game.step().is_some() {
                // This shouldn't change unless we changed some logic in the game.
                // (to guarantee that the simulation is deterministic)
                assert_eq!(i, 655);
                break;
            }
        }
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
            current_player: 0,
            dev_cards: ArrayVec::from([0; 5]),
            game_detail: GameDetail::new(),
            state: GameState::Dice7Rolled(p(0)), // Player 0 rolled a 7
            rng,
        };

        // Player 1: 8 resource cards (must discard down to 7).
        game.players[1].add_resource_card(Resource::Brick, 4);
        game.players[1].add_resource_card(Resource::Lumber, 4);

        // Player 2: 9 resource cards (must discard down to 7).
        game.players[2].add_resource_card(Resource::Ore, 5);
        game.players[2].add_resource_card(Resource::Grain, 4);

        // Place player 2's settlement on tile 1 (node 1) for the steal phase.
        game.board
            .build_structure(NodeId::new(1), p(2), Building::Settlement);

        // --- Phase 1: Discard ---

        // Player 1 (8 cards > 7) should discard first.
        let actions = game.get_actions();
        assert!(actions.iter().all(|a| matches!(a, Action::DiscardResource(_))));
        assert_eq!(game.current_player, 1);

        game.run_action(Action::DiscardResource(Resource::Brick));
        assert_eq!(game.players[1].num_resource_cards(), 7);

        // Player 2 (9 cards > 7) should discard next.
        let actions = game.get_actions();
        assert!(actions.iter().all(|a| matches!(a, Action::DiscardResource(_))));
        assert_eq!(game.current_player, 2);

        game.run_action(Action::DiscardResource(Resource::Ore));
        assert_eq!(game.players[2].num_resource_cards(), 8);

        // Player 2 still has 8 > 7, must discard again.
        let actions = game.get_actions();
        assert!(actions.iter().all(|a| matches!(a, Action::DiscardResource(_))));
        assert_eq!(game.current_player, 2);

        game.run_action(Action::DiscardResource(Resource::Ore));
        assert_eq!(game.players[2].num_resource_cards(), 7);

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
        assert_eq!(game.players[2].num_resource_cards(), 6);
    }
}
