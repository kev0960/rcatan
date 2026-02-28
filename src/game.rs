use std::array::from_fn;

use arrayvec::ArrayVec;

use crate::action::Action;
use crate::action::Building;
use crate::action::Resource;
use crate::board::Board;
use crate::id::PlayerId;
use crate::player::Player;
use crate::road::find_longest_road;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

static WINNING_VP: u8 = 10;

struct GameDetail {
    longest_road: u8,
    largest_army: u8,
}

impl GameDetail {
    fn new() -> Self {
        Self {
            longest_road: 0,
            largest_army: 0,
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

    fn get_largest_army() -> Option<(PlayerId, u8)> {
        None
    }
}

enum GameState {
    // current player should roll the dice.
    RollDice,
    // current player should continue action.
    ContinueAction,
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
            state: GameState::RollDice,
            rng,
        }
    }

    // Run "one step". Returns the player id when the game is ended. Otherwise returns None.
    pub fn step(&mut self) -> Option<PlayerId> {
        match self.state {
            GameState::RollDice => {
                let dice =
                    self.rng.random_range(1..=6) + self.rng.random_range(1..=6);

                if dice == 7 {
                    // No-op for now.
                    self.current_player = (self.current_player + 1) % 4;
                    return None;
                }

                self.handle_non_7_dice_roll(dice);
                self.state = GameState::ContinueAction;
            }
            GameState::ContinueAction => {}
        };

        let mut actions = self.players[self.current_player]
            .get_available_actions(&self.board);

        let action_index = self.rng.random_range(0..actions.len());
        let action = actions.swap_remove(action_index);

        match action {
            Action::EndTurn => {
                self.current_player = (self.current_player + 1) % 4;
                self.state = GameState::RollDice;
                return None;
            }
            Action::BuildStructure(_, _) | Action::BuildRoad(_) => {
                self.run_action(action);

                // Check the win condition.
                let vps = self.compute_victory_point();
                for (p, vp) in vps.into_iter().enumerate() {
                    if vp >= WINNING_VP {
                        return Some(PlayerId::from(p));
                    }
                }
            }
            action => self.run_action(action),
        }

        None
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
        PlayerId::from(self.current_player)
    }

    fn current_player_mut(&mut self) -> &mut Player {
        &mut self.players[self.current_player]
    }

    fn run_action(&mut self, action: Action) {
        match action {
            Action::BuildRoad(r) => {
                self.board.build_road(r, self.current_player_id());

                let current_player = self.current_player_mut();
                current_player.num_road += 1;

                current_player.cards[Resource::Brick as usize] -= 1;
                current_player.cards[Resource::Lumber as usize] -= 1;
            }
            Action::BuildStructure(building, node_id) => {
                let current_player = self.current_player_mut();
                match building {
                    Building::Settlement => {
                        current_player.num_settlement += 1;

                        current_player.cards[Resource::Brick as usize] -= 1;
                        current_player.cards[Resource::Lumber as usize] -= 1;
                        current_player.cards[Resource::Grain as usize] -= 1;
                        current_player.cards[Resource::Wool as usize] -= 1;
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
            }
            Action::MaritimeTrade(lose, get) => {
                let current_player = self.current_player_mut();
                current_player.cards[lose.0 as usize] -= lose.1;
                current_player.cards[get as usize] += 1;
            }
            Action::EndTurn => {} /*
                                  Action::BuyDevCard => {
                                      let dev_card_index =
                                          choose_weighted(&self.dev_cards, &mut self.rng);
                                      self.dev_cards[dev_card_index] -= 1;

                                      false
                                  }
                                  Action::UseDevCard(dev_card) => false,
                                  */
        }
    }
}
