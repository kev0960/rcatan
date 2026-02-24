mod action;
mod board;
mod id;
mod player;
mod sample;
mod table;

use std::array::from_fn;

use arrayvec::ArrayVec;
use rand::rngs::ThreadRng;

use crate::action::{Action, Resource};
use crate::board::Board;
use crate::id::PlayerId;
use crate::player::Player;
use rand::prelude::*;

struct Game {
    dev_cards: ArrayVec<u8, 5>,
    current_player: usize,
    board: Board,
    players: ArrayVec<Player, 4>,
    rng: ThreadRng,
}

impl Game {
    fn new() -> Self {
        Self {
            board: Board::new(&[]),
            players: ArrayVec::from(from_fn(|i| Player::new(i.into()))),
            rng: rand::rng(),
            current_player: 0,
            dev_cards: ArrayVec::from([0; 5]),
        }
    }

    fn handle_non_7_dice_roll(&mut self, dice: u8) {
        assert!(dice != 7);

        let resources = self.board.get_matching_resources(dice);

        for (player, resource, num) in resources {
            self.players[usize::from(player)].add_resource_card(resource, num);
        }
    }

    fn game_loop(&mut self) {
        let mut actions = self.players[self.current_player]
            .get_available_actions(&self.board);

        let action_index = self.rng.random_range(0..actions.len());
        let action = actions.swap_remove(action_index);

        self.run_action(action);
    }

    fn current_player_id(&self) -> PlayerId {
        PlayerId::from(self.current_player)
    }

    fn current_player_mut(&mut self) -> &mut Player {
        &mut self.players[self.current_player]
    }

    fn run_action(&mut self, action: Action) -> bool {
        match action {
            Action::BuildRoad(r) => {
                self.board.build_road(r, self.current_player_id());

                let current_player = self.current_player_mut();
                current_player.num_road += 1;

                current_player.cards[Resource::Brick as usize] -= 1;
                current_player.cards[Resource::Lumber as usize] -= 1;

                false
            }
            Action::BuildStructure(building, node_id) => {
                let current_player = self.current_player_mut();
                match building {
                    action::Building::Settlement => {
                        current_player.num_settlement += 1;

                        current_player.cards[Resource::Brick as usize] -= 1;
                        current_player.cards[Resource::Lumber as usize] -= 1;
                        current_player.cards[Resource::Grain as usize] -= 1;
                        current_player.cards[Resource::Wool as usize] -= 1;
                    }
                    action::Building::City => {
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

                false
            }
            Action::MaritimeTrade(lose, get) => {
                let current_player = self.current_player_mut();
                current_player.cards[lose.0 as usize] -= lose.1;
                current_player.cards[get as usize] += 1;

                false
            }
            Action::EndTurn => true,
            /*
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

fn main() {
    let mut g = Game::new();
    g.handle_non_7_dice_roll(3);
    g.game_loop();

    println!("Hello, world!");
}
