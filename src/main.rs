mod action;
mod board;
mod id;
mod player;
mod table;

use std::array::from_fn;

use arrayvec::ArrayVec;
use rand::rngs::ThreadRng;

use crate::action::Action;
use crate::board::Board;
use crate::id::PlayerId;
use crate::player::Player;
use rand::prelude::*;

struct Game {
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
        let actions = self.players[self.current_player]
            .get_available_actions(&self.board);

        let action = actions.choose(&mut self.rng);
        self.run_action(action);
    }

    fn current_player(&self) -> PlayerId {
        PlayerId::from(self.current_player)
    }

    fn run_action(&mut self, action: Action) {
        match action {
            Action::BuildRoad(r) => {
                self.board.build_road(r, self.current_player());
            }
            Action::BuildStructure(building, id) => todo!(),
            Action::BuyDevCard => todo!(),
            Action::UseDevCard(dev_card) => todo!(),
            Action::MaritimeTrade(resources, resource) => todo!(),
        }
    }
}

fn main() {
    let mut g = Game::new();
    g.handle_non_7_dice_roll(3);
    g.game_loop();

    println!("Hello, world!");
}
