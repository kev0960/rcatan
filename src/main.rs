use crate::game::Game;

mod action;
mod board;
mod game;
mod id;
mod player;
mod road;
mod sample;
mod table;

fn main() {
    let mut g = Game::new(1234);
    g.step();

    println!("Hello, world!");
}
