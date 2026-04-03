use crate::{action::Action, game::Game};

pub struct Engine {
    games: Vec<Game>,
    next_seed: u64,
}

impl Engine {
    pub fn new(batch_size: usize, seed: u64) -> Self {
        Self {
            games: (0..batch_size)
                .map(|index| Game::new(seed + index as u64))
                .collect(),
            next_seed: seed + batch_size as u64,
        }
    }

    pub fn get_actions(&mut self) -> Vec<Vec<Action>> {
        self.games.iter_mut().map(|g| g.get_actions()).collect()
    }

    pub fn run_actions(&mut self, actions: Vec<Action>) {
        for (i, action) in actions.into_iter().enumerate() {
            self.games[i].run_action(action);
        }

        for i in 0..self.games.len() {
            if self.games[i].is_game_ended().is_some() {
                self.games[i] = Game::new(self.next_seed);
                self.next_seed += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn run_32_games_1000_steps() {
        let mut engine = Engine::new(32, /*seed=*/ 42);
        let mut rng = ChaCha8Rng::seed_from_u64(615);

        for _ in 0..1000 {
            let actions_per_game = engine.get_actions();
            let chosen: Vec<Action> = actions_per_game
                .into_iter()
                .map(|actions| {
                    let idx = rng.random_range(0..actions.len());
                    actions.into_iter().nth(idx).unwrap()
                })
                .collect();
            engine.run_actions(chosen);
        }
    }
}
