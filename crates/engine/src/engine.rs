use crate::game::Game;

struct Engine {
    games: Vec<Game>,
}

impl Engine {
    fn new(batch_size: usize, seed: u64) -> Self {
        Self {
            games: (0..batch_size)
                .into_iter()
                .map(|index| Game::new(seed))
                .collect(),
        }
    }
}
