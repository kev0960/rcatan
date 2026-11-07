//! Plays random legal moves in a batch of games, with no Python and no
//! network, and prints how long a step takes as the games go on. For timing
//! and profiling the engine on its own:
//!
//!     cargo run --release -p rcatan --example random_play -- 32 1500
//!     samply record target/release/examples/random_play 32 1500
//!
//! Arguments: batch size (default 32) and number of steps (default 1500).
//! Finished games restart automatically, like in training.

use std::time::Instant;

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rcatan::Engine;

const REPORT_EVERY: usize = 250;

fn main() {
    let mut args = std::env::args().skip(1);
    let batch: usize = args.next().map_or(32, |a| a.parse().expect("batch size"));
    let steps: usize = args.next().map_or(1500, |a| a.parse().expect("steps"));

    let mut engine = Engine::new(batch, 0);
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let mut games_finished = 0;
    let start = Instant::now();
    let mut chunk_start = Instant::now();

    for step in 1..=steps {
        let actions = engine
            .get_actions()
            .into_iter()
            .map(|legal| *legal.choose(&mut rng).expect("a legal action"))
            .collect();
        let (_, ended) = engine.run_actions(actions);
        games_finished += ended.iter().filter(|e| e.is_some()).count();

        if step % REPORT_EVERY == 0 {
            let per_env = chunk_start.elapsed().as_secs_f64() / (REPORT_EVERY * batch) as f64;
            println!(
                "steps {:>5}-{:<5} {:>8.1} us per game-step  ({games_finished} games finished)",
                step - REPORT_EVERY + 1,
                step,
                per_env * 1e6,
            );
            chunk_start = Instant::now();
        }
    }
    println!(
        "total {:.2}s for {} game-steps",
        start.elapsed().as_secs_f64(),
        steps * batch
    );
}
