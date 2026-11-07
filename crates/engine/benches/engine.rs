//! Engine benchmarks (Criterion). From the repo root:
//!
//!     cargo bench -p rcatan
//!     cargo bench -p rcatan -- --save-baseline before   # then, after a change:
//!     cargo bench -p rcatan -- --baseline before        # compare against it
//!
//! Reports land in target/criterion/report/index.html.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rcatan::Engine;

/// Games played side by side, like a (small) training batch.
const BATCH: usize = 4;

/// Plays `steps` random legal moves in `BATCH` fresh games. Same seeds every
/// time, so every run does exactly the same work.
fn play(steps: usize) -> Engine {
    let mut engine = Engine::new(BATCH, 0);
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    for _ in 0..steps {
        let actions = engine
            .get_actions()
            .into_iter()
            .map(|legal| *legal.choose(&mut rng).expect("a legal action"))
            .collect();
        black_box(engine.run_actions(actions));
    }
    engine
}

/// Whole engine (rules, action lists, card estimates), from the first move
/// to `steps` moves in. If the cost per step is constant, time grows linearly
/// with `steps`; faster than linear means later steps are more expensive.
fn random_play(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_play");
    // Long games can take a while per iteration.
    group.sample_size(10);
    for steps in [100, 300, 1000] {
        group.bench_with_input(BenchmarkId::from_parameter(steps), &steps, |b, &steps| {
            b.iter(|| play(steps))
        });
    }
    group.finish();
}

criterion_group!(benches, random_play);
criterion_main!(benches);
