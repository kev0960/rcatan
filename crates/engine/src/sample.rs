use rand::prelude::*;

pub fn choose_weighted<R: Rng>(counts: &[u8], rng: &mut R) -> usize {
    let total = counts.iter().sum::<u8>();

    let mut r = rng.random_range(0..total);
    for (i, count) in counts.iter().enumerate() {
        if r < *count {
            return i;
        }

        r -= *count;
    }

    panic!("Weight is not correct?")
}
