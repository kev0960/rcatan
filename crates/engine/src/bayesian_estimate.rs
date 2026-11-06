use ahash::AHashMap;

use crate::Resource;

pub struct BayesianCardEstimate {
    cards: Vec<(ResourceCards, f32)>,
}

impl BayesianCardEstimate {
    pub fn new() -> Self {
        // Start from a known empty hand: every opponent begins with no cards.
        Self {
            cards: vec![(ResourceCards::new(&[0; 5]), 1.0)],
        }
    }

    pub fn observe_gain_card(&mut self, cards: &[u8]) {
        for (card, _) in &mut self.cards {
            *card = card.add_cards(&cards);
        }
    }

    pub fn observe_discard(&mut self, cards: &[u8]) {
        self.cards.retain_mut(|(card, _)| {
            let new_card = card.remove_cards(&cards);
            if let Some(new_card) = new_card {
                *card = new_card;
                true
            } else {
                false
            }
        });

        if self.cards.is_empty() {
            return;
        }

        // Normalize probabilities
        normalize_probabilities(&mut self.cards);
    }

    pub fn observe_discard_unknown(&mut self) {
        let mut cards = AHashMap::new();

        for (card, probability) in &self.cards {
            // A stolen card is drawn uniformly from the hand, so each resource
            // is taken with probability count / total.
            let total = card.total() as f32;
            for resource in card.not_zero_resources() {
                let weight = probability * card.count(resource) as f32 / total;

                cards
                    .entry(card.remove_resource(resource))
                    .and_modify(|p| *p += weight)
                    .or_insert(weight);
            }
        }

        let mut cards = cards.into_iter().collect::<Vec<_>>();
        normalize_probabilities(&mut cards);

        self.cards = cards;
    }

    /// Someone played monopoly on `resource`: this player loses every card of
    /// that type and is left with `remaining_cards` cards in total (hand size
    /// is public). Hands that would not end at that size are ruled out.
    pub fn observe_monopoly(&mut self, resource: Resource, remaining_cards: u32) {
        let mut cards = AHashMap::new();

        for (card, probability) in &self.cards {
            let card = card.remove_all(resource);
            if card.total() != remaining_cards {
                continue;
            }

            // Hands that differ only in `resource` collapse into one.
            cards
                .entry(card)
                .and_modify(|p| *p += probability)
                .or_insert(*probability);
        }

        let mut cards = cards.into_iter().collect::<Vec<_>>();
        if !cards.is_empty() {
            normalize_probabilities(&mut cards);
        }

        self.cards = cards;
    }

    pub fn observe_gain_card_unknown(&mut self) {
        let mut cards = AHashMap::new();

        for (card, probability) in &self.cards {
            for resource in Resource::ALL {
                cards
                    .entry(card.add_resource(resource))
                    .and_modify(|p| {
                        *p += (probability / Resource::ALL.len() as f32)
                    })
                    .or_insert((*probability / Resource::ALL.len() as f32));
            }
        }

        let mut cards = cards.into_iter().collect::<Vec<_>>();
        normalize_probabilities(&mut cards);

        self.cards = cards;
    }
}

#[derive(Eq, Hash, PartialEq)]
pub struct ResourceCards {
    cards: u64,
}

impl ResourceCards {
    pub fn new(cards: &[u8]) -> Self {
        Self {
            cards: u64::from_le_bytes([
                cards[0], cards[1], cards[2], cards[3], cards[4], 0, 0, 0,
            ]),
        }
    }

    pub fn count(&self, resource: Resource) -> u8 {
        self.cards.to_le_bytes()[resource as usize]
    }

    pub fn total(&self) -> u32 {
        self.cards.to_le_bytes().iter().map(|&b| b as u32).sum()
    }

    pub fn not_zero_resources(&self) -> Vec<Resource> {
        self.cards
            .to_le_bytes()
            .iter()
            .enumerate()
            .filter_map(
                |(i, &b)| if b != 0 { Some(Resource::ALL[i]) } else { None },
            )
            .collect()
    }

    pub fn add_cards(&self, cards: &[u8]) -> Self {
        let mut bytes = self.cards.to_le_bytes();

        for i in 0..5 {
            bytes[i] += cards[i];
        }

        ResourceCards::new(&bytes)
    }

    pub fn add_resource(&self, resource: Resource) -> Self {
        let mut bytes = self.cards.to_le_bytes();
        bytes[resource as usize] += 1;

        ResourceCards::new(&bytes)
    }

    pub fn remove_cards(&self, cards: &[u8]) -> Option<Self> {
        let mut bytes = self.cards.to_le_bytes();

        for i in 0..5 {
            if bytes[i] < cards[i] {
                return None;
            }

            bytes[i] -= cards[i];
        }

        Some(ResourceCards::new(&bytes))
    }

    pub fn remove_resource(&self, resource: Resource) -> Self {
        let mut bytes = self.cards.to_le_bytes();

        if bytes[resource as usize] == 0 {
            panic!("Removing resource that is not present");
        }

        bytes[resource as usize] -= 1;
        ResourceCards::new(&bytes)
    }

    pub fn remove_all(&self, resource: Resource) -> Self {
        let mut bytes = self.cards.to_le_bytes();
        bytes[resource as usize] = 0;

        ResourceCards::new(&bytes)
    }

    pub fn is_empty(&self) -> bool {
        self.cards == 0
    }
}

fn normalize_probabilities(cards: &mut Vec<(ResourceCards, f32)>) {
    let total_probability = cards
        .iter()
        .map(|(_, probability)| *probability)
        .sum::<f32>();

    for (_, probability) in cards {
        *probability /= total_probability;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-6;

    fn prob(estimate: &BayesianCardEstimate, hand: [u8; 5]) -> f32 {
        let target = ResourceCards::new(&hand);
        estimate
            .cards
            .iter()
            .filter(|(card, _)| *card == target)
            .map(|(_, p)| *p)
            .sum()
    }

    fn total_prob(estimate: &BayesianCardEstimate) -> f32 {
        estimate.cards.iter().map(|(_, p)| *p).sum()
    }

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < EPS,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn starts_with_certain_empty_hand() {
        let estimate = BayesianCardEstimate::new();
        assert_eq!(estimate.cards.len(), 1);
        assert_close(prob(&estimate, [0; 5]), 1.0);
    }

    #[test]
    fn known_gain_and_discard_are_exact() {
        let mut estimate = BayesianCardEstimate::new();
        estimate.observe_gain_card(&[2, 1, 0, 0, 3]);
        estimate.observe_discard(&[1, 0, 0, 0, 2]);

        assert_eq!(estimate.cards.len(), 1);
        assert_close(prob(&estimate, [1, 1, 0, 0, 1]), 1.0);
    }

    #[test]
    fn unknown_gain_on_empty_hand_is_uniform() {
        let mut estimate = BayesianCardEstimate::new();
        estimate.observe_gain_card_unknown();

        assert_eq!(estimate.cards.len(), 5);
        for i in 0..5 {
            let mut hand = [0; 5];
            hand[i] = 1;
            assert_close(prob(&estimate, hand), 0.2);
        }
    }

    #[test]
    fn unknown_gain_merges_identical_hands() {
        let mut estimate = BayesianCardEstimate::new();
        estimate.observe_gain_card_unknown();
        estimate.observe_gain_card_unknown();

        // 5 doubles + C(5, 2) = 10 mixed pairs.
        assert_eq!(estimate.cards.len(), 15);
        assert_close(prob(&estimate, [2, 0, 0, 0, 0]), 1.0 / 25.0);
        assert_close(prob(&estimate, [1, 1, 0, 0, 0]), 2.0 / 25.0);
        assert_close(total_prob(&estimate), 1.0);
    }

    #[test]
    fn unknown_discard_is_weighted_by_card_count() {
        let mut estimate = BayesianCardEstimate::new();
        estimate.observe_gain_card(&[3, 0, 1, 0, 0]);
        estimate.observe_discard_unknown();

        // Brick is stolen 3 times out of 4, ore 1 time out of 4.
        assert_close(prob(&estimate, [2, 0, 1, 0, 0]), 0.75);
        assert_close(prob(&estimate, [3, 0, 0, 0, 0]), 0.25);
    }

    #[test]
    fn unknown_discard_from_single_resource_hand_is_certain() {
        let mut estimate = BayesianCardEstimate::new();
        estimate.observe_gain_card(&[0, 0, 0, 2, 0]);
        estimate.observe_discard_unknown();

        assert_eq!(estimate.cards.len(), 1);
        assert_close(prob(&estimate, [0, 0, 0, 1, 0]), 1.0);
    }

    #[test]
    fn known_discard_rules_out_impossible_hands() {
        let mut estimate = BayesianCardEstimate::new();
        estimate.observe_gain_card_unknown();
        // Only the hand holding a brick can pay this.
        estimate.observe_discard(&[1, 0, 0, 0, 0]);

        assert_eq!(estimate.cards.len(), 1);
        assert_close(prob(&estimate, [0; 5]), 1.0);
    }

    #[test]
    fn known_discard_renormalizes_remaining_hands() {
        let mut estimate = BayesianCardEstimate::new();
        estimate.observe_gain_card(&[1, 0, 0, 0, 0]);
        estimate.observe_gain_card_unknown();
        // Hands: {2B}, {B,L}, {B,O}, {B,G}, {B,W} each 0.2.
        // Paying a lumber leaves only {B}.
        estimate.observe_discard(&[0, 1, 0, 0, 0]);
        assert_close(prob(&estimate, [1, 0, 0, 0, 0]), 1.0);

        let mut estimate = BayesianCardEstimate::new();
        estimate.observe_gain_card_unknown();
        estimate.observe_gain_card_unknown();
        // Paying a brick keeps the 9 hands with a brick, weighted 1:2:2:2:2.
        estimate.observe_discard(&[1, 0, 0, 0, 0]);
        assert_close(total_prob(&estimate), 1.0);
        assert_close(prob(&estimate, [1, 0, 0, 0, 0]), 1.0 / 9.0);
        assert_close(prob(&estimate, [0, 1, 0, 0, 0]), 2.0 / 9.0);
    }

    #[test]
    fn steal_after_unknown_gain() {
        let mut estimate = BayesianCardEstimate::new();
        estimate.observe_gain_card(&[1, 0, 0, 0, 0]);
        estimate.observe_gain_card_unknown();
        estimate.observe_discard_unknown();

        // Each of the 5 hands has 0.2. {2B} always leaves {B};
        // {B, X} leaves {B} or {X} with 0.5 each.
        // P({B}) = 0.2 + 4 * 0.1 = 0.6; P({X}) = 0.1 for each other X.
        assert_close(prob(&estimate, [1, 0, 0, 0, 0]), 0.6);
        assert_close(prob(&estimate, [0, 1, 0, 0, 0]), 0.1);
        assert_close(prob(&estimate, [0, 0, 0, 0, 1]), 0.1);
        assert_close(total_prob(&estimate), 1.0);
    }

    #[test]
    fn impossible_discard_empties_estimate() {
        let mut estimate = BayesianCardEstimate::new();
        estimate.observe_gain_card(&[1, 0, 0, 0, 0]);
        estimate.observe_discard(&[0, 0, 1, 0, 0]);

        assert!(estimate.cards.is_empty());
    }

    #[test]
    fn monopoly_on_known_hand_removes_all_of_resource() {
        let mut estimate = BayesianCardEstimate::new();
        estimate.observe_gain_card(&[2, 1, 3, 0, 0]);
        estimate.observe_monopoly(Resource::Ore, 3);

        assert_eq!(estimate.cards.len(), 1);
        assert_close(prob(&estimate, [2, 1, 0, 0, 0]), 1.0);
    }

    #[test]
    fn monopoly_rules_out_hands_with_wrong_remaining_count() {
        let mut estimate = BayesianCardEstimate::new();
        estimate.observe_gain_card(&[1, 0, 0, 0, 0]);
        estimate.observe_gain_card_unknown();
        // Hands: {2B}, {B,L}, {B,O}, {B,G}, {B,W} each 0.2.
        // Monopoly on wool leaving 2 cards: only {B,W} loses a card, so it
        // is ruled out and the other four stay with 0.25 each.
        estimate.observe_monopoly(Resource::Wool, 2);

        assert_eq!(estimate.cards.len(), 4);
        assert_close(prob(&estimate, [2, 0, 0, 0, 0]), 0.25);
        assert_close(prob(&estimate, [1, 1, 0, 0, 0]), 0.25);
        assert_close(prob(&estimate, [1, 0, 0, 0, 1]), 0.0);
    }

    #[test]
    fn monopoly_reveals_hidden_card() {
        let mut estimate = BayesianCardEstimate::new();
        estimate.observe_gain_card(&[1, 0, 0, 0, 0]);
        estimate.observe_gain_card_unknown();
        // Monopoly on wool leaving 1 card: the hidden card must have been wool.
        estimate.observe_monopoly(Resource::Wool, 1);

        assert_eq!(estimate.cards.len(), 1);
        assert_close(prob(&estimate, [1, 0, 0, 0, 0]), 1.0);
    }

    #[test]
    fn monopoly_merges_hands_that_differ_only_in_resource() {
        let mut estimate = BayesianCardEstimate::new();
        estimate.observe_gain_card_unknown();
        estimate.observe_gain_card_unknown();
        // Monopoly on brick leaving 1 card: {B,X} for X != B (each 2/25)
        // all become {X}; {2B} and the 2-card non-brick hands are ruled out.
        estimate.observe_monopoly(Resource::Brick, 1);

        assert_eq!(estimate.cards.len(), 4);
        for i in 1..5 {
            let mut hand = [0; 5];
            hand[i] = 1;
            assert_close(prob(&estimate, hand), 0.25);
        }
        assert_close(total_prob(&estimate), 1.0);
    }

    #[test]
    fn monopoly_merges_different_counts_of_taken_resource() {
        let mut estimate = BayesianCardEstimate::new();
        estimate.observe_gain_card(&[0, 1, 0, 0, 0]);
        estimate.observe_gain_card_unknown();
        estimate.observe_gain_card_unknown();
        // Lumber is known; whatever else they held was brick or not.
        // Monopoly on brick leaving 1 card means both hidden cards were
        // brick, so {1L} is the only result.
        estimate.observe_monopoly(Resource::Brick, 1);

        assert_eq!(estimate.cards.len(), 1);
        assert_close(prob(&estimate, [0, 1, 0, 0, 0]), 1.0);
    }

    #[test]
    fn impossible_monopoly_empties_estimate() {
        let mut estimate = BayesianCardEstimate::new();
        estimate.observe_gain_card(&[1, 1, 0, 0, 0]);
        estimate.observe_monopoly(Resource::Ore, 1);

        assert!(estimate.cards.is_empty());
    }
}
