use ahash::AHashMap;
use arrayvec::ArrayVec;
use std::array::from_fn;

use crate::{PlayerId, RelativePlayerId, Resource};

pub struct OtherPlayerCardEstimates {
    enable_estimation: bool,
    pub estimates: ArrayVec<ArrayVec<BayesianCardEstimate, 3>, 4>,
}

impl Default for OtherPlayerCardEstimates {
    fn default() -> Self {
        Self::new(true)
    }
}

impl OtherPlayerCardEstimates {
    pub fn new(enable_estimation: bool) -> Self {
        Self {
            enable_estimation,
            estimates: ArrayVec::from(from_fn(|_| {
                ArrayVec::from([
                    BayesianCardEstimate::new(),
                    BayesianCardEstimate::new(),
                    BayesianCardEstimate::new(),
                ])
            })),
        }
    }

    fn apply_actions<Action: Fn(&mut BayesianCardEstimate) -> ()>(
        &mut self,
        player_id: PlayerId,
        action: Action,
    ) {
        for i in 0..4 {
            let target_id = PlayerId::from(i);
            if target_id == player_id {
                continue;
            }

            let relative = RelativePlayerId::from(target_id, player_id);
            action(&mut self.estimates[i][usize::from(relative) - 1]);
        }
    }

    pub fn observe_gain_resource(
        &mut self,
        player_id: PlayerId,
        resource: Resource,
        count: u8,
    ) {
        if !self.enable_estimation {
            return;
        }

        self.apply_actions(player_id, |estimate| {
            estimate.observe_gain_resource(resource, count)
        });
    }

    pub fn observe_discard_resource(
        &mut self,
        player_id: PlayerId,
        resource: Resource,
        count: u8,
    ) {
        if !self.enable_estimation {
            return;
        }

        self.apply_actions(player_id, |estimate| {
            estimate.observe_discard_resource(resource, count)
        });
    }

    pub fn observe_monopoly(
        &mut self,
        monopoly_player_id: PlayerId,
        resource: Resource,
        remaining_cards_per_player: &[u8],
    ) {
        if !self.enable_estimation {
            return;
        }

        for i in 0..4usize {
            let current = PlayerId::from(i);
            for j in 0..4usize {
                let target = PlayerId::from(j);

                if current == target {
                    continue;
                }

                // What `current` believes about `target`'s hand.
                let relative = RelativePlayerId::from(current, target);
                let estimate =
                    &mut self.estimates[i][usize::from(relative) - 1];

                if target == monopoly_player_id {
                    // The monopoly player gained every card the others lost,
                    // so the hand grew by exactly that many of `resource`.
                    let num_cards = estimate.get_num_cards();
                    let remaining = remaining_cards_per_player[j];
                    if remaining > num_cards {
                        estimate.observe_gain_resource(
                            resource,
                            remaining - num_cards,
                        );
                    }
                } else {
                    // The other player lost every card of `resource`.
                    estimate.observe_losing_given_resource(
                        resource,
                        remaining_cards_per_player[j],
                    );
                }
            }
        }
    }

    /// What `observer` expects each other player to hold: the expected count
    /// of each resource, indexed like `Resource::ALL`. Rows are in relative
    /// order (index 0 = the next player after `observer`).
    pub fn expected_cards(&self, observer: PlayerId) -> [[f32; 5]; 3] {
        let row = &self.estimates[usize::from(observer)];
        from_fn(|i| row[i].expected_cards())
    }

    /// `thief` stole `resource` from `victim`. Only the two of them know
    /// which card moved; everyone else sees an unknown card change hands.
    pub fn observe_steal(
        &mut self,
        thief: PlayerId,
        victim: PlayerId,
        resource: Resource,
    ) {
        if !self.enable_estimation {
            return;
        }

        for i in 0..4usize {
            let observer = PlayerId::from(i);
            let slot = |target: PlayerId| {
                usize::from(RelativePlayerId::from(observer, target)) - 1
            };

            if observer == thief {
                // The thief saw which card it drew at random from the victim.
                self.estimates[i][slot(victim)]
                    .observe_stolen_resource(resource);
            } else if observer == victim {
                // The victim knows which card it lost.
                self.estimates[i][slot(thief)]
                    .observe_gain_resource(resource, 1);
            } else {
                // A bystander only knows a random card moved. What the thief
                // gained follows from this observer's view of the victim, so
                // read it before removing the card.
                let victim_estimate = &mut self.estimates[i][slot(victim)];
                let transferred = victim_estimate.stolen_card_distribution();
                victim_estimate.observe_discard_unknown_single_card();

                self.estimates[i][slot(thief)]
                    .observe_gain_unknown_card(&transferred);
            }
        }
    }
}

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

    pub fn observe_gain_resource(&mut self, resource: Resource, count: u8) {
        for (card, _) in &mut self.cards {
            *card = card.add_resource(resource, count);
        }
    }

    pub fn observe_discard_resource(&mut self, resource: Resource, count: u8) {
        self.cards.retain_mut(|(card, _)| {
            let new_card = card.remove_resources(resource, count);
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

    pub fn observe_discard_unknown_single_card(&mut self) {
        let mut cards = AHashMap::new();

        for (card, probability) in &self.cards {
            // A stolen card is drawn uniformly from the hand, so each resource
            // is taken with probability count / total.
            let total = card.total() as f32;
            for resource in card.not_zero_resources() {
                let weight = probability * card.count(resource) as f32 / total;

                cards
                    // Will not panic because we checked it is non-zero above.
                    .entry(card.remove_resources(resource, 1).unwrap())
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
    pub fn observe_losing_given_resource(
        &mut self,
        resource: Resource,
        remaining_cards: u8,
    ) {
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

    /// The victim's card at `resource` was stolen at random and this observer
    /// saw which one. A hand holding more of `resource` was more likely to
    /// give it up, so each hand is weighted by count / total (Bayes' rule)
    /// before the card is removed.
    pub fn observe_stolen_resource(&mut self, resource: Resource) {
        let mut cards = Vec::with_capacity(self.cards.len());

        for (card, probability) in &self.cards {
            let Some(new_card) = card.remove_resources(resource, 1) else {
                continue;
            };

            let likelihood = card.count(resource) as f32 / card.total() as f32;
            cards.push((new_card, probability * likelihood));
        }

        if !cards.is_empty() {
            normalize_probabilities(&mut cards);
        }

        self.cards = cards;
    }

    /// Probability of each resource being the card a random steal would take
    /// from this hand: sum over hands of P(hand) * count / total.
    pub fn stolen_card_distribution(&self) -> [f32; 5] {
        let mut distribution = [0.0; 5];

        for (card, probability) in &self.cards {
            let total = card.total() as f32;
            for resource in card.not_zero_resources() {
                distribution[resource as usize] +=
                    probability * card.count(resource) as f32 / total;
            }
        }

        distribution
    }

    /// Expected count of each resource: sum over hands of P(hand) * count.
    /// The counts add up to the hand size. All zero if every hand was ruled
    /// out.
    pub fn expected_cards(&self) -> [f32; 5] {
        let mut expected = [0.0; 5];

        for (card, probability) in &self.cards {
            for resource in Resource::ALL {
                expected[resource as usize] +=
                    probability * card.count(resource) as f32;
            }
        }

        expected
    }

    /// Gain one card whose resource follows `distribution` (indexed like
    /// `Resource::ALL`). Falls back to uniform when the distribution is empty,
    /// e.g. when the source estimate had already been ruled out entirely.
    pub fn observe_gain_unknown_card(&mut self, distribution: &[f32; 5]) {
        let uniform = [1.0 / Resource::ALL.len() as f32; 5];
        let distribution = if distribution.iter().sum::<f32>() > 0.0 {
            distribution
        } else {
            &uniform
        };

        let mut cards = AHashMap::new();

        for (card, probability) in &self.cards {
            for resource in Resource::ALL {
                let weight = probability * distribution[resource as usize];
                if weight == 0.0 {
                    continue;
                }

                cards
                    .entry(card.add_resource(resource, 1))
                    .and_modify(|p| *p += weight)
                    .or_insert(weight);
            }
        }

        let mut cards = cards.into_iter().collect::<Vec<_>>();
        if !cards.is_empty() {
            normalize_probabilities(&mut cards);
        }

        self.cards = cards;
    }

    /// Gain one card of a completely unknown resource.
    #[cfg(test)]
    pub fn observe_gain_card_unknown_single_card(&mut self) {
        self.observe_gain_unknown_card(&[1.0 / Resource::ALL.len() as f32; 5]);
    }

    pub fn get_num_cards(&self) -> u8 {
        if self.cards.is_empty() {
            return 0;
        }

        self.cards[0].0.total()
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

    pub fn total(&self) -> u8 {
        self.cards.to_le_bytes().iter().map(|&b| b as u8).sum()
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

    pub fn add_resource(&self, resource: Resource, count: u8) -> Self {
        let mut bytes = self.cards.to_le_bytes();
        bytes[resource as usize] += count;

        ResourceCards::new(&bytes)
    }

    pub fn remove_resources(
        &self,
        resource: Resource,
        count: u8,
    ) -> Option<Self> {
        let mut bytes = self.cards.to_le_bytes();

        if bytes[resource as usize] < count {
            return None;
        }

        bytes[resource as usize] -= count;
        Some(ResourceCards::new(&bytes))
    }

    pub fn remove_all(&self, resource: Resource) -> Self {
        let mut bytes = self.cards.to_le_bytes();
        bytes[resource as usize] = 0;

        ResourceCards::new(&bytes)
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

    /// Give a known hand: `hand[i]` cards of `Resource::ALL[i]`.
    fn gain_cards(estimate: &mut BayesianCardEstimate, hand: &[u8; 5]) {
        for (resource, &count) in Resource::ALL.into_iter().zip(hand) {
            estimate.observe_gain_resource(resource, count);
        }
    }

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < EPS,
            "expected {expected}, got {actual}"
        );
    }

    // ---------------------------------------------------------------------
    // ResourceCards
    // ---------------------------------------------------------------------

    #[test]
    fn resource_cards_count_and_total() {
        let cards = ResourceCards::new(&[2, 0, 1, 0, 4]);
        assert_eq!(cards.count(Resource::Brick), 2);
        assert_eq!(cards.count(Resource::Lumber), 0);
        assert_eq!(cards.count(Resource::Wool), 4);
        assert_eq!(cards.total(), 7);
        assert_eq!(
            cards.not_zero_resources(),
            vec![Resource::Brick, Resource::Ore, Resource::Wool]
        );
    }

    #[test]
    fn resource_cards_add_resource_with_count() {
        let cards = ResourceCards::new(&[1, 0, 0, 0, 0]);
        let cards = cards.add_resource(Resource::Grain, 3);
        assert!(cards == ResourceCards::new(&[1, 0, 0, 3, 0]));
    }

    #[test]
    fn resource_cards_remove_resources() {
        let cards = ResourceCards::new(&[0, 3, 0, 0, 0]);
        assert!(
            cards.remove_resources(Resource::Lumber, 2)
                == Some(ResourceCards::new(&[0, 1, 0, 0, 0]))
        );
        assert!(
            cards.remove_resources(Resource::Lumber, 3)
                == Some(ResourceCards::new(&[0; 5]))
        );
        assert!(cards.remove_resources(Resource::Lumber, 4).is_none());
        assert!(cards.remove_resources(Resource::Ore, 1).is_none());
    }

    #[test]
    fn resource_cards_remove_all() {
        let cards = ResourceCards::new(&[2, 1, 3, 0, 0]);
        assert!(
            cards.remove_all(Resource::Ore)
                == ResourceCards::new(&[2, 1, 0, 0, 0])
        );
        assert!(
            cards.remove_all(Resource::Wool)
                == ResourceCards::new(&[2, 1, 3, 0, 0])
        );
    }

    // ---------------------------------------------------------------------
    // BayesianCardEstimate
    // ---------------------------------------------------------------------

    #[test]
    fn starts_with_certain_empty_hand() {
        let estimate = BayesianCardEstimate::new();
        assert_eq!(estimate.cards.len(), 1);
        assert_close(prob(&estimate, [0; 5]), 1.0);
        assert_eq!(estimate.get_num_cards(), 0);
    }

    #[test]
    fn known_gain_and_discard_are_exact() {
        let mut estimate = BayesianCardEstimate::new();
        gain_cards(&mut estimate, &[2, 1, 0, 0, 3]);
        estimate.observe_discard_resource(Resource::Brick, 1);
        estimate.observe_discard_resource(Resource::Wool, 2);

        assert_eq!(estimate.cards.len(), 1);
        assert_close(prob(&estimate, [1, 1, 0, 0, 1]), 1.0);
    }

    #[test]
    fn gain_resource_adds_to_every_hand() {
        let mut estimate = BayesianCardEstimate::new();
        estimate.observe_gain_card_unknown_single_card();
        estimate.observe_gain_resource(Resource::Ore, 2);

        // Each of the 5 single-card hands now also holds 2 ore.
        assert_eq!(estimate.cards.len(), 5);
        assert_close(prob(&estimate, [1, 0, 2, 0, 0]), 0.2);
        assert_close(prob(&estimate, [0, 0, 3, 0, 0]), 0.2);
        assert_close(prob(&estimate, [0, 0, 2, 0, 1]), 0.2);
        assert_close(total_prob(&estimate), 1.0);
    }

    #[test]
    fn gain_resource_with_zero_count_is_noop() {
        let mut estimate = BayesianCardEstimate::new();
        gain_cards(&mut estimate, &[1, 0, 0, 0, 0]);
        estimate.observe_gain_resource(Resource::Grain, 0);

        assert_close(prob(&estimate, [1, 0, 0, 0, 0]), 1.0);
    }

    #[test]
    fn num_cards_tracks_hand_size() {
        let mut estimate = BayesianCardEstimate::new();
        estimate.observe_gain_resource(Resource::Brick, 2);
        assert_eq!(estimate.get_num_cards(), 2);

        estimate.observe_gain_card_unknown_single_card();
        assert_eq!(estimate.get_num_cards(), 3);

        estimate.observe_discard_unknown_single_card();
        assert_eq!(estimate.get_num_cards(), 2);

        estimate.observe_discard_resource(Resource::Brick, 1);
        assert_eq!(estimate.get_num_cards(), 1);
    }

    #[test]
    fn num_cards_of_empty_estimate_is_zero() {
        let mut estimate = BayesianCardEstimate::new();
        gain_cards(&mut estimate, &[1, 0, 0, 0, 0]);
        estimate.observe_discard_resource(Resource::Ore, 1);

        assert!(estimate.cards.is_empty());
        assert_eq!(estimate.get_num_cards(), 0);
    }

    #[test]
    fn unknown_gain_on_empty_hand_is_uniform() {
        let mut estimate = BayesianCardEstimate::new();
        estimate.observe_gain_card_unknown_single_card();

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
        estimate.observe_gain_card_unknown_single_card();
        estimate.observe_gain_card_unknown_single_card();

        // 5 doubles + C(5, 2) = 10 mixed pairs.
        assert_eq!(estimate.cards.len(), 15);
        assert_close(prob(&estimate, [2, 0, 0, 0, 0]), 1.0 / 25.0);
        assert_close(prob(&estimate, [1, 1, 0, 0, 0]), 2.0 / 25.0);
        assert_close(total_prob(&estimate), 1.0);
    }

    #[test]
    fn unknown_discard_is_weighted_by_card_count() {
        let mut estimate = BayesianCardEstimate::new();
        gain_cards(&mut estimate, &[3, 0, 1, 0, 0]);
        estimate.observe_discard_unknown_single_card();

        // Brick is stolen 3 times out of 4, ore 1 time out of 4.
        assert_close(prob(&estimate, [2, 0, 1, 0, 0]), 0.75);
        assert_close(prob(&estimate, [3, 0, 0, 0, 0]), 0.25);
    }

    #[test]
    fn unknown_discard_from_single_resource_hand_is_certain() {
        let mut estimate = BayesianCardEstimate::new();
        gain_cards(&mut estimate, &[0, 0, 0, 2, 0]);
        estimate.observe_discard_unknown_single_card();

        assert_eq!(estimate.cards.len(), 1);
        assert_close(prob(&estimate, [0, 0, 0, 1, 0]), 1.0);
    }

    #[test]
    fn known_discard_rules_out_impossible_hands() {
        let mut estimate = BayesianCardEstimate::new();
        estimate.observe_gain_card_unknown_single_card();
        // Only the hand holding a brick can pay this.
        estimate.observe_discard_resource(Resource::Brick, 1);

        assert_eq!(estimate.cards.len(), 1);
        assert_close(prob(&estimate, [0; 5]), 1.0);
    }

    #[test]
    fn known_discard_of_multiple_cards_rules_out_short_hands() {
        let mut estimate = BayesianCardEstimate::new();
        estimate.observe_gain_card_unknown_single_card();
        estimate.observe_gain_card_unknown_single_card();
        // Only {2B} can pay two bricks.
        estimate.observe_discard_resource(Resource::Brick, 2);

        assert_eq!(estimate.cards.len(), 1);
        assert_close(prob(&estimate, [0; 5]), 1.0);
    }

    #[test]
    fn known_discard_renormalizes_remaining_hands() {
        let mut estimate = BayesianCardEstimate::new();
        gain_cards(&mut estimate, &[1, 0, 0, 0, 0]);
        estimate.observe_gain_card_unknown_single_card();
        // Hands: {2B}, {B,L}, {B,O}, {B,G}, {B,W} each 0.2.
        // Paying a lumber leaves only {B}.
        estimate.observe_discard_resource(Resource::Lumber, 1);
        assert_close(prob(&estimate, [1, 0, 0, 0, 0]), 1.0);

        let mut estimate = BayesianCardEstimate::new();
        estimate.observe_gain_card_unknown_single_card();
        estimate.observe_gain_card_unknown_single_card();
        // Paying a brick keeps the 9 hands with a brick, weighted 1:2:2:2:2.
        estimate.observe_discard_resource(Resource::Brick, 1);
        assert_close(total_prob(&estimate), 1.0);
        assert_close(prob(&estimate, [1, 0, 0, 0, 0]), 1.0 / 9.0);
        assert_close(prob(&estimate, [0, 1, 0, 0, 0]), 2.0 / 9.0);
    }

    #[test]
    fn steal_after_unknown_gain() {
        let mut estimate = BayesianCardEstimate::new();
        gain_cards(&mut estimate, &[1, 0, 0, 0, 0]);
        estimate.observe_gain_card_unknown_single_card();
        estimate.observe_discard_unknown_single_card();

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
        gain_cards(&mut estimate, &[1, 0, 0, 0, 0]);
        estimate.observe_discard_resource(Resource::Ore, 1);

        assert!(estimate.cards.is_empty());
    }

    #[test]
    fn losing_resource_on_known_hand_removes_all_of_it() {
        let mut estimate = BayesianCardEstimate::new();
        gain_cards(&mut estimate, &[2, 1, 3, 0, 0]);
        estimate.observe_losing_given_resource(Resource::Ore, 3);

        assert_eq!(estimate.cards.len(), 1);
        assert_close(prob(&estimate, [2, 1, 0, 0, 0]), 1.0);
    }

    #[test]
    fn losing_resource_rules_out_hands_with_wrong_remaining_count() {
        let mut estimate = BayesianCardEstimate::new();
        gain_cards(&mut estimate, &[1, 0, 0, 0, 0]);
        estimate.observe_gain_card_unknown_single_card();
        // Hands: {2B}, {B,L}, {B,O}, {B,G}, {B,W} each 0.2.
        // Monopoly on wool leaving 2 cards: only {B,W} loses a card, so it
        // is ruled out and the other four stay with 0.25 each.
        estimate.observe_losing_given_resource(Resource::Wool, 2);

        assert_eq!(estimate.cards.len(), 4);
        assert_close(prob(&estimate, [2, 0, 0, 0, 0]), 0.25);
        assert_close(prob(&estimate, [1, 1, 0, 0, 0]), 0.25);
        assert_close(prob(&estimate, [1, 0, 0, 0, 1]), 0.0);
    }

    #[test]
    fn losing_resource_reveals_hidden_card() {
        let mut estimate = BayesianCardEstimate::new();
        gain_cards(&mut estimate, &[1, 0, 0, 0, 0]);
        estimate.observe_gain_card_unknown_single_card();
        // Monopoly on wool leaving 1 card: the hidden card must have been wool.
        estimate.observe_losing_given_resource(Resource::Wool, 1);

        assert_eq!(estimate.cards.len(), 1);
        assert_close(prob(&estimate, [1, 0, 0, 0, 0]), 1.0);
    }

    #[test]
    fn losing_resource_merges_hands_that_differ_only_in_resource() {
        let mut estimate = BayesianCardEstimate::new();
        estimate.observe_gain_card_unknown_single_card();
        estimate.observe_gain_card_unknown_single_card();
        // Monopoly on brick leaving 1 card: {B,X} for X != B (each 2/25)
        // all become {X}; {2B} and the 2-card non-brick hands are ruled out.
        estimate.observe_losing_given_resource(Resource::Brick, 1);

        assert_eq!(estimate.cards.len(), 4);
        for i in 1..5 {
            let mut hand = [0; 5];
            hand[i] = 1;
            assert_close(prob(&estimate, hand), 0.25);
        }
        assert_close(total_prob(&estimate), 1.0);
    }

    #[test]
    fn losing_resource_merges_different_counts_of_taken_resource() {
        let mut estimate = BayesianCardEstimate::new();
        gain_cards(&mut estimate, &[0, 1, 0, 0, 0]);
        estimate.observe_gain_card_unknown_single_card();
        estimate.observe_gain_card_unknown_single_card();
        // Lumber is known; whatever else they held was brick or not.
        // Monopoly on brick leaving 1 card means both hidden cards were
        // brick, so {1L} is the only result.
        estimate.observe_losing_given_resource(Resource::Brick, 1);

        assert_eq!(estimate.cards.len(), 1);
        assert_close(prob(&estimate, [0, 1, 0, 0, 0]), 1.0);
    }

    #[test]
    fn impossible_losing_resource_empties_estimate() {
        let mut estimate = BayesianCardEstimate::new();
        gain_cards(&mut estimate, &[1, 1, 0, 0, 0]);
        estimate.observe_losing_given_resource(Resource::Ore, 1);

        assert!(estimate.cards.is_empty());
    }

    // ---------------------------------------------------------------------
    // OtherPlayerCardEstimates
    // ---------------------------------------------------------------------

    /// What `observer` believes about `target`'s hand.
    fn view(
        estimates: &OtherPlayerCardEstimates,
        observer: usize,
        target: usize,
    ) -> &BayesianCardEstimate {
        let relative = RelativePlayerId::from(
            PlayerId::from(observer),
            PlayerId::from(target),
        );
        &estimates.estimates[observer][usize::from(relative) - 1]
    }

    fn assert_all_views_empty_hand(estimates: &OtherPlayerCardEstimates) {
        for observer in 0..4 {
            for target in 0..4 {
                if observer == target {
                    continue;
                }
                assert_close(
                    prob(view(estimates, observer, target), [0; 5]),
                    1.0,
                );
            }
        }
    }

    #[test]
    fn other_player_estimates_start_empty() {
        let estimates = OtherPlayerCardEstimates::new(true);
        assert_eq!(estimates.estimates.len(), 4);
        for row in &estimates.estimates {
            assert_eq!(row.len(), 3);
        }
        assert_all_views_empty_hand(&estimates);
    }

    #[test]
    fn disabled_estimates_ignore_all_observations() {
        let mut estimates = OtherPlayerCardEstimates::new(false);
        let p1 = PlayerId::from(1usize);

        estimates.observe_gain_resource(p1, Resource::Brick, 2);
        estimates.observe_discard_resource(p1, Resource::Brick, 1);
        estimates.observe_monopoly(p1, Resource::Wool, &[0, 0, 0, 0]);

        assert_all_views_empty_hand(&estimates);
    }

    #[test]
    fn gain_resource_updates_every_observer_except_self() {
        let mut estimates = OtherPlayerCardEstimates::new(true);
        estimates.observe_gain_resource(
            PlayerId::from(2usize),
            Resource::Ore,
            3,
        );

        for observer in [0, 1, 3] {
            assert_close(
                prob(view(&estimates, observer, 2), [0, 0, 3, 0, 0]),
                1.0,
            );
            // Nobody else's hand changed.
            for target in 0..4 {
                if target == observer || target == 2 {
                    continue;
                }
                assert_close(
                    prob(view(&estimates, observer, target), [0; 5]),
                    1.0,
                );
            }
        }
        // Player 2's own row is untouched.
        for estimate in &estimates.estimates[2] {
            assert_close(prob(estimate, [0; 5]), 1.0);
        }
    }

    #[test]
    fn discard_resource_updates_every_observer() {
        let mut estimates = OtherPlayerCardEstimates::new(true);
        let p0 = PlayerId::from(0usize);
        estimates.observe_gain_resource(p0, Resource::Lumber, 2);
        estimates.observe_discard_resource(p0, Resource::Lumber, 1);

        for observer in 1..4 {
            assert_close(
                prob(view(&estimates, observer, 0), [0, 1, 0, 0, 0]),
                1.0,
            );
        }
    }

    #[test]
    fn monopoly_takes_resource_from_victims() {
        let mut estimates = OtherPlayerCardEstimates::new(true);
        // Player 1 holds 2 wool + 1 brick, player 2 holds 1 grain.
        estimates.observe_gain_resource(
            PlayerId::from(1usize),
            Resource::Wool,
            2,
        );
        estimates.observe_gain_resource(
            PlayerId::from(1usize),
            Resource::Brick,
            1,
        );
        estimates.observe_gain_resource(
            PlayerId::from(2usize),
            Resource::Grain,
            1,
        );

        // Player 0 monopolizes wool and ends with the 2 stolen cards.
        estimates.observe_monopoly(
            PlayerId::from(0usize),
            Resource::Wool,
            &[2, 1, 1, 0],
        );

        for observer in [0, 2, 3] {
            assert_close(
                prob(view(&estimates, observer, 1), [1, 0, 0, 0, 0]),
                1.0,
            );
        }
        for observer in [0, 1, 3] {
            assert_close(
                prob(view(&estimates, observer, 2), [0, 0, 0, 1, 0]),
                1.0,
            );
        }
        for observer in [0, 1, 2] {
            assert_close(prob(view(&estimates, observer, 3), [0; 5]), 1.0);
        }
    }

    #[test]
    fn monopoly_adds_stolen_cards_to_monopoly_player() {
        let mut estimates = OtherPlayerCardEstimates::new(true);
        let p0 = PlayerId::from(0usize);
        // Player 0 already holds 1 ore; players 1 and 2 hold 1 wool each.
        estimates.observe_gain_resource(p0, Resource::Ore, 1);
        estimates.observe_gain_resource(
            PlayerId::from(1usize),
            Resource::Wool,
            1,
        );
        estimates.observe_gain_resource(
            PlayerId::from(2usize),
            Resource::Wool,
            1,
        );

        estimates.observe_monopoly(p0, Resource::Wool, &[3, 0, 0, 0]);

        for observer in 1..4 {
            assert_close(
                prob(view(&estimates, observer, 0), [0, 0, 1, 0, 2]),
                1.0,
            );
        }
    }

    #[test]
    fn stolen_resource_weights_hands_by_likelihood() {
        // {2B} and {B,O} each 0.5. Drawing a brick at random is certain from
        // {2B} but only 50% from {B,O}, so P(remaining = B) = 0.5 / 0.75.
        let mut estimate = BayesianCardEstimate::new();
        estimate.observe_gain_resource(Resource::Brick, 1);
        estimate.observe_gain_unknown_card(&[0.5, 0.0, 0.5, 0.0, 0.0]);
        estimate.observe_stolen_resource(Resource::Brick);

        assert_close(prob(&estimate, [1, 0, 0, 0, 0]), 2.0 / 3.0);
        assert_close(prob(&estimate, [0, 0, 1, 0, 0]), 1.0 / 3.0);
    }

    #[test]
    fn stolen_resource_rules_out_hands_without_it() {
        let mut estimate = BayesianCardEstimate::new();
        estimate.observe_gain_card_unknown_single_card();
        estimate.observe_stolen_resource(Resource::Wool);

        assert_eq!(estimate.cards.len(), 1);
        assert_close(prob(&estimate, [0; 5]), 1.0);

        let mut estimate = BayesianCardEstimate::new();
        gain_cards(&mut estimate, &[1, 0, 0, 0, 0]);
        estimate.observe_stolen_resource(Resource::Wool);
        assert!(estimate.cards.is_empty());
    }

    #[test]
    fn stolen_card_distribution_is_weighted_by_count() {
        let mut estimate = BayesianCardEstimate::new();
        gain_cards(&mut estimate, &[3, 0, 1, 0, 0]);
        let distribution = estimate.stolen_card_distribution();
        assert_close(distribution[Resource::Brick as usize], 0.75);
        assert_close(distribution[Resource::Ore as usize], 0.25);
        assert_close(distribution.iter().sum(), 1.0);

        // Mixed over uncertain hands: {2B} and {B,O} each 0.5
        // -> P(B) = 0.5 * 1 + 0.5 * 0.5 = 0.75.
        let mut estimate = BayesianCardEstimate::new();
        estimate.observe_gain_resource(Resource::Brick, 1);
        estimate.observe_gain_unknown_card(&[0.5, 0.0, 0.5, 0.0, 0.0]);
        let distribution = estimate.stolen_card_distribution();
        assert_close(distribution[Resource::Brick as usize], 0.75);
        assert_close(distribution[Resource::Ore as usize], 0.25);

        // Empty hand: nothing to steal.
        let distribution =
            BayesianCardEstimate::new().stolen_card_distribution();
        assert_close(distribution.iter().sum(), 0.0);
    }

    #[test]
    fn gain_unknown_card_follows_distribution() {
        let mut estimate = BayesianCardEstimate::new();
        gain_cards(&mut estimate, &[0, 1, 0, 0, 0]);
        estimate.observe_gain_unknown_card(&[0.75, 0.0, 0.25, 0.0, 0.0]);

        // Zero-probability resources are not added as hands.
        assert_eq!(estimate.cards.len(), 2);
        assert_close(prob(&estimate, [1, 1, 0, 0, 0]), 0.75);
        assert_close(prob(&estimate, [0, 1, 1, 0, 0]), 0.25);
    }

    #[test]
    fn expected_cards_average_over_hands() {
        // {2B} and {B,O} each 0.5 -> 1.5 brick, 0.5 ore.
        let mut estimate = BayesianCardEstimate::new();
        estimate.observe_gain_resource(Resource::Brick, 1);
        estimate.observe_gain_unknown_card(&[0.5, 0.0, 0.5, 0.0, 0.0]);

        let expected = estimate.expected_cards();
        assert_close(expected[Resource::Brick as usize], 1.5);
        assert_close(expected[Resource::Ore as usize], 0.5);
        assert_close(expected.iter().sum(), 2.0);

        // Every hand ruled out: nothing expected.
        let mut estimate = BayesianCardEstimate::new();
        gain_cards(&mut estimate, &[1, 0, 0, 0, 0]);
        estimate.observe_discard_resource(Resource::Ore, 1);
        assert_close(estimate.expected_cards().iter().sum(), 0.0);
    }

    #[test]
    fn expected_cards_are_in_relative_order() {
        let mut estimates = OtherPlayerCardEstimates::new(true);
        estimates.observe_gain_resource(
            PlayerId::from(2usize),
            Resource::Ore,
            3,
        );

        // Player 1 sees player 2 as the next player, player 3 sees it as the
        // third.
        let from_1 = estimates.expected_cards(PlayerId::from(1usize));
        assert_close(from_1[0][Resource::Ore as usize], 3.0);
        let from_3 = estimates.expected_cards(PlayerId::from(3usize));
        assert_close(from_3[2][Resource::Ore as usize], 3.0);
        assert_close(from_3[0].iter().sum(), 0.0);
    }

    #[test]
    fn gain_unknown_card_with_empty_distribution_is_uniform() {
        let mut estimate = BayesianCardEstimate::new();
        estimate.observe_gain_unknown_card(&[0.0; 5]);

        assert_eq!(estimate.cards.len(), 5);
        assert_close(prob(&estimate, [0, 0, 0, 1, 0]), 0.2);
    }

    #[test]
    fn steal_is_exact_for_thief_and_victim_only() {
        let mut estimates = OtherPlayerCardEstimates::new(true);
        let (p0, p1) = (PlayerId::from(0usize), PlayerId::from(1usize));
        // Player 1 holds 1 brick + 1 ore; player 0 steals the ore.
        estimates.observe_gain_resource(p1, Resource::Brick, 1);
        estimates.observe_gain_resource(p1, Resource::Ore, 1);
        estimates.observe_steal(p0, p1, Resource::Ore);

        // The thief knows exactly what the victim has left.
        assert_close(prob(view(&estimates, 0, 1), [1, 0, 0, 0, 0]), 1.0);
        // The victim knows exactly what the thief gained.
        assert_close(prob(view(&estimates, 1, 0), [0, 0, 1, 0, 0]), 1.0);

        for observer in [2, 3] {
            // Bystanders only know one of the victim's two cards was taken...
            let victim = view(&estimates, observer, 1);
            assert_close(prob(victim, [1, 0, 0, 0, 0]), 0.5);
            assert_close(prob(victim, [0, 0, 1, 0, 0]), 0.5);
            // ...so the thief gained brick or ore, never anything else.
            let thief = view(&estimates, observer, 0);
            assert_eq!(thief.cards.len(), 2);
            assert_close(prob(thief, [1, 0, 0, 0, 0]), 0.5);
            assert_close(prob(thief, [0, 0, 1, 0, 0]), 0.5);
        }
    }

    #[test]
    fn bystander_infers_stolen_card_from_known_victim_hand() {
        let mut estimates = OtherPlayerCardEstimates::new(true);
        let (p0, p1) = (PlayerId::from(0usize), PlayerId::from(1usize));
        // Player 1 holds exactly one brick, so the stolen card is a brick.
        estimates.observe_gain_resource(p1, Resource::Brick, 1);
        estimates.observe_steal(p0, p1, Resource::Brick);

        for observer in [2, 3] {
            assert_close(
                prob(view(&estimates, observer, 0), [1, 0, 0, 0, 0]),
                1.0,
            );
            assert_close(prob(view(&estimates, observer, 1), [0; 5]), 1.0);
        }
    }

    #[test]
    fn thief_applies_likelihood_to_victim_hand() {
        let mut estimates = OtherPlayerCardEstimates::new(true);
        let (p0, p1, p2) = (
            PlayerId::from(0usize),
            PlayerId::from(1usize),
            PlayerId::from(2usize),
        );
        // Player 1 holds a known brick plus a card stolen from player 2,
        // who held {B, O}. Player 0 now steals a brick from player 1.
        estimates.observe_gain_resource(p2, Resource::Brick, 1);
        estimates.observe_gain_resource(p2, Resource::Ore, 1);
        estimates.observe_gain_resource(p1, Resource::Brick, 1);
        estimates.observe_steal(p1, p2, Resource::Ore);
        // Player 0 (bystander) sees player 1 as {2B} or {B,O}, 0.5 each.
        assert_close(prob(view(&estimates, 0, 1), [2, 0, 0, 0, 0]), 0.5);

        estimates.observe_steal(p0, p1, Resource::Brick);
        assert_close(prob(view(&estimates, 0, 1), [1, 0, 0, 0, 0]), 2.0 / 3.0);
        assert_close(prob(view(&estimates, 0, 1), [0, 0, 1, 0, 0]), 1.0 / 3.0);
    }

    #[test]
    fn disabled_estimates_ignore_steal() {
        let mut estimates = OtherPlayerCardEstimates::new(false);
        estimates.observe_steal(
            PlayerId::from(0usize),
            PlayerId::from(1usize),
            Resource::Ore,
        );
        assert_all_views_empty_hand(&estimates);
    }
}
