// Resource card bookkeeping.
//
// Every resource card is always held by either the bank or a player, and the
// other players' hand estimates have to follow each movement. The functions
// here are the only ones that should move resource cards, so the hands, the
// bank and the estimates can't drift apart. They apply a movement; deciding
// whether it is legal is up to the caller.

use super::Game;
use crate::action::Resource;
use crate::util::id::PlayerId;

// Number of cards of each resource in the game.
pub(super) const CARDS_PER_RESOURCE: u8 = 19;

impl Game {
    // Bank -> player. Everyone sees which cards were taken.
    pub(super) fn gain_from_bank(
        &mut self,
        player: PlayerId,
        resource: Resource,
        count: u8,
    ) {
        debug_assert!(
            self.bank[resource as usize] >= count,
            "the bank does not have {count} {resource:?}"
        );

        self.bank[resource as usize] -= count;
        self.players[usize::from(player)].add_resource_card(resource, count);
        self.other_player_card_estimates
            .observe_gain_resource(player, resource, count);
    }

    // Player -> bank. Everyone sees which cards were paid.
    pub(super) fn pay_to_bank(
        &mut self,
        player: PlayerId,
        resource: Resource,
        count: u8,
    ) {
        self.players[usize::from(player)].use_resource_card(resource, count);
        self.bank[resource as usize] += count;
        self.other_player_card_estimates
            .observe_discard_resource(player, resource, count);
    }

    // The current player pays `cost` to the bank.
    pub(super) fn spend_resources(&mut self, cost: &[(Resource, u8)]) {
        let player = self.current_player_id();
        for &(resource, count) in cost {
            self.pay_to_bank(player, resource, count);
        }
    }

    // Victim -> thief, one card. Only the two of them know which card moved.
    pub(super) fn steal(
        &mut self,
        thief: PlayerId,
        victim: PlayerId,
        resource: Resource,
    ) {
        self.players[usize::from(victim)].use_resource_card(resource, 1);
        self.players[usize::from(thief)].add_resource_card(resource, 1);
        self.other_player_card_estimates
            .observe_steal(thief, victim, resource);
    }

    // Every other player -> `player`, all of their `resource` cards.
    pub(super) fn monopolize(&mut self, player: PlayerId, resource: Resource) {
        let mut total_cards = 0;
        for other in &mut self.players {
            if other.id == player {
                continue;
            }

            let num_cards = other.cards[resource as usize];
            total_cards += num_cards;
            other.use_resource_card(resource, num_cards);
        }

        self.players[usize::from(player)]
            .add_resource_card(resource, total_cards);

        let hand_sizes: Vec<u8> = self
            .players
            .iter()
            .map(|p| p.num_resource_cards())
            .collect();
        self.other_player_card_estimates.observe_monopoly(
            player,
            resource,
            &hand_sizes,
        );
    }

    // Cards of each resource across the bank and every hand. No action may
    // change this.
    pub(super) fn total_resource_cards(&self) -> [u8; 5] {
        let mut total = self.bank;
        for player in &self.players {
            for r in 0..5 {
                total[r] += player.cards[r];
            }
        }
        total
    }
}
