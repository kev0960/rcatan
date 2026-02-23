use std::collections::HashSet;

use arrayvec::ArrayVec;

use crate::action::Action;
use crate::action::Building;
use crate::action::DevCard;
use crate::action::Resource;
use crate::board::Board;
use crate::id::NodeId;
use crate::id::PlayerId;
use crate::id::RoadId;
use crate::table::NODE_TO_NEARBY_NODES;
use crate::table::ROAD_TO_CONNECTED_ROADS;
use crate::table::ROAD_TO_NODES;
use rand::prelude::*;

pub struct Player {
    id: PlayerId,
    cards: ArrayVec<i32, 10>,

    num_road: u8,
    num_settlement: u8,
    num_city: u8,
}

impl Player {
    pub fn new(id: PlayerId) -> Self {
        Self {
            cards: ArrayVec::from([0; 10]),
            id,
            num_road: 0,
            num_settlement: 0,
            num_city: 0,
        }
    }

    pub fn add_resource_card(&mut self, resource: Resource, num_cards: u8) {
        self.cards[resource as usize] += num_cards as i32;
    }

    pub fn add_dev_card(&mut self, dev_card: DevCard) {
        self.cards[dev_card as usize] += 1;
    }

    // Find all roads where the player can build a new road.
    //   1. It should be connected to the existing roads.
    //   2. It should not be occupied.
    pub fn get_available_roads(&self, board: &Board) -> Vec<RoadId> {
        // Find all roads occupied by the player, and get the list of un-occupied nearby roads.
        board
            .roads
            .iter()
            .enumerate()
            .filter(|(_, r)| r.get() == Some(self.id))
            // Find the roads that is connected to the road occupied by the player.
            .flat_map(|(r, _)| &(*ROAD_TO_CONNECTED_ROADS)[r])
            // Among those connected roads, find the one that is not occupied yet.
            .filter(|r| {
                let road_index = usize::from(**r);
                if board.roads[road_index].get().is_some() {
                    return false;
                }

                let [start, end] = &(*ROAD_TO_NODES)[road_index];

                // The node shouldn't be occupied by the other player.
                if let Some((_, p)) = board.structure(*start).get()
                    && p != self.id
                {
                    return false;
                }

                if let Some((_, p)) = board.structure(*end).get()
                    && p != self.id
                {
                    return false;
                }

                true
            })
            .copied()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect()
    }

    // Find all nodes that is available for building.
    pub fn get_available_nodes(&self, board: &Board) -> Vec<NodeId> {
        board
            .roads
            .iter()
            .enumerate()
            // Find roads that is owned by the player.
            .filter(|(_, r)| r.get() == Some(self.id))
            // Find the all available "nodes" connected from the roads that are not occupied yet.
            .flat_map(|(r, _)| {
                (*ROAD_TO_NODES)[r]
                    .iter()
                    .filter(|n| board.structure(**n).get().is_none())
            })
            .collect::<HashSet<_>>()
            .into_iter()
            // Now make sure there is no structure nearby.
            .filter_map(|n| {
                if (*NODE_TO_NEARBY_NODES)[usize::from(*n)]
                    .iter()
                    .all(|n| board.structure(*n).get().is_none())
                {
                    Some(*n)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
    }

    pub fn get_existing_settlements(&self, board: &Board) -> Vec<NodeId> {
        board
            .structures
            .iter()
            .enumerate()
            .filter(|(_, s)| s.get() == Some((Building::Settlement, self.id)))
            .map(|(n, _)| n.into())
            .collect()
    }

    pub fn get_available_actions(&self, board: &Board) -> Vec<Action> {
        let mut actions = vec![];

        // Check road building.
        if self.cards[Resource::Brick as usize] != 0
            && self.cards[Resource::Lumber as usize] != 0
            && self.num_road < 15
        {
            let available_roads = self.get_available_roads(board);
            actions.extend(available_roads.into_iter().map(Action::BuildRoad));
        }

        if self.num_settlement < 5
            && self.cards[Resource::Brick as usize] != 0
            && self.cards[Resource::Lumber as usize] != 0
            && self.cards[Resource::Grain as usize] != 0
            && self.cards[Resource::Wool as usize] != 0
        {
            let available_nodes = self.get_available_nodes(board);
            actions.extend(
                available_nodes
                    .into_iter()
                    .map(|n| Action::BuildStructure(Building::Settlement, n)),
            )
        }

        if self.num_city < 4
            && self.num_settlement != 0
            && self.cards[Resource::Ore as usize] >= 3
            && self.cards[Resource::Grain as usize] >= 2
        {
            let settlements = self.get_existing_settlements(board);
            actions.extend(
                settlements
                    .into_iter()
                    .map(|n| Action::BuildStructure(Building::City, n)),
            )
        }

        if self.cards[Resource::Ore as usize] != 0
            && self.cards[Resource::Grain as usize] != 0
            && self.cards[Resource::Wool as usize] != 0
        {
            actions.push(Action::BuyDevCard);
        }

        actions
    }
}
