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

pub struct Player {
    id: PlayerId,
    dev_card_bought: ArrayVec<u8, 5>,

    pub cards: ArrayVec<u8, 10>,
    pub num_road: u8,
    pub num_settlement: u8,
    pub num_city: u8,
}

impl Player {
    pub fn new(id: PlayerId) -> Self {
        Self {
            cards: ArrayVec::from([0; 10]),
            dev_card_bought: ArrayVec::from([0; 5]),
            id,
            num_road: 0,
            num_settlement: 0,
            num_city: 0,
        }
    }

    pub fn add_resource_card(&mut self, resource: Resource, num_cards: u8) {
        self.cards[resource as usize] += num_cards;
    }

    pub fn add_dev_card(&mut self, dev_card: DevCard) {
        self.dev_card_bought[dev_card as usize - 5] += 1;
    }

    // Find all roads where the player can build a new road.
    //   1. It should be connected to the existing roads.
    //   2. It should not be occupied.
    pub fn get_available_roads(&self, board: &Board) -> Vec<RoadId> {
        let roads_owned_by_player: Vec<_> = board
            .roads
            .iter()
            .enumerate()
            .filter(|(_, r)| r.get() == Some(self.id))
            .collect();

        // All "nodes" that is covered by the player's roads.
        let road_node_touched: HashSet<_> = roads_owned_by_player
            .iter()
            .flat_map(|&(r, _)| (*ROAD_TO_NODES)[r])
            .collect();

        // Find all roads occupied by the player, and get the list of un-occupied nearby roads.
        roads_owned_by_player
            .iter()
            // Find the roads that is connected to the road occupied by the player.
            .flat_map(|(r, _)| &(*ROAD_TO_CONNECTED_ROADS)[*r])
            // Among those connected roads, find the one that is not occupied yet.
            .filter(|r| {
                let road_index = usize::from(**r);
                if board.roads[road_index].get().is_some() {
                    return false;
                }

                let [start, end] = &(*ROAD_TO_NODES)[road_index];

                // <-- Road --> (Other Settlement) <-- New Road ? -->
                //                                     ^^^^^^^^^
                //                                    Cannot build here
                if road_node_touched.contains(start) {
                    // If the new road "extension" point is occupied by the other player, then it
                    // cannot be extended from there
                    if let Some((_, p)) = board.structure(*start).get()
                        && p != self.id
                    {
                        return false;
                    }
                }

                if road_node_touched.contains(end) {
                    // If the new road "extension" point is occupied by the other player, then it
                    // cannot be extended from there
                    if let Some((_, p)) = board.structure(*end).get()
                        && p != self.id
                    {
                        return false;
                    }
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
        let mut actions = vec![Action::EndTurn];

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

        /*
        if self.cards[Resource::Ore as usize] != 0
            && self.cards[Resource::Grain as usize] != 0
            && self.cards[Resource::Wool as usize] != 0
        {
            actions.push(Action::BuyDevCard);
        }
        */

        for i in 0..5 {
            if self.cards[i] >= 4 {
                let resource = Resource::ALL[i];
                actions.extend(
                    Resource::ALL
                        .iter()
                        .map(|r| Action::MaritimeTrade((resource, 4), *r)),
                )
            }
        }

        actions
    }

    pub fn end_player_turn(&mut self) {
        for i in 0..5 {
            self.cards[i + 5] += self.dev_card_bought[i];
            self.dev_card_bought[i] = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Road, Structure};

    fn p0() -> PlayerId {
        PlayerId::new(0)
    }
    fn p1() -> PlayerId {
        PlayerId::new(1)
    }

    fn make_board() -> Board {
        let mut board = Board::new(&[]);
        for _ in 0..72 {
            board.roads.push(Road::default());
        }
        for _ in 0..54 {
            board.structures.push(Structure::default());
        }
        board
    }

    #[test]
    fn new_player_has_zero_cards_and_counts() {
        let p = Player::new(p0());
        assert_eq!(p.cards[..], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(p.num_road, 0);
        assert_eq!(p.num_settlement, 0);
        assert_eq!(p.num_city, 0);
    }

    #[test]
    fn add_resource_card() {
        let mut p = Player::new(p0());
        p.add_resource_card(Resource::Ore, 3);
        assert_eq!(p.cards[Resource::Ore as usize], 3);
        p.add_resource_card(Resource::Ore, 2);
        assert_eq!(p.cards[Resource::Ore as usize], 5);
    }

    #[test]
    fn add_dev_card() {
        let mut p = Player::new(p0());
        p.add_dev_card(DevCard::Knight);
        assert_eq!(p.cards[DevCard::Knight as usize], 0);
        assert_eq!(p.dev_card_bought[0], 1);
    }

    #[test]
    fn end_player_turn_moves_bought_dev_cards() {
        let mut p = Player::new(p0());
        p.add_dev_card(DevCard::Knight);
        assert_eq!(p.cards[DevCard::Knight as usize], 0);

        p.end_player_turn();
        assert_eq!(p.cards[DevCard::Knight as usize], 1);
        assert_eq!(p.dev_card_bought[0], 0);
    }

    // Road 0 connects nodes [0, 3]. Connected roads are 1 and 6.
    #[test]
    fn get_available_roads_from_single_road() {
        let mut board = make_board();
        board.build_road(RoadId::new(0), p0());

        let p = Player::new(p0());
        let mut roads = p.get_available_roads(&board);
        roads.sort();
        assert_eq!(roads, [RoadId::new(1), RoadId::new(6)]);
    }

    #[test]
    fn get_available_roads_blocked_by_opponent_structure() {
        let mut board = make_board();
        board.build_road(RoadId::new(0), p0());
        board.build_structure(NodeId::new(3), p1(), Building::Settlement);

        let p = Player::new(p0());
        let roads = p.get_available_roads(&board);
        assert_eq!(roads, [RoadId::new(1)]);
    }

    #[test]
    fn get_available_roads_blocked_by_opponent_structure2() {
        let mut board = make_board();
        board.build_road(RoadId::new(57), p0());
        board.build_road(RoadId::new(51), p0());

        board.build_structure(NodeId::new(44), p1(), Building::Settlement);
        board.build_structure(NodeId::new(30), p1(), Building::Settlement);

        let p = Player::new(p0());
        let mut roads = p.get_available_roads(&board);
        roads.sort();

        assert_eq!(roads, [RoadId::new(43), RoadId::new(44), RoadId::new(58)]);
    }

    #[test]
    fn get_available_roads_blocked_by_opponent_structure3() {
        let mut board = make_board();
        board.build_road(RoadId::new(27), p0());
        board.build_road(RoadId::new(28), p0());

        board.build_structure(NodeId::new(23), p1(), Building::Settlement);
        board.build_road(RoadId::new(35), p1());

        let p = Player::new(p0());
        let mut roads = p.get_available_roads(&board);
        roads.sort();

        assert_eq!(roads, [RoadId::new(20), RoadId::new(29), RoadId::new(36)]);
    }

    // Road 0 connects nodes [0, 3]. Both endpoints are available.
    #[test]
    fn get_available_nodes_from_single_road() {
        let mut board = make_board();
        board.build_road(RoadId::new(0), p0());

        let p = Player::new(p0());
        let mut nodes = p.get_available_nodes(&board);
        nodes.sort();
        assert_eq!(nodes, [NodeId::new(0), NodeId::new(3)]);
    }

    // Node 0's nearby nodes are [3, 4]. Structure on 4 blocks node 0.
    #[test]
    fn get_available_nodes_respects_distance_rule() {
        let mut board = make_board();
        board.build_road(RoadId::new(0), p0());
        board.build_structure(NodeId::new(4), p0(), Building::Settlement);

        let p = Player::new(p0());
        let nodes = p.get_available_nodes(&board);
        assert_eq!(nodes, [NodeId::new(3)]);
    }

    #[test]
    fn get_existing_settlements_finds_only_own() {
        let mut board = make_board();
        board.build_structure(NodeId::new(0), p0(), Building::Settlement);
        board.build_structure(NodeId::new(10), p1(), Building::Settlement);
        board.build_structure(NodeId::new(20), p0(), Building::City);

        let p = Player::new(p0());
        assert_eq!(p.get_existing_settlements(&board), [NodeId::new(0)]);
    }

    #[test]
    fn get_available_actions_no_resources_only_end_turn() {
        let board = make_board();
        let p = Player::new(p0());
        assert_eq!(p.get_available_actions(&board), [Action::EndTurn]);
    }

    #[test]
    fn get_available_actions_road_building() {
        let mut board = make_board();
        board.build_road(RoadId::new(0), p0());

        let mut p = Player::new(p0());
        p.add_resource_card(Resource::Brick, 1);
        p.add_resource_card(Resource::Lumber, 1);

        let mut actions = p.get_available_actions(&board);
        actions.sort();
        assert_eq!(
            actions,
            [
                Action::BuildRoad(RoadId::new(1)),
                Action::BuildRoad(RoadId::new(6)),
                Action::EndTurn,
            ]
        );
    }

    #[test]
    fn get_available_actions_road_no_resources() {
        let mut board = make_board();
        board.build_road(RoadId::new(0), p0());

        let mut p = Player::new(p0());
        p.add_resource_card(Resource::Brick, 1);

        assert_eq!(p.get_available_actions(&board), [Action::EndTurn]);
    }

    #[test]
    fn get_available_actions_settlement() {
        let mut board = make_board();
        board.build_road(RoadId::new(0), p0());

        let mut p = Player::new(p0());
        p.add_resource_card(Resource::Brick, 1);
        p.add_resource_card(Resource::Lumber, 1);
        p.add_resource_card(Resource::Grain, 1);
        p.add_resource_card(Resource::Wool, 1);

        let mut actions = p.get_available_actions(&board);
        actions.sort();
        assert_eq!(
            actions,
            [
                Action::BuildRoad(RoadId::new(1)),
                Action::BuildRoad(RoadId::new(6)),
                Action::BuildStructure(Building::Settlement, NodeId::new(0)),
                Action::BuildStructure(Building::Settlement, NodeId::new(3)),
                Action::EndTurn,
            ]
        );
    }

    #[test]
    fn get_available_actions_city() {
        let mut board = make_board();
        board.build_structure(NodeId::new(0), p0(), Building::Settlement);

        let mut p = Player::new(p0());
        p.num_settlement = 1;
        p.add_resource_card(Resource::Ore, 3);
        p.add_resource_card(Resource::Grain, 2);

        let mut actions = p.get_available_actions(&board);
        actions.sort();
        assert_eq!(
            actions,
            [
                Action::BuildStructure(Building::City, NodeId::new(0)),
                Action::EndTurn,
            ]
        );
    }

    #[test]
    fn get_available_actions_city_requires_settlement() {
        let board = make_board();
        let mut p = Player::new(p0());
        p.add_resource_card(Resource::Ore, 3);
        p.add_resource_card(Resource::Grain, 2);

        assert_eq!(p.get_available_actions(&board), [Action::EndTurn]);
    }

    #[test]
    fn get_available_actions_maritime_trade_4_for_1() {
        let board = make_board();
        let mut p = Player::new(p0());
        p.add_resource_card(Resource::Brick, 4);

        let mut actions = p.get_available_actions(&board);
        actions.sort();
        assert_eq!(
            actions,
            [
                Action::MaritimeTrade((Resource::Brick, 4), Resource::Brick),
                Action::MaritimeTrade((Resource::Brick, 4), Resource::Lumber),
                Action::MaritimeTrade((Resource::Brick, 4), Resource::Ore),
                Action::MaritimeTrade((Resource::Brick, 4), Resource::Grain),
                Action::MaritimeTrade((Resource::Brick, 4), Resource::Wool),
                Action::EndTurn,
            ]
        );
    }

    #[test]
    fn get_available_actions_no_maritime_trade_with_3() {
        let board = make_board();
        let mut p = Player::new(p0());
        p.add_resource_card(Resource::Brick, 3);

        assert_eq!(p.get_available_actions(&board), [Action::EndTurn]);
    }
}
