use std::collections::HashSet;
use std::collections::VecDeque;

use arrayvec::ArrayVec;

use crate::board::Board;
use crate::table::ROAD_TO_CONNECTED_ROADS;
use crate::table::ROAD_TO_NODES;
use crate::table::ROAD_TO_ROAD_CONNECTION;
use crate::util::id::NodeId;
use crate::util::id::PlayerId;
use crate::util::id::RoadId;

pub fn find_longest_road(board: &Board) -> ArrayVec<u8, 4> {
    let mut checked = HashSet::<RoadId>::new();

    let mut connected_road_per_player =
        ArrayVec::<Vec<HashSet<RoadId>>, 4>::new();
    for _ in 0..4 {
        connected_road_per_player.push(Vec::new());
    }

    for (road_id, road) in board.roads.iter().enumerate() {
        let road_id = RoadId::from(road_id);
        if checked.contains(&road_id) {
            continue;
        }

        if let Some(road_owner) = road.get() {
            connected_road_per_player[usize::from(road_owner)]
                .push(bfs_on_graph(board, road_owner, road_id, &mut checked));
        } else {
            checked.insert(road_id);
        }
    }

    // Now given the "connected" set, find the longest road.
    connected_road_per_player
        .into_iter()
        .enumerate()
        .map(|(p, connected_sets)| {
            let player = PlayerId::from(p);

            connected_sets
                .into_iter()
                .map(|set| {
                    find_longest_road_in_connected_set(board, player, set)
                })
                .max()
                .unwrap_or(0)
        })
        .collect()
}

fn find_longest_road_in_connected_set(
    board: &Board,
    player: PlayerId,
    connected_roads: HashSet<RoadId>,
) -> u8 {
    // We can ignore the "pass-through" roads.
    //
    // ... <-- road --> <-- road --> <-- road -->
    //                      ^^^ Pass through.
    //              There is no other road / settelment in its nodes.
    let mut starting_roads = connected_roads
        .iter()
        .filter(|r| {
            let road_id = *r;

            // One of the road end is occupied by the other settlement.
            let nodes = (*ROAD_TO_NODES)[usize::from(*road_id)];
            if nodes.iter().any(|n| {
                if let Some((_, owner)) = board.get_structure(*n).get()
                    && owner != player
                {
                    true
                } else {
                    false
                }
            }) {
                return true;
            }

            let nearby_roads = (*ROAD_TO_CONNECTED_ROADS)
                [usize::from(*road_id)]
            .iter()
            .filter(|r| connected_roads.contains(*r))
            .collect::<Vec<_>>();

            if nearby_roads.len() != 2 {
                return true;
            }

            // The road is only connected to two roads, and they are connected to the different ends of
            // the road, then this road is a "PASS THROUGH" node.
            ROAD_TO_ROAD_CONNECTION.get(road_id, nearby_roads[0])
                == ROAD_TO_ROAD_CONNECTION.get(road_id, nearby_roads[1])
        })
        .collect::<Vec<_>>();

    if starting_roads.is_empty() {
        starting_roads = connected_roads.iter().collect::<Vec<_>>();
    }

    // Now find the longest path starting from each of the "starting_nodes".
    starting_roads
        .into_iter()
        .map(|starting_node| {
            let mut current_checked = HashSet::new();
            dfs_for_longest_road(
                board,
                player,
                *starting_node,
                None,
                &mut current_checked,
                &connected_roads,
            )
        })
        .max()
        .unwrap_or(0u8)
}

fn dfs_for_longest_road(
    board: &Board,
    current_player: PlayerId,
    current_road: RoadId,
    node_coming_from: Option<NodeId>,
    current_checked: &mut HashSet<RoadId>,
    connected_roads: &HashSet<RoadId>,
) -> u8 {
    current_checked.insert(current_road);

    let nearby_roads = &(*ROAD_TO_CONNECTED_ROADS)[usize::from(current_road)];

    let mut max_length = None;

    for road in nearby_roads {
        if !connected_roads.contains(road) || current_checked.contains(road) {
            continue;
        }

        let connecting_node = ROAD_TO_ROAD_CONNECTION.get(&current_road, road);

        // Cannot come from the same node.
        if let Some(node_coming_from) = node_coming_from
            && connecting_node == node_coming_from
        {
            continue;
        }

        let structure = board.get_structure(connecting_node).get();
        if let Some((_, owner)) = structure
            && owner != current_player
        {
            continue;
        }

        let length = dfs_for_longest_road(
            board,
            current_player,
            *road,
            Some(connecting_node),
            current_checked,
            connected_roads,
        );

        max_length = Some(max_length.map_or(length, |l: u8| l.max(length)))
    }

    current_checked.remove(&current_road);
    max_length.unwrap_or(0) + 1
}

fn bfs_on_graph(
    board: &Board,
    current_player: PlayerId,
    start: RoadId,
    checked: &mut HashSet<RoadId>,
) -> HashSet<RoadId> {
    let mut connected = HashSet::new();
    connected.insert(start);
    let mut roads_to_check = VecDeque::<RoadId>::from([start]);

    while let Some(current) = roads_to_check.pop_front() {
        checked.insert(current);

        // Get the roads connected to the road.
        let nearby_roads = &(*ROAD_TO_CONNECTED_ROADS)[usize::from(current)];
        for next_road_id in nearby_roads {
            if connected.contains(next_road_id) {
                continue;
            }

            // Go to the road only if it is connected.
            let next_road = board.get_road(*next_road_id).get();
            if next_road.is_none() || next_road.unwrap() != current_player {
                continue;
            }

            let connecting_node =
                ROAD_TO_ROAD_CONNECTION.get(&current, next_road_id);

            let structure = board.get_structure(connecting_node).get();
            if let Some((_, owner)) = structure
                && owner != current_player
            {
                continue;
            }

            connected.insert(*next_road_id);
            roads_to_check.push_back(*next_road_id);
        }
    }

    connected
}

trait CountUniqueAtMost2: Iterator + Sized
where
    Self::Item: PartialEq,
{
    fn count_unique_at_most2(mut self) -> usize {
        let Some(first) = self.next() else { return 0 };

        for x in self {
            if x != first {
                return 2;
            }
        }
        1
    }
}

impl<I> CountUniqueAtMost2 for I
where
    I: Iterator,
    I::Item: PartialEq,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::Building;
    use crate::board::Board;
    use crate::player::Player;

    // Road layout reference (road_id → [node, node]):
    //   0:[0,3]  1:[0,4]  2:[1,4]  3:[1,5]  4:[2,5]  5:[2,6]
    //   6:[3,7]  7:[4,8]  8:[5,9]  9:[6,10]
    //  10:[7,11] 11:[7,12] 12:[8,12] 13:[8,13]
    //  19:[12,17] 25:[17,22] 34:[22,28]

    #[test]
    fn no_roads() {
        let board = Board::new(&[]);
        let result = find_longest_road(&board);
        assert_eq!(result.as_slice(), &[0, 0, 0, 0]);
    }

    #[test]
    fn single_road() {
        let mut board = Board::new(&[]);
        let p0 = PlayerId::new(0);
        board.build_road(RoadId::new(0), p0);

        let result = find_longest_road(&board);
        assert_eq!(result[0], 1);
    }

    #[test]
    fn two_connected_roads() {
        let mut board = Board::new(&[]);
        let p0 = PlayerId::new(0);
        // Road 0 [0,3] and road 6 [3,7] share node 3.
        board.build_road(RoadId::new(0), p0);
        board.build_road(RoadId::new(6), p0);

        let result = find_longest_road(&board);
        assert_eq!(result[0], 2);
    }

    #[test]
    fn linear_chain_of_five() {
        let mut board = Board::new(&[]);
        let p0 = PlayerId::new(0);
        // Chain: 0[0,3] → 6[3,7] → 11[7,12] → 19[12,17] → 25[17,22]
        for id in [0, 6, 11, 19, 25] {
            board.build_road(RoadId::new(id), p0);
        }

        let result = find_longest_road(&board);
        assert_eq!(result[0], 5);
    }

    #[test]
    fn full_cycle() {
        let mut board = Board::new(&[]);
        let p0 = PlayerId::new(0);
        for id in [27, 28, 36, 44, 43, 35] {
            board.build_road(RoadId::new(id), p0);
        }

        let result = find_longest_road(&board);
        assert_eq!(result[0], 6);
    }

    #[test]
    fn double_cycle() {
        let mut board = Board::new(&[]);
        let p0 = PlayerId::new(0);
        for id in [54, 55, 56, 57, 51, 43, 42, 41, 40, 49, 50, 51] {
            board.build_road(RoadId::new(id), p0);
        }

        let result = find_longest_road(&board);
        assert_eq!(result[0], 11);
    }

    #[test]
    fn triple_cycle() {
        let mut board = Board::new(&[]);
        let p0 = PlayerId::new(0);
        for id in [29, 30, 31, 32, 36, 37, 38, 45, 46, 47, 48, 52, 53, 60, 61] {
            board.build_road(RoadId::new(id), p0);
        }

        let result = find_longest_road(&board);
        assert_eq!(result[0], 14);

        // Enemy building at the center
        let p1 = PlayerId::new(1);
        board.build_structure(NodeId::new(31), p1, Building::City);

        let result = find_longest_road(&board);
        assert_eq!(result[0], 13);
    }

    #[test]
    fn dumbell_shape() {
        let mut board = Board::new(&[]);
        let p0 = PlayerId::new(0);
        for id in [18, 24, 25, 19, 11, 10, 12, 13, 7, 14, 20, 28, 29, 21, 15] {
            board.build_road(RoadId::new(id), p0);
        }

        let result = find_longest_road(&board);
        assert_eq!(result[0], 14);
    }

    #[test]
    fn two_groups() {
        let mut board = Board::new(&[]);
        let p0 = PlayerId::new(0);
        for id in [27, 28, 36, 44, 43, 35] {
            board.build_road(RoadId::new(id), p0);
        }

        for id in [57, 63, 68, 69, 64, 58, 70] {
            board.build_road(RoadId::new(id), p0);
        }

        let result = find_longest_road(&board);
        assert_eq!(result[0], 7);
    }

    #[test]
    fn two_disconnected_roads() {
        let mut board = Board::new(&[]);
        let p0 = PlayerId::new(0);
        // Road 0 [0,3] and road 4 [2,5] share no node.
        board.build_road(RoadId::new(0), p0);
        board.build_road(RoadId::new(4), p0);

        let result = find_longest_road(&board);
        assert_eq!(result[0], 1);
    }

    #[test]
    fn branch_takes_longest_path() {
        let mut board = Board::new(&[]);
        let p0 = PlayerId::new(0);
        // Y-shape: road 1[0,4] ← road 0[0,3] → road 6[3,7]
        // Longest path = 1→0→6 = 3
        board.build_road(RoadId::new(0), p0);
        board.build_road(RoadId::new(1), p0);
        board.build_road(RoadId::new(6), p0);

        let result = find_longest_road(&board);
        assert_eq!(result[0], 3);
    }

    #[test]
    fn enemy_settlement_blocks_road() {
        let mut board = Board::new(&[]);
        let p0 = PlayerId::new(0);
        let p1 = PlayerId::new(1);
        // Chain: 0[0,3] → 6[3,7] → 11[7,12]
        board.build_road(RoadId::new(0), p0);
        board.build_road(RoadId::new(6), p0);
        board.build_road(RoadId::new(11), p0);

        // Enemy settlement at node 7 blocks between roads 6 and 11.
        board.build_structure(NodeId::new(7), p1, Building::Settlement);

        let result = find_longest_road(&board);
        assert_eq!(result[0], 2);
    }

    #[test]
    fn cycle_but_has_one_enemy_settlement() {
        let mut board = Board::new(&[]);
        let p0 = PlayerId::new(0);
        let p1 = PlayerId::new(1);

        for id in [23, 24, 34, 40, 39, 33] {
            board.build_road(RoadId::new(id), p0);
        }

        // There is an enemy settlement at road 23 <-> 24 connection. However, it shouldn't affect
        // the cycle to be connected as it can still traverse other direction.
        board.build_structure(NodeId::new(16), p1, Building::Settlement);

        let result = find_longest_road(&board);
        assert_eq!(result[0], 6);

        // But if there is another building, then the cycle is now broken.
        board.build_structure(NodeId::new(28), p1, Building::Settlement);
        let result = find_longest_road(&board);
        assert_eq!(result[0], 4);
    }

    #[test]
    fn own_settlement_does_not_block() {
        let mut board = Board::new(&[]);
        let p0 = PlayerId::new(0);
        // Chain: 0[0,3] → 6[3,7] → 11[7,12]
        board.build_road(RoadId::new(0), p0);
        board.build_road(RoadId::new(6), p0);
        board.build_road(RoadId::new(11), p0);

        // Own settlement at node 7 should not block.
        board.build_structure(NodeId::new(7), p0, Building::Settlement);

        let result = find_longest_road(&board);
        assert_eq!(result[0], 3);
    }

    #[test]
    fn complex_dfs_start_road_matters() {
        let mut board = Board::new(&[]);
        let p0 = PlayerId::new(0);

        for id in [
            34, 41, 42, 35, 26, 25, 43, 44, 36, 28, 27, 45, 46, 37, 30, 29,
        ] {
            board.build_road(RoadId::new(id), p0);
        }

        for id in [20, 13, 14, 51, 52] {
            board.build_road(RoadId::new(id), p0);
        }

        let result = find_longest_road(&board);
        assert_eq!(result[0], 16);
    }

    #[test]
    fn two_players() {
        let mut board = Board::new(&[]);
        let p0 = PlayerId::new(0);
        let p1 = PlayerId::new(1);

        // Player 0: roads 0[0,3], 6[3,7] → length 2
        board.build_road(RoadId::new(0), p0);
        board.build_road(RoadId::new(6), p0);

        // Player 1: roads 4[2,5], 8[5,9] → length 2
        board.build_road(RoadId::new(4), p1);
        board.build_road(RoadId::new(8), p1);

        let result = find_longest_road(&board);
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 2);
    }
}
