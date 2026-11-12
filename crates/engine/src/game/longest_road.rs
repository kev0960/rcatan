use std::collections::VecDeque;

use ahash::AHashSet;
use bitvec::prelude::*;

use crate::NodeId;
use crate::PlayerId;
use crate::ROAD_TO_NODES;
use crate::RoadId;
use crate::board::Board;
use crate::table::NODE_TO_CONTAINING_ROADS;
use crate::table::ROAD_TO_CONNECTED_ROADS;
use crate::table::ROAD_TO_ROAD_CONNECTION;

/// Computes each player's longest road, and remembers the board it was
/// computed for so that the next call only searches the players whose roads
/// could have changed.
pub struct LongestRoadTracker {
    last_computed: CompactBoardState,
    last_computed_longest_roads: [u8; 4],
}

impl LongestRoadTracker {
    pub fn new() -> Self {
        Self {
            last_computed: CompactBoardState::new(),
            last_computed_longest_roads: [0; 4],
        }
    }

    pub fn find_longest_road(
        &mut self,
        current: &CompactBoardState,
    ) -> [u8; 4] {
        if current == &self.last_computed {
            return self.last_computed_longest_roads;
        }

        // Then let's find what's modified compared to the last computed.
        let players_to_check =
            self.last_computed.find_players_to_check(current);

        // Only the modified players are searched; the others keep their
        // previously computed length.
        let mut checked = AHashSet::<RoadId>::new();
        for player in players_to_check {
            let player_index = usize::from(player);

            // Bits `72 * player .. 72 * (player + 1)` are this player's
            // roads, so the index within the slice is the road id.
            let player_roads =
                &current.roads[72 * player_index..72 * (player_index + 1)];

            let mut longest_road = 0;
            for road_id in player_roads.iter_ones() {
                let road_id = RoadId::from(road_id);
                if checked.contains(&road_id) {
                    continue;
                }

                let connected =
                    bfs_on_graph(current, player, road_id, &mut checked);
                longest_road =
                    longest_road.max(find_longest_road_in_connected_set(
                        current, player, &connected,
                    ));
            }

            self.last_computed_longest_roads[player_index] = longest_road;
        }

        self.last_computed = current.clone();
        return self.last_computed_longest_roads;
    }
}

fn find_longest_road_in_connected_set(
    board: &CompactBoardState,
    player: PlayerId,
    connected_roads: &AHashSet<RoadId>,
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
                if let Some(owner) = board.get_node_owner(*n)
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
            let mut current_checked = AHashSet::<RoadId>::new();
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
    current: &CompactBoardState,
    current_player: PlayerId,
    current_road: RoadId,
    node_coming_from: Option<NodeId>,
    current_checked: &mut AHashSet<RoadId>,
    connected_roads: &AHashSet<RoadId>,
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

        let structure = current.get_node_owner(connecting_node);
        if let Some(owner) = structure
            && owner != current_player
        {
            continue;
        }

        let length = dfs_for_longest_road(
            current,
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
    board: &CompactBoardState,
    current_player: PlayerId,
    start: RoadId,
    checked: &mut AHashSet<RoadId>,
) -> AHashSet<RoadId> {
    let mut connected = AHashSet::<RoadId>::new();
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
            if board.get_road_owner(*next_road_id) != Some(current_player) {
                continue;
            }

            let connecting_node =
                ROAD_TO_ROAD_CONNECTION.get(&current, &next_road_id);

            let structure = board.get_node_owner(connecting_node);
            if let Some(owner) = structure
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

/// Who owns each road and node, as one bit per (player, road) and
/// (player, node). The bit index is `road + 72 * player` and
/// `node + 54 * player`.
#[derive(Clone, Copy, Eq, PartialEq)]
pub struct CompactBoardState {
    roads: BitArr!(for 72 * 4, in u32),
    nodes: BitArr!(for 54 * 4, in u32),
}

impl From<&Board> for CompactBoardState {
    fn from(board: &Board) -> Self {
        let mut state = Self::new();

        for (road_index, road) in board.roads.iter().enumerate() {
            if let Some(owner) = road.get() {
                state.roads.set(road_index + 72 * usize::from(owner), true);
            }
        }

        for (node_index, structure) in board.structures.iter().enumerate() {
            if let Some((_, owner)) = structure.get() {
                state.nodes.set(node_index + 54 * usize::from(owner), true);
            }
        }

        state
    }
}

impl CompactBoardState {
    pub fn new() -> Self {
        Self {
            roads: bitarr![u32, Lsb0; 0; 72 * 4],
            nodes: bitarr![u32, Lsb0; 0; 54 * 4],
        }
    }

    pub fn find_players_to_check(
        &self,
        check_against: &CompactBoardState,
    ) -> AHashSet<PlayerId> {
        let mut players_to_check = AHashSet::<PlayerId>::new();

        // Changes in the nodes;
        let modified_nodes = self.nodes ^ check_against.nodes;
        let modified_nodes_indices =
            modified_nodes.iter_ones().collect::<Vec<_>>();

        for node_index in modified_nodes_indices {
            let node_player = PlayerId::from(node_index / 54);
            let node_id = NodeId::from(node_index % 54);

            // Check if there are any connecting roads;
            let connecting_roads =
                &(*NODE_TO_CONTAINING_ROADS)[usize::from(node_id)];

            let enemy_road_connected_to_node = connecting_roads
                .iter()
                .map(|road_id| check_against.get_road_owner(*road_id))
                .flatten()
                .filter(|owner| owner != &node_player)
                .collect::<Vec<_>>();

            players_to_check.extend(enemy_road_connected_to_node);
        }

        let modified_roads = self.roads ^ check_against.roads;
        let modified_roads_indices =
            modified_roads.iter_ones().collect::<Vec<_>>();

        for road_index in modified_roads_indices {
            let road_player = PlayerId::from(road_index / 72);
            players_to_check.insert(road_player);
        }

        players_to_check
    }

    pub fn get_node_owner(&self, node_id: NodeId) -> Option<PlayerId> {
        let index = usize::from(node_id);

        let player_index = (self.nodes[index] as usize) * 1
            + (self.nodes[index + 54] as usize) * 2
            + (self.nodes[index + 54 * 2] as usize) * 3
            + (self.nodes[index + 54 * 3] as usize) * 4;

        if player_index == 0 {
            return None;
        }

        Some(PlayerId::from(player_index - 1))
    }

    pub fn get_road_owner(&self, road_id: RoadId) -> Option<PlayerId> {
        let index = usize::from(road_id);

        let player_index = (self.roads[index] as usize) * 1
            + (self.roads[index + 72] as usize) * 2
            + (self.roads[index + 72 * 2] as usize) * 3
            + (self.roads[index + 72 * 3] as usize) * 4;

        if player_index == 0 {
            return None;
        }

        Some(PlayerId::from(player_index - 1))
    }
}

#[cfg(test)]
mod tests {
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;

    use super::*;
    use crate::action::Building;
    use crate::board::Board;

    // Road layout reference (road_id → [node, node]):
    //   0:[0,3]  1:[0,4]  2:[1,4]  3:[1,5]  4:[2,5]  5:[2,6]
    //   6:[3,7]  7:[4,8]  8:[5,9]  9:[6,10]
    //  10:[7,11] 11:[7,12] 12:[8,12] 13:[8,13]
    //  19:[12,17] 25:[17,22] 34:[22,28]

    impl CompactBoardState {
        fn build_road(&mut self, road_id: RoadId, player: PlayerId) {
            self.roads
                .set(usize::from(road_id) + 72 * usize::from(player), true);
        }

        fn build_structure(&mut self, node_id: NodeId, player: PlayerId) {
            self.nodes
                .set(usize::from(node_id) + 54 * usize::from(player), true);
        }
    }

    /// A board together with a tracker that is kept across the calls, so
    /// every `longest_roads` call after the first one is an incremental
    /// update.
    struct Fixture {
        board: CompactBoardState,
        tracker: LongestRoadTracker,
    }

    impl Fixture {
        fn new() -> Self {
            Self {
                board: CompactBoardState::new(),
                tracker: LongestRoadTracker::new(),
            }
        }

        fn build_roads(&mut self, road_ids: &[u8], player: PlayerId) {
            for id in road_ids {
                self.board.build_road(RoadId::new(*id), player);
            }
        }

        fn build_structure(&mut self, node_id: u8, player: PlayerId) {
            self.board.build_structure(NodeId::new(node_id), player);
        }

        /// The incremental result, which must always match the one that is
        /// computed from scratch.
        fn longest_roads(&mut self) -> [u8; 4] {
            let incremental = self.tracker.find_longest_road(&self.board);
            let from_scratch =
                LongestRoadTracker::new().find_longest_road(&self.board);
            assert_eq!(incremental, from_scratch);

            incremental
        }
    }

    fn players_to_check(
        from: &CompactBoardState,
        to: &CompactBoardState,
    ) -> Vec<PlayerId> {
        let mut players = from
            .find_players_to_check(to)
            .into_iter()
            .collect::<Vec<_>>();
        players.sort();
        players
    }

    #[test]
    fn no_roads() {
        let mut f = Fixture::new();
        assert_eq!(f.longest_roads(), [0, 0, 0, 0]);
    }

    #[test]
    fn single_road() {
        let mut f = Fixture::new();
        f.build_roads(&[0], PlayerId::new(0));

        assert_eq!(f.longest_roads()[0], 1);
    }

    #[test]
    fn two_connected_roads() {
        let mut f = Fixture::new();
        // Road 0 [0,3] and road 6 [3,7] share node 3.
        f.build_roads(&[0, 6], PlayerId::new(0));

        assert_eq!(f.longest_roads()[0], 2);
    }

    #[test]
    fn linear_chain_of_five() {
        let mut f = Fixture::new();
        // Chain: 0[0,3] → 6[3,7] → 11[7,12] → 19[12,17] → 25[17,22]
        f.build_roads(&[0, 6, 11, 19, 25], PlayerId::new(0));

        assert_eq!(f.longest_roads()[0], 5);
    }

    #[test]
    fn full_cycle() {
        let mut f = Fixture::new();
        f.build_roads(&[27, 28, 36, 44, 43, 35], PlayerId::new(0));

        assert_eq!(f.longest_roads()[0], 6);
    }

    #[test]
    fn double_cycle() {
        let mut f = Fixture::new();
        f.build_roads(
            &[54, 55, 56, 57, 51, 43, 42, 41, 40, 49, 50, 51],
            PlayerId::new(0),
        );

        assert_eq!(f.longest_roads()[0], 11);
    }

    #[test]
    fn triple_cycle() {
        let mut f = Fixture::new();
        f.build_roads(
            &[29, 30, 31, 32, 36, 37, 38, 45, 46, 47, 48, 52, 53, 60, 61],
            PlayerId::new(0),
        );

        assert_eq!(f.longest_roads()[0], 14);

        // Enemy building at the center
        f.build_structure(31, PlayerId::new(1));

        assert_eq!(f.longest_roads()[0], 13);
    }

    #[test]
    fn dumbell_shape() {
        let mut f = Fixture::new();
        f.build_roads(
            &[18, 24, 25, 19, 11, 10, 12, 13, 7, 14, 20, 28, 29, 21, 15],
            PlayerId::new(0),
        );

        assert_eq!(f.longest_roads()[0], 14);
    }

    #[test]
    fn two_groups() {
        let mut f = Fixture::new();
        let p0 = PlayerId::new(0);
        f.build_roads(&[27, 28, 36, 44, 43, 35], p0);
        f.build_roads(&[57, 63, 68, 69, 64, 58, 70], p0);

        assert_eq!(f.longest_roads()[0], 7);
    }

    #[test]
    fn two_disconnected_roads() {
        let mut f = Fixture::new();
        // Road 0 [0,3] and road 4 [2,5] share no node.
        f.build_roads(&[0, 4], PlayerId::new(0));

        assert_eq!(f.longest_roads()[0], 1);
    }

    #[test]
    fn branch_takes_longest_path() {
        let mut f = Fixture::new();
        // Y-shape: road 1[0,4] ← road 0[0,3] → road 6[3,7]
        // Longest path = 1→0→6 = 3
        f.build_roads(&[0, 1, 6], PlayerId::new(0));

        assert_eq!(f.longest_roads()[0], 3);
    }

    #[test]
    fn enemy_settlement_blocks_road() {
        let mut f = Fixture::new();
        // Chain: 0[0,3] → 6[3,7] → 11[7,12]
        f.build_roads(&[0, 6, 11], PlayerId::new(0));

        // Enemy settlement at node 7 blocks between roads 6 and 11.
        f.build_structure(7, PlayerId::new(1));

        assert_eq!(f.longest_roads()[0], 2);
    }

    #[test]
    fn cycle_but_has_one_enemy_settlement() {
        let mut f = Fixture::new();
        let p1 = PlayerId::new(1);
        f.build_roads(&[23, 24, 34, 40, 39, 33], PlayerId::new(0));

        // There is an enemy settlement at road 23 <-> 24 connection. However, it shouldn't affect
        // the cycle to be connected as it can still traverse other direction.
        f.build_structure(16, p1);

        assert_eq!(f.longest_roads()[0], 6);

        // But if there is another building, then the cycle is now broken.
        f.build_structure(28, p1);

        assert_eq!(f.longest_roads()[0], 4);
    }

    #[test]
    fn own_settlement_does_not_block() {
        let mut f = Fixture::new();
        let p0 = PlayerId::new(0);
        // Chain: 0[0,3] → 6[3,7] → 11[7,12]
        f.build_roads(&[0, 6, 11], p0);

        // Own settlement at node 7 should not block.
        f.build_structure(7, p0);

        assert_eq!(f.longest_roads()[0], 3);
    }

    #[test]
    fn complex_dfs_start_road_matters() {
        let mut f = Fixture::new();
        let p0 = PlayerId::new(0);
        f.build_roads(
            &[
                34, 41, 42, 35, 26, 25, 43, 44, 36, 28, 27, 45, 46, 37, 30, 29,
            ],
            p0,
        );
        f.build_roads(&[20, 13, 14, 51, 52], p0);

        assert_eq!(f.longest_roads()[0], 16);
    }

    #[test]
    fn two_players() {
        let mut f = Fixture::new();

        // Player 0: roads 0[0,3], 6[3,7] → length 2
        f.build_roads(&[0, 6], PlayerId::new(0));

        // Player 1: roads 4[2,5], 8[5,9] → length 2
        f.build_roads(&[4, 8], PlayerId::new(1));

        assert_eq!(f.longest_roads(), [2, 2, 0, 0]);
    }

    #[test]
    fn incremental_updates_step_by_step() {
        let mut f = Fixture::new();
        let p0 = PlayerId::new(0);
        let p1 = PlayerId::new(1);
        let p3 = PlayerId::new(3);

        // Chain: 0[0,3] → 6[3,7] → 11[7,12] → 19[12,17] → 25[17,22], built
        // one road at a time.
        for (built, id) in [0, 6, 11, 19, 25].into_iter().enumerate() {
            f.build_roads(&[id], p0);
            assert_eq!(f.longest_roads(), [built as u8 + 1, 0, 0, 0]);
        }

        // The other players build while player 0 does not change.
        f.build_roads(&[4, 8], p1);
        assert_eq!(f.longest_roads(), [5, 2, 0, 0]);

        f.build_roads(&[5], p3);
        assert_eq!(f.longest_roads(), [5, 2, 0, 1]);

        // Player 1 cuts the chain of player 0 at node 12 (roads 11 and 19).
        f.build_structure(12, p1);
        assert_eq!(f.longest_roads(), [3, 2, 0, 1]);

        // Nothing is modified.
        assert_eq!(f.longest_roads(), [3, 2, 0, 1]);

        // Player 0 extends the longer half: 1[0,4] → 0 → 6 → 11.
        f.build_roads(&[1], p0);
        assert_eq!(f.longest_roads(), [4, 2, 0, 1]);
    }

    #[test]
    fn only_modified_players_are_checked() {
        let p0 = PlayerId::new(0);
        let p1 = PlayerId::new(1);
        let p2 = PlayerId::new(2);

        let empty = CompactBoardState::new();

        // Chain of player 0: 0[0,3] → 6[3,7] → 11[7,12]
        let mut roads = empty;
        for id in [0, 6, 11] {
            roads.build_road(RoadId::new(id), p0);
        }
        // Player 2: road 4[2,5]
        roads.build_road(RoadId::new(4), p2);
        assert_eq!(players_to_check(&empty, &roads), vec![p0, p2]);

        // A new road only affects its owner.
        let mut next = roads;
        next.build_road(RoadId::new(8), p2);
        assert_eq!(players_to_check(&roads, &next), vec![p2]);

        // An enemy settlement on the road affects the road owner only.
        let mut next = roads;
        next.build_structure(NodeId::new(7), p1);
        assert_eq!(players_to_check(&roads, &next), vec![p0]);

        // An own settlement on the road affects no one.
        let mut next = roads;
        next.build_structure(NodeId::new(7), p0);
        assert_eq!(players_to_check(&roads, &next), vec![]);

        // A settlement away from every road affects no one.
        let mut next = roads;
        next.build_structure(NodeId::new(40), p1);
        assert_eq!(players_to_check(&roads, &next), vec![]);
    }

    #[test]
    fn stale_length_is_kept_for_unmodified_players() {
        // The tracker reuses the stored length for the players that are not
        // modified, so a deliberately wrong one must survive the update.
        let mut f = Fixture::new();
        f.build_roads(&[0, 6], PlayerId::new(0));
        f.tracker.find_longest_road(&f.board);

        f.tracker.last_computed_longest_roads[0] = 42;
        f.build_roads(&[4], PlayerId::new(1));

        assert_eq!(f.tracker.find_longest_road(&f.board), [42, 1, 0, 0]);
    }

    /// The longest road that starts from `node`, by trying every road that
    /// is not used yet. Shares nothing with the tracker, so it is the
    /// reference of the random test.
    fn brute_force_from_node(
        board: &CompactBoardState,
        player: PlayerId,
        node: NodeId,
        used_roads: u128,
    ) -> u8 {
        let mut longest = 0;

        for road in &(*NODE_TO_CONTAINING_ROADS)[usize::from(node)] {
            let road_bit = 1u128 << usize::from(*road);
            if board.get_road_owner(*road) != Some(player)
                || used_roads & road_bit != 0
            {
                continue;
            }

            let nodes = (*ROAD_TO_NODES)[usize::from(*road)];
            let next_node = if nodes[0] == node { nodes[1] } else { nodes[0] };

            // The road can end at the enemy structure, but cannot go past it.
            let blocked = board
                .get_node_owner(next_node)
                .is_some_and(|owner| owner != player);
            let rest = if blocked {
                0
            } else {
                brute_force_from_node(
                    board,
                    player,
                    next_node,
                    used_roads | road_bit,
                )
            };

            longest = longest.max(1 + rest);
        }

        longest
    }

    fn brute_force_longest_roads(board: &CompactBoardState) -> [u8; 4] {
        std::array::from_fn(|player| {
            (0..54usize)
                .map(|node| {
                    brute_force_from_node(
                        board,
                        PlayerId::from(player),
                        NodeId::from(node),
                        0,
                    )
                })
                .max()
                .unwrap()
        })
    }

    #[test]
    fn incremental_matches_brute_force_on_random_boards() {
        let mut rng = ChaCha8Rng::seed_from_u64(615);

        for _ in 0..200 {
            // The game converts its board on every computation, so the test
            // goes through the same conversion.
            let mut board = Board::new(&[]);
            let mut tracker = LongestRoadTracker::new();
            let mut num_roads = [0u8; 4];

            for _ in 0..70 {
                let player = PlayerId::from(rng.random_range(0..4usize));

                if rng.random_range(0..4) == 0 {
                    let node_id = NodeId::from(rng.random_range(0..54usize));
                    if board.get_structure(node_id).get().is_none() {
                        board.build_structure(
                            node_id,
                            player,
                            Building::Settlement,
                        );
                    }
                } else {
                    let road_id = RoadId::from(rng.random_range(0..72usize));
                    let num_roads = &mut num_roads[usize::from(player)];
                    if board.get_road(road_id).get().is_none()
                        && *num_roads < 15
                    {
                        board.build_road(road_id, player);
                        *num_roads += 1;
                    }
                }

                let compact = CompactBoardState::from(&board);
                assert_eq!(
                    tracker.find_longest_road(&compact),
                    brute_force_longest_roads(&compact)
                );
            }
        }
    }

    #[test]
    fn city_upgrade_keeps_the_node_owner() {
        let p1 = PlayerId::new(1);
        let mut board = Board::new(&[]);
        board.build_road(RoadId::new(0), PlayerId::new(0));
        board.build_structure(NodeId::new(3), p1, Building::Settlement);
        let with_settlement = CompactBoardState::from(&board);

        board.build_structure(NodeId::new(3), p1, Building::City);
        let with_city = CompactBoardState::from(&board);

        assert!(with_settlement == with_city);
        assert_eq!(with_city.get_node_owner(NodeId::new(3)), Some(p1));
        assert_eq!(
            with_city.get_road_owner(RoadId::new(0)),
            Some(PlayerId::new(0))
        );
    }
}
