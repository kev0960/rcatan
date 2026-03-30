use std::array::from_fn;
use std::collections::HashSet;

use arrayvec::ArrayVec;
use once_cell::sync::Lazy;

use crate::id::NodeId;
use crate::id::RoadId;
use crate::id::TileId;

const fn nodes<const N: usize>(a: [u8; N]) -> [NodeId; N] {
    let mut result = [NodeId::new(0); N];
    let mut i = 0;
    while i < N {
        result[i] = NodeId::new(a[i]);
        i += 1;
    }
    result
}

const fn roads<const N: usize>(a: [u8; N]) -> [RoadId; N] {
    let mut result = [RoadId::new(0); N];
    let mut i = 0;
    while i < N {
        result[i] = RoadId::new(a[i]);
        i += 1;
    }
    result
}

static TILE_INDEX_TO_START_COORDS: [(u8, u8); 19] = [
    (3, 0),
    (5, 0),
    (7, 0),
    (2, 2),
    (4, 2),
    (6, 2),
    (8, 2),
    (1, 4),
    (3, 4),
    (5, 4),
    (7, 4),
    (9, 4),
    (2, 6),
    (4, 6),
    (6, 6),
    (8, 6),
    (3, 8),
    (5, 8),
    (7, 8),
];

pub static TILE_TO_NODE: [[NodeId; 6]; 19] = [
    nodes([0, 3, 4, 7, 8, 12]),
    nodes([1, 4, 5, 8, 9, 13]),
    nodes([2, 5, 6, 9, 10, 14]),
    nodes([7, 11, 12, 16, 17, 22]),
    nodes([8, 12, 13, 17, 18, 23]),
    nodes([9, 13, 14, 18, 19, 24]),
    nodes([10, 14, 15, 19, 20, 25]),
    nodes([16, 21, 22, 27, 28, 33]),
    nodes([17, 22, 23, 28, 29, 34]),
    nodes([18, 23, 24, 29, 30, 35]),
    nodes([19, 24, 25, 30, 31, 36]),
    nodes([20, 25, 26, 31, 32, 37]),
    nodes([28, 33, 34, 38, 39, 43]),
    nodes([29, 34, 35, 39, 40, 44]),
    nodes([30, 35, 36, 40, 41, 45]),
    nodes([31, 36, 37, 41, 42, 46]),
    nodes([39, 43, 44, 47, 48, 51]),
    nodes([40, 44, 45, 48, 49, 52]),
    nodes([41, 45, 46, 49, 50, 53]),
];

pub static NODE_TO_TILE: Lazy<[ArrayVec<TileId, 3>; 54]> =
    Lazy::new(|| build_reverse_map(&TILE_TO_NODE));

pub static TILE_TO_ROAD: [[RoadId; 6]; 19] = [
    roads([0, 1, 6, 7, 11, 12]),
    roads([2, 3, 7, 8, 13, 14]),
    roads([4, 5, 8, 9, 15, 16]),
    roads([10, 11, 18, 19, 24, 25]),
    roads([12, 13, 19, 20, 26, 27]),
    roads([14, 15, 20, 21, 28, 29]),
    roads([16, 17, 21, 22, 30, 31]),
    roads([23, 24, 33, 34, 39, 40]),
    roads([25, 26, 34, 35, 41, 42]),
    roads([27, 28, 35, 36, 43, 44]),
    roads([29, 30, 36, 37, 45, 46]),
    roads([31, 32, 37, 38, 47, 48]),
    roads([40, 41, 49, 50, 54, 55]),
    roads([42, 43, 50, 51, 56, 57]),
    roads([44, 45, 51, 52, 58, 59]),
    roads([46, 47, 52, 53, 60, 61]),
    roads([55, 56, 62, 63, 66, 67]),
    roads([57, 58, 63, 64, 68, 69]),
    roads([59, 60, 64, 65, 70, 71]),
];

pub static ROAD_TO_TILE: Lazy<[ArrayVec<TileId, 2>; 72]> =
    Lazy::new(|| build_reverse_map(&TILE_TO_ROAD));

fn build_reverse_map<const N: usize, const K: usize, T: Into<usize> + Clone>(
    arr: &[[T; 6]],
) -> [ArrayVec<TileId, K>; N] {
    let mut m = from_fn(|_| ArrayVec::new());
    for (i, row) in arr.iter().enumerate() {
        for idx in row {
            m[idx.clone().into()].push(TileId::from(i));
        }
    }
    m
}

// Map between the Road index and the index of two nodes that forms the node.
pub static ROAD_TO_NODES: Lazy<[[NodeId; 2]; 72]> =
    Lazy::new(build_road_to_nodes);

fn build_road_to_nodes() -> [[NodeId; 2]; 72] {
    // 3, 4, 4, 5, 5, 6
    let mut road_to_nodes: [[NodeId; 2]; 72] = [nodes([0, 0]); 72];

    let mut road_count = 0;
    for road_start in 0u8..54 {
        let pairs: Vec<[NodeId; 2]> = match road_start {
            0..3 => {
                vec![
                    nodes([road_start, road_start + 3]),
                    nodes([road_start, road_start + 4]),
                ]
            }
            3..7 => vec![nodes([road_start, road_start + 4])],
            7..11 => {
                vec![
                    nodes([road_start, road_start + 4]),
                    nodes([road_start, road_start + 5]),
                ]
            }
            11..16 => vec![nodes([road_start, road_start + 5])],
            16..21 => {
                vec![
                    nodes([road_start, road_start + 5]),
                    nodes([road_start, road_start + 6]),
                ]
            }
            21..27 => vec![nodes([road_start, road_start + 6])],
            33..38 => {
                vec![
                    nodes([road_start, road_start - 6]),
                    nodes([road_start, road_start - 5]),
                ]
            }
            38..43 => vec![nodes([road_start, road_start - 5])],
            43..47 => {
                vec![
                    nodes([road_start, road_start - 5]),
                    nodes([road_start, road_start - 4]),
                ]
            }
            47..51 => vec![nodes([road_start, road_start - 4])],
            51..54 => {
                vec![
                    nodes([road_start, road_start - 4]),
                    nodes([road_start, road_start - 3]),
                ]
            }
            _ => vec![],
        };

        for pair in pairs {
            road_to_nodes[road_count] = pair;
            road_count += 1;
        }
    }

    assert!(road_count == 72);

    road_to_nodes
}

pub static NODE_TO_NEARBY_NODES: Lazy<[ArrayVec<NodeId, 3>; 54]> =
    Lazy::new(compute_nearby_nodes);

fn compute_nearby_nodes() -> [ArrayVec<NodeId, 3>; 54] {
    let mut nearby_nodes: Vec<ArrayVec<NodeId, 3>> =
        vec![ArrayVec::<NodeId, 3>::new(); 54];

    for [start, end] in *ROAD_TO_NODES {
        nearby_nodes[usize::from(start)].push(end);
        nearby_nodes[usize::from(end)].push(start);
    }

    nearby_nodes.try_into().unwrap()
}

pub static NODE_TO_CONTAINING_ROADS: Lazy<[ArrayVec<RoadId, 3>; 54]> =
    Lazy::new(compute_node_to_containing_roads);

fn compute_node_to_containing_roads() -> [ArrayVec<RoadId, 3>; 54] {
    let mut node_to_containing_roads = vec![ArrayVec::<RoadId, 3>::new(); 54];

    for (road, [start, end]) in (*ROAD_TO_NODES).into_iter().enumerate() {
        node_to_containing_roads[usize::from(start)].push(road.into());
        node_to_containing_roads[usize::from(end)].push(road.into());
    }

    node_to_containing_roads.try_into().unwrap()
}

pub static ROAD_TO_CONNECTED_ROADS: Lazy<[ArrayVec<RoadId, 4>; 72]> =
    Lazy::new(compute_road_to_connected_roads);

fn compute_road_to_connected_roads() -> [ArrayVec<RoadId, 4>; 72] {
    let mut road_to_connected_roads = vec![HashSet::<RoadId>::new(); 72];

    for roads in &*NODE_TO_CONTAINING_ROADS {
        for i in 0..roads.len() {
            for j in (i + 1)..roads.len() {
                let r1 = roads[i];
                let r2 = roads[j];

                road_to_connected_roads[usize::from(r1)].insert(r2);
                road_to_connected_roads[usize::from(r2)].insert(r1);
            }
        }
    }

    road_to_connected_roads
        .into_iter()
        .map(ArrayVec::<RoadId, 4>::from_iter)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

pub static ROAD_TO_ROAD_CONNECTION: Lazy<RoadToRoadConnection> =
    Lazy::new(RoadToRoadConnection::new);

pub struct RoadToRoadConnection {
    data: ArrayVec<ArrayVec<u8, 72>, 72>,
}

impl RoadToRoadConnection {
    fn new() -> Self {
        Self {
            data: compute_road_to_road_intersection(),
        }
    }

    // Use when you two roads are expected to overlap. This removes unnecessary unwrap step.
    pub fn get(&self, a: &RoadId, b: &RoadId) -> NodeId {
        let overlap = self.data[usize::from(*a)][usize::from(*b)];
        debug_assert!(overlap & 0b1000_0000 != 0);
        NodeId::from(overlap & 0b0111_1111)
    }

    pub fn maybe_get(&self, a: &RoadId, b: &RoadId) -> Option<NodeId> {
        let overlap = self.data[usize::from(*a)][usize::from(*b)];
        if overlap & 0b1000_0000 == 0 {
            None
        } else {
            Some(NodeId::from(
                self.data[usize::from(*a)][usize::from(*b)] & 0b0111_1111,
            ))
        }
    }
}

fn compute_road_to_road_intersection() -> ArrayVec<ArrayVec<u8, 72>, 72> {
    let mut road_to_road_connection: ArrayVec<ArrayVec<u8, 72>, 72> =
        ArrayVec::from(from_fn(|_| ArrayVec::from([0; 72])));

    for (road_id, connecting_roads) in
        (*ROAD_TO_CONNECTED_ROADS).iter().enumerate()
    {
        let current_road_nodes = (*ROAD_TO_NODES)[road_id];
        for connecting_road in connecting_roads {
            let connecting_road_nodes =
                &(*ROAD_TO_NODES)[usize::from(*connecting_road)];
            let node_id =
                get_overlap_node(&current_road_nodes, connecting_road_nodes);

            road_to_road_connection[road_id][usize::from(*connecting_road)] =
                0b1000_0000 | (u8::from(node_id));
        }
    }

    road_to_road_connection
}

fn get_overlap_node(left: &[NodeId; 2], right: &[NodeId; 2]) -> NodeId {
    let [l_1, l_2] = left;
    let [r_1, r_2] = right;

    if l_1 == r_1 || l_1 == r_2 {
        *l_1
    } else {
        debug_assert!(l_2 == r_1 || l_2 == r_2);
        *l_2
    }
}
