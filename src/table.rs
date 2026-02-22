use std::collections::HashSet;

use arrayvec::ArrayVec;
use once_cell::sync::Lazy;

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

pub static TILE_TO_NODE: [[u8; 6]; 19] = [
    [0, 3, 4, 7, 8, 12],
    [1, 4, 5, 8, 9, 13],
    [2, 5, 6, 9, 10, 14],
    [7, 11, 12, 16, 17, 22],
    [8, 12, 13, 17, 18, 23],
    [9, 13, 14, 18, 19, 24],
    [10, 14, 15, 19, 20, 25],
    [16, 21, 22, 27, 28, 33],
    [17, 22, 23, 28, 29, 34],
    [18, 23, 24, 29, 30, 35],
    [19, 24, 25, 30, 31, 36],
    [20, 25, 26, 31, 32, 37],
    [28, 33, 34, 38, 39, 43],
    [29, 34, 35, 39, 40, 44],
    [30, 35, 36, 40, 41, 45],
    [31, 36, 37, 41, 42, 46],
    [39, 43, 44, 47, 48, 51],
    [40, 44, 45, 48, 49, 52],
    [41, 45, 46, 49, 50, 53],
];

pub static EDGE_TO_TILE: Lazy<[u8; 54]> = Lazy::new(|| build_reverse_map(&TILE_TO_NODE));

pub static TILE_TO_ROAD: [[u8; 6]; 19] = [
    [0, 1, 6, 7, 11, 12],
    [2, 3, 7, 8, 13, 14],
    [4, 5, 8, 9, 15, 16],
    [10, 11, 18, 19, 24, 25],
    [12, 13, 19, 20, 26, 27],
    [14, 15, 20, 21, 28, 29],
    [16, 17, 21, 22, 30, 31],
    [23, 24, 33, 34, 39, 40],
    [25, 26, 34, 35, 41, 42],
    [27, 28, 35, 36, 43, 44],
    [29, 30, 36, 37, 45, 46],
    [31, 32, 37, 38, 47, 48],
    [40, 41, 49, 50, 54, 55],
    [42, 43, 50, 51, 56, 57],
    [44, 45, 51, 52, 58, 59],
    [46, 47, 52, 53, 60, 61],
    [55, 56, 62, 63, 66, 67],
    [57, 58, 63, 64, 68, 69],
    [59, 60, 64, 65, 70, 71],
];

pub static ROAD_TO_TILE: Lazy<[u8; 72]> = Lazy::new(|| build_reverse_map(&TILE_TO_ROAD));

fn build_reverse_map<const N: usize>(arr: &[[u8; 6]]) -> [u8; N] {
    let mut m = [0; N];
    for (i, row) in arr.iter().enumerate() {
        for &idx in row {
            m[idx as usize] = i as u8;
        }
    }
    m
}

// Map between the Road index and the index of two nodes that forms the node.
pub static ROAD_TO_NODES: Lazy<[[u8; 2]; 72]> = Lazy::new(build_road_to_nodes);

fn build_road_to_nodes() -> [[u8; 2]; 72] {
    // 3, 4, 4, 5, 5, 6
    let mut road_to_nodes = [[0, 0]; 72];

    let mut road_count = 0;
    for road_start in 0..54 {
        let roads = match road_start {
            0..3 => vec![[road_start, road_start + 3], [road_start, road_start + 4]],
            3..7 => vec![[road_start, road_start + 4]],
            7..11 => vec![[road_start, road_start + 4], [road_start, road_start + 5]],
            11..16 => vec![[road_start, road_start + 5]],
            16..21 => vec![[road_start, road_start + 5], [road_start, road_start + 6]],
            21..27 => vec![[road_start, road_start + 6]],
            33..38 => vec![[road_start, road_start - 6], [road_start, road_start - 5]],
            38..43 => vec![[road_start, road_start - 5]],
            43..47 => vec![[road_start, road_start - 5], [road_start, road_start - 4]],
            47..51 => vec![[road_start, road_start - 4]],
            51..54 => vec![[road_start, road_start - 4], [road_start, road_start - 3]],
            _ => vec![],
        };

        for road in roads {
            road_to_nodes[road_count] = road;
            road_count += 1;
        }
    }

    assert!(road_count == 72);

    road_to_nodes
}

pub static NODE_TO_NEARBY_NODES: Lazy<[ArrayVec<u8, 9>; 54]> =
    Lazy::new(compute_node_to_node_with_distance_2);

fn compute_node_to_node_with_distance_2() -> [ArrayVec<u8, 9>; 54] {
    let mut node_to_distance_1: Vec<Vec<u8>> = vec![Vec::new(); 54];
    for [start, end] in *ROAD_TO_NODES {
        node_to_distance_1[start as usize].push(end);
        node_to_distance_1[end as usize].push(start);
    }

    let mut node_to_distance_1_or_2 = vec![HashSet::<u8>::new(); 54];
    for (start, ends) in node_to_distance_1.iter().enumerate() {
        for end in ends {
            node_to_distance_1_or_2[start].extend(
                node_to_distance_1[*end as usize]
                    .iter()
                    .filter(|s| **s != start as u8),
            );
        }

        node_to_distance_1_or_2[start].extend(ends);
    }

    node_to_distance_1_or_2
        .into_iter()
        .map(ArrayVec::<u8, 9>::from_iter)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

pub static NODE_TO_CONTAINING_ROADS: Lazy<[ArrayVec<u8, 3>; 54]> =
    Lazy::new(compute_node_to_containing_roads);

fn compute_node_to_containing_roads() -> [ArrayVec<u8, 3>; 54] {
    let mut node_to_containing_roads = vec![ArrayVec::<u8, 3>::new(); 54];

    for (road, [start, end]) in (*ROAD_TO_NODES).into_iter().enumerate() {
        node_to_containing_roads[start as usize].push(road as u8);
        node_to_containing_roads[end as usize].push(road as u8);
    }

    node_to_containing_roads.try_into().unwrap()
}

pub static ROAD_TO_CONNECTED_ROADS: Lazy<[ArrayVec<u8, 4>; 72]> =
    Lazy::new(compute_road_to_connected_roads);

fn compute_road_to_connected_roads() -> [ArrayVec<u8, 4>; 72] {
    let mut road_to_connected_roads = vec![HashSet::<u8>::new(); 72];

    for roads in &*NODE_TO_CONTAINING_ROADS {
        for i in 0..roads.len() {
            for j in (i + 1)..roads.len() {
                let r1 = roads[i];
                let r2 = roads[j];

                road_to_connected_roads[r1 as usize].insert(r2);
                road_to_connected_roads[r2 as usize].insert(r1);
            }
        }
    }

    road_to_connected_roads
        .into_iter()
        .map(ArrayVec::<u8, 4>::from_iter)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}
