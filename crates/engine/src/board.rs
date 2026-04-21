use std::array::from_fn;

use arrayvec::ArrayVec;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;

use crate::action::Building;
use crate::action::Resource;
use crate::id::NodeId;
use crate::id::PlayerId;
use crate::id::RoadId;
use crate::id::TileId;
use crate::table::NODE_TO_TILE;
use crate::table::TILE_TO_NODE;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TileType {
    Hills,
    Forest,
    Mountains,
    Fields,
    Pasture,
    Desert,
}

#[derive(Clone, Copy, Debug)]
struct Tile {
    tile_type: TileType,
    has_robber: bool,
    tile_index: TileId,
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Road {
    // Uses the last 3 bits
    // _____       1                  11
    //             ^                  ^^
    // Whether the road is occupied
    //                            Who occupies
    data: u8,
}

impl Road {
    pub fn get(&self) -> Option<PlayerId> {
        if self.data & 0b0000_0100 == 0 {
            return None;
        }

        Some(PlayerId::from(self.data & 0b0000_0011))
    }

    pub fn set_player(&mut self, player: PlayerId) {
        self.data = u8::from(player) | 0b0000_0100;
    }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Structure {
    // Uses the last 4 bits
    // ____1       1                  11
    //     ^        ^                 ^^
    // Structure    |                 ||
    // Whether the structure exists   ||
    //                            Who occupies
    data: u8,
}

impl Structure {
    pub fn get(&self) -> Option<(Building, PlayerId)> {
        if self.data & 0b0000_0100 == 0 {
            return None;
        }

        let structure = match self.data & 0b0000_1000 {
            0 => Building::Settlement,
            _ => Building::City,
        };

        Some((structure, PlayerId::from(self.data & 0b0000_0011)))
    }

    pub fn set(&mut self, player: PlayerId, structure: Building) {
        let building = match structure {
            Building::Settlement => 0b0000_0000,
            Building::City => 0b0000_1000,
        };

        self.data = u8::from(player) | building | 0b0000_0100;
    }
}

// Question:
//  -- Given the dice role, who will be getting the resource?
//  -- Given the dice role, which resource will be given?
//
//  ==> We need the mapping between the digit and the list of houses & resource.
//
//  -- Is the given vertex occupied?
//  ==> Need a maping between vertex coord and the occupancy (user + (town or city))
//
//  -- Is the given edge occupied?
//  ==> Need a mapping between the edge and the occupancy (user)

#[derive(Debug, Clone)]
pub struct Board {
    // Coord to road.
    pub roads: ArrayVec<Road, 72>,

    // Coord to the structure
    pub structures: ArrayVec<Structure, 54>,

    // Dice number to the tiles.
    // 2 to 12.
    dice_to_tiles: ArrayVec<ArrayVec<Tile, 2>, 11>,

    tiles: ArrayVec<TileType, 19>,
}

impl Board {
    pub fn new(tiles: &[(/*dice=*/ u8, TileType)]) -> Self {
        let mut dice_to_tiles: ArrayVec<ArrayVec<Tile, 2>, 11> =
            ArrayVec::from(from_fn(|_| ArrayVec::new()));

        for (tile_index, tile) in tiles.iter().enumerate() {
            let (dice, tile_type) = *tile;
            dice_to_tiles[(dice - 2) as usize].push(Tile {
                tile_type,
                has_robber: tile_type == TileType::Desert,
                tile_index: TileId::from(tile_index),
            });
        }

        Self {
            roads: ArrayVec::from([Road::default(); 72]),
            structures: ArrayVec::from([Structure::default(); 54]),
            dice_to_tiles,
            tiles: ArrayVec::from_iter(tiles.iter().map(|(_, t)| *t)),
        }
    }

    pub fn get_tile_without_robber(&self) -> ArrayVec<TileId, 18> {
        let mut v = ArrayVec::<TileId, 18>::new();

        for tiles in &self.dice_to_tiles {
            for t in tiles {
                if !t.has_robber {
                    v.push(t.tile_index);
                }
            }
        }

        v
    }

    pub fn get_robber_position(&self) -> TileId {
        for tiles in &self.dice_to_tiles {
            for t in tiles {
                if t.has_robber {
                    return t.tile_index;
                }
            }
        }

        panic!("The board does not have a robber")
    }

    pub fn move_robber_to(&mut self, tile_id: TileId) {
        for tiles in &mut self.dice_to_tiles {
            for t in tiles {
                if t.has_robber {
                    t.has_robber = false;
                } else if t.tile_index == tile_id {
                    t.has_robber = true
                }
            }
        }
    }

    pub fn new_from_rand(rng: &mut ChaCha8Rng) -> Self {
        let tile_orders = [
            0, 1, 2, 6, 11, 15, 18, 17, 16, 12, 7, 3, 4, 5, 10, 14, 13, 8, 9,
        ];

        let dice_digits =
            [5, 6, 11, 5, 8, 10, 9, 2, 10, 12, 9, 8, 3, 4, 3, 4, 6, 11];

        let mut resources = vec![
            vec![TileType::Hills; 3],
            vec![TileType::Forest; 4],
            vec![TileType::Mountains; 3],
            vec![TileType::Fields; 4],
            vec![TileType::Pasture; 4],
            vec![TileType::Desert; 1],
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

        resources.shuffle(rng);

        let mut digit_index = 0;
        let mut tiles = vec![(0, TileType::Desert); 19];

        for i in 0..19 {
            let tile_index = tile_orders[i];
            if resources[i] == TileType::Desert {
                // 7 gets automatically assigned to the desert.
                tiles[tile_index].0 = 7;
            } else {
                tiles[tile_index] = (dice_digits[digit_index], resources[i]);
                digit_index += 1;
            }
        }

        Self::new(&tiles)
    }

    pub fn get_road(&self, road_id: RoadId) -> &Road {
        &self.roads[usize::from(road_id)]
    }

    pub fn build_road(&mut self, r: RoadId, player: PlayerId) {
        self.roads[usize::from(r)].set_player(player);
    }

    pub fn structure(&self, s: impl Into<usize>) -> &Structure {
        &self.structures[s.into()]
    }

    pub fn get_structure(&self, node_id: NodeId) -> &Structure {
        &self.structures[usize::from(node_id)]
    }

    pub fn build_structure(
        &mut self,
        n: NodeId,
        player: PlayerId,
        building: Building,
    ) {
        self.structures[usize::from(n)].set(player, building)
    }

    pub fn get_resources_near_node_id(&self, node_id: NodeId) -> Vec<Resource> {
        let tile_ids = &NODE_TO_TILE[usize::from(node_id)];

        tile_ids
            .into_iter()
            .filter_map(|id| {
                tile_type_to_resource(self.tiles[usize::from(*id)])
            })
            .collect()
    }

    pub fn get_matching_resources(
        &self,
        dice: u8,
    ) -> Vec<(PlayerId, Resource, u8)> {
        let tiles = &self.dice_to_tiles[(dice - 2) as usize];

        tiles
            .iter()
            .filter(|t| !t.has_robber)
            .flat_map(|t| {
                TILE_TO_NODE[usize::from(t.tile_index)]
                    .into_iter()
                    .map(|n| (t.tile_type, &self.structures[usize::from(n)]))
            })
            .filter_map(|(tile_type, structure)| {
                structure.get().map(|(building, player)| {
                    let resource =
                        tile_type_to_resource(tile_type).expect("not possible");

                    let num_cards = match building {
                        Building::Settlement => 1,
                        Building::City => 2,
                    };

                    (player, resource, num_cards)
                })
            })
            .collect::<Vec<_>>()
    }
}

fn tile_type_to_resource(tile_type: TileType) -> Option<Resource> {
    match tile_type {
        TileType::Hills => Some(Resource::Brick),
        TileType::Forest => Some(Resource::Lumber),
        TileType::Mountains => Some(Resource::Ore),
        TileType::Fields => Some(Resource::Grain),
        TileType::Pasture => Some(Resource::Wool),
        TileType::Desert => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn road_default_is_unoccupied() {
        let road = Road { data: 0 };
        assert_eq!(road.get(), None);
    }

    #[test]
    fn road_set_and_get() {
        let mut road = Road { data: 0 };
        let player = PlayerId::new(2);
        road.set_player(player);
        assert_eq!(road.get(), Some(player));
    }

    #[test]
    fn structure_default_is_empty() {
        let s = Structure { data: 0 };
        assert_eq!(s.get(), None);
    }

    #[test]
    fn structure_set_settlement() {
        let mut s = Structure { data: 0 };
        let player = PlayerId::new(1);
        s.set(player, Building::Settlement);
        assert_eq!(s.get(), Some((Building::Settlement, player)));
    }

    #[test]
    fn structure_set_city() {
        let mut s = Structure { data: 0 };

        let player = PlayerId::new(3);
        s.set(player, Building::City);
        assert_eq!(s.get(), Some((Building::City, player)));
    }

    #[test]
    fn structure_upgrade_settlement_to_city() {
        let mut s = Structure { data: 0 };
        let player = PlayerId::new(0);

        s.set(player, Building::Settlement);
        assert_eq!(s.get(), Some((Building::Settlement, player)));

        s.set(player, Building::City);
        assert_eq!(s.get(), Some((Building::City, player)));
    }

    #[test]
    fn build_road_on_board() {
        let mut board = Board::new(&[]);
        let player = PlayerId::new(0);
        let road_id = RoadId::new(5);

        assert!(board.roads[5].get().is_none());
        board.build_road(road_id, player);
        assert_eq!(board.roads[5].get(), Some(player));
    }

    #[test]
    fn build_structure_on_board() {
        let mut board = Board::new(&[]);
        let player = PlayerId::new(1);
        let node = NodeId::new(10);

        assert!(board.structure(node).get().is_none());
        board.build_structure(node, player, Building::Settlement);
        assert_eq!(
            board.structure(node).get(),
            Some((Building::Settlement, player))
        );
    }

    #[test]
    fn get_matching_resources_empty_board() {
        let board = Board::new(&[(3, TileType::Hills)]);
        let result = board.get_matching_resources(3);
        assert!(result.is_empty());
    }

    #[test]
    fn get_matching_resources_with_settlement() {
        let mut board = Board::new(&[(3, TileType::Forest)]);
        let player = PlayerId::new(0);

        // Tile 0 at dice=3 has nodes [0, 3, 4, 7, 8, 12].
        // Place a settlement on node 0.
        board.build_structure(NodeId::new(0), player, Building::Settlement);

        let result = board.get_matching_resources(3);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], (player, Resource::Lumber, 1));
    }

    #[test]
    fn get_matching_resources_city_gives_two() {
        let mut board = Board::new(&[(3, TileType::Mountains)]);
        let player = PlayerId::new(2);

        board.build_structure(NodeId::new(3), player, Building::City);

        let result = board.get_matching_resources(3);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], (player, Resource::Ore, 2));
    }

    #[test]
    fn get_matching_resources_robber_blocks() {
        let mut board = Board::new(&[(5, TileType::Fields)]);
        let player = PlayerId::new(0);

        board.build_structure(NodeId::new(0), player, Building::Settlement);

        // Manually set the robber on tile 0
        board.dice_to_tiles[(5 - 2) as usize][0].has_robber = true;

        let result = board.get_matching_resources(5);
        assert!(result.is_empty());
    }
}
