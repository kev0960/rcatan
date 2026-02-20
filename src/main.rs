mod table;

use std::array::from_fn;

use arrayvec::ArrayVec;
use table::TILE_TO_NODE;

#[repr(u8)]
enum Resource {
    Brick = 0,
    Lumber = 1,
    Ore = 2,
    Grain = 3,
    Wool = 4,
}

#[derive(Clone, Copy)]
enum TileType {
    Hills,
    Forest,
    Mountains,
    Fields,
    Pasture,
    Desert,
}

enum Building {
    Settlement,
    City,
}

type PlayerId = u8;

#[derive(Clone, Copy)]
struct Tile {
    tile_type: TileType,
    has_robber: bool,
    tile_index: u8,
}

struct Road {
    // Uses the last 3 bits
    // _____       1                  11
    //             ^                  ^^
    // Whether the road is occupied
    //                            Who occupies
    data: u8,
}

impl Road {
    fn get(&self) -> Option<PlayerId> {
        if self.data & 0b0000_0100 == 0 {
            return None;
        }

        Some(self.data & 0b0000_0011)
    }
}

struct Structure {
    // Uses the last 4 bits
    // ____1       1                  11
    //     ^        ^                 ^^
    // Structure    |                 ||
    // Whether the structure exists   ||
    //                            Who occupies
    data: u8,
}

impl Structure {
    fn get(&self) -> Option<(Building, PlayerId)> {
        if self.data & 0b0000_1000 == 0 {
            return None;
        }

        let structure = match self.data & 0b0000_0100 {
            0 => Building::Settlement,
            _ => Building::City,
        };

        Some((structure, self.data & 0b0000_0011))
    }
}

struct Player {
    cards: ArrayVec<i32, 5>,
}

impl Player {
    fn new() -> Self {
        Self {
            cards: ArrayVec::from([0; 5]),
        }
    }

    fn add_resource_card(&mut self, resource: Resource, num_cards: i32) {
        self.cards[resource as usize] += num_cards;
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
struct Game {
    // Coord to road.
    roads: ArrayVec<Road, 72>,

    // Coord to the structure
    structure: ArrayVec<Structure, 54>,

    // Dice number to the tiles.
    // 2 to 12.
    dice_to_tiles: ArrayVec<ArrayVec<Tile, 2>, 11>,

    players: ArrayVec<Player, 4>,
}

impl Game {
    fn new(tiles: &[(/*dice=*/ u8, TileType)]) -> Self {
        let mut dice_to_tiles: ArrayVec<ArrayVec<Tile, 2>, 11> =
            ArrayVec::from(from_fn(|_| ArrayVec::new()));

        for (tile_index, tile) in tiles.iter().enumerate() {
            let (dice, tile_type) = *tile;
            dice_to_tiles[(dice - 2) as usize].push(Tile {
                tile_type,
                has_robber: false,
                tile_index: tile_index as u8,
            });
        }

        Self {
            roads: ArrayVec::new(),
            structure: ArrayVec::new(),
            dice_to_tiles,
            players: ArrayVec::from(from_fn(|_| Player::new())),
        }
    }

    fn handle_non_7_dice_roll(&mut self, rolled_number: u8) {
        assert!(rolled_number != 7);

        let tiles = &self.dice_to_tiles[(rolled_number - 2) as usize];

        let structures = tiles
            .iter()
            .filter(|t| !t.has_robber)
            .flat_map(|t| {
                TILE_TO_NODE[t.tile_index as usize]
                    .into_iter()
                    .map(|n| (t.tile_type, &self.structure[n as usize]))
            })
            .filter_map(|(tile_type, structure)| structure.get().map(|b| (tile_type, b)))
            .collect::<Vec<_>>();

        // Now add cards to the players.
        for (tile_type, (building, player)) in structures {
            let resource = match tile_type {
                TileType::Hills => Resource::Brick,
                TileType::Forest => Resource::Lumber,
                TileType::Mountains => Resource::Ore,
                TileType::Fields => Resource::Grain,
                TileType::Pasture => Resource::Wool,
                TileType::Desert => panic!("Not possible"),
            };

            let num_cards = match building {
                Building::Settlement => 1,
                Building::City => 2,
            };

            self.players[player as usize].add_resource_card(resource, num_cards);
        }
    }
}

fn main() {
    let mut g = Game::new(&[]);
    g.handle_non_7_dice_roll(3);
    println!("Hello, world!");
}
