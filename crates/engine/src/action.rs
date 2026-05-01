use crate::util::id::NodeId;
use crate::util::id::PlayerId;
use crate::util::id::RoadId;
use crate::util::id::TileId;

#[derive(PartialEq, Eq, PartialOrd, Ord, Copy, Clone, Debug)]
pub enum Building {
    Settlement,
    City,
}

#[repr(u8)]
#[derive(PartialEq, Eq, PartialOrd, Ord, Copy, Clone, Debug, Hash)]
pub enum Resource {
    Brick = 0,
    Lumber = 1,
    Ore = 2,
    Grain = 3,
    Wool = 4,
}

impl Resource {
    pub const ALL: [Resource; 5] = [
        Resource::Brick,
        Resource::Lumber,
        Resource::Ore,
        Resource::Grain,
        Resource::Wool,
    ];
}

#[repr(u8)]
#[derive(Copy, Clone)]
pub enum DevCard {
    Knight = 5,
    VictoryPoint = 6,
    RoadBuilding = 7,
    YearOfPlenty = 8,
    Monopoly = 9,
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Copy, Clone)]
pub enum Action {
    BuildRoad(RoadId),                // 72
    BuildStructure(Building, NodeId), // 54 * 2

    DiscardResource(Resource),
    MoveRobber(TileId),
    StealCard(PlayerId),

    // BuyDevCard,
    // UseDevCard(DevCard),

    // Trades *not* between the player.
    MaritimeTrade((Resource, u8), Resource),
    EndTurn,
}
