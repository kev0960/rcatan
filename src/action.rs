use crate::id::{NodeId, RoadId};

#[derive(PartialEq)]
pub enum Building {
    Settlement,
    City,
}

#[repr(u8)]
pub enum Resource {
    Brick = 0,
    Lumber = 1,
    Ore = 2,
    Grain = 3,
    Wool = 4,
}

#[repr(u8)]
pub enum DevCard {
    Knight = 5,
    VictoryPoint = 6,
    RoadBuilding = 7,
    YearOfPlenty = 8,
    Monopoly = 9,
}

pub enum Action {
    BuildRoad(RoadId),
    BuildStructure(Building, NodeId),
    BuyDevCard,
    UseDevCard(DevCard),

    // Trades *not* between the player.
    MaritimeTrade(Vec<Resource>, Resource),
}
