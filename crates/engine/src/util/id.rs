use std::marker::PhantomData;

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy, Debug)]
pub struct Id<const COUNT: u8, Tag = ()>(u8, PhantomData<Tag>);

impl<const COUNT: u8, Tag: FromRawU8> Id<COUNT, Tag> {
    pub const fn new(v: u8) -> Self {
        debug_assert!(v < COUNT);
        Self(v, PhantomData)
    }
}

impl<const COUNT: u8, Tag> From<Id<COUNT, Tag>> for u8 {
    fn from(v: Id<COUNT, Tag>) -> u8 {
        v.0
    }
}

impl<const COUNT: u8, Tag> From<Id<COUNT, Tag>> for usize {
    fn from(v: Id<COUNT, Tag>) -> usize {
        v.0 as usize
    }
}

// Converting the u8/usize to Id is only allowed for absolute tags.
impl<const COUNT: u8, Tag: FromRawU8> From<u8> for Id<COUNT, Tag> {
    fn from(v: u8) -> Self {
        Self::new(v)
    }
}

impl<const COUNT: u8, Tag: FromRawU8> From<usize> for Id<COUNT, Tag> {
    fn from(v: usize) -> Self {
        Self::new(v as u8)
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy, Debug)]
pub struct Absolute;

pub trait FromRawU8 {}
impl FromRawU8 for () {}
impl FromRawU8 for Absolute {}

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy, Debug)]
pub struct Relative;

// "Absolute" ids that can be directly converted from u8.
pub type RoadId = Id<72>;
pub type NodeId = Id<54>;
pub type TileId = Id<19>;
pub type PlayerId = Id<4, Absolute>;

// Player id **relative** to the current player that does the action.
pub type RelativePlayerId = Id<4, Relative>;

impl RelativePlayerId {
    /// A relative id decoded from an action index (1 = next player, ...).
    pub fn from_index(v: u8) -> Self {
        debug_assert!(v < 4);
        Self(v, PhantomData)
    }

    pub fn from(current_player: PlayerId, target_player: PlayerId) -> Self {
        if target_player.0 > current_player.0 {
            Self(target_player.0 - current_player.0, PhantomData)
        } else {
            Self(target_player.0 + 4 - current_player.0, PhantomData)
        }
    }

    pub fn to(&self, current_player: PlayerId) -> usize {
        if self.0 + current_player.0 < 4 {
            usize::from(self.0 + current_player.0)
        } else {
            usize::from(self.0 + current_player.0 - 4)
        }
    }
}
