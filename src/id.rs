#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy, Debug)]
pub struct Id<const COUNT: u8>(u8);

impl<const COUNT: u8> Id<COUNT> {
    pub const fn new(v: u8) -> Self {
        debug_assert!(v < COUNT);
        Self(v)
    }
}

impl<const COUNT: u8> From<Id<COUNT>> for u8 {
    fn from(v: Id<COUNT>) -> u8 {
        v.0
    }
}

impl<const COUNT: u8> From<Id<COUNT>> for usize {
    fn from(v: Id<COUNT>) -> usize {
        v.0 as usize
    }
}

impl<const COUNT: u8> From<u8> for Id<COUNT> {
    fn from(v: u8) -> Self {
        Self::new(v)
    }
}

impl<const COUNT: u8> From<usize> for Id<COUNT> {
    fn from(v: usize) -> Self {
        Self::new(v as u8)
    }
}

pub type RoadId = Id<72>;
pub type NodeId = Id<54>;
pub type PlayerId = Id<4>;
