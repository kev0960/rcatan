use arrayvec::ArrayVec;
use std::cmp::max;

use crate::PlayerId;

#[derive(PartialEq, Eq, PartialOrd, Ord, Copy, Clone, Debug)]
pub struct VictoryPoint {
    vp: u16,
}

impl VictoryPoint {
    pub fn new(vps: ArrayVec<u8, 4>) -> Self {
        let vp = vps
            .into_iter()
            .enumerate()
            // By default, all players have 2 VPs. This is to prevent initial
            // settlement set ups being highly rewarded.
            .fold(0, |acc, (i, v)| acc | (v.max(2) as u16) << (4 * i));
        Self { vp }
    }

    pub fn get_vp(&self, player_id: PlayerId) -> u8 {
        ((self.vp >> (4 * usize::from(player_id))) & 0b1111) as u8
    }
}

pub struct Reward {
    pub reward: ArrayVec<f32, 4>,
}

impl Reward {
    pub fn from_win(player_id: PlayerId) -> Self {
        // When the game ends, every losers get -1 reward, and the winner gets
        // 1.
        let mut reward = ArrayVec::<f32, 4>::from([-1.0; 4]);
        reward[usize::from(player_id)] = 1.0;

        Self { reward }
    }

    pub fn from_vp_change(
        prev_vps: VictoryPoint,
        updated_vps: VictoryPoint,
    ) -> Self {
        if prev_vps == updated_vps {
            return Self {
                reward: ArrayVec::<f32, 4>::from([0.0; 4]),
            };
        }

        let mut reward = ArrayVec::<f32, 4>::from([0.0; 4]);

        for i in 0..4 {
            let prev_vp = prev_vps.get_vp(PlayerId::from(i));
            let updated_vp = updated_vps.get_vp(PlayerId::from(i));

            if prev_vp != updated_vp {
                // 1 VP change is worth 0.1.
                reward[i] = (updated_vp as f32 - prev_vp as f32) * 0.1;
            }
        }

        Self { reward }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(id: usize) -> PlayerId {
        PlayerId::new(id as u8)
    }

    // ---- VictoryPoint tests ----

    #[test]
    fn vp_new_clamps_below_2() {
        let vp = VictoryPoint::new(ArrayVec::from([0, 1, 0, 1]));
        for i in 0..4 {
            assert_eq!(vp.get_vp(p(i)), 2);
        }
    }

    #[test]
    fn vp_new_stores_each_player() {
        let vp = VictoryPoint::new(ArrayVec::from([2, 5, 9, 3]));
        assert_eq!(vp.get_vp(p(0)), 2);
        assert_eq!(vp.get_vp(p(1)), 5);
        assert_eq!(vp.get_vp(p(2)), 9);
        assert_eq!(vp.get_vp(p(3)), 3);
    }

    #[test]
    fn vp_new_max_value_per_player() {
        // 4 bits per player => max 15
        let vp = VictoryPoint::new(ArrayVec::from([15, 15, 15, 15]));
        for i in 0..4 {
            assert_eq!(vp.get_vp(p(i)), 15);
        }
    }

    #[test]
    fn vp_equality() {
        let a = VictoryPoint::new(ArrayVec::from([2, 3, 4, 5]));
        let b = VictoryPoint::new(ArrayVec::from([2, 3, 4, 5]));
        let c = VictoryPoint::new(ArrayVec::from([2, 3, 4, 6]));
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    // ---- Reward::from_win tests ----

    #[test]
    fn from_win_gives_winner_positive() {
        let r = Reward::from_win(p(2));
        assert_eq!(r.reward[2], 1.0);
        assert_eq!(r.reward[0], -1.0);
        assert_eq!(r.reward[1], -1.0);
        assert_eq!(r.reward[3], -1.0);
    }

    // ---- Reward::from_vp_change tests ----

    #[test]
    fn from_vp_change_no_change_returns_zeros() {
        let vp = VictoryPoint::new(ArrayVec::from([3, 3, 3, 3]));
        let r = Reward::from_vp_change(vp, vp);
        assert_eq!(r.reward.as_slice(), &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn from_vp_change_increase() {
        let prev = VictoryPoint::new(ArrayVec::from([2, 3, 2, 2]));
        let next = VictoryPoint::new(ArrayVec::from([2, 4, 2, 2]));
        let r = Reward::from_vp_change(prev, next);
        assert_eq!(r.reward[1], 0.1);
        assert_eq!(r.reward[0], 0.0);
    }

    #[test]
    fn from_vp_change_decrease() {
        // Player 0 loses longest road (VP goes from 4 to 2)
        let prev = VictoryPoint::new(ArrayVec::from([4, 2, 2, 2]));
        let next = VictoryPoint::new(ArrayVec::from([2, 4, 2, 2]));
        let r = Reward::from_vp_change(prev, next);
        assert_eq!(r.reward[0], -0.2);
        assert_eq!(r.reward[1], 0.2);
        assert_eq!(r.reward[2], 0.0);
        assert_eq!(r.reward[3], 0.0);
    }
}
