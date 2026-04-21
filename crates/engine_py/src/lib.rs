use pyo3::prelude::*;
use rcatan::{Action, Building, Engine, PlayerId, Resource, State};

// Action space layout (284 total):
//   [  0,  72)  BuildRoad(road_id)
//   [ 72, 126)  BuildStructure(Settlement, node_id)
//   [126, 180)  BuildStructure(City, node_id)
//   [180, 185)  DiscardResource(resource)
//   [185, 204)  MoveRobber(tile_id)
//   [204, 208)  StealCard(player_id)
//   [208, 283)  MaritimeTrade  (give_resource * 15 + amount_idx * 5 + want_resource)
//               amount_idx: 0 = give 2, 1 = give 3, 2 = give 4
//   283         EndTurn
const NUM_ACTIONS: usize = 284;

fn action_to_index(action: &Action) -> usize {
    match action {
        Action::BuildRoad(id) => u8::from(*id) as usize,
        Action::BuildStructure(Building::Settlement, id) => {
            72 + u8::from(*id) as usize
        }
        Action::BuildStructure(Building::City, id) => {
            126 + u8::from(*id) as usize
        }
        Action::DiscardResource(r) => 180 + *r as usize,
        Action::MoveRobber(id) => 185 + u8::from(*id) as usize,
        Action::StealCard(id) => 204 + u8::from(*id) as usize,
        Action::MaritimeTrade((give_r, give_amt), want_r) => {
            let amt_idx = (*give_amt - 2) as usize;
            208 + (*give_r as usize) * 15 + amt_idx * 5 + (*want_r as usize)
        }
        Action::EndTurn => 283,
    }
}

fn index_to_action(index: usize) -> Action {
    match index {
        0..72 => Action::BuildRoad((index as u8).into()),
        72..126 => Action::BuildStructure(
            Building::Settlement,
            ((index - 72) as u8).into(),
        ),
        126..180 => {
            Action::BuildStructure(Building::City, ((index - 126) as u8).into())
        }
        180..185 => Action::DiscardResource(Resource::ALL[index - 180]),
        185..204 => Action::MoveRobber(((index - 185) as u8).into()),
        204..208 => Action::StealCard(((index - 204) as u8).into()),
        208..283 => {
            let offset = index - 208;
            let give_r = Resource::ALL[offset / 15];
            let amt_idx = (offset % 15) / 5;
            let want_r = Resource::ALL[offset % 5];
            Action::MaritimeTrade((give_r, (amt_idx + 2) as u8), want_r)
        }
        283 => Action::EndTurn,
        _ => panic!("invalid action index: {index}"),
    }
}

// State vector layout (732 floats), encoded from the current player's perspective.
// "Rotated player index" = (player_id - current_player + 4) % 4, so index 0 is
// always the current player, index 1 is the next player to act, etc.
//
//   [  0,  288)  Road ownership:        72 * one-hot[4]
//   [288,  504)  Settlement ownership:  54 * one-hot[4]
//   [504,  720)  City ownership:        54 * one-hot[4]
//   [720,  725)  Current player resource counts per resource (count / 10, clipped at 1)
//   [725,  728)  Other players' total resource counts (count / 20, clipped at 1),
//                rotated so index 0 is the next player after current
//   [728,  732)  Longest road holder: one-hot[4] (all zero if nobody holds it)
const STATE_VEC_LEN: usize = 732;

fn state_to_vec(state: State) -> Vec<f32> {
    let current = u8::from(state.current_player);
    let rotate = |p: PlayerId| ((u8::from(p) + 4 - current) % 4) as usize;

    let mut v = Vec::with_capacity(STATE_VEC_LEN);

    // Roads: 72 * one-hot[4].
    for road in &state.board.roads {
        let start = v.len();
        v.extend([0.0f32; 4]);
        if let Some(p) = road.get() {
            v[start + rotate(p)] = 1.0;
        }
    }

    // Settlements: 54 * one-hot[4].
    for structure in &state.board.structures {
        let start = v.len();
        v.extend([0.0f32; 4]);
        if let Some((Building::Settlement, p)) = structure.get() {
            v[start + rotate(p)] = 1.0;
        }
    }

    // Cities: 54 * one-hot[4].
    for structure in &state.board.structures {
        let start = v.len();
        v.extend([0.0f32; 4]);
        if let Some((Building::City, p)) = structure.get() {
            v[start + rotate(p)] = 1.0;
        }
    }

    // Current player's resource breakdown.
    let current_idx = current as usize;
    for i in 0..5 {
        let count = state.player_cards[current_idx][i] as f32;
        v.push((count / 10.0).min(1.0));
    }

    // Other players' total resources, rotated (next player first).
    for offset in 1..4 {
        let p_idx = (current_idx + offset) % 4;
        let total: u8 = state.player_cards[p_idx].iter().sum();
        v.push((total as f32 / 20.0).min(1.0));
    }

    // Longest road holder: one-hot[4].
    let start = v.len();
    v.extend([0.0f32; 4]);
    if let Some((p, _)) = state.longest_road {
        v[start + rotate(p)] = 1.0;
    }

    debug_assert_eq!(v.len(), STATE_VEC_LEN);
    v
}

#[pyclass(name = "Engine")]
struct PyEngine {
    inner: Engine,
}

#[pymethods]
impl PyEngine {
    #[new]
    fn new(batch_size: usize, seed: u64) -> Self {
        Self {
            inner: Engine::new(batch_size, seed),
        }
    }

    #[classattr]
    const NUM_ACTIONS: usize = NUM_ACTIONS;

    fn state(&self) -> (Vec<Vec<f32>>, Vec<u32>) {
        let current_players = self
            .inner
            .state()
            .iter()
            .map(|s| u8::from(s.current_player) as u32)
            .collect();

        let states = self.inner.state().into_iter().map(state_to_vec).collect();

        (states, current_players)
    }

    /// Returns a boolean action mask per game: list[list[bool]] of shape (batch_size, 284).
    fn get_action_mask(&mut self) -> Vec<Vec<bool>> {
        self.inner
            .get_actions()
            .into_iter()
            .map(|actions| {
                let mut mask = vec![false; NUM_ACTIONS];
                for action in &actions {
                    mask[action_to_index(action)] = true;
                }
                mask
            })
            .collect()
    }

    /// Advance each game by one action. Takes a list of action indices (one per game).
    /// Returns rewards: list[tuple[f32, f32, f32, f32]] of shape (batch_size, 4).
    fn step(
        &mut self,
        actions: Vec<usize>,
    ) -> (Vec<(f32, f32, f32, f32)>, Vec<bool>) {
        let actions: Vec<Action> =
            actions.into_iter().map(index_to_action).collect();
        let (rewards, game_ended_mask) = self.inner.run_actions(actions);

        let rewards = rewards
            .into_iter()
            .map(|r| (r.reward[0], r.reward[1], r.reward[2], r.reward[3]))
            .collect();

        (rewards, game_ended_mask)
    }
}

#[pymodule]
fn catan_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEngine>()?;
    m.add("NUM_ACTIONS", NUM_ACTIONS)?;
    Ok(())
}
