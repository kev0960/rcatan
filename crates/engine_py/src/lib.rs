use pyo3::prelude::*;
use rcatan::{Action, Building, Engine, Resource};

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
        Action::BuildStructure(Building::Settlement, id) => 72 + u8::from(*id) as usize,
        Action::BuildStructure(Building::City, id) => 126 + u8::from(*id) as usize,
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
        72..126 => Action::BuildStructure(Building::Settlement, ((index - 72) as u8).into()),
        126..180 => Action::BuildStructure(Building::City, ((index - 126) as u8).into()),
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
    fn step(&mut self, actions: Vec<usize>) {
        let actions: Vec<Action> = actions.into_iter().map(index_to_action).collect();
        self.inner.run_actions(actions);
    }
}

#[pymodule]
fn catan_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEngine>()?;
    m.add("NUM_ACTIONS", NUM_ACTIONS)?;
    Ok(())
}
