use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rcatan::{
    Action, Building, DevCard, Engine, FullState, Phase, PlayerId, Resource,
    State, TileType,
};

// Action space layout (300 total):
//   [  0,  72)  BuildRoad(road_id)
//   [ 72, 126)  BuildStructure(Settlement, node_id)
//   [126, 180)  BuildStructure(City, node_id)
//   [180, 185)  DiscardResource(resource)
//   [185, 204)  MoveRobber(tile_id)
//   [204, 208)  StealCard(player_id)
//   [208, 283)  MaritimeTrade  (give_resource * 15 + amount_idx * 5 + want_resource)
//               amount_idx: 0 = give 2, 1 = give 3, 2 = give 4
//   283         EndTurn
//   284         BuyDevCard
//   [285, 290)  UseDevCard(dev_card)               (DevCard::ALL order)
//   [290, 295)  YearOfPlentyGetResource(resource)
//   [295, 300)  MonopolyGetResource(resource)
const NUM_ACTIONS: usize = 300;

fn dev_card_index(d: DevCard) -> usize {
    d as usize - 5
}

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
        Action::BuyDevCard => 284,
        Action::UseDevCard(d) => 285 + dev_card_index(*d),
        Action::YearOfPlentyGetResource(r) => 290 + *r as usize,
        Action::MonopolyGetResource(r) => 295 + *r as usize,
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
        284 => Action::BuyDevCard,
        285..290 => Action::UseDevCard(DevCard::from_array_index(index - 285)),
        290..295 => Action::YearOfPlentyGetResource(Resource::ALL[index - 290]),
        295..300 => Action::MonopolyGetResource(Resource::ALL[index - 295]),
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
    batch_size: usize,
}

#[pymethods]
impl PyEngine {
    #[new]
    fn new(batch_size: usize, seed: u64) -> Self {
        Self {
            batch_size,
            inner: Engine::new(batch_size, seed),
        }
    }

    #[classattr]
    const NUM_ACTIONS: usize = NUM_ACTIONS;

    fn state<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<u32>>)> {
        let current_players: Vec<u32> = self
            .inner
            .state()
            .iter()
            .map(|s| u8::from(s.current_player) as u32)
            .collect();

        let states: Vec<f32> = self
            .inner
            .state()
            .into_iter()
            .map(state_to_vec)
            .flatten()
            .collect();

        let states_array =
            Array2::from_shape_vec((self.batch_size, STATE_VEC_LEN), states)
                .map_err(|err| {
                    pyo3::exceptions::PyValueError::new_err(err.to_string())
                })?;

        Ok((
            states_array.into_pyarray(py),
            current_players.into_pyarray(py),
        ))
    }

    /// Returns a boolean action mask per game: list[list[bool]] of shape (batch_size, 284).
    fn get_action_mask<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray2<bool>>> {
        let masks: Vec<bool> = self
            .inner
            .get_actions()
            .into_iter()
            .map(|actions| {
                let mut mask = vec![false; NUM_ACTIONS];
                for action in &actions {
                    mask[action_to_index(action)] = true;
                }
                mask
            })
            .flatten()
            .collect();

        let masks_array =
            Array2::from_shape_vec((self.batch_size, NUM_ACTIONS), masks)
                .map_err(|err| {
                    pyo3::exceptions::PyValueError::new_err(err.to_string())
                })?;

        Ok(masks_array.into_pyarray(py))
    }

    /// Advance each game by one action. Takes a list of action indices (one per game).
    /// Returns:
    ///   rewards:      ndarray (batch_size, 4) f32
    ///   ended_mask:   ndarray (batch_size,)   bool — True if the game ended this step
    ///   final_vps:    ndarray (batch_size, 4) u8  — final VPs for ended games, 0 elsewhere
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: Vec<usize>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyArray2<u8>>,
    )> {
        let actions: Vec<Action> =
            actions.into_iter().map(index_to_action).collect();
        let (rewards, game_ended_mask) = self.inner.run_actions(actions);

        let rewards_flat: Vec<f32> =
            rewards.into_iter().flat_map(|r| r.reward).collect();
        let rewards_arr =
            Array2::from_shape_vec((self.batch_size, 4), rewards_flat)
                .map_err(|err| {
                    pyo3::exceptions::PyValueError::new_err(err.to_string())
                })?;

        let ended: Vec<bool> =
            game_ended_mask.iter().map(|v| v.is_some()).collect();

        let mut vps_flat: Vec<u8> = Vec::with_capacity(self.batch_size * 4);
        for v in &game_ended_mask {
            match v {
                Some(arr) => vps_flat.extend_from_slice(arr),
                None => vps_flat.extend_from_slice(&[0u8; 4]),
            }
        }
        let vps_arr = Array2::from_shape_vec((self.batch_size, 4), vps_flat)
            .map_err(|err| {
                pyo3::exceptions::PyValueError::new_err(err.to_string())
            })?;

        Ok((
            rewards_arr.into_pyarray(py),
            ended.into_pyarray(py),
            vps_arr.into_pyarray(py),
        ))
    }

    /// Returns each game's victory points as ndarray (batch_size, 4) u8.
    fn vps<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<u8>>> {
        let flat: Vec<u8> = self.inner.vps().into_iter().flatten().collect();
        Array2::from_shape_vec((self.batch_size, 4), flat)
            .map(|a| a.into_pyarray(py))
            .map_err(|err| {
                pyo3::exceptions::PyValueError::new_err(err.to_string())
            })
    }

    /// Rich snapshot of game `idx` as a Python dict, intended for rendering
    /// (UI, REST responses). Shape:
    ///   {
    ///     "current_player": int,
    ///     "phase": str,
    ///     "phase_data": dict | None,
    ///     "last_dice_roll": int | None,
    ///     "winner": int | None,
    ///     "longest_road": {"player": int, "length": int} | None,
    ///     "tiles": list of {tile_id, resource, dice, has_robber},
    ///     "roads": list of {road_id, player},
    ///     "structures": list of {node_id, building, player},
    ///     "players": list of {id, vp, cards: {brick, lumber, ore, grain, wool}},
    ///   }
    fn full_state<'py>(
        &self,
        py: Python<'py>,
        idx: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        full_state_to_dict(py, &self.inner.full_state(idx))
    }
}

fn resource_name(r: Resource) -> &'static str {
    match r {
        Resource::Brick => "brick",
        Resource::Lumber => "lumber",
        Resource::Ore => "ore",
        Resource::Grain => "grain",
        Resource::Wool => "wool",
    }
}

fn tile_resource_name(t: TileType) -> &'static str {
    match t {
        TileType::Hills => "brick",
        TileType::Forest => "lumber",
        TileType::Mountains => "ore",
        TileType::Fields => "grain",
        TileType::Pasture => "wool",
        TileType::Desert => "desert",
    }
}

fn full_state_to_dict<'py>(
    py: Python<'py>,
    fs: &FullState,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("current_player", u8::from(fs.current_player))?;
    d.set_item("last_dice_roll", fs.last_dice_roll)?;
    d.set_item("winner", fs.winner.map(u8::from))?;

    let (phase_name, phase_data): (&str, Option<Bound<'py, PyDict>>) =
        match fs.phase {
            Phase::InitialSettlementPlacement => {
                ("initial_settlement_placement", None)
            }
            Phase::InitialRoadPlacement => ("initial_road_placement", None),
            Phase::Discard { target_counts } => {
                let pd = PyDict::new(py);
                pd.set_item("target_counts", target_counts.to_vec())?;
                ("discard", Some(pd))
            }
            Phase::MoveRobber => ("move_robber", None),
            Phase::StealResource => ("steal_resource", None),
            Phase::ContinueAction => ("continue_action", None),
            Phase::GameEnded => ("game_ended", None),
        };
    d.set_item("phase", phase_name)?;
    d.set_item("phase_data", phase_data)?;

    let longest_road = match fs.longest_road {
        Some((p, len)) => {
            let lr = PyDict::new(py);
            lr.set_item("player", u8::from(p))?;
            lr.set_item("length", len)?;
            Some(lr)
        }
        None => None,
    };
    d.set_item("longest_road", longest_road)?;

    // Tiles: walk dice_to_tiles (dice 2..=12) and emit each tile with its
    // canonical tile_id. Desert tiles live at the dice==7 slot.
    let mut tiles: Vec<Bound<'py, PyDict>> = Vec::with_capacity(19);
    for (dice_offset, bucket) in fs.board.dice_to_tiles.iter().enumerate() {
        let dice = (dice_offset as u8) + 2;
        for tile in bucket {
            let t = PyDict::new(py);
            t.set_item("tile_id", u8::from(tile.tile_index))?;
            t.set_item("resource", tile_resource_name(tile.tile_type))?;
            t.set_item("dice", dice)?;
            t.set_item("has_robber", tile.has_robber)?;
            tiles.push(t);
        }
    }
    tiles.sort_by_key(|t| {
        t.get_item("tile_id")
            .ok()
            .and_then(|x| x.and_then(|v| v.extract::<u8>().ok()))
            .unwrap_or(0)
    });
    d.set_item("tiles", tiles)?;

    let mut roads: Vec<Bound<'py, PyDict>> = Vec::new();
    for (i, road) in fs.board.roads.iter().enumerate() {
        if let Some(p) = road.get() {
            let r = PyDict::new(py);
            r.set_item("road_id", i as u8)?;
            r.set_item("player", u8::from(p))?;
            roads.push(r);
        }
    }
    d.set_item("roads", roads)?;

    let mut structures: Vec<Bound<'py, PyDict>> = Vec::new();
    for (i, s) in fs.board.structures.iter().enumerate() {
        if let Some((building, p)) = s.get() {
            let st = PyDict::new(py);
            st.set_item("node_id", i as u8)?;
            st.set_item(
                "building",
                match building {
                    Building::Settlement => "settlement",
                    Building::City => "city",
                },
            )?;
            st.set_item("player", u8::from(p))?;
            structures.push(st);
        }
    }
    d.set_item("structures", structures)?;

    let mut players: Vec<Bound<'py, PyDict>> = Vec::with_capacity(4);
    for i in 0..4 {
        let p = PyDict::new(py);
        p.set_item("id", i as u8)?;
        p.set_item("vp", fs.vps[i])?;
        let cards = PyDict::new(py);
        for r in Resource::ALL {
            cards.set_item(resource_name(r), fs.player_cards[i][r as usize])?;
        }
        p.set_item("cards", cards)?;
        players.push(p);
    }
    d.set_item("players", players)?;

    Ok(d)
}

#[pymodule]
fn catan_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEngine>()?;
    m.add("NUM_ACTIONS", NUM_ACTIONS)?;
    Ok(())
}
