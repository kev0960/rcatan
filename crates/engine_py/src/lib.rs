use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rcatan::{
    Action, Building, DevCard, Engine, FullState, Phase, PlayerId, Resource,
    State, TileType, port_at_node,
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

// State vector layout (1976 floats), encoded from the current player's perspective.
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
//   [732, 1920)  Node features: 54 * [22]. Per node:
//                  [0, 5)   pips per resource from adjacent tiles / 15, in
//                           Resource::ALL order (pips = ways two dice roll
//                           the tile's number: 2 and 12 -> 1, 6 and 8 -> 5)
//                  [5, 15)  adjacent tiles per dice number / 2, in order
//                           2, 3, 4, 5, 6, 8, 9, 10, 11, 12
//                  15       1 if the robber is on an adjacent producing tile
//                  [16, 22) port at this node: one-hot over brick, lumber,
//                           ore, grain, wool (2:1) and 3:1; all zero if none
//   [1920, 1970) Current player's cards received per (resource, dice number):
//                5 * [10], resource-major in Resource::ALL order, same dice
//                order as above (count / 4, clipped at 1). Respects the robber,
//                so it shows what the player is paid in on each roll.
//   [1970, 1976) Ports the current player owns: multi-hot, same port order.
//   [1976, 1982) Phase: one-hot over initial settlement, initial road,
//                discard, move robber, steal, normal turn
//   [1982, 1985) Dev card being resolved: one-hot over road building, year of
//                plenty, monopoly (all zero if none)
//   1985         Cards the current player still has to discard / 10, clipped
//   [1986, 1991) Current player's playable dev cards / 5, clipped, in
//                DevCard::ALL order (knight, VP, road building, year of
//                plenty, monopoly)
//   [1991, 1996) Current player's dev cards bought this turn / 2, clipped
//   1996         1 if the current player already played a dev card this turn
//   [1997, 2001) VPs / 10, rotated. Index 0 is the current player's full VP
//                (including its VP cards); the others are public VPs only
//                (buildings, longest road, largest army)
//   [2001, 2005) Knights played / 7, clipped, rotated
//   [2005, 2009) Largest army holder: one-hot[4], rotated
//   [2009, 2013) Longest road length / 15, rotated
//   [2013, 2016) Other players' dev card counts (held + bought this turn) / 5,
//                clipped, rotated so index 0 is the next player after current
const STATE_VEC_LEN: usize = 2016;

// Pips for dice numbers 2..=12: the number of two-dice combinations that roll
// each number.
const PIPS: [f32; 11] = [1., 2., 3., 4., 5., 6., 5., 4., 3., 2., 1.];

// Slot of a dice number in the 10-wide per-number features (7 has none).
fn dice_slot(dice: u8) -> usize {
    if dice < 7 { dice as usize - 2 } else { dice as usize - 3 }
}

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

    // Node features: 54 * [22]. Pips per resource say what a node produces
    // and how often; the per-number counts say which rolls pay out, so the
    // agent can tell spreading its numbers from stacking them.
    for node in 0..54u8 {
        let start = v.len();
        v.extend([0.0f32; 22]);
        if let Some(port) = port_at_node(node.into()) {
            v[start + 16 + port as usize] = 1.0;
        }
        for (dice, resource, has_robber) in
            state.board.get_tiles_near_node_id(node.into())
        {
            v[start + resource as usize] += PIPS[dice as usize - 2] / 15.0;
            v[start + 5 + dice_slot(dice)] += 0.5;
            if has_robber {
                v[start + 15] = 1.0;
            }
        }
    }

    // Current player's cards received per (resource, dice number).
    let start = v.len();
    v.extend([0.0f32; 50]);
    for dice in (2..=12u8).filter(|&d| d != 7) {
        for (player, resource, cards) in state.board.get_matching_resources(dice)
        {
            if player == state.current_player {
                v[start + resource as usize * 10 + dice_slot(dice)] +=
                    cards as f32 / 4.0;
            }
        }
    }
    for x in &mut v[start..] {
        *x = x.min(1.0);
    }

    // Ports the current player owns.
    let start = v.len();
    v.extend([0.0f32; 6]);
    for (node, structure) in state.board.structures.iter().enumerate() {
        if let Some((_, p)) = structure.get()
            && p == state.current_player
            && let Some(port) = port_at_node((node as u8).into())
        {
            v[start + port as usize] = 1.0;
        }
    }

    // Phase: one-hot[6]. All zero once the game has ended.
    let phase_slot = match state.phase {
        Phase::InitialSettlementPlacement => Some(0),
        Phase::InitialRoadPlacement => Some(1),
        Phase::Discard { .. } => Some(2),
        Phase::MoveRobber => Some(3),
        Phase::StealResource => Some(4),
        Phase::ContinueAction => Some(5),
        Phase::GameEnded => None,
    };
    let start = v.len();
    v.extend([0.0f32; 6]);
    if let Some(slot) = phase_slot {
        v[start + slot] = 1.0;
    }

    // Dev card being resolved: one-hot[3].
    let start = v.len();
    v.extend([0.0f32; 3]);
    match state.pending_dev_card {
        Some(DevCard::RoadBuilding) => v[start] = 1.0,
        Some(DevCard::YearOfPlenty) => v[start + 1] = 1.0,
        Some(DevCard::Monopoly) => v[start + 2] = 1.0,
        _ => {}
    }

    // Cards the current player still has to discard.
    let to_discard = match state.phase {
        Phase::Discard { target_counts } => {
            let hand: u8 = state.player_cards[current_idx].iter().sum();
            hand.saturating_sub(target_counts[current_idx])
        }
        _ => 0,
    };
    v.push((to_discard as f32 / 10.0).min(1.0));

    // Current player's dev cards: playable, then bought this turn.
    for &count in &state.dev_cards[current_idx] {
        v.push((count as f32 / 5.0).min(1.0));
    }
    for &count in &state.dev_cards_bought[current_idx] {
        v.push((count as f32 / 2.0).min(1.0));
    }
    v.push(if state.used_dev_card_this_turn { 1.0 } else { 0.0 });

    // VPs, rotated. Only the current player's own VP cards are visible.
    for offset in 0..4 {
        let p_idx = (current_idx + offset) % 4;
        let vp = match offset {
            0 => state.vps[p_idx],
            _ => state.public_vps[p_idx],
        };
        v.push(vp as f32 / 10.0);
    }

    // Knights played, rotated.
    for offset in 0..4 {
        let p_idx = (current_idx + offset) % 4;
        v.push((state.knights_played[p_idx] as f32 / 7.0).min(1.0));
    }

    // Largest army holder: one-hot[4].
    let start = v.len();
    v.extend([0.0f32; 4]);
    if let Some((p, _)) = state.largest_army {
        v[start + rotate(p)] = 1.0;
    }

    // Longest road length per player, rotated.
    for offset in 0..4 {
        let p_idx = (current_idx + offset) % 4;
        v.push(state.road_lengths[p_idx] as f32 / 15.0);
    }

    // Other players' dev card counts, rotated (next player first).
    for offset in 1..4 {
        let p_idx = (current_idx + offset) % 4;
        let held: u8 = state.dev_cards[p_idx].iter().sum();
        let bought: u8 = state.dev_cards_bought[p_idx].iter().sum();
        v.push(((held + bought) as f32 / 5.0).min(1.0));
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

    // Returns a boolean action mask per game of shape (batch_size,
    // NUM_ACTIONS). Idempotent — calling twice in a row returns the same mask
    // without changing engine state.
    fn get_action_mask<'py>(
        &self,
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
                // pyo3 maps Vec<u8> to Python `bytes`, which json.dumps
                // can't serialize. Widen to u32 so it lands as list[int].
                let counts: Vec<u32> =
                    target_counts.iter().map(|&c| c as u32).collect();
                pd.set_item("target_counts", counts)?;
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
