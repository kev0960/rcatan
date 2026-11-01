import pytest
import torch
from catan_engine import Engine

from train.network_graph import (
    CITIES,
    NODE_FEATURES,
    NUM_NODES,
    NUM_ROADS,
    ROADS,
    SETTLEMENTS,
    STATE_LEN,
    GraphActor,
    GraphCritic,
)
from train.ppo import ACTOR_INPUT_SIZE, PPO, load_actor


@pytest.fixture
def endpoints() -> torch.Tensor:
    return torch.from_numpy(Engine.road_endpoints())


def _real_states(batch_size: int, steps: int = 40) -> torch.Tensor:
    """States from a few random moves into real games, so features are
    realistic (buildings, roads, income)."""
    engine = Engine(batch_size=batch_size, seed=0)
    gen = torch.Generator().manual_seed(0)
    for _ in range(steps):
        masks = torch.from_numpy(engine.get_action_mask())
        actions = torch.multinomial(masks.float(), 1, generator=gen).squeeze(-1)
        engine.step(actions.tolist())
    states, _ = engine.state()
    return torch.from_numpy(states)


def test_state_len_matches_engine():
    assert STATE_LEN == ACTOR_INPUT_SIZE
    assert _real_states(1, steps=0).shape == (1, STATE_LEN)


def test_output_shapes(endpoints):
    states = _real_states(5)
    assert GraphActor(endpoints)(states).shape == (5, 300)
    assert GraphCritic(endpoints)(states).shape == (5, 1)


def test_heads_land_at_their_action_indices(endpoints):
    actor = GraphActor(endpoints)
    with torch.no_grad():
        for head, bias in [
            (actor.road_head[-1], 3.0),
            (actor.pooled_head[-1], 4.0),
        ]:
            head.weight.zero_()
            head.bias.fill_(bias)
        actor.node_head.weight.zero_()
        actor.node_head.bias.copy_(torch.tensor([1.0, 2.0]))

        logits = actor(_real_states(2))

    assert torch.all(logits[:, 0:72] == 3.0)  # BuildRoad
    assert torch.all(logits[:, 72:126] == 1.0)  # BuildStructure(Settlement)
    assert torch.all(logits[:, 126:180] == 2.0)  # BuildStructure(City)
    assert torch.all(logits[:, 180:300] == 4.0)  # everything else


def _permute_state(state, node_perm, road_perm):
    """Relabel nodes and roads: old node n becomes node_perm[n], old road r
    becomes road_perm[r]."""
    out = state.clone()
    b = state.shape[0]
    for block, count, width, perm in [
        (ROADS, NUM_ROADS, 4, road_perm),
        (SETTLEMENTS, NUM_NODES, 4, node_perm),
        (CITIES, NUM_NODES, 4, node_perm),
        (NODE_FEATURES, NUM_NODES, 22, node_perm),
    ]:
        rows = state[:, block].view(b, count, width)
        moved = torch.empty_like(rows)
        moved[:, perm] = rows
        out[:, block] = moved.reshape(b, -1)
    return out


def test_relabeling_the_board_relabels_the_outputs(endpoints):
    """The network has no notion of node or road ids, only of the graph. So
    relabeling nodes and roads (and swapping which end of a road is listed
    first) must relabel the outputs the same way and change nothing else."""
    gen = torch.Generator().manual_seed(1)
    node_perm = torch.randperm(NUM_NODES, generator=gen)
    road_perm = torch.randperm(NUM_ROADS, generator=gen)
    flip = torch.rand(NUM_ROADS, generator=gen) < 0.5

    relabeled = torch.empty_like(endpoints)
    relabeled[road_perm] = node_perm[endpoints]
    relabeled[road_perm[flip]] = relabeled[road_perm[flip]].flip(-1)

    actor = GraphActor(endpoints).eval()
    actor_relabeled = GraphActor(relabeled).eval()
    actor_relabeled.load_state_dict(actor.state_dict())
    critic = GraphCritic(endpoints).eval()
    critic_relabeled = GraphCritic(relabeled).eval()
    critic_relabeled.load_state_dict(critic.state_dict())

    states = _real_states(3)
    states_relabeled = _permute_state(states, node_perm, road_perm)

    with torch.no_grad():
        out = actor(states)
        out_relabeled = actor_relabeled(states_relabeled)
        value = critic(states)
        value_relabeled = critic_relabeled(states_relabeled)

    atol = 1e-4
    torch.testing.assert_close(
        out_relabeled[:, road_perm], out[:, 0:72], atol=atol, rtol=0
    )
    torch.testing.assert_close(
        out_relabeled[:, 72 + node_perm], out[:, 72:126], atol=atol, rtol=0
    )
    torch.testing.assert_close(
        out_relabeled[:, 126 + node_perm], out[:, 126:180], atol=atol, rtol=0
    )
    torch.testing.assert_close(
        out_relabeled[:, 180:], out[:, 180:], atol=atol, rtol=0
    )
    torch.testing.assert_close(value_relabeled, value, atol=atol, rtol=0)


def test_batching_matches_one_board_at_a_time(endpoints):
    actor = GraphActor(endpoints).eval()
    states = _real_states(4)
    with torch.no_grad():
        batched = actor(states)
        single = torch.cat([actor(states[i : i + 1]) for i in range(4)])
    torch.testing.assert_close(batched, single, atol=1e-5, rtol=0)


@pytest.mark.parametrize("network", ["graph", "ff"])
def test_ppo_smoke_and_checkpoint_round_trip(network, tmp_path):
    ppo = PPO(batch_size=4, network=network)
    ppo.observe(30)
    metrics = ppo.train(mini_batch_size=32)
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())

    path = tmp_path / "actor.pt"
    ppo.save_actor(path)
    loaded = load_actor(path, device="cpu")
    assert type(loaded) is type(ppo.actor)

    states = _real_states(2).to(next(ppo.actor.parameters()).device)
    ppo.actor.eval()
    with torch.no_grad():
        torch.testing.assert_close(
            loaded(states.cpu()), ppo.actor(states).cpu(), atol=1e-5, rtol=0
        )


def test_load_actor_rejects_a_different_state_size(tmp_path):
    from train.network_ff import FeedForwardNN

    path = tmp_path / "old.pt"
    torch.save(FeedForwardNN(input_size=1002, output_size=300).state_dict(), path)
    with pytest.raises(ValueError, match="1002-float state"):
        load_actor(path, device="cpu")
