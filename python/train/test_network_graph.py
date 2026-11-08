import json
import sys

import pytest
import torch
from catan_engine import Engine

from train.network_graph import (
    CITIES,
    LEGACY_STATE_LEN,
    NODE_FEATURES,
    NUM_NODES,
    NUM_ROADS,
    NUM_TILES,
    ROADS,
    SETTLEMENTS,
    STATE_LEN,
    GraphActor,
    GraphCritic,
    GraphEncoder,
)
from train.ppo import (
    ACTOR_INPUT_SIZE,
    PPO,
    TRAINING_STATE,
    find_resume_checkpoint,
    load_actor,
)


@pytest.fixture
def endpoints() -> torch.Tensor:
    return torch.from_numpy(Engine.road_endpoints())


@pytest.fixture
def tile_nodes() -> torch.Tensor:
    return torch.from_numpy(Engine.tile_nodes())


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


def test_card_estimates_add_up_to_the_public_hand_sizes():
    """The estimate of each opponent's hand ([2016, 2031), expected count /
    10) must add up to its public hand size ([725, 728), total / 20), in the
    same rotated order."""
    states = _real_states(16, steps=300)
    totals = states[:, 725:728] * 20
    estimates = states[:, LEGACY_STATE_LEN:STATE_LEN].view(-1, 3, 5) * 10
    # Both are clipped (20 cards in total, 10 of one resource); skip those.
    unclipped = (totals < 20).all(-1) & (estimates < 10).all(-1).all(-1)
    assert unclipped.sum() > 8
    assert totals.sum() > 0
    torch.testing.assert_close(
        estimates.sum(-1)[unclipped], totals[unclipped], atol=1e-3, rtol=0
    )


def test_legacy_actors_ignore_the_card_estimates(tmp_path, endpoints, tile_nodes):
    """Actors trained on the 2016-float state (230211, 115058, ...) still
    load, and run on the current state by reading only its first 2016 floats."""
    torch.manual_seed(0)
    actor = GraphActor(endpoints, tile_nodes=tile_nodes, state_len=LEGACY_STATE_LEN)
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "network": "graph",
            "num_layers": 3,
            "actor_head": "entity-robber",
            "state_len": LEGACY_STATE_LEN,
            "state_dict": actor.state_dict(),
        },
        path,
    )
    loaded = load_actor(path, device="cpu")

    states = _real_states(4)
    noisy = states.clone()
    noisy[:, LEGACY_STATE_LEN:] = torch.rand(4, STATE_LEN - LEGACY_STATE_LEN)
    with torch.no_grad():
        out = loaded(states)
        torch.testing.assert_close(loaded(noisy), out)
        torch.testing.assert_close(actor.eval()(states[:, :LEGACY_STATE_LEN]), out)


def test_output_shapes(endpoints):
    states = _real_states(5)
    assert GraphActor(endpoints)(states).shape == (5, 300)
    assert GraphCritic(endpoints)(states).shape == (5, 1)


@pytest.mark.parametrize("robber_head", [False, True])
def test_heads_land_at_their_action_indices(endpoints, tile_nodes, robber_head):
    actor = GraphActor(endpoints, tile_nodes=tile_nodes if robber_head else None)
    heads = [(actor.road_head[-1], 3.0), (actor.pooled_head[-1], 4.0)]
    if robber_head:
        heads.append((actor.robber_head[-1], 5.0))
    with torch.no_grad():
        for head, bias in heads:
            head.weight.zero_()
            head.bias.fill_(bias)
        actor.node_head.weight.zero_()
        actor.node_head.bias.copy_(torch.tensor([1.0, 2.0]))

        logits = actor(_real_states(2))

    assert torch.all(logits[:, 0:72] == 3.0)  # BuildRoad
    assert torch.all(logits[:, 72:126] == 1.0)  # BuildStructure(Settlement)
    assert torch.all(logits[:, 126:180] == 2.0)  # BuildStructure(City)
    assert torch.all(logits[:, 180:185] == 4.0)  # DiscardResource
    robber = 5.0 if robber_head else 4.0
    assert torch.all(logits[:, 185:204] == robber)  # MoveRobber(tile 0-18)
    assert torch.all(logits[:, 204:300] == 4.0)  # everything else


def test_tile_nodes_match_the_board(endpoints, tile_nodes):
    """Each tile is a hexagon: six distinct corners joined in a ring by six
    roads. Every node is a corner of one to three tiles."""
    assert tile_nodes.shape == (NUM_TILES, 6)
    roads = {frozenset(r) for r in endpoints.tolist()}
    for corners in tile_nodes.tolist():
        assert len(set(corners)) == 6
        assert sum(r <= set(corners) for r in roads) == 6
    counts = torch.bincount(tile_nodes.flatten(), minlength=NUM_NODES)
    assert counts.min() >= 1 and counts.max() <= 3


def test_robber_moves_only_read_their_own_tile(endpoints, tile_nodes):
    """A robber move's logit sees the board only through its tile's six
    corner embeddings (and the globals): perturbing every other node's
    embedding leaves it unchanged."""
    torch.manual_seed(0)
    actor = GraphActor(endpoints, tile_nodes=tile_nodes).eval()
    states = _real_states(1)
    tile = 4
    corners = set(tile_nodes[tile].tolist())
    others = [n for n in range(NUM_NODES) if n not in corners]

    original_forward = actor.encoder.forward

    def perturbed(state):
        h, roads, globals_ = original_forward(state)
        h = h.clone()
        h[:, others] += torch.randn_like(h[:, others])
        return h, roads, globals_

    with torch.no_grad():
        before = actor(states)
        actor.encoder.forward = perturbed
        after = actor(states)
    torch.testing.assert_close(after[:, 185 + tile], before[:, 185 + tile])
    assert not torch.allclose(after[:, 185:204], before[:, 185:204])


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


@pytest.mark.parametrize("robber_head", [False, True])
def test_relabeling_the_board_relabels_the_outputs(
    endpoints, tile_nodes, robber_head
):
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

    # Tile ids stay; their corners are relabeled with the nodes.
    tiles = tile_nodes if robber_head else None
    tiles_relabeled = node_perm[tile_nodes] if robber_head else None
    actor = GraphActor(endpoints, tile_nodes=tiles).eval()
    actor_relabeled = GraphActor(relabeled, tile_nodes=tiles_relabeled).eval()
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


def test_messages_travel_one_road_per_convolution(endpoints):
    """Each convolution only passes messages along roads: the second conv's
    input at node 0 can't depend on nodes more than one road away."""
    near = {0}
    for _ in range(2):
        near |= {
            n
            for a, b in endpoints.tolist()
            if a in near or b in near
            for n in (a, b)
        }
    distant = next(n for n in range(NUM_NODES) if n not in near)

    torch.manual_seed(0)
    encoder = GraphEncoder(endpoints, num_layers=2)
    states = _real_states(1).requires_grad_()
    second_conv_inputs = []
    hook = encoder.convs[1].register_forward_pre_hook(
        lambda module, args: second_conv_inputs.append(args[0])
    )
    try:
        encoder(states)
    finally:
        hook.remove()
    (gradient,) = torch.autograd.grad(second_conv_inputs[0][0, 0, 0], states)
    distant_gradient = gradient[:, NODE_FEATURES].view(1, NUM_NODES, 22)[0, distant]
    assert torch.count_nonzero(distant_gradient) == 0


@pytest.mark.parametrize("network", ["graph", "ff"])
def test_ppo_smoke_and_checkpoint_round_trip(network, tmp_path):
    ppo = PPO(batch_size=4, network=network)
    ppo.observe(30)
    metrics = ppo.train(mini_batch_size=32)
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())

    path = tmp_path / "actor_iter_10.pt"
    ppo.save_actor(path)
    if network == "graph":
        # Checkpoints from 2026-11-03 to 2026-11-05 record global_fusion=False.
        saved = torch.load(path, weights_only=True)
        saved["global_fusion"] = False
        torch.save(saved, path)
    head = "entity-robber" if network == "graph" else "entity"
    assert find_resume_checkpoint(tmp_path) == (path, network, 3, head)
    loaded = load_actor(path, device="cpu")
    assert type(loaded) is type(ppo.actor)

    states = _real_states(2).to(next(ppo.actor.parameters()).device)
    ppo.actor.eval()
    # Same device for both: with TF32 matmuls (see ppo.py), the GPU and the
    # CPU round differently.
    loaded.to(states.device)
    with torch.no_grad():
        torch.testing.assert_close(
            loaded(states), ppo.actor(states), atol=1e-5, rtol=0
        )

    training_path = tmp_path / TRAINING_STATE
    ppo.save_training_state(training_path, iteration=10)
    assert find_resume_checkpoint(tmp_path) == (training_path, network, 3, head)
    resumed = PPO(batch_size=4, network=network)
    assert resumed.load_checkpoint(training_path) == (10, True)
    torch.testing.assert_close(resumed.actor.state_dict(), ppo.actor.state_dict())
    torch.testing.assert_close(resumed.critic.state_dict(), ppo.critic.state_dict())
    torch.testing.assert_close(
        resumed.actor_optimizer.state_dict(), ppo.actor_optimizer.state_dict()
    )
    torch.testing.assert_close(
        resumed.critic_optimizer.state_dict(), ppo.critic_optimizer.state_dict()
    )


def test_checkpoints_without_a_head_load_as_entity(tmp_path):
    """Graph checkpoints from before the actor_head setting (230211, 115058,
    ...) don't record one. They load as "entity", with the robber moves in
    the pooled head, and resume as "entity"."""
    ppo = PPO(batch_size=2, network="graph", actor_head="entity")
    assert ppo.actor.robber_head is None
    path = tmp_path / TRAINING_STATE
    ppo.save_training_state(path, iteration=3)
    saved = torch.load(path, weights_only=True)
    del saved["actor_head"]
    torch.save(saved, path)

    assert find_resume_checkpoint(tmp_path)[3] == "entity"
    assert PPO(batch_size=2, network="graph", actor_head="entity").load_checkpoint(
        path
    ) == (3, True)
    with pytest.raises(ValueError, match="actor_head=entity"):
        PPO(batch_size=2, network="graph").load_checkpoint(path)

    actor_path = tmp_path / "actor.pt"
    ppo.save_actor(actor_path)
    saved = torch.load(actor_path, weights_only=True)
    del saved["actor_head"]
    torch.save(saved, actor_path)
    loaded = load_actor(actor_path, device="cpu")
    assert isinstance(loaded, GraphActor) and loaded.robber_head is None


def test_global_fusion_checkpoints_are_rejected(tmp_path):
    """Global fusion was removed; its checkpoints fail with a clear error
    instead of a state-dict mismatch."""
    ppo = PPO(batch_size=1, network="graph")
    actor_path = tmp_path / "actor_iter_10.pt"
    ppo.save_actor(actor_path)
    training_path = tmp_path / TRAINING_STATE
    ppo.save_training_state(training_path, iteration=10)
    for path in (actor_path, training_path):
        saved = torch.load(path, weights_only=True)
        saved["global_fusion"] = True
        torch.save(saved, path)

    with pytest.raises(ValueError, match="global fusion"):
        load_actor(actor_path, device="cpu")
    with pytest.raises(ValueError, match="global fusion"):
        find_resume_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="global fusion"):
        PPO(batch_size=1, network="graph").load_checkpoint(training_path)


@pytest.mark.parametrize("flags", [[], ["--network", "ff"]])
def test_cli_resume_continues_the_run(flags, tmp_path, monkeypatch):
    from train import __main__ as cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "NUM_STEPS", 6)
    monkeypatch.setattr(cli, "MINI_BATCH_SIZE", 8)
    monkeypatch.setattr(cli, "BASELINE_CHECKPOINT", None)
    common = ["--iterations", "1", "--batch-size", "2", "--actor-passes", "1", "--no-compile"]
    monkeypatch.setattr(sys, "argv", ["train", *common, *flags])
    cli.main()

    run_dir = cli.CHECKPOINT_ROOT / cli.RUN_TIMESTAMP
    config_path = run_dir / cli.RUN_CONFIG
    assert "global_fusion" not in json.loads(config_path.read_text())
    monkeypatch.setattr(sys, "argv", ["train", *common, "--resume", str(run_dir)])
    cli.main()
    config = json.loads(config_path.read_text())
    assert config["iterations"] == 2
    assert config["network"] == (flags[1] if flags else "graph")
    saved = torch.load(run_dir / TRAINING_STATE, weights_only=True)
    assert saved["iteration"] == 2


def test_load_actor_rejects_a_different_state_size(tmp_path):
    from train.network_ff import FeedForwardNN

    path = tmp_path / "old.pt"
    torch.save(FeedForwardNN(input_size=1002, output_size=300).state_dict(), path)
    with pytest.raises(ValueError, match="1002-float state"):
        load_actor(path, device="cpu")



def test_gradient_accumulation_matches_one_big_minibatch():
    """micro_batch_size only changes memory use: 4 micro-batches of 8 must give
    the gradient of one 32 minibatch. The optimizer step is replaced with a
    recorder, so the weights stay put and both runs see the same actor."""
    # No clipping, so the recorded gradients are the raw ones.
    ppo = PPO(batch_size=4, network="graph", max_grad_norm=None)
    ppo.observe(30)
    buffers = {k: list(v) for k, v in vars(ppo).items() if k.startswith("buf_")}

    def gradients(**kwargs):
        recorded = []
        ppo.actor_optimizer.step = lambda: recorded.append(
            [p.grad.clone() for p in ppo.actor.parameters()]
        )
        ppo.critic_optimizer.step = lambda: None
        for k, v in buffers.items():
            getattr(ppo, k)[:] = v
        torch.manual_seed(0)  # same minibatch order
        metrics = ppo.train(mini_batch_size=32, **kwargs)
        return recorded, metrics

    # Full float32 matmuls: TF32 rounding differs with the batch shape.
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        whole, whole_metrics = gradients()
        split, split_metrics = gradients(micro_batch_size=8)
    finally:
        torch.set_float32_matmul_precision(precision)
    assert len(whole) == len(split) > 1
    for a, b in zip(whole, split):
        for ga, gb in zip(a, b):
            torch.testing.assert_close(gb, ga, atol=1e-5, rtol=1e-3)
    for key in ("actor_loss", "critic_loss", "entropy", "approx_kl"):
        assert split_metrics[key] == pytest.approx(whole_metrics[key], abs=1e-5)


def test_gradient_clipping_caps_every_step():
    """With a tiny max_grad_norm every step is clipped: the gradient the
    optimizer sees has exactly that norm, while the logged norm is the one
    from before clipping."""
    ppo = PPO(batch_size=4, network="graph", max_grad_norm=1e-3)
    ppo.observe(30)
    seen = []
    step = ppo.actor_optimizer.step

    def recording_step():
        grads = [p.grad for p in ppo.actor.parameters() if p.grad is not None]
        seen.append(torch.nn.utils.get_total_norm(grads).item())
        step()

    ppo.actor_optimizer.step = recording_step
    metrics = ppo.train(mini_batch_size=32)
    assert seen and all(n == pytest.approx(1e-3, rel=1e-3) for n in seen)
    assert metrics["actor_grad_clipped"] == 1.0
    assert metrics["actor_grad_norm"] > 1e-3
    assert metrics["actor_grad_norm_max"] >= metrics["actor_grad_norm"]
    assert metrics["critic_grad_norm"] > 0


def test_no_clipping_still_logs_the_norm():
    ppo = PPO(batch_size=4, network="graph", max_grad_norm=None)
    ppo.observe(30)
    metrics = ppo.train(mini_batch_size=32)
    assert metrics["actor_grad_norm"] > 0
    assert metrics["actor_grad_clipped"] == 0.0
