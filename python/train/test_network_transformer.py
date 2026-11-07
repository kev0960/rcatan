"""Checks for `TransformerActor`.

    uv run --no-sync python -m pytest -q train/test_network_transformer.py
"""

import pytest
import torch
from catan_engine import Engine

from train.network_graph import STATE_LEN
from train.network_transformer import (
    NUM_ACTIONS,
    NUM_BOARD_TOKENS,
    TransformerActor,
)
from train.ppo import PPO, TRAINING_STATE, find_resume_checkpoint, load_actor


@pytest.fixture
def endpoints() -> torch.Tensor:
    return torch.from_numpy(Engine.road_endpoints())


@pytest.fixture
def states() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(5, STATE_LEN)


def test_sizes_match_the_engine():
    assert NUM_ACTIONS == Engine.NUM_ACTIONS == 300
    assert NUM_BOARD_TOKENS == 54 + 72 + 1


def test_output_is_one_logit_per_action(endpoints, states):
    assert TransformerActor(endpoints)(states).shape == (5, NUM_ACTIONS)


def test_starts_as_a_uniform_policy(endpoints, states):
    """A zero-initialized readout makes every logit 0, so the first policy is
    uniform over the legal actions whatever the random embeddings are."""
    logits = TransformerActor(endpoints)(states)
    assert torch.count_nonzero(logits) == 0


def _with_random_readout(actor: TransformerActor) -> TransformerActor:
    # Past the zero init, as after a few updates.
    torch.manual_seed(1)
    torch.nn.init.normal_(actor.readout.weight)
    torch.nn.init.normal_(actor.readout.bias)
    return actor


def test_actions_get_different_logits(endpoints, states):
    """Nothing but the learned action tokens tells the 300 actions apart."""
    logits = _with_random_readout(TransformerActor(endpoints))(states)
    assert logits[0].unique().numel() == NUM_ACTIONS


def test_logits_depend_on_the_board(endpoints, states):
    actor = _with_random_readout(TransformerActor(endpoints)).eval()
    a, b = actor(states[:1]), actor(states[1:2])
    assert not torch.allclose(a, b)


def test_every_parameter_is_trained(endpoints, states):
    """Each embedding, the transformer, the readout and the encoder all get a
    gradient. A parameter without one was built but never used in forward."""
    actor = _with_random_readout(TransformerActor(endpoints))
    actor(states).logsumexp(-1).sum().backward()
    unused = [
        name
        for name, p in actor.named_parameters()
        if p.grad is None or torch.count_nonzero(p.grad) == 0
    ]
    assert unused == []


def test_batch_rows_are_independent(endpoints, states):
    actor = _with_random_readout(TransformerActor(endpoints)).eval()
    # Batched and single-row matmuls round differently (about 1e-5 here with
    # pos_emb at std 1). Rows leaking into each other would differ by far more.
    torch.testing.assert_close(
        actor(states)[2:3], actor(states[2:3]), atol=1e-4, rtol=1e-4
    )


def test_ppo_checkpoint_round_trip(tmp_path):
    ppo = PPO(batch_size=2, network="graph", actor_head="transformer")
    assert isinstance(ppo.actor, TransformerActor)

    path = tmp_path / TRAINING_STATE
    ppo.save_training_state(path, iteration=5)
    assert find_resume_checkpoint(tmp_path)[3] == "transformer"
    resumed = PPO(batch_size=2, network="graph", actor_head="transformer")
    assert resumed.load_checkpoint(path) == (5, True)

    with pytest.raises(ValueError, match="actor_head=transformer"):
        PPO(batch_size=2, network="graph").load_checkpoint(path)

    actor_path = tmp_path / "actor.pt"
    ppo.save_actor(actor_path)
    assert isinstance(load_actor(actor_path, device="cpu"), TransformerActor)


def test_ppo_collects_and_trains():
    """One short collection and update through the real PPO loop."""
    ppo = PPO(batch_size=4, network="graph", actor_head="transformer")
    readout_before = ppo.actor.readout.weight.detach().clone()
    ppo.observe(30)
    metrics = ppo.train(mini_batch_size=32)
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert not torch.equal(readout_before, ppo.actor.readout.weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bfloat16 autocast is GPU only")
def test_bfloat16_on_the_gpu_stays_close_to_float32(endpoints, states):
    """On the GPU the transformers run in bfloat16, but the logits come out in
    float32 and give nearly the same policy as the CPU's all-float32 ones.
    PPO compares log-probabilities from collection and training, so rounding
    noise must stay far below a real update's KL (about 0.003-0.01)."""
    actor = TransformerActor(endpoints).eval()
    torch.manual_seed(1)
    # Logits of a realistic size (std around 0.5), not the zero init.
    torch.nn.init.normal_(actor.readout.weight, std=0.1)
    with torch.no_grad():
        reference = actor(states).log_softmax(-1)
        actor.cuda()
        logits = actor(states.cuda()).cpu()
    assert logits.dtype == torch.float32
    log_probs = logits.log_softmax(-1)
    assert (log_probs - reference).abs().max() < 0.05
    kl = (reference.exp() * (reference - log_probs)).sum(-1)
    assert kl.max() < 1e-4
