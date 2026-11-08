from pathlib import Path

import numpy as np
import torch
from catan_engine import Engine
from torch import nn
from torch.distributions import Categorical
from tqdm import tqdm

from train.network_ff import FeedForwardNN
from train.network_graph import (
    LEGACY_STATE_LEN,
    NUM_LAYERS,
    STATE_LEN,
    GraphActor,
    GraphCritic,
)
from train.network_transformer import TransformerActor

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Let float32 matmuls use TF32 tensor cores: float32's range with about 10
# mantissa bits. Several times faster on matmul-heavy networks (the transformer
# actor), and the rounding is far below minibatch gradient noise. Runs from
# before 2026-11-04 used full float32.
torch.set_float32_matmul_precision("high")
DISCOUNT_FACTOR = 0.996
EPSILON = 0.2
# Gradient-norm clipping, applied to the actor's and the critic's gradients
# separately, just before each optimizer step: if the gradient's total L2 norm
# (over all of that network's parameters) is above this, the whole gradient is
# scaled down to this norm. Its direction is unchanged; below the threshold it
# does nothing.
#
# This is not the same as PPO's ratio clip (EPSILON). The ratio clip works per
# sample, inside the loss: once a sample's ratio has left [1 - EPSILON,
# 1 + EPSILON] in the direction its advantage pushes, that sample's gradient
# is 0. But it is checked against the weights before the step, and it never
# scales down the samples still inside the range. So one minibatch with an
# unusually large gradient (all ratios still near 1) can take one huge step,
# and Adam can enlarge it further for weights whose recent gradients were
# small. Norm clipping caps every single step. It is standard in PPO
# implementations (OpenAI baselines, CleanRL, Stable-Baselines3), at 0.5.
#
# Added 2026-11-05 after KL spikes in the transformer runs (0.043 at iteration
# 221 of 20261104-230947, 0.050 at iteration 46 of 20261105-210942, against a
# usual 0.005-0.007). Earlier runs had no clipping. None disables it.
#
# Raised from 0.5 to 2.0 on 2026-11-06. In 20261105-222120 (0.5) the actor's
# mean norm grew from 0.05 to about 0.95 as training went on, so from
# iteration 51 every single step was clipped (scaled by about 0.5). With Adam a
# steady scale like that changes little, but it means the clip no longer tells
# outliers apart. Within an iteration the largest step was usually only about
# 1.5x the mean, while the outliers were 5-35x (25.2 at iteration 41). 2.0 is
# about 2x the typical actor norm: ordinary steps pass, outliers are still
# capped. The critic's norm (about 0.2-0.3) is then practically never clipped.
MAX_GRAD_NORM = 2.0
GAE_LAMBDA = 0.95

# How advantages and critic targets are computed: "gae" (GAE(lambda), see
# `compute_gae`) or "mc" (Monte Carlo discounted returns).
ADVANTAGES = ("gae", "mc")
DEFAULT_ADVANTAGE = "gae"

# Actor, critic and optimizer state saved in each run's checkpoint dir so
# training can resume; see `PPO.save_training_state`.
TRAINING_STATE = "train_latest.pt"

ACTOR_INPUT_SIZE = STATE_LEN
# State sizes `load_actor` accepts: the current one, and the one before the
# card estimates were appended (actors trained on it read only its floats).
LOADABLE_STATE_LENS = (STATE_LEN, LEGACY_STATE_LEN)
ACTOR_OUTPUT_SIZE = 300

# Network architectures for actor and critic: "graph" or "ff" (feed-forward).
NETWORKS = ("graph", "ff")
DEFAULT_NETWORK = "graph"

# How the graph actor turns the encoder's output into the 300 logits:
# - "entity-robber" (GraphActor with tile_nodes): per-road, per-node and
#   per-tile (robber) heads, plus a pooled head for the rest. Added 2026-11-06.
# - "entity" (GraphActor): the same, but the robber moves come from the
#   pooled head. Every graph run up to 2026-11-06 (230211, 115058, ...).
# - "transformer" (TransformerActor: learned action tokens attend over the
#   board).
# - "transformer-wired" (TransformerActor with tile_nodes): the same, but each
#   action token starts from the board tokens it acts on. Added 2026-11-07.
ACTOR_HEADS = ("entity-robber", "entity", "transformer", "transformer-wired")
DEFAULT_ACTOR_HEAD = "entity-robber"
# The head of a checkpoint that doesn't record one: they all predate the
# actor_head setting, so they are "entity". Feed-forward runs record it too,
# though their single head ignores it.
LEGACY_ACTOR_HEAD = "entity"

# Passes over each collected buffer per `train` call. The critic does fewer on
# the graph network: more passes let it memorize the buffer's games instead of
# learning values that carry over to new ones.
ACTOR_PASSES = 4
# Adam learning rate for the actor at the start of training.
LEARNING_RATE = 1e-4
# The critic's learning rate is this multiple of the actor's, so it follows the
# same schedule. 10x made the critic's predictions worse (explained_var_mc
# about 0.04 lower) and the actor a few points weaker than 1x.
CRITIC_LR_SCALE = 1.0
CRITIC_PASSES = {"graph": 2, "ff": 4}


def build_actor(
    network: str,
    num_layers: int = NUM_LAYERS,
    actor_head: str = DEFAULT_ACTOR_HEAD,
    state_len: int = ACTOR_INPUT_SIZE,
) -> nn.Module:
    """Build the actor; layer count and the actor head apply only to graph.
    `state_len` is the state the actor was trained on (see
    LOADABLE_STATE_LENS); it always runs on the current state."""
    if network == "graph":
        if actor_head not in ACTOR_HEADS:
            raise ValueError(
                f"unknown actor head: {actor_head!r} (expected {ACTOR_HEADS})"
            )
        road_endpoints = torch.from_numpy(Engine.road_endpoints())
        tile_nodes = (
            torch.from_numpy(Engine.tile_nodes())
            if actor_head in ("entity-robber", "transformer-wired")
            else None
        )
        if actor_head.startswith("transformer"):
            return TransformerActor(road_endpoints, num_layers, state_len, tile_nodes)
        return GraphActor(road_endpoints, num_layers, tile_nodes, state_len)
    if network == "ff":
        return FeedForwardNN(input_size=state_len, output_size=ACTOR_OUTPUT_SIZE)
    raise ValueError(f"unknown network: {network!r} (expected 'graph' or 'ff')")


def build_critic(network: str) -> nn.Module:
    if network == "graph":
        return GraphCritic(torch.from_numpy(Engine.road_endpoints()))
    if network == "ff":
        return FeedForwardNN(input_size=ACTOR_INPUT_SIZE, output_size=1)
    raise ValueError(f"unknown network: {network!r} (expected 'graph' or 'ff')")


def load_actor(path: str | Path, device: str | torch.device = DEVICE) -> nn.Module:
    """Load a trained actor from `path` for inference (sets eval mode).

    Checkpoints record their network and state size. Older checkpoints are a
    bare feed-forward state dict. Actors trained on the legacy 2016-float
    state (before the card estimates) load too and ignore the estimates.
    Raises ValueError for any other state size."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if "state_dict" in checkpoint:
        network = checkpoint["network"]
        state_len = checkpoint["state_len"]
        state_dict = checkpoint["state_dict"]
        num_layers = checkpoint.get("num_layers", NUM_LAYERS)
    else:
        network, state_dict, num_layers = "ff", checkpoint, NUM_LAYERS
        state_len = state_dict["fc1.weight"].shape[1]

    if state_len not in LOADABLE_STATE_LENS:
        raise ValueError(
            f"{path} was trained on a {state_len}-float state, but the engine "
            f"now produces {ACTOR_INPUT_SIZE}; retrain it"
        )

    _reject_global_fusion(checkpoint, path)
    actor = build_actor(
        network,
        num_layers,
        checkpoint.get("actor_head", LEGACY_ACTOR_HEAD),
        state_len,
    )
    actor.load_state_dict(state_dict)
    actor.to(device)
    actor.eval()
    return actor


def play_matches(
    actor: nn.Module,
    opponent_actor: nn.Module | None,
    num_matches: int,
    desc: str,
    seed: int = SEED + 10_000,
    num_seats: int = 1,
) -> dict[str, float]:
    """Play `num_matches` games with `actor` in `num_seats` randomized
    seats. Other seats are filled by `opponent_actor` (argmax) if provided,
    otherwise by a uniform-random agent over valid actions. Both actors'
    train/eval modes are restored afterwards.

    A match counts as a win if any of `actor`'s seats wins. `avg_vp` and
    `avg_opp_vp` are per-seat averages."""
    if not 1 <= num_seats <= 3:
        raise ValueError(f"num_seats must be 1..3, got {num_seats}")
    engine = Engine(batch_size=num_matches, seed=seed)
    rng = np.random.default_rng(seed + 1)
    # trained_seats[b, p]: seat p of match b is played by `actor`.
    seat_order = rng.permuted(np.tile(np.arange(4), (num_matches, 1)), axis=1)
    trained_seats = np.zeros((num_matches, 4), dtype=bool)
    np.put_along_axis(trained_seats, seat_order[:, :num_seats], True, axis=1)
    batch_idx = np.arange(num_matches)

    wins = 0
    finished = 0
    total_steps = 0
    total_vp = 0
    total_opp_vp = 0
    steps_since_start = np.zeros(num_matches, dtype=np.int64)

    def argmax_actions(actor, states_np, masks_np, idx):
        state_tensor = torch.tensor(states_np[idx]).float().to(DEVICE)
        logits = actor(state_tensor)
        mask_tensor = torch.tensor(masks_np[idx]).bool().to(DEVICE)
        logits[~mask_tensor] = -float("inf")
        return torch.argmax(logits, dim=-1).cpu().numpy()

    was_training = actor.training
    opp_was_training = opponent_actor is not None and opponent_actor.training
    actor.eval()
    if opponent_actor is not None:
        opponent_actor.eval()
    try:
        with (
            torch.no_grad(),
            tqdm(total=num_matches, desc=desc, leave=False) as pbar,
        ):
            while finished < num_matches:
                states, players = engine.state()
                masks = engine.get_action_mask()

                actions = np.zeros(num_matches, dtype=np.int64)

                is_trained = trained_seats[batch_idx, players]
                trained_idx = np.where(is_trained)[0]
                if trained_idx.size > 0:
                    actions[trained_idx] = argmax_actions(
                        actor, states, masks, trained_idx
                    )

                opp_idx = np.where(~is_trained)[0]
                if opp_idx.size > 0:
                    if opponent_actor is None:
                        for i in opp_idx:
                            valid = np.flatnonzero(masks[i])
                            actions[i] = rng.choice(valid)
                    else:
                        actions[opp_idx] = argmax_actions(
                            opponent_actor, states, masks, opp_idx
                        )

                rewards, ended_mask, final_vps = engine.step(actions.tolist())
                steps_since_start += 1

                for i in range(len(ended_mask)):
                    if not ended_mask[i]:
                        continue
                    if finished < num_matches:
                        mine = trained_seats[i]
                        if any(rewards[i][s] > 0 for s in np.flatnonzero(mine)):
                            wins += 1
                        total_vp += int(final_vps[i][mine].sum())
                        total_opp_vp += int(final_vps[i][~mine].sum())
                        total_steps += int(steps_since_start[i])
                        finished += 1
                        pbar.update(1)
                    steps_since_start[i] = 0
    finally:
        actor.train(was_training)
        if opponent_actor is not None:
            opponent_actor.train(opp_was_training)

    return {
        "win_rate": wins / finished if finished > 0 else 0.0,
        "avg_steps": total_steps / finished if finished > 0 else 0.0,
        "avg_vp": total_vp / (finished * num_seats) if finished > 0 else 0.0,
        "avg_opp_vp": (
            total_opp_vp / (finished * (4 - num_seats)) if finished > 0 else 0.0
        ),
    }


def find_resume_checkpoint(
    run_dir: str | Path,
) -> tuple[Path, str, int, bool, str]:
    """Checkpoint to resume `run_dir` from, and its network, actor layer
    count, global fusion setting and actor head. Prefers the full training state; older
    runs fall back to their newest `actor_iter_N.pt` (actor only)."""
    run_dir = Path(run_dir)
    path = run_dir / TRAINING_STATE
    if not path.is_file():
        actors = sorted(
            run_dir.glob("actor_iter_*.pt"), key=lambda p: _checkpoint_iter(p)
        )
        if not actors:
            raise FileNotFoundError(
                f"no {TRAINING_STATE} or actor_iter_*.pt in {run_dir}"
            )
        path = actors[-1]
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    _reject_global_fusion(checkpoint, path)
    network = checkpoint["network"] if "network" in checkpoint else "ff"
    return (
        path,
        network,
        checkpoint.get("num_layers", NUM_LAYERS),
        checkpoint.get("actor_head", LEGACY_ACTOR_HEAD),
    )


def _reject_global_fusion(checkpoint: dict, path: str | Path) -> None:
    """Global fusion (pooling the board into every node after each graph
    convolution) was removed on 2026-11-05: its only run, 20261103-185950,
    ended at parity with the runs without it. Checkpoints from 2026-11-03 to
    2026-11-05 record `global_fusion`; those with it off still load, and those
    with it on can't, since their encoder has layers this code no longer has."""
    if checkpoint.get("global_fusion", False):
        raise ValueError(
            f"{path} was trained with global fusion, which was removed; "
            "it can't be loaded by this version"
        )


def _checkpoint_iter(path: Path) -> int:
    """N from `actor_iter_N.pt`."""
    return int(path.stem.rsplit("_", 1)[1])


class PPO:
    def __init__(
        self,
        batch_size: int,
        network: str = DEFAULT_NETWORK,
        compile: bool = False,
        advantage: str = DEFAULT_ADVANTAGE,
        gae_lambda: float = GAE_LAMBDA,
        critic_passes: int | None = None,
        actor_passes: int = ACTOR_PASSES,
        actor_layers: int = NUM_LAYERS,
        critic_lr_scale: float = CRITIC_LR_SCALE,
        actor_head: str = DEFAULT_ACTOR_HEAD,
        max_grad_norm: float | None = MAX_GRAD_NORM,
    ):
        if advantage not in ADVANTAGES:
            raise ValueError(f"unknown advantage: {advantage!r} (expected {ADVANTAGES})")
        if critic_passes is None:
            critic_passes = min(CRITIC_PASSES[network], actor_passes)
        if actor_passes < 1:
            raise ValueError(f"actor_passes must be at least 1, got {actor_passes}")
        # The critic trains on the first `critic_passes` of the actor's passes.
        if not 1 <= critic_passes <= actor_passes:
            raise ValueError(
                f"critic_passes must be between 1 and actor_passes "
                f"({actor_passes}), got {critic_passes}"
            )
        if max_grad_norm is not None and max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm must be positive or None, got {max_grad_norm}")
        self.actor_passes = actor_passes
        self.critic_passes = critic_passes
        self.max_grad_norm = max_grad_norm
        self.network = network
        self.advantage = advantage
        self.gae_lambda = gae_lambda
        self.actor_layers = actor_layers
        # The feed-forward actor has a single head.
        self.actor_head = actor_head if network == "graph" else LEGACY_ACTOR_HEAD
        self.actor = build_actor(
            network, actor_layers, self.actor_head
        ).to(DEVICE)
        self.critic = build_critic(network).to(DEVICE)
        # Compiled wrappers share parameters with the modules above; collection
        # and training run through them. Checkpoints and validation use the
        # plain modules, so saved state dicts keep their usual key names.
        # dynamic=True: collection, mini-batches and the last partial
        # mini-batch all have different batch sizes.
        if compile:
            self._actor = torch.compile(self.actor, dynamic=True)
            self._critic = torch.compile(self.critic, dynamic=True)
        else:
            self._actor, self._critic = self.actor, self.critic
        self.engine = Engine(batch_size=batch_size, seed=SEED)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=LEARNING_RATE
        )
        self.critic_lr_scale = critic_lr_scale
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=LEARNING_RATE * critic_lr_scale
        )

        # The run's config, saved into every checkpoint (see `train.__main__`).
        self.config: dict | None = None

        self.buf_states = []
        self.buf_players = []
        self.buf_actions = []
        self.buf_log_probs = []
        self.buf_values = []
        self.buf_rewards = []
        self.buf_game_ended_mask = []
        self.buf_masks = []

    def save_actor(self, path: str | Path) -> None:
        """Persist the actor's weights to `path` (parent dirs created if needed),
        along with its network, layer count and state size so `load_actor`
        can rebuild it."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "network": self.network,
                "num_layers": self.actor_layers,
                "actor_head": self.actor_head,
                "state_len": ACTOR_INPUT_SIZE,
                "config": self.config,
                "state_dict": self.actor.state_dict(),
            },
            path,
        )

    @property
    def actor_lr(self) -> float:
        """The actor's current learning rate."""
        return self.actor_optimizer.param_groups[0]["lr"]

    @property
    def critic_lr(self) -> float:
        return self.critic_optimizer.param_groups[0]["lr"]

    def set_actor_lr(self, lr: float) -> None:
        """Set the actor's learning rate to `lr` and the critic's to
        `lr * critic_lr_scale`."""
        for group in self.actor_optimizer.param_groups:
            group["lr"] = lr
        for group in self.critic_optimizer.param_groups:
            group["lr"] = lr * self.critic_lr_scale

    def save_training_state(self, path: str | Path, iteration: int) -> None:
        """Persist everything needed to resume training after `iteration`
        completed iterations: actor, critic and both optimizers."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "network": self.network,
                "num_layers": self.actor_layers,
                "actor_head": self.actor_head,
                "state_len": ACTOR_INPUT_SIZE,
                "iteration": iteration,
                "config": self.config,
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> tuple[int, bool]:
        """Load a training state or an actor checkpoint from `path`. Returns
        (completed iterations, whether the critic was restored). An actor
        checkpoint only restores the actor; the critic and optimizers start
        fresh and the iteration comes from its `actor_iter_N.pt` name."""
        path = Path(path)
        # Load onto the CPU: load_state_dict copies weights and Adam's moments
        # to the parameters' device, but leaves Adam's step counter where it
        # is, and Adam expects it on the CPU (on the GPU it syncs every step).
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("network", "ff") != self.network:
            raise ValueError(
                f"{path} is a {checkpoint.get('network', 'ff')} checkpoint, "
                f"but this run uses {self.network}"
            )
        if checkpoint.get("num_layers", NUM_LAYERS) != self.actor_layers:
            raise ValueError(
                f"{path} has a {checkpoint.get('num_layers', NUM_LAYERS)}-layer "
                f"actor, but this run uses {self.actor_layers}"
            )
        _reject_global_fusion(checkpoint, path)
        saved_head = checkpoint.get("actor_head", LEGACY_ACTOR_HEAD)
        if saved_head != self.actor_head:
            raise ValueError(
                f"{path} has actor_head={saved_head}, but this run uses "
                f"actor_head={self.actor_head}; resume with the checkpoint's "
                "architecture or start a new run"
            )
        if checkpoint.get("state_len", ACTOR_INPUT_SIZE) != ACTOR_INPUT_SIZE:
            raise ValueError(
                f"{path} was trained on a {checkpoint['state_len']}-float "
                f"state, but the engine now produces {ACTOR_INPUT_SIZE}"
            )

        if "critic" in checkpoint:
            self.actor.load_state_dict(checkpoint["actor"])
            self.critic.load_state_dict(checkpoint["critic"])
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
            return checkpoint["iteration"], True

        self.actor.load_state_dict(checkpoint.get("state_dict", checkpoint))
        return _checkpoint_iter(path), False

    # Collection never backpropagates, so skip building autograd graphs.
    @torch.no_grad()
    def observe(self, steps: int):
        for step in tqdm(range(steps), desc="observe", leave=False):
            (
                states,
                players,
                actions,
                log_probs,
                value,
                rewards,
                game_ended_mask,
                masks,
            ) = self.step()

            self.buf_states.append(states)
            self.buf_players.append(players)
            self.buf_actions.append(actions)
            self.buf_log_probs.append(log_probs)
            self.buf_values.append(value)
            self.buf_rewards.append(rewards)
            self.buf_game_ended_mask.append(game_ended_mask)
            self.buf_masks.append(masks)

    def train(self, mini_batch_size: int, micro_batch_size: int | None = None):
        """One PPO update on the collected buffer.

        mini_batch_size: samples per optimizer step.
        micro_batch_size: samples per forward/backward pass, at most
            mini_batch_size (default: the same, i.e. no accumulation). Each
            minibatch is split into micro-batches whose gradients add up in
            `.grad` before the single optimizer step (gradient accumulation).
            Only one micro-batch's activations are in memory at a time, so a
            4096 update fits in the memory of a 1024 one. The step is the same
            as one 4096 forward/backward (up to float rounding): every loss
            here is a per-sample mean, so each micro-batch's mean is weighted
            by its share of the minibatch, and the advantages are normalized
            over the whole buffer, not per minibatch.

        Bigger minibatches mean fewer, less noisy steps per iteration. With a
        fixed buffer, 4096 instead of 1024 is 4x fewer actor steps per pass,
        each averaging 4x more samples, so the policy drifts less from the one
        that collected the data (lower KL and clip_frac per iteration).
        """
        micro_batch_size = micro_batch_size or mini_batch_size
        if not 1 <= micro_batch_size <= mini_batch_size:
            raise ValueError(
                f"micro_batch_size={micro_batch_size} must be between 1 and "
                f"mini_batch_size={mini_batch_size}"
            )
        # stack(values) is (T, B, 1)
        values = torch.stack(self.buf_values).squeeze(-1).to(DEVICE)

        # Monte Carlo discounted returns: what actually happened.
        mc_returns = torch.from_numpy(self.compute_discounted_rewards()).to(DEVICE)

        # `returns` are the critic's targets: lambda-returns for GAE, discounted
        # returns for Monte Carlo.
        if self.advantage == "gae":
            advantages, returns = self.compute_gae(
                values.cpu().numpy(), self.gae_lambda
            )
            advantages = torch.from_numpy(advantages).to(DEVICE)
            returns = torch.from_numpy(returns).to(DEVICE)
        else:
            returns = mc_returns
            advantages = returns - values

        # How much of the target variance the critic explained, using the
        # values predicted while collecting (before training on this batch).
        # 1 = perfect, 0 = no better than predicting the mean. GAE targets are
        # built partly from the critic's own values, so against them this
        # flatters the critic and is not comparable across the two modes.
        explained_var = 1 - (returns - values).var() / (returns.var() + 1e-8)
        # The same against the Monte Carlo returns, which the critic cannot
        # influence: comparable across modes and runs. Equal to explained_var
        # in Monte Carlo mode.
        explained_var_mc = 1 - (mc_returns - values).var() / (mc_returns.var() + 1e-8)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        flat_states = torch.stack(self.buf_states).flatten(0, 1)
        flat_mask = torch.stack(self.buf_masks).flatten(0, 1)
        flat_actions = torch.stack(self.buf_actions).flatten(0, 1)
        flat_log_probs_old = torch.stack(self.buf_log_probs).flatten(0, 1)

        flat_returns = returns.flatten(0, 1)
        flat_advantages = advantages.flatten(0, 1)

        N = flat_states.shape[0]

        # Summed on the GPU and read once at the end: calling .item() per
        # minibatch makes the CPU wait for the GPU before it can queue the
        # next one.
        zero = torch.zeros((), device=DEVICE)
        sum_actor_loss = zero.clone()
        sum_critic_loss = zero.clone()
        sum_entropy = zero.clone()
        sum_loss = zero.clone()
        sum_approx_kl = zero.clone()
        sum_clip_frac = zero.clone()
        # Gradient norms before clipping (see MAX_GRAD_NORM): the mean and max
        # over this iteration's steps, and the share of actor steps that were
        # clipped. A KL spike from one huge step shows up as a max far above
        # the mean. If nearly every step is clipped, the threshold is too low
        # and is acting as a lower learning rate.
        sum_actor_grad_norm = zero.clone()
        max_actor_grad_norm = zero.clone()
        sum_actor_clipped = zero.clone()
        sum_critic_grad_norm = zero.clone()
        max_critic_grad_norm = zero.clone()
        n_updates = 0
        n_critic_updates = 0

        for pass_ in tqdm(range(self.actor_passes), desc="train", leave=False):
            # The critic only trains on the first `critic_passes` passes.
            train_critic = pass_ < self.critic_passes
            # On the GPU, so indexing doesn't copy the indices over each time.
            perm = torch.randperm(N, device=DEVICE)
            for start in range(0, N, mini_batch_size):
                idx = perm[start : start + mini_batch_size]
                # A Python int (slicing is resolved on the CPU), so no GPU sync.
                mb_len = idx.shape[0]

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                # Gradient accumulation: forward/backward one micro-batch at a
                # time; .backward() adds into .grad, and the optimizers step
                # once below. With micro_batch_size == mini_batch_size this
                # loop runs once and is a plain minibatch update.
                for micro_start in range(0, mb_len, micro_batch_size):
                    micro_idx = idx[micro_start : micro_start + micro_batch_size]
                    # This micro-batch's share of the minibatch. Its losses
                    # are means over its own samples, so scaling each by its
                    # share and summing gives the mean over the minibatch,
                    # also when the last micro-batch is short.
                    weight = micro_idx.shape[0] / mb_len

                    mb_states = flat_states[micro_idx]
                    mb_mask = flat_mask[micro_idx]
                    mb_actions = flat_actions[micro_idx]
                    mb_log_probs_old = flat_log_probs_old[micro_idx]
                    mb_returns = flat_returns[micro_idx]
                    mb_advantages = flat_advantages[micro_idx]

                    # Computing the pi_theta (a_t | s_t) on the updated actor
                    # masked_fill, unlike boolean-index assignment, doesn't
                    # wait on the GPU to count the masked entries. Argument
                    # validation would too (it checks values on the CPU);
                    # collection keeps it.
                    logits = self._actor(mb_states).masked_fill(
                        ~mb_mask, -float("inf")
                    )
                    dist = Categorical(logits=logits, validate_args=False)
                    logp_new = dist.log_prob(mb_actions)

                    # Compute pi_theta (a_t | s_t) / pi_theta_old (a_t | s_t)
                    ratio = torch.exp(logp_new - mb_log_probs_old)

                    unclipped = ratio * mb_advantages
                    clipped = (
                        torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON) * mb_advantages
                    )

                    # Minimize the PPO-Clip objective.
                    actor_loss = -torch.min(unclipped, clipped).mean()

                    entropy = dist.entropy().mean()

                    loss = actor_loss - 0.01 * entropy
                    if train_critic:
                        # Value function fitting.
                        critic_loss = (
                            (self._critic(mb_states).squeeze(-1) - mb_returns) ** 2
                        ).mean()
                        loss = loss + 0.5 * critic_loss

                    (loss * weight).backward()

                    # Metrics: weighted the same way, so each update adds its
                    # minibatch mean, and the logged value is the average over
                    # updates as before.
                    if train_critic:
                        sum_critic_loss += critic_loss.detach() * weight
                    sum_actor_loss += actor_loss.detach() * weight
                    sum_entropy += entropy.detach() * weight
                    sum_loss += loss.detach() * weight

                    # How far this update moved the policy: KL(old || new)
                    # estimated as E[(r - 1) - log r], and the share of
                    # samples where the ratio left the clip range.
                    with torch.no_grad():
                        log_ratio = logp_new - mb_log_probs_old
                        sum_approx_kl += ((ratio - 1) - log_ratio).mean() * weight
                        sum_clip_frac += (
                            ((ratio - 1).abs() > EPSILON).float().mean() * weight
                        )

                # The gradient of the whole minibatch is complete here (all
                # micro-batches added up), so the norm is the minibatch's, the
                # same with or without accumulation. Clipped per network: the
                # critic's gradient is typically much larger than the actor's
                # and would otherwise dominate a joint norm.
                actor_grad_norm = self._clip_gradients(self.actor)
                sum_actor_grad_norm += actor_grad_norm
                max_actor_grad_norm = torch.maximum(max_actor_grad_norm, actor_grad_norm)
                if self.max_grad_norm is not None:
                    sum_actor_clipped += (actor_grad_norm > self.max_grad_norm).float()
                self.actor_optimizer.step()
                if train_critic:
                    critic_grad_norm = self._clip_gradients(self.critic)
                    sum_critic_grad_norm += critic_grad_norm
                    max_critic_grad_norm = torch.maximum(
                        max_critic_grad_norm, critic_grad_norm
                    )
                    self.critic_optimizer.step()
                    n_critic_updates += 1
                n_updates += 1

        self.buf_states.clear()
        self.buf_players.clear()
        self.buf_actions.clear()
        self.buf_log_probs.clear()
        self.buf_values.clear()
        self.buf_rewards.clear()
        self.buf_game_ended_mask.clear()
        self.buf_masks.clear()

        return {
            "loss": sum_loss.item() / n_updates,
            "actor_loss": sum_actor_loss.item() / n_updates,
            "critic_loss": sum_critic_loss.item() / n_critic_updates,
            "entropy": sum_entropy.item() / n_updates,
            "return_mean": flat_returns.mean().item(),
            "return_std": flat_returns.std().item(),
            "value_mean": values.mean().item(),
            "value_std": values.std().item(),
            "explained_var": explained_var.item(),
            "explained_var_mc": explained_var_mc.item(),
            "approx_kl": sum_approx_kl.item() / n_updates,
            "clip_frac": sum_clip_frac.item() / n_updates,
            "actor_grad_norm": sum_actor_grad_norm.item() / n_updates,
            "actor_grad_norm_max": max_actor_grad_norm.item(),
            "actor_grad_clipped": sum_actor_clipped.item() / n_updates,
            "critic_grad_norm": sum_critic_grad_norm.item() / n_critic_updates,
            "critic_grad_norm_max": max_critic_grad_norm.item(),
        }

    def _clip_gradients(self, network: nn.Module) -> torch.Tensor:
        """Scale `network`'s gradients down to `max_grad_norm` if their total
        norm is above it (see MAX_GRAD_NORM). Returns the norm before clipping,
        as a GPU tensor (no sync). With clipping off, only measures it."""
        params = [p for p in network.parameters() if p.grad is not None]
        if self.max_grad_norm is None:
            return torch.nn.utils.get_total_norm([p.grad for p in params])
        return torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)

    # Denote the state as s_(player)_(turn) where player is 0-3.
    # Denote the action as a_(player)_(turn) where player is 0-3.
    #
    # Each state change returns a reward for each player.
    #
    #  s_0_0, a_0_0 -> (r_0_0, r_1_0, r_2_0, r_3_0)
    #  s_1_1, a_1_1 -> (r_0_1, r_1_1, r_2_1, r_3_1)
    #
    # The player may not change after the action is taken. That means the next
    # state can still be
    #
    #  s_1_2, a_1_2 -> (r_0_2, r_1_2, r_2_2, r_3_2)
    #  s_1_3, a_1_3 -> (r_0_3, r_1_3, r_2_3, r_3_3)
    #  s_2_4, a_2_4 -> (r_0_4, r_1_4, r_2_4, r_3_4)
    #  s_3_5, a_3_5 -> (r_0_5, r_1_5, r_2_5, r_3_5)
    #
    # Now when the player 0's turn starts again, then we can finally calculate the
    # reward from the initial state s_0_0, a_0_0.
    #
    # Reward for s_0_0, a_0_0 =
    #    r_0_0 + r_0_1 + r_0_2 + r_0_3 + r_0_4 + r_0_5
    #
    #  s_0_6, a_0_6 -> (r_0_6, r_1_6, r_2_6, r_3_6)
    #
    # Reward for s_1_1, a_1_1 = r_1_0 + r_1_1 and
    #            s_1_2, a_1_2 = r_1_2 + r_1_3 + r_1_4 + r_1_5 + r_1_6
    #
    # And so on until the game ends.
    #  s_1_7, a_1_7 -> (r_0_7, r_1_7, r_2_7, r_3_7)
    #  s_2_8, a_2_8 -> (r_0_8, r_1_8, r_2_8, r_3_8) <-- Game ends, player 2 wins
    #
    # Then we immediately resolve all rewards for each player.
    def compute_rewards(self):
        T = len(self.buf_states)
        B = len(self.buf_players[0])

        players = np.asarray(self.buf_players)
        gamed_ended_mask = np.asarray(self.buf_game_ended_mask, dtype=bool)

        accum = np.zeros((B, 4), dtype=np.float32)
        prev_action_index = np.full((B, 4), -1, dtype=np.int32)
        result = np.ndarray((T, B), dtype=np.float32)
        b_arr = np.arange(B)

        for i in range(T):
            """
            current_players = self.buf_players[i]
            for batch in range(B):
                current_player = current_players[batch]

                last_action_index = prev_player_last_action_index[batch][current_player]
                reward = self.buf_rewards[i][batch]

                if last_action_index is None:
                    prev_player_last_action_index[batch][current_player] = i
                    current_per_player_rewards[batch] += reward
                    continue

                # Compute the rewards for the last (state, action).
                actual_rewards_for_steps[(last_action_index, batch)] = (
                    current_per_player_rewards[batch][current_player]
                )

                current_per_player_rewards[batch][current_player] = 0

                # This should be the numpy sum.
                current_per_player_rewards[batch] += reward
                prev_player_last_action_index[batch][current_player] = i
            """

            # Below is the vectorized version of the above.

            # Pick the last action index
            current_players = players[i]  # [B] int

            # Pick the previous actions corresponding to each player.
            prev_actions = prev_action_index[b_arr, current_players]

            # We are using -1 to denote None.
            need_to_finalize = prev_actions >= 0
            if need_to_finalize.any():
                batch_index_to_finalize = b_arr[need_to_finalize]
                player_index_to_finalize = current_players[need_to_finalize]

                result[
                    prev_actions[batch_index_to_finalize], batch_index_to_finalize
                ] = accum[batch_index_to_finalize, player_index_to_finalize]

                accum[batch_index_to_finalize, player_index_to_finalize] = 0

            accum += self.buf_rewards[i]
            prev_action_index[b_arr, current_players] = i

            """
            # Handle the game ending condition.
            for batch in range(B):
                # The game is either ended or it is the last step of the batch,
                # we have to finalize the rewards.
                if self.buf_game_ended_mask[i][batch] or i == len(self.buf_states) - 1:
                    for player in range(4):
                        last_action_index = prev_player_last_action_index[batch][player]

                        # Rare case where the player has never acted before the
                        # batch ends.
                        if last_action_index is None:
                            continue

                        # When the game ends, then the last action index won't be None.
                        actual_rewards_for_steps[(last_action_index, batch)] = (
                            current_per_player_rewards[batch][player]
                        )

                        current_per_player_rewards[batch][player] = 0
                        prev_player_last_action_index[batch][player] = None
            """

            # Below is the vectorized version of the above.
            game_ended = gamed_ended_mask[i]
            if i == T - 1:
                game_ended = np.ones_like(game_ended)

            # Makes (B, 4).
            end_mask = game_ended[:, None] & (prev_action_index >= 0)

            # When the game ends or the batch ends, we have to finalize all
            # rewards.
            if end_mask.any():
                batch_index, player_index = np.where(end_mask)

                result[prev_action_index[batch_index, player_index], batch_index] = (
                    accum[batch_index, player_index]
                )

                accum[batch_index, player_index] = 0
                prev_action_index[batch_index, player_index] = -1

        return result

    def compute_discounted_rewards(self):
        B = len(self.buf_players[0])

        # Compute the discounted rewards for each step.
        actual_rewards_for_steps = self.compute_rewards()

        # Here R_i_j is the actual reward computed through compute_rewards.
        # s_0_0, a_0_0 -> R_0_0    <-- R_0_0 + r R_0_5 + r^2 R_0_10
        # s_1_1, a_1_1 -> R_1_1    <-- R_1_1 + r R_1_2 + r^2 R_1_6
        # s_1_2, a_1_2 -> R_1_2    <-- R_1_2 + r R_1_6
        # s_2_3, a_2_3 -> R_2_3    <-- R_2_3 + r R_2_7 + r^2 R_2_8
        # s_3_4, a_3_4 -> R_3_4    <-- R_3_4 + r R_3_9
        # s_0_5, a_0_5 -> R_0_5    <-- R_0_5 + r R_0_10
        # s_1_6, a_1_6 -> R_1_6    <-- R_1_6
        # s_2_7, a_2_7 -> R_2_7    <-- R_2_7 + r R_2_8
        # s_2_8, a_2_8 -> R_2_8    <-- R_2_8
        # s_3_9, a_3_9 -> R_3_9    <-- R_3_9
        # s_0_10, a_0_10 -> R_0_10 <-- R_0_10

        players = np.asarray(self.buf_players)
        game_ended_mask = np.asarray(self.buf_game_ended_mask, dtype=bool)
        accum = np.zeros((B, 4), dtype=np.float32)
        b_arr = np.arange(B)

        for i in reversed(range(len(self.buf_rewards))):
            # Handle the game ending condition.
            if game_ended_mask[i].any():
                # If the game ends, reset rewards.
                finished_batch = np.where(game_ended_mask[i])
                accum[finished_batch, :] = 0

            current_players = players[i]

            # For all batch, update the accumulator.
            accum[b_arr, current_players] = (
                DISCOUNT_FACTOR * accum[b_arr, current_players]
                + actual_rewards_for_steps[i]
            )

            actual_rewards_for_steps[i] = accum[b_arr, current_players]

        return actual_rewards_for_steps

    def compute_gae(
        self, values: np.ndarray, gae_lambda: float = GAE_LAMBDA
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generalized Advantage Estimation over the buffer.

        Args:
            values: (T, B) critic values V(s) recorded during collection, each
                from the perspective of the player acting at that step.
            gae_lambda: 0 = one-step TD, 1 = Monte Carlo returns.

        Returns:
            advantages (T, B) and critic targets (T, B) = advantages + values.

        Works per (game, player) like `compute_discounted_rewards`: "next"
        means the same player's next decision, and the reward is everything
        that happened to that player in between (from `compute_rewards`).
        Walking backwards, for the player acting at step i:

            delta_i = R_i + gamma * V_next - V_i
            A_i     = delta_i + gamma * lambda * A_next

        After a game ends the future is worth 0. A game still running when the
        buffer ends has no next state for its players; the engine only encodes
        the state for the player to move, so each player's own last value
        stands in for their next one.
        """
        T, B = values.shape
        rewards = self.compute_rewards()
        players = np.asarray(self.buf_players)
        game_ended_mask = np.asarray(self.buf_game_ended_mask, dtype=bool)

        next_value = np.zeros((B, 4), dtype=np.float32)
        next_advantage = np.zeros((B, 4), dtype=np.float32)
        # Whether the player has a later decision in the same game.
        has_next = np.zeros((B, 4), dtype=bool)
        # Still inside the game that the buffer cut off.
        unfinished = np.ones((B, 4), dtype=bool)

        advantages = np.zeros((T, B), dtype=np.float32)
        b_arr = np.arange(B)

        for i in reversed(range(T)):
            # Step i ended its game: rows after it belong to the next game.
            ended = game_ended_mask[i]
            if ended.any():
                has_next[ended] = False
                unfinished[ended] = False

            # p = player acting at step i; i' = p's next decision in this game.
            p = players[i]
            v = values[i]
            # V(s_i'): p's value at i' if there is one; otherwise 0 when the
            # game ended, or p's own V(s_i) when the buffer cut the game off.
            bootstrap = np.where(unfinished[b_arr, p], v, 0)
            v_next = np.where(has_next[b_arr, p], next_value[b_arr, p], bootstrap)
            # A_i': 0 when p has no later decision.
            a_next = np.where(has_next[b_arr, p], next_advantage[b_arr, p], 0)

            # delta_i = R_i + gamma * V(s_i') - V(s_i)
            delta = rewards[i] + DISCOUNT_FACTOR * v_next - v
            # A_i = delta_i + gamma * lambda * A_i'
            #     = delta_i + (gamma*lambda) delta_i' + (gamma*lambda)^2 delta_i'' + ...
            advantages[i] = delta + DISCOUNT_FACTOR * gae_lambda * a_next

            # Step i is p's "next" decision for p's earlier steps.
            next_value[b_arr, p] = v
            next_advantage[b_arr, p] = advantages[i]
            has_next[b_arr, p] = True

        return advantages, advantages + values

    def _play_eval(
        self,
        num_matches: int,
        opponent_actor: nn.Module | None,
        desc: str,
    ) -> dict[str, float]:
        return play_matches(self.actor, opponent_actor, num_matches, desc)

    def validate(
        self,
        current_iter: int,
        num_matches_random: int = 100,
        num_matches_checkpoint: int = 200,
        checkpoint_dir: str | Path = "checkpoints",
        checkpoint_step: int = 10,
        baseline_checkpoint: str | Path | None = None,
    ) -> dict[str, float]:
        """Validate the trained actor against a random agent and past
        checkpoints at 75% and 50% of `current_iter` (each floored to a
        multiple of `checkpoint_step`). Plays `num_matches_random` against the
        random agent and `num_matches_checkpoint` per checkpoint opponent.

        If `baseline_checkpoint` is given, also plays `num_matches_checkpoint`
        games against it with two trained seats and two baseline seats, so
        parity is a 50% win rate (a win for either trained seat counts). This
        compares runs, e.g. graph vs feed-forward; the other numbers only
        compare a run with itself. NaN if not given.

        Up to 2026-11-07 this was one trained seat vs three baseline seats
        (`*_vs_baseline` columns, parity 25%). The lone seat there loses a few
        points to whichever policy holds the other three seats, whatever their
        strength, so 2 vs 2 measures strength more fairly. The new columns are
        named `*_vs_baseline_2v2` so the two are never mixed up.

        Results are cached by target iter so duplicates aren't re-run:
        percentages that floor to the same checkpoint share a single eval.
        A percentage that floors to 0 falls back to the random agent and is
        re-run at `num_matches_checkpoint` so its sample size matches the
        other checkpoint comparisons. Win rate / avg steps are NaN if a
        positive target's checkpoint is missing."""
        checkpoint_dir = Path(checkpoint_dir)
        out: dict[str, float] = {}

        random_metrics = self._play_eval(
            num_matches_random, opponent_actor=None, desc="validate vs random"
        )
        out["win_rate_vs_random"] = random_metrics["win_rate"]
        out["avg_steps_vs_random"] = random_metrics["avg_steps"]
        out["avg_vp_vs_random"] = random_metrics["avg_vp"]
        out["avg_opp_vp_vs_random"] = random_metrics["avg_opp_vp"]

        cache: dict[int, dict[str, float]] = {}

        for pct, label in [(0.75, "75"), (0.5, "50")]:
            target_iter = int(pct * current_iter)
            target_iter = (target_iter // checkpoint_step) * checkpoint_step

            if target_iter not in cache:
                if target_iter == 0:
                    cache[target_iter] = self._play_eval(
                        num_matches_checkpoint,
                        opponent_actor=None,
                        desc=f"validate vs {label}% (random)",
                    )
                else:
                    ckpt_path = checkpoint_dir / f"actor_iter_{target_iter}.pt"
                    if not ckpt_path.exists():
                        cache[target_iter] = {
                            "win_rate": float("nan"),
                            "avg_steps": float("nan"),
                            "avg_vp": float("nan"),
                            "avg_opp_vp": float("nan"),
                        }
                    else:
                        opponent = load_actor(ckpt_path, DEVICE)
                        cache[target_iter] = self._play_eval(
                            num_matches_checkpoint,
                            opponent_actor=opponent,
                            desc=f"validate vs {label}% (iter {target_iter})",
                        )

            metrics = cache[target_iter]
            out[f"win_rate_vs_{label}"] = metrics["win_rate"]
            out[f"avg_steps_vs_{label}"] = metrics["avg_steps"]
            out[f"avg_vp_vs_{label}"] = metrics["avg_vp"]
            out[f"avg_opp_vp_vs_{label}"] = metrics["avg_opp_vp"]

        if baseline_checkpoint is not None:
            metrics = play_matches(
                self.actor,
                load_actor(baseline_checkpoint, DEVICE),
                num_matches_checkpoint,
                desc="validate 2 vs 2 baseline",
                num_seats=2,
            )
        else:
            metrics = dict.fromkeys(
                ("win_rate", "avg_steps", "avg_vp", "avg_opp_vp"), float("nan")
            )
        out["win_rate_vs_baseline_2v2"] = metrics["win_rate"]
        out["avg_steps_vs_baseline_2v2"] = metrics["avg_steps"]
        out["avg_vp_vs_baseline_2v2"] = metrics["avg_vp"]
        out["avg_opp_vp_vs_baseline_2v2"] = metrics["avg_opp_vp"]

        return out

    def step(self):
        states, players = self.engine.state()

        # (batch_size, 284) bool
        masks = torch.from_numpy(self.engine.get_action_mask()).bool().to(DEVICE)

        state_tensor = torch.from_numpy(states).float().to(DEVICE)
        logits = self._actor(state_tensor)

        # Set invalid actions to -inf.
        logits[~masks] = -float("inf")
        dist = Categorical(logits=logits)

        # (batch_size,)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        value = self._critic(state_tensor)

        rewards, game_ended_mask, _ = self.engine.step(actions.cpu().numpy())

        return (
            state_tensor,
            players,
            actions,
            log_probs,
            value,
            rewards,
            game_ended_mask,
            masks,
        )
