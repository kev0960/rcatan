from pathlib import Path

import numpy as np
import torch
from catan_engine import Engine
from torch import nn
from torch.distributions import Categorical
from tqdm import tqdm

from train.network_ff import FeedForwardNN
from train.network_graph import GraphActor, GraphCritic

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DISCOUNT_FACTOR = 0.996
EPSILON = 0.2
GAE_LAMBDA = 0.95

# How advantages and critic targets are computed: "gae" (GAE(lambda), see
# `compute_gae`) or "mc" (Monte Carlo discounted returns).
ADVANTAGES = ("gae", "mc")
DEFAULT_ADVANTAGE = "gae"

# Actor, critic and optimizer state saved in each run's checkpoint dir so
# training can resume; see `PPO.save_training_state`.
TRAINING_STATE = "train_latest.pt"

ACTOR_INPUT_SIZE = 2016
ACTOR_OUTPUT_SIZE = 300

# Network architectures for actor and critic: "graph" or "ff" (feed-forward).
NETWORKS = ("graph", "ff")
DEFAULT_NETWORK = "graph"


def build_actor(network: str) -> nn.Module:
    if network == "graph":
        return GraphActor(torch.from_numpy(Engine.road_endpoints()))
    if network == "ff":
        return FeedForwardNN(input_size=ACTOR_INPUT_SIZE, output_size=ACTOR_OUTPUT_SIZE)
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
    bare feed-forward state dict. Raises ValueError if the checkpoint was
    trained on a different state size than the engine now produces."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if "state_dict" in checkpoint:
        network = checkpoint["network"]
        state_len = checkpoint["state_len"]
        state_dict = checkpoint["state_dict"]
    else:
        network, state_dict = "ff", checkpoint
        state_len = state_dict["fc1.weight"].shape[1]

    if state_len != ACTOR_INPUT_SIZE:
        raise ValueError(
            f"{path} was trained on a {state_len}-float state, but the engine "
            f"now produces {ACTOR_INPUT_SIZE}; retrain it"
        )

    actor = build_actor(network)
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


def find_resume_checkpoint(run_dir: str | Path) -> tuple[Path, str]:
    """Checkpoint to resume `run_dir` from, and the network it was trained
    with. Prefers the full training state; runs saved before it existed fall
    back to their newest `actor_iter_N.pt` (actor only)."""
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
    network = checkpoint["network"] if "network" in checkpoint else "ff"
    return path, network


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
    ):
        if advantage not in ADVANTAGES:
            raise ValueError(f"unknown advantage: {advantage!r} (expected {ADVANTAGES})")
        self.network = network
        self.advantage = advantage
        self.gae_lambda = gae_lambda
        self.actor = build_actor(network).to(DEVICE)
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

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)

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
        along with its network and state size so `load_actor` can rebuild it."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "network": self.network,
                "state_len": ACTOR_INPUT_SIZE,
                "state_dict": self.actor.state_dict(),
            },
            path,
        )

    def save_training_state(self, path: str | Path, iteration: int) -> None:
        """Persist everything needed to resume training after `iteration`
        completed iterations: actor, critic and both optimizers."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "network": self.network,
                "state_len": ACTOR_INPUT_SIZE,
                "iteration": iteration,
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
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
        if checkpoint.get("network", "ff") != self.network:
            raise ValueError(
                f"{path} is a {checkpoint.get('network', 'ff')} checkpoint, "
                f"but this run uses {self.network}"
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
            self.buf_log_probs.append(log_probs.detach())
            self.buf_values.append(value.detach())
            self.buf_rewards.append(rewards)
            self.buf_game_ended_mask.append(game_ended_mask)
            self.buf_masks.append(masks)

    def train(self, mini_batch_size: int):
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

        sum_actor_loss = 0.0
        sum_critic_loss = 0.0
        sum_entropy = 0.0
        sum_loss = 0.0
        sum_approx_kl = 0.0
        sum_clip_frac = 0.0
        n_updates = 0

        for _ in tqdm(range(4), desc="train", leave=False):
            perm = torch.randperm(N)
            for start in range(0, N, mini_batch_size):
                idx = perm[start : start + mini_batch_size]
                mb_states = flat_states[idx]
                mb_mask = flat_mask[idx]
                mb_actions = flat_actions[idx]
                mb_log_probs_old = flat_log_probs_old[idx]
                mb_returns = flat_returns[idx]
                mb_advantages = flat_advantages[idx]

                # Computing the pi_theta (a_t | s_t) on the updated actor
                logits = self._actor(mb_states)
                logits[~mb_mask] = -float("inf")
                dist = Categorical(logits=logits)
                logp_new = dist.log_prob(mb_actions)

                # Compute pi_theta (a_t | s_t) / pi_theta_old (a_t | s_t)
                ratio = torch.exp(logp_new - mb_log_probs_old)

                unclipped = ratio * mb_advantages
                clipped = torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON) * mb_advantages

                # Minimize the PPO-Clip objective.
                actor_loss = -torch.min(unclipped, clipped).mean()

                # Value function fitting.
                critic_loss = (
                    (self._critic(mb_states).squeeze(-1) - mb_returns) ** 2
                ).mean()

                entropy = dist.entropy().mean()

                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                loss.backward()

                self.actor_optimizer.step()
                self.critic_optimizer.step()

                sum_actor_loss += actor_loss.item()
                sum_critic_loss += critic_loss.item()
                sum_entropy += entropy.item()
                sum_loss += loss.item()

                # How far this update moved the policy: KL(old || new)
                # estimated as E[(r - 1) - log r], and the share of samples
                # where the ratio left the clip range.
                with torch.no_grad():
                    log_ratio = logp_new - mb_log_probs_old
                    sum_approx_kl += ((ratio - 1) - log_ratio).mean().item()
                    sum_clip_frac += (
                        ((ratio - 1).abs() > EPSILON).float().mean().item()
                    )
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
            "loss": sum_loss / n_updates,
            "actor_loss": sum_actor_loss / n_updates,
            "critic_loss": sum_critic_loss / n_updates,
            "entropy": sum_entropy / n_updates,
            "return_mean": flat_returns.mean().item(),
            "return_std": flat_returns.std().item(),
            "value_mean": values.mean().item(),
            "value_std": values.std().item(),
            "explained_var": explained_var.item(),
            "explained_var_mc": explained_var_mc.item(),
            "approx_kl": sum_approx_kl / n_updates,
            "clip_frac": sum_clip_frac / n_updates,
        }

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
        games against it (one trained seat vs three baseline seats, so parity
        is a 25% win rate). This compares runs, e.g. graph vs feed-forward;
        the other numbers only compare a run with itself. NaN if not given.

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
            metrics = self._play_eval(
                num_matches_checkpoint,
                opponent_actor=load_actor(baseline_checkpoint, DEVICE),
                desc="validate vs baseline",
            )
        else:
            metrics = dict.fromkeys(
                ("win_rate", "avg_steps", "avg_vp", "avg_opp_vp"), float("nan")
            )
        out["win_rate_vs_baseline"] = metrics["win_rate"]
        out["avg_steps_vs_baseline"] = metrics["avg_steps"]
        out["avg_vp_vs_baseline"] = metrics["avg_vp"]
        out["avg_opp_vp_vs_baseline"] = metrics["avg_opp_vp"]

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
