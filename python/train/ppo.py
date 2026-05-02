from pathlib import Path

import numpy as np
from catan_engine import Engine
from train.network import FeedForwardNN
from torch.distributions import Categorical
from tqdm import tqdm
import torch

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DISCOUNT_FACTOR = 0.996
EPSILON = 0.2

ACTOR_INPUT_SIZE = 732
ACTOR_OUTPUT_SIZE = 300


def load_actor(path: str | Path, device: str | torch.device = DEVICE) -> FeedForwardNN:
    """Load a trained actor from `path` for inference (sets eval mode)."""
    actor = FeedForwardNN(input_size=ACTOR_INPUT_SIZE, output_size=ACTOR_OUTPUT_SIZE)
    actor.load_state_dict(torch.load(path, map_location=device))
    actor.to(device)
    actor.eval()
    return actor


class PPO:
    def __init__(self, batch_size: int):
        self.actor = FeedForwardNN(
            input_size=ACTOR_INPUT_SIZE, output_size=ACTOR_OUTPUT_SIZE
        ).to(DEVICE)
        self.critic = FeedForwardNN(input_size=ACTOR_INPUT_SIZE, output_size=1).to(
            DEVICE
        )
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
        """Persist the actor's weights to `path` (parent dirs created if needed)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), path)

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
        returns = torch.from_numpy(self.compute_discounted_rewards()).to(DEVICE)

        # stack(values) is (T, B, 1)
        values = torch.stack(self.buf_values).squeeze(-1).to(DEVICE)

        advantages = returns - values
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
                logits = self.actor(mb_states)
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
                    (self.critic(mb_states).squeeze(-1) - mb_returns) ** 2
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

    def _play_eval(
        self,
        num_matches: int,
        opponent_actor: FeedForwardNN | None,
        desc: str,
    ) -> dict[str, float]:
        """Play `num_matches` games with the trained actor at a randomized
        seat. Other seats are filled by `opponent_actor` (argmax) if provided,
        otherwise by a uniform-random agent over valid actions."""
        engine = Engine(batch_size=num_matches, seed=SEED + 10_000)
        rng = np.random.default_rng(SEED + 10_001)
        trained_seats = rng.integers(0, 4, size=num_matches, dtype=np.int64)

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

        self.actor.eval()
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

                    trained_idx = np.where(players == trained_seats)[0]
                    if trained_idx.size > 0:
                        actions[trained_idx] = argmax_actions(
                            self.actor, states, masks, trained_idx
                        )

                    opp_idx = np.where(players != trained_seats)[0]
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
                            seat = int(trained_seats[i])
                            if rewards[i][seat] > 0:
                                wins += 1
                            total_vp += int(final_vps[i, seat])
                            total_opp_vp += int(final_vps[i].sum()) - int(
                                final_vps[i, seat]
                            )
                            total_steps += int(steps_since_start[i])
                            finished += 1
                            pbar.update(1)
                        steps_since_start[i] = 0
        finally:
            self.actor.train()

        return {
            "win_rate": wins / finished if finished > 0 else 0.0,
            "avg_steps": total_steps / finished if finished > 0 else 0.0,
            "avg_vp": total_vp / finished if finished > 0 else 0.0,
            "avg_opp_vp": total_opp_vp / (finished * 3) if finished > 0 else 0.0,
        }

    def validate(
        self,
        current_iter: int,
        num_matches_random: int = 100,
        num_matches_checkpoint: int = 200,
        checkpoint_dir: str | Path = "checkpoints",
        checkpoint_step: int = 10,
    ) -> dict[str, float]:
        """Validate the trained actor against a random agent and past
        checkpoints at 90%, 75%, and 50% of `current_iter` (each floored to a
        multiple of `checkpoint_step`). Plays `num_matches_random` against the
        random agent and `num_matches_checkpoint` per checkpoint opponent.

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

        for pct, label in [(0.9, "90"), (0.75, "75"), (0.5, "50")]:
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

        return out

    def step(self):
        states, players = self.engine.state()

        # (batch_size, 284) bool
        masks = torch.from_numpy(self.engine.get_action_mask()).bool().to(DEVICE)

        state_tensor = torch.from_numpy(states).float().to(DEVICE)
        logits = self.actor.forward(state_tensor)

        # Set invalid actions to -inf.
        logits[~masks] = -float("inf")
        dist = Categorical(logits=logits)

        # (batch_size,)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        value = self.critic.forward(state_tensor)

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
