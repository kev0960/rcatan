import numpy as np
from catan_engine import Engine
from train.network import FeedForwardNN
from torch.distributions import Categorical
from tqdm import tqdm
import torch

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DISCOUNT_FACTOR = 0.99
EPSILON = 0.2


class PPO:
    def __init__(self, batch_size: int):
        self.actor = FeedForwardNN(input_size=732, output_size=284).to(DEVICE)
        self.critic = FeedForwardNN(input_size=732, output_size=1).to(DEVICE)
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
            self.buf_rewards.append(np.array(rewards))
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

        for _ in tqdm(range(10), desc="train", leave=False):
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

    def validate(self, num_matches: int = 100) -> dict[str, float]:
        """Play `num_matches` games with the trained actor at a randomized seat
        and random agents at the other seats. Returns the trained agent's win
        rate and the average number of engine steps per finished game."""
        engine = Engine(batch_size=num_matches, seed=SEED + 10_000)
        rng = np.random.default_rng(SEED + 10_001)
        trained_seats = rng.integers(0, 4, size=num_matches, dtype=np.int64)

        wins = 0
        finished = 0
        total_steps = 0
        steps_since_start = np.zeros(num_matches, dtype=np.int64)

        self.actor.eval()
        try:
            with torch.no_grad():
                while finished < num_matches:
                    states, players = engine.state()
                    masks_np = np.asarray(engine.get_action_mask(), dtype=bool)
                    players_np = np.asarray(players)

                    actions = np.zeros(num_matches, dtype=np.int64)

                    trained_idx = np.where(players_np == trained_seats)[0]
                    if trained_idx.size > 0:
                        state_tensor = (
                            torch.tensor(np.asarray(states)[trained_idx])
                            .float()
                            .to(DEVICE)
                        )
                        logits = self.actor(state_tensor)
                        mask_tensor = (
                            torch.tensor(masks_np[trained_idx]).bool().to(DEVICE)
                        )
                        logits[~mask_tensor] = -float("inf")
                        actions[trained_idx] = (
                            torch.argmax(logits, dim=-1).cpu().numpy()
                        )

                    for i in np.where(players_np != trained_seats)[0]:
                        valid = np.flatnonzero(masks_np[i])
                        actions[i] = rng.choice(valid)

                    rewards, game_ended_mask = engine.step(actions.tolist())
                    steps_since_start += 1

                    for i, ended in enumerate(game_ended_mask):
                        if not ended:
                            continue
                        if finished < num_matches:
                            if rewards[i][trained_seats[i]] > 0:
                                wins += 1
                            total_steps += int(steps_since_start[i])
                            finished += 1
                        steps_since_start[i] = 0
        finally:
            self.actor.train()

        return {
            "win_rate": wins / finished if finished > 0 else 0.0,
            "avg_steps": total_steps / finished if finished > 0 else 0.0,
        }

    def step(self):
        states, players = self.engine.state()

        # (batch_size, 284) bool
        masks = torch.tensor(self.engine.get_action_mask()).bool().to(DEVICE)

        state_tensor = torch.tensor(states).float().to(DEVICE)
        logits = self.actor.forward(state_tensor)

        # Set invalid actions to -inf.
        logits[~masks] = -float("inf")
        dist = Categorical(logits=logits)

        # (batch_size,)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        value = self.critic.forward(state_tensor)

        rewards, game_ended_mask = self.engine.step(actions.cpu().numpy())

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
