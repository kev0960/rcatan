import numpy as np
from catan_engine import Engine
from train.network import FeedForwardNN
from torch.distributions import Categorical
import torch

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DISCOUNT_FACTOR = 0.99
EPSILON = 0.1

"""
  actor  = FeedForwardNN(732, 284)   # outputs logits
  critic = FeedForwardNN(732, 1)     # outputs V(s)
  engine = Engine(batch_size=B, seed=...)

  for iteration in range(NUM_ITERS):
      # ---- 1. Rollout: collect T steps of on-policy data ----
      buf_states, buf_players, buf_actions, buf_masks = [], [], [], []
      buf_logp, buf_values, buf_rewards               = [], [], []

      for t in range(T):
          states, players = engine.state()              # (B,732), (B,)
          mask   = engine.get_action_mask()             # (B,284) bool

          logits = actor(states)
          logits[~mask] = -inf                          # mask invalid actions
          dist   = Categorical(logits=logits)
          action = dist.sample()                        # (B,)
          logp   = dist.log_prob(action)                # (B,)
          value  = critic(states).squeeze(-1)           # (B,)

          reward = engine.step(action)                  # (B,4) per-player

          # Detach logp/value: at update time they are rollout-time
          # *constants* (logp is the denominator of the ratio; value is
          # used only to form the detached advantage). Storing them with
          # grad history would keep the rollout graph alive across T steps.
          buf_states.append(states);   buf_players.append(players)
          buf_actions.append(action);  buf_masks.append(mask)
          buf_logp.append(logp.detach())
          buf_values.append(value.detach()); buf_rewards.append(reward)

      # ---- 2. Compute per-action returns & advantages ----
      # (T,B) reward attributed to each action via the logic we built
      returns    = PPO.compute_rewards(stack(buf_players), stack(buf_rewards))
      advantages = returns - stack(buf_values)
      advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

      # Flatten (T,B,*) -> (T*B,*) for the update. All tensors must be
      # flattened the same way so indices line up across states/masks/
      # actions/logp/returns/adv.
      flat_states   = stack(buf_states).flatten(0,1)    # (N, 732)   N = T*B
      flat_mask     = stack(buf_masks).flatten(0,1)     # (N, 284)
      flat_actions  = stack(buf_actions).flatten(0,1)   # (N,)
      flat_logp_old = stack(buf_logp).flatten(0,1)      # (N,)
      flat_returns  = returns.flatten(0,1)              # (N,)
      flat_adv      = advantages.flatten(0,1)           # (N,)

      # ---- 3. K epochs of minibatch updates over the SAME buffer ----
      # This is what makes PPO "proximal". On epoch=0 minibatch=0 the
      # weights still match the behavior policy, so ratio == 1 exactly
      # and the clip is inactive — a single step here is just vanilla
      # policy gradient. The clip only starts doing work on subsequent
      # minibatches, once the policy has drifted from π_θ_old. Without
      # this loop, we'd throw away the rollout after one update; the
      # ratio/clip machinery would be dead weight.
      N = flat_states.shape[0]
      for epoch in range(K):                            # e.g. K = 4..10
          perm = randperm(N)
          for start in range(0, N, M):                  # M = minibatch size
              idx = perm[start:start+M]
              mb_states   = flat_states[idx]
              mb_mask     = flat_mask[idx]
              mb_actions  = flat_actions[idx]
              mb_logp_old = flat_logp_old[idx]
              mb_returns  = flat_returns[idx]
              mb_adv      = flat_adv[idx]

              logits = actor(mb_states)
              logits[~mb_mask] = -inf                   # same mask as rollout
              dist_new = Categorical(logits=logits)
              logp_new = dist_new.log_prob(mb_actions)  # π_θ(a_t | s_t)

              ratio = exp(logp_new - mb_logp_old)       # ==1 only on epoch=0,mb=0
              unclipped = ratio * mb_adv
              clipped   = clip(ratio, 1-eps, 1+eps) * mb_adv
              actor_loss  = -min(unclipped, clipped).mean()
              critic_loss = ((critic(mb_states).squeeze(-1) - mb_returns)**2).mean()
              entropy     = dist_new.entropy().mean()

              loss = actor_loss + 0.5*critic_loss - 0.01*entropy
              optimizer.zero_grad(); loss.backward(); optimizer.step()
"""


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
        for step in range(steps):
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

    def train(self):
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

        M = 128
        N = flat_states.shape[0]

        for epoch in range(10):
            perm = torch.randperm(N)
            for start in range(0, N, M):
                idx = perm[start : start + M]
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
