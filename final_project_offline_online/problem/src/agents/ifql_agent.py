from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Callable, Optional, Sequence, Tuple, List


class IFQLAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_actor_flow,
        make_actor_flow_optimizer,
        make_critic,
        make_critic_optimizer,
        make_value,
        make_value_optimizer,

        discount: float,
        target_update_rate: float,
        flow_steps: int,
        online_training: bool = False,
        num_samples: int = 32,
        expectile: float = 0.9,
        rho: float = 0.5,
    ):
        super().__init__()

        self.action_dim = action_dim
        
        # TODO(student): Create flow actor
        self.online_training = online_training
        self.actor_flow = make_actor_flow(observation_shape, action_dim)

        # TODO(student): Create critic (ensemble of Q-functions), target critic (ensemble of Q-functions), and value function
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.value = make_value(observation_shape)

        # TODO(student): Create optimizers for all the above models
        self.actor_flow_optimizer = make_actor_flow_optimizer(self.actor_flow.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())
        self.value_optimizer = make_value_optimizer(self.value.parameters())

        self.discount = discount
        self.target_update_rate = target_update_rate
        self.flow_steps = flow_steps
        self.num_samples = num_samples
        self.expectile = expectile

    @staticmethod
    def expectile_loss(adv: torch.Tensor, expectile: float) -> torch.Tensor:
        """
        Compute the expectile loss for IFQL
        adv = V(s) - min_i Q̄_i(s, a)
        ℓ²_τ(x) = |τ - I(x > 0)| * x²
        """
        # TODO(student): Implement the expectile loss
        weights = torch.where(adv > 0,
                              torch.ones_like(adv) * (1 - expectile),
                              torch.ones_like(adv) * expectile)
        return (weights * adv.pow(2)).mean()

    @torch.compile
    def update_value(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict:
        """
        Update value function
        """
        # TODO(student): Implement the value function update
        with torch.no_grad():
            q_min = self.target_critic(observations, actions).min(dim=0).values

        v = self.value(observations)
        adv = v - q_min
        loss = self.expectile_loss(adv, self.expectile)

        # TODO(student): Update value function
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()

        return {
            "value_loss": loss,
            "value_mean": v.mean(),
            "adv_mean": adv.mean(),
        }

    @torch.no_grad()
    def sample_actions(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Rejection / best-of-n sampling using the flow policy and critic.

        We:
          1. Sample multiple candidate actions via the BC flow.
          2. Evaluate them with the critic.
          3. Pick the action with the highest Q-value.
        """
        # TODO(student): Implement rejection sampling
        B = observations.shape[0]
        N = self.num_samples

        obs_exp = observations.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)
        z = torch.randn(B * N, self.action_dim, dtype=observations.dtype, device=observations.device)

        cand_actions = self.get_flow_action(obs_exp, z)  # (B*N, ac_dim)

        qs = self.target_critic(obs_exp, cand_actions)  # (2, B*N)
        q_min = qs.min(dim=0).values  # (B*N,)
        q_min = q_min.reshape(B, N)

        best_idx = q_min.argmax(dim=1)  # (B,)
        cand_actions = cand_actions.reshape(B, N, self.action_dim)
        best_actions = cand_actions[torch.arange(B, device=observations.device), best_idx]

        return best_actions

    def get_action(self, observation: np.ndarray):
        """
        Used for evaluation.
        """
        # TODO(student): Implement get action
        observation = ptu.from_numpy(np.asarray(observation))[None]
        action = self.sample_actions(observation)
        return ptu.to_numpy(action[0])

    @torch.compile
    def get_flow_action(self, observation: torch.Tensor, noise: torch.Tensor):
        """
        Compute the flow action using Euler integration for `self.flow_steps` steps.
        """
        # TODO(student): Implement euler integration to get flow action
        action_x = noise
        dt = 1.0 / self.flow_steps
        for step in range(self.flow_steps):
            t = torch.full(
                (observation.shape[0], 1),
                step * dt,
                device=observation.device,
                dtype=observation.dtype,
            )
            v = self.actor_flow(observation, action_x, t)
            action_x = action_x + dt * v
        return torch.clamp(action_x, -1, 1)

    @torch.compile
    def update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """
        Update Q(s, a) using the learned value function for bootstrapping,
        as in IFQL / IQL-style critic training.
        """
        # TODO(student): Implement Q-function update
        with torch.no_grad():
            next_v = self.value(next_observations)
            y = rewards + self.discount * (1 - dones) * next_v

        q = self.critic(observations, actions)  # (2, B)
        loss = ((q - y.unsqueeze(0)) ** 2).mean()

        # TODO(student): Update Q-function
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
            "q_loss": loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }

    @torch.compile
    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the flow actor using the velocity matching loss.
        """
        # TODO(student): Implement flow actor update
        z = torch.randn_like(actions)
        t = torch.rand(actions.shape[0], 1, device=actions.device, dtype=actions.dtype)
        a_tilde = (1 - t) * z + t * actions
        v_pred = self.actor_flow(observations, a_tilde, t)
        target = actions - z
        loss = (v_pred - target).pow(2).mean(dim=-1).mean()

        # TODO(student): Update flow actor
        self.actor_flow_optimizer.zero_grad()
        loss.backward()
        self.actor_flow_optimizer.step()

        return {
            "actor_loss": loss,
        }

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        metrics_v = self.update_value(observations, actions)
        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        metrics_actor = self.update_actor(observations, actions)
        metrics = {
            **{f"value/{k}": v.item() for k, v in metrics_v.items()},
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"actor/{k}": v.item() for k, v in metrics_actor.items()},
        }

        self.update_target_critic()

        return metrics

    def update_target_critic(self) -> None:
        # TODO(student): Update target_critic using Polyak averaging with self.target_update_rate
        tau = self.target_update_rate
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
                tp.data.mul_(1 - tau).add_(tau * p.data)
