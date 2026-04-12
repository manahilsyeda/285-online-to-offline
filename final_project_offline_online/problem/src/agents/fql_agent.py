from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Callable, Optional, Sequence, Tuple, List


class FQLAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_bc_actor,
        make_bc_actor_optimizer,
        make_onestep_actor,
        make_onestep_actor_optimizer,
        make_critic,
        make_critic_optimizer,

        discount: float,
        target_update_rate: float,
        flow_steps: int,
        alpha: float,
    ):
        super().__init__()

        self.action_dim = action_dim

        self.bc_actor = make_bc_actor(observation_shape, action_dim)
        self.onestep_actor = make_onestep_actor(observation_shape, action_dim)
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.bc_actor_optimizer = make_bc_actor_optimizer(self.bc_actor.parameters())
        self.onestep_actor_optimizer = make_onestep_actor_optimizer(self.onestep_actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())

        self.discount = discount
        self.target_update_rate = target_update_rate
        self.flow_steps = flow_steps
        self.alpha = alpha

    def _onestep_action(self, observation: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """One-step flow policy: single Euler step from noise z with step size 1."""
        return noise + self.onestep_actor(observation, noise)

    def get_action(self, observation: np.ndarray):
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]
        # TODO(student): Compute the action for evaluation
        # Hint: Unlike SAC+BC and IQL, the evaluation action is *sampled* (i.e., not the mode or mean) from the policy
        z = torch.randn(1, self.action_dim, dtype=observation.dtype, device=observation.device)
        one_step_action = self._onestep_action(observation, z)
        action = torch.clamp(one_step_action, -1, 1)
        return ptu.to_numpy(action)[0]

    @torch.compile
    def get_bc_action(self, observation: torch.Tensor, noise: torch.Tensor):
        """
        Used for training.
        """
        # TODO(student): Compute the BC flow action using the Euler method for `self.flow_steps` steps
        # Hint: This function should *only* be used in `update_onestep_actor`        
        action_x = noise
        dt = 1.0 / self.flow_steps
        for step in range(self.flow_steps):
            t = torch.full(
                (observation.shape[0], 1),
                step * dt,
                device=observation.device,
                dtype=observation.dtype,
            )
            v = self.bc_actor(observation, action_x, t)
            action_x = action_x + dt * v
        action = torch.clamp(action_x, -1, 1)
        return action

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
        Update Q(s, a)
        """
        # TODO(student): Compute the Q loss
        # Hint: Use the one-step actor to compute next actions
        # Hint: Remember to clamp the actions to be in [-1, 1] when feeding them to the critic!
        with torch.no_grad():
            z = torch.randn_like(actions)
            next_actions = self._onestep_action(next_observations, z)
            next_actions = torch.clamp(next_actions, -1, 1)
            next_qs = self.target_critic(next_observations, next_actions)
            y = rewards.squeeze(-1) + (1.0 - dones.squeeze(-1)) * self.discount * 0.5 * next_qs.sum(
                dim=0
            )

        q = self.critic(observations, actions)
        loss = ((q - y.unsqueeze(0)) ** 2).mean()

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
    def update_bc_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the BC actor
        """
        # TODO(student): Compute the BC flow loss
        z = torch.randn_like(actions)
        t = torch.rand(actions.shape[0], 1, device=actions.device, dtype=actions.dtype)
        a_tilde = (1.0 - t) * z + t * actions
        v_pred = self.bc_actor(observations, a_tilde, t)
        target = actions - z
        loss = (v_pred - target).pow(2).mean(dim=-1).mean()

        self.bc_actor_optimizer.zero_grad()
        loss.backward()
        self.bc_actor_optimizer.step()

        return {
            "loss": loss,
        }

    @torch.compile
    def update_onestep_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the one-step actor
        """
        # TODO(student): Compute the one-step actor loss
        # Hint: Do *not* clip the one-step actor actions when computing the distillation loss
                
        z = torch.randn_like(actions)
        pi_omega = self._onestep_action(observations, z)

        with torch.no_grad():
            pi_v = self.get_bc_action(observations, z)

        distill_loss = self.alpha * (pi_omega - pi_v).pow(2).mean(dim=-1).mean()

        pi_for_q = torch.clamp(pi_omega, -1, 1)
        qs = self.critic(observations, pi_for_q)
        # Hint: *Do* clip the one-step actor actions when feeding them to the critic
        q_loss = -(0.5 * qs.sum(dim=0)).mean()

        # Total loss.
        loss = distill_loss + q_loss

        # Additional metrics for logging.
        mse = (pi_omega - pi_v).pow(2).mean()

        self.onestep_actor_optimizer.zero_grad()
        loss.backward()
        self.onestep_actor_optimizer.step()

        return {
            "total_loss": loss,
            "distill_loss": distill_loss,
            "q_loss": q_loss,
            "mse": mse,
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
        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        metrics_bc_actor = self.update_bc_actor(observations, actions)
        metrics_onestep_actor = self.update_onestep_actor(observations, actions)
        metrics = {
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"bc_actor/{k}": v.item() for k, v in metrics_bc_actor.items()},
            **{f"onestep_actor/{k}": v.item() for k, v in metrics_onestep_actor.items()},
        }

        self.update_target_critic()

        return metrics

    def update_target_critic(self) -> None:
        # TODO(student): Update target_critic using Polyak averaging with self.target_update_rate
        tau = self.target_update_rate
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
                tp.data.mul_(1.0 - tau).add_(tau * p.data)
