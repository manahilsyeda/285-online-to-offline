from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Sequence


class DSRLAgent(nn.Module):
    """DSRL agent - https://arxiv.org/abs/2506.15799"""

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_bc_flow_actor,
        make_bc_flow_actor_optimizer,
        make_noise_actor,
        make_noise_actor_optimizer,
        make_critic,
        make_critic_optimizer,
        make_z_critic,
        make_z_critic_optimizer,

        discount: float,
        target_update_rate: float,
        flow_steps: int,
        noise_scale: float = 1.0,

        online_training: bool = False,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.discount = discount
        self.target_update_rate = target_update_rate
        self.flow_steps = flow_steps
        self.noise_scale = noise_scale
        self.target_entropy = -action_dim

        # TODO(student): Create BC flow actor and target BC flow actor
        self.bc_flow_actor = make_bc_flow_actor(observation_shape, action_dim)
        self.target_bc_flow_actor = make_bc_flow_actor(observation_shape, action_dim)
        self.target_bc_flow_actor.load_state_dict(self.bc_flow_actor.state_dict())

        # TODO(student): Create noise policy
        self.noise_actor = make_noise_actor(observation_shape, action_dim)

        # TODO(student): Create critic (ensemble of Q-functions), target critic (ensemble of Q-functions), and z critic (for noise policy)
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.z_critic = make_z_critic(observation_shape, action_dim)

        # TODO(student): Create learnable entropy coefficient
        self.log_alpha = nn.Parameter(torch.zeros(1, device=ptu.device))

        # TODO(student): Create optimizers for all the above models
        self.bc_flow_actor_optimizer = make_bc_flow_actor_optimizer(self.bc_flow_actor.parameters())
        self.noise_actor_optimizer = make_noise_actor_optimizer(self.noise_actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())
        self.z_critic_optimizer = make_z_critic_optimizer(self.z_critic.parameters())
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)

        self.to(ptu.device)

    @property
    def alpha(self):
        # TODO(student): Allow access to the learnable entropy coefficient (tip: if you are learning log alpha, as in HW3, then when we want to use alpha, you should return the exponential of the log alpha)
        return self.log_alpha.exp()

    @torch.compiler.disable
    def sample_flow_actions(self, observations: torch.Tensor, noises: torch.Tensor) -> torch.Tensor:
        """Euler integration of BC flow from t=0 to t=1."""
        # TODO(student): Implement Euler integration of BC flow. Keep in mind that the target BC flow actor should be used
        # Also note that we can control what we use as the noise input (could be sampled from a noise policy or from a normal distribution)
        action_x = noises
        dt = 1.0 / self.flow_steps
        for step in range(self.flow_steps):
            t = torch.full(
                (observations.shape[0], 1),
                step * dt,
                device=observations.device,
                dtype=observations.dtype,
            )
            v = self.target_bc_flow_actor(observations, action_x, t)
            action_x = action_x + dt * v
        return torch.clamp(action_x, -1, 1)

    @torch.no_grad()
    def sample_actions(self, observations: torch.Tensor) -> torch.Tensor:
        """Sample actions using noise policy for noise input to BC flow policy."""
        # TODO(student): Sample noise from the noise policy and use to sample actions from the BC flow policy
        noise_dist = self.noise_actor(observations)
        z = noise_dist.rsample()
        scaled_z = self.noise_scale * z
        return self.sample_flow_actions(observations, scaled_z)

    def get_action(self, observation: np.ndarray):
        """Used for evaluation."""
        # TODO(student): Implement get action
        observation = ptu.from_numpy(np.asarray(observation))[None]
        action = self.sample_actions(observation)
        return ptu.to_numpy(action[0])

    def update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """Update critic"""
        # TODO(student): Implement critic loss
        with torch.no_grad():
            next_noise_dist = self.noise_actor(next_observations)
            next_z = next_noise_dist.rsample()
            next_scaled_z = self.noise_scale * next_z
            next_actions = self.sample_flow_actions(next_observations, next_scaled_z)
            next_qs = self.target_critic(next_observations, next_actions)
            next_v = next_qs.mean(dim=0)
            y = rewards + self.discount * (1 - dones) * next_v

        q = self.critic(observations, actions)
        loss = ((q - y.unsqueeze(0)) ** 2).mean()

        # TODO(student): Update critic
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
            "q_loss": loss,
            "q_mean": q.mean(),
        }

    def update_qz(self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        noises: torch.Tensor,
    ) -> dict:
        """Update z_critic."""

        # TODO(student): Implement z_critic loss
        with torch.no_grad():
            bc_actions = self.sample_flow_actions(observations, noises)
            target_qs = self.critic(observations, bc_actions)
            y = target_qs.mean(dim=0)

        qz = self.z_critic(observations, noises)
        loss = ((qz - y.unsqueeze(0)) ** 2).mean()

        # TODO(student): Update z_critic
        self.z_critic_optimizer.zero_grad()
        loss.backward()
        self.z_critic_optimizer.step()

        return {
            "qz_loss": loss,
            "qz_mean": qz.mean(),
        }

    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict:
        """Update BC flow actor"""
        # TODO(student): Implement BC flow loss
        z = torch.randn_like(actions)
        t = torch.rand(actions.shape[0], 1, device=actions.device, dtype=actions.dtype)
        a_tilde = (1 - t) * z + t * actions
        v_pred = self.bc_flow_actor(observations, a_tilde, t)
        target = actions - z
        loss = (v_pred - target).pow(2).mean(dim=-1).mean()

        # TODO(student): Update BC flow actor
        self.bc_flow_actor_optimizer.zero_grad()
        loss.backward()
        self.bc_flow_actor_optimizer.step()

        return {
            "actor_loss": loss,
        }

    def update_noise_actor(self,
        observations: torch.Tensor,
    ) -> dict:
        """Update noise actor."""
        # TODO(student): Implement noise actor loss
        noise_dist = self.noise_actor(observations)
        z = noise_dist.rsample()
        log_prob = noise_dist.log_prob(z)

        scaled_z = self.noise_scale * z
        qz = self.z_critic(observations, scaled_z)
        qz_min = qz.min(dim=0).values

        loss = (self.alpha.detach() * log_prob - qz_min).mean()

        # TODO(student): Update noise actor
        self.noise_actor_optimizer.zero_grad()
        loss.backward()
        self.noise_actor_optimizer.step()

        return {
            "noise_actor_loss": loss,
            "entropy": -log_prob.mean().detach(),
            "log_prob": log_prob.mean().detach(),
        }

    def update_alpha(self, observations: torch.Tensor) -> dict:
        """Update alpha."""
        # TODO(student): Implement alpha loss
        with torch.no_grad():
            noise_dist = self.noise_actor(observations)
            z = noise_dist.rsample()
            log_prob = noise_dist.log_prob(z)

        # Dual: L(alpha) = -alpha * (E[log pi(z|s)] + H_target)
        loss = -self.alpha * (log_prob + self.target_entropy).mean()

        # TODO(student): Update alpha
        self.alpha_optimizer.zero_grad()
        loss.backward()
        self.alpha_optimizer.step()
        with torch.no_grad():
            self.log_alpha.clamp_(min=-3.0)  # keep alpha >= ~0.05 so entropy never dies

        return {
            "alpha_loss": loss,
            "alpha": self.alpha,
        }

    def update_target_critic(self) -> None:
        # TODO(student): Implement target critic update
        tau = self.target_update_rate
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
                tp.data.mul_(1 - tau).add_(tau * p.data)

    def update_target_bc_flow_actor(self) -> None:
        # TODO(student): Implement target BC flow actor update
        tau = self.target_update_rate
        with torch.no_grad():
            for p, tp in zip(self.bc_flow_actor.parameters(), self.target_bc_flow_actor.parameters()):
                tp.data.mul_(1 - tau).add_(tau * p.data)

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        # TODO(student): Update critic, z_critic, actor, noise actor, and alpha - feel free to modify this code according to your setup!
        z = torch.randn(observations.shape[0], self.action_dim, device=observations.device, dtype=observations.dtype)
        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        metrics_qz = self.update_qz(observations, actions, self.noise_scale * z)
        metrics_actor = self.update_actor(observations, actions)
        metrics_noise_actor = self.update_noise_actor(observations)
        metrics_alpha = self.update_alpha(observations)
        metrics = {
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"z_critic/{k}": v.item() for k, v in metrics_qz.items()},
            **{f"actor/{k}": v.item() for k, v in metrics_actor.items()},
            **{f"noise_actor/{k}": v.item() for k, v in metrics_noise_actor.items()},
            **{f"alpha/{k}": v.item() for k, v in metrics_alpha.items()},
        }

        self.update_target_critic()
        self.update_target_bc_flow_actor()

        return metrics
