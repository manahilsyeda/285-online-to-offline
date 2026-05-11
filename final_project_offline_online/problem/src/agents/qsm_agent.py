from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Callable, Optional, Sequence, Tuple, List

class QSMAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_actor,
        make_actor_optimizer,
        make_critic,
        make_critic_optimizer,

        discount: float,
        target_update_rate: float,
        alpha: float,
        inv_temp: float,
        flow_steps: int,
    ):
        super().__init__()

        self.action_dim = action_dim
        
        # TODO(student): Create actor
        self.actor = make_actor(observation_shape, action_dim)
        
        # TODO(student): Create critic (ensemble of Q-functions), target critic (ensemble of Q-functions)
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # TODO(student): Create optimizers for all the above models
        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())
        
        self.discount = discount
        self.target_update_rate = target_update_rate
        self.alpha = alpha
        self.inv_temp = inv_temp
        self.flow_steps = flow_steps
        self.max_grad_norm = 1.0

        betas = self.cosine_beta_schedule(flow_steps)
        self.register_buffer("betas", betas) # TODO(student): Implement betas
        self.register_buffer("alphas", 1.0 - betas) # TODO(student): Implement alphas
        self.register_buffer("alpha_hats", torch.cumprod(1.0 - betas, dim=0)) # TODO(student): Implement alpha_hats

        self.to(ptu.device)
    
    def cosine_beta_schedule(self, timesteps):
        """
        Cosine annealing beta schedule
        """
        # TODO(student): Implement cosine annealing beta schedule
        s = 0.08
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
        alpha_hats = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5).pow(2)
        alpha_hats = alpha_hats / alpha_hats[0]
        betas = 1.0 - (alpha_hats[1:] / alpha_hats[:-1])
        return torch.clamp(betas, 1e-4, 0.999)
    
    @torch.compiler.disable
    def ddpm_sampler(self, observations: torch.Tensor, noise: torch.Tensor):
        """
        DDPM sampling
        """
        # TODO(student): Implement DDPM sampling
        x = noise
        for step in reversed(range(self.flow_steps)):
            t = torch.full(
                (observations.shape[0], 1),
                step / max(self.flow_steps - 1, 1),
                device=observations.device,
                dtype=observations.dtype,
            )
            eps_pred = self.actor(observations, x, t)
            alpha_t = self.alphas[step].to(dtype=observations.dtype)
            alpha_hat_t = self.alpha_hats[step].to(dtype=observations.dtype)
            beta_t = self.betas[step].to(dtype=observations.dtype)
            mean = (x - beta_t / torch.sqrt(1.0 - alpha_hat_t) * eps_pred) / torch.sqrt(alpha_t)
            if step > 0:
                x = mean + torch.sqrt(beta_t) * torch.randn_like(x)
            else:
                x = mean
        return torch.clamp(x, -1, 1)
    
    def get_action(self, observation: torch.Tensor):
        """
        Used for evaluation.
        """
        # TODO(student): Implement get_action
        observation = ptu.from_numpy(np.asarray(observation))[None]
        with torch.no_grad():
            num_candidates = 8
            observations = observation.repeat(num_candidates, 1)
            noise = torch.randn(
                num_candidates,
                self.action_dim,
                device=observation.device,
                dtype=observation.dtype,
            )
            actions = self.ddpm_sampler(observations, noise)
            qs = self.critic(observations, actions).mean(dim=0)
            action = actions[torch.argmax(qs)]
        return ptu.to_numpy(action)

    @torch.compiler.disable
    def update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """
        Update Critic
        """
        # TODO(student): Implement critic update
        with torch.no_grad():
            num_target_candidates = 8
            batch_size = next_observations.shape[0]
            next_observations_repeated = next_observations.repeat_interleave(num_target_candidates, dim=0)
            noise = torch.randn(
                batch_size * num_target_candidates,
                self.action_dim,
                device=next_observations.device,
                dtype=next_observations.dtype,
            )
            next_actions = self.ddpm_sampler(next_observations_repeated, noise)
            next_qs = self.target_critic(next_observations_repeated, next_actions)
            next_qs = next_qs.min(dim=0).values.view(batch_size, num_target_candidates)
            next_q = next_qs.max(dim=1).values
            target = rewards.squeeze(-1) + self.discount * (1.0 - dones.squeeze(-1)) * next_q

        qs = self.critic(observations, actions)
        loss = (qs - target.unsqueeze(0)).pow(2).mean()
        
        # TODO(student): Update critic
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        
        return {
            "q_loss": loss,
            "q_mean": qs.mean(),
            "q_max": qs.max(),
            "q_min": qs.min(),
            "target_q_mean": target.mean(),
        }
        
    @torch.compiler.disable
    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        step: int,
    ):
        """
        Update the actor
        """

        # TODO(student): Implement actor update
        z = torch.randn_like(actions)
        t_idx = torch.randint(0, self.flow_steps, (actions.shape[0],), device=actions.device)
        t = (t_idx.float() / max(self.flow_steps - 1, 1)).unsqueeze(-1).to(dtype=actions.dtype)
        alpha_hat = self.alpha_hats[t_idx].to(dtype=actions.dtype).unsqueeze(-1)
        noisy_actions = torch.sqrt(alpha_hat) * actions + torch.sqrt(1.0 - alpha_hat) * z
        eps_pred = self.actor(observations, noisy_actions, t)

        q_actions = noisy_actions.detach().clone().requires_grad_(True)
        qs = self.target_critic(observations, q_actions)
        q = qs.mean(dim=0)
        q_grad = torch.autograd.grad(q.sum(), q_actions, retain_graph=False, create_graph=False)[0].detach()

        qsm_target = -self.inv_temp * q_grad
        qsm_target_norm = qsm_target.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        qsm_target = qsm_target * torch.clamp(1.0 / qsm_target_norm, max=1.0)
        qsm_loss = (eps_pred - qsm_target).pow(2).mean(dim=-1).mean()
        ddpm_loss = (z - eps_pred).pow(2).mean(dim=-1).mean()
        qsm_weight = min(max((step - 20000) / 80000, 0.0), 1.0)
        loss = qsm_weight * qsm_loss + self.alpha * ddpm_loss
        
        # TODO(student): Update actor
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        
        return {
            "actor_loss": loss,
            "qsm_loss": qsm_loss,
            "ddpm_loss": ddpm_loss,
            "qsm_weight": torch.as_tensor(qsm_weight, device=actions.device, dtype=actions.dtype),
            "q_grad_norm": q_grad.norm(dim=-1).mean(),
            "qsm_target_norm": qsm_target.norm(dim=-1).mean(),
            "eps_norm": eps_pred.norm(dim=-1).mean(),
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
        metrics_actor = self.update_actor(observations, actions, step)
        metrics = {
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"actor/{k}": v.item() for k, v in metrics_actor.items()},
        }

        self.update_target_critic()

        return metrics

    def update_target_critic(self) -> None:
        # TODO(student): Update target_critic using Polyak averaging with self.target_update_rate
        tau = self.target_update_rate
        with torch.no_grad():
            for p, p_target in zip(self.critic.parameters(), self.target_critic.parameters()):
                p_target.data.mul_(1.0 - tau).add_(tau * p.data)
