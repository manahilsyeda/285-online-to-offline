import argparse
import os
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import tqdm

import configs
from agents import agents
from infrastructure import utils
from infrastructure import pytorch_util as ptu
from infrastructure.log_utils import setup_wandb, Logger, dump_log
from infrastructure.replay_buffer import ReplayBuffer


def replay_buffer_from_offline(offline: ReplayBuffer, capacity: int) -> ReplayBuffer:
    """
    Copy a fixed offline dataset into a larger circular replay buffer so online RL can append
    new transitions without dropping offline data until capacity is exceeded.
    """
    # TODO(student): Implement offline training loop
    n = offline.size
    cap = max(int(capacity), n)
    rb = ReplayBuffer(capacity=cap)
    rb.observations = np.empty((cap, *offline.observations.shape[1:]), dtype=offline.observations.dtype)
    rb.next_observations = np.empty(
        (cap, *offline.next_observations.shape[1:]), dtype=offline.next_observations.dtype
    )
    rb.actions = np.empty((cap, *offline.actions.shape[1:]), dtype=offline.actions.dtype)
    rb.rewards = np.empty((cap,) + offline.rewards.shape[1:], dtype=offline.rewards.dtype)
    rb.dones = np.empty((cap,) + offline.dones.shape[1:], dtype=offline.dones.dtype)
    rb.observations[:n] = offline.observations[:n]
    rb.next_observations[:n] = offline.next_observations[:n]
    rb.actions[:n] = offline.actions[:n]
    rb.rewards[:n] = offline.rewards[:n]
    rb.dones[:n] = offline.dones[:n]
    rb.size = n
    return rb


def run_offline_training_loop(
    config: dict,
    train_logger: Logger,
    eval_logger: Logger,
    args: argparse.Namespace,
    start_step: int = 0,
) -> Tuple[str, ReplayBuffer, nn.Module]:
    """
    Offline RL: sample transitions from the offline dataset (via a replay buffer), update the agent,
    and log/evaluate on the same global step axis as the later online phase.

    Returns the path to a saved agent checkpoint, the replay buffer (for online finetuning), and the
    trained agent module so the caller can continue training without reloading weights from disk.
    """
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    _, offline_dataset = config["make_env_and_dataset"]()
    replay_buffer = replay_buffer_from_offline(offline_dataset, args.replay_buffer_capacity)

    example_batch = replay_buffer.sample(1)
    agent_cls = agents[config["agent"]]
    agent_kwargs = dict(config["agent_kwargs"])
    if config["agent"] in ("ifql", "dsrl"):
        agent_kwargs["online_training"] = False
    agent = agent_cls(
        example_batch["observations"].shape[1:],
        example_batch["actions"].shape[-1],
        **agent_kwargs,
    )
    agent.to(ptu.device)

    offline_steps = config["offline_training_steps"]

    eval_env, _ = config["make_env_and_dataset"]()
    eval_ep_len = eval_env.spec.max_episode_steps or eval_env.max_episode_steps

    for step in tqdm.trange(offline_steps, dynamic_ncols=True):
        global_step = start_step + step
        batch = replay_buffer.sample(config["batch_size"])
        batch = {k: ptu.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in batch.items()}

        metrics = agent.update(
            batch["observations"],
            batch["actions"],
            batch["rewards"],
            batch["next_observations"],
            batch["dones"],
            step,
        )

        if step % args.log_interval == 0:
            train_logger.log(metrics, step=global_step)

        if step % args.eval_interval == 0:
            trajectories = utils.sample_n_trajectories(
                eval_env,
                agent,
                args.num_eval_trajectories,
                eval_ep_len,
            )
            successes = [t["episode_statistics"]["s"] for t in trajectories]
            eval_logger.log(
                {"eval/success_rate": float(np.mean(successes))},
                step=global_step,
            )

    agent_path = os.path.join(args.save_dir, "agent.pt")
    torch.save(agent.state_dict(), agent_path)
    return agent_path, replay_buffer, agent


def run_online_training_loop(
    config: dict,
    train_logger: Logger,
    eval_logger: Logger,
    args: argparse.Namespace,
    agent_path: Optional[str],
    # agent_path: str,
    replay_buffer: Optional[ReplayBuffer],
    agent: Optional[nn.Module],
    start_step: int = 0,
) -> nn.Module:
    # TODO(student): Implement online training loop
    """
    Online RL: finetune the agent on a mixture of offline data (already in replay_buffer) and new
    on-policy data collected in the simulator.
    """
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    env, offline_dataset = config["make_env_and_dataset"]()
    if replay_buffer is None:
        replay_buffer = replay_buffer_from_offline(offline_dataset, args.replay_buffer_capacity)

    example_batch = replay_buffer.sample(1)
    agent_cls = agents[config["agent"]]
    agent_kwargs = dict(config["agent_kwargs"])
    if config["agent"] in ("ifql", "dsrl"):
        agent_kwargs["online_training"] = True

    if agent is not None:
        state_dict = agent.state_dict()
        agent = agent_cls(
            example_batch["observations"].shape[1:],
            example_batch["actions"].shape[-1],
            **agent_kwargs,
        )
        agent.load_state_dict(state_dict)
        agent.to(ptu.device)
    else:
        agent = agent_cls(
            example_batch["observations"].shape[1:],
            example_batch["actions"].shape[-1],
            **agent_kwargs,
        )
        agent.to(ptu.device)
        if agent_path is not None and os.path.isfile(agent_path):
            state = torch.load(agent_path, map_location=ptu.device)
            agent.load_state_dict(state)

    ep_len = env.spec.max_episode_steps or env.max_episode_steps
    online_steps = config["online_training_steps"]
    obs, _ = env.reset()

    eval_env, _ = config["make_env_and_dataset"]()
    eval_ep_len = eval_env.spec.max_episode_steps or eval_env.max_episode_steps

    for step in tqdm.trange(online_steps, dynamic_ncols=True):
        global_step = start_step + step

        if replay_buffer.size >= config["batch_size"]:
            batch = replay_buffer.sample(config["batch_size"])
            batch = {k: ptu.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in batch.items()}
            metrics = agent.update(
                batch["observations"],
                batch["actions"],
                batch["rewards"],
                batch["next_observations"],
                batch["dones"],
                step,
            )
            if step % args.log_interval == 0:
                train_logger.log(metrics, step=global_step)

        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        replay_buffer.insert(
            observation=obs,
            action=action,
            reward=reward,
            next_observation=next_obs,
            done=done,
        )
        obs = next_obs
        if done:
            obs, _ = env.reset()

        if step % args.eval_interval == 0:
            trajectories = utils.sample_n_trajectories(
                eval_env,
                agent,
                args.num_eval_trajectories,
                eval_ep_len,
            )
            successes = [t["episode_statistics"]["s"] for t in trajectories]
            eval_logger.log(
                {"eval/success_rate": float(np.mean(successes))},
                step=global_step,
            )

    return agent


def setup_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", type=str, default='sacbc')
    parser.add_argument("--env_name", type=str, default='cube-single-play-singletask-task1-v0')
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_group", type=str, default='Debug')
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--which_gpu", default=0)
    parser.add_argument("--offline_training_steps", type=int, default=500000)  # Should be 500k to pass the autograder
    parser.add_argument("--online_training_steps", type=int, default=100000)  # Should be 100k to pass the autograder
    parser.add_argument("--replay_buffer_capacity", type=int, default=1000000)
    parser.add_argument("--log_interval", type=int, default=10000)
    parser.add_argument("--eval_interval", type=int, default=100000)
    parser.add_argument("--num_eval_trajectories", type=int, default=25)  # Should be greater than or equal to 20 to pass autograder
    
    # Online retention of offline data
    # TODO(student): If desired, add arguments for online retention of offline data
    
    # WSRL
    # TODO (student): If desired, add arguments for WSRL
    

    # IFQL
    parser.add_argument("--expectile", type=float, default=None)

    # FQL / QSM
    parser.add_argument("--alpha", type=float, default=None)

    # QSM
    parser.add_argument("--inv_temp", type=float, default=None)

    # DSRL
    parser.add_argument("--noise_scale", type=float, default=None)

    # For njobs mode (optional)
    parser.add_argument("--njobs", type=int, default=None)
    parser.add_argument("job_specs", nargs="*")

    args = parser.parse_args(args=args)

    return args


def main(args):
    # Create directory for logging
    logdir_prefix = "exp"  # Keep for autograder

    config = configs.configs[args.base_config](args.env_name)

    # Set common config values from args for autograder
    config['seed'] = args.seed
    config['run_group'] = args.run_group
    config['offline_training_steps'] = args.offline_training_steps
    config['online_training_steps'] = args.online_training_steps
    config['log_interval'] = args.log_interval
    config['eval_interval'] = args.eval_interval
    config['num_eval_trajectories'] = args.num_eval_trajectories
    config['replay_buffer_capacity'] = args.replay_buffer_capacity
    
    # TODO(student): If necessary, add additional config values

    exp_name = f"sd{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config['log_name']}"

    # Override agent hyperparameters if specified
    if args.expectile is not None:
        config['agent_kwargs']['expectile'] = args.expectile
        exp_name = f"{exp_name}_e{args.expectile}"
    if args.alpha is not None:
        config['agent_kwargs']['alpha'] = args.alpha
        exp_name = f"{exp_name}_a{args.alpha}"
    if args.inv_temp is not None:
        config['agent_kwargs']['inv_temp'] = args.inv_temp
        exp_name = f"{exp_name}_i{args.inv_temp}"
    if args.noise_scale is not None:
        config['agent_kwargs']['noise_scale'] = args.noise_scale
        exp_name = f"{exp_name}_n{args.noise_scale}"
    if args.online_training_steps > 0:
        exp_name = f"{exp_name}_online"
    if args.offline_training_steps > 0:
        exp_name = f"{exp_name}_offline"

    setup_wandb(project='cs185_default_project', name=exp_name, group=args.run_group, config=config)
    args.save_dir = os.path.join(logdir_prefix, args.run_group, exp_name)
    os.makedirs(args.save_dir, exist_ok=True)
    train_logger = Logger(os.path.join(args.save_dir, 'train.csv'))
    eval_logger = Logger(os.path.join(args.save_dir, 'eval.csv'))

    start_step = 0
    agent_path: Optional[str] = None
    replay_buffer: Optional[ReplayBuffer] = None
    agent: Optional[nn.Module] = None

    if args.offline_training_steps > 0:
        print(f"Running offline training loop with {args.offline_training_steps} steps")
        # TODO(student): Implement offline training loop
        # Hint: You might consider passing the agent's path to the online training loop
        agent_path, replay_buffer, agent = run_offline_training_loop(
            config, train_logger, eval_logger, args, start_step=start_step
        )
        start_step = args.offline_training_steps
    
    if args.online_training_steps > 0:
        print(f"Running online training loop with {args.online_training_steps} steps")
         # TODO(student): Implement online training loop
        agent = run_online_training_loop(
            config,
            train_logger,
            eval_logger,
            args,
            agent_path=agent_path,
            replay_buffer=replay_buffer,
            agent=agent,
            start_step=start_step,
        )

    # if agent is not None:
    #     dump_log(agent, train_logger, eval_logger, config, args.save_dir)


if __name__ == "__main__":
    args = setup_arguments()
    if args.njobs is not None and len(args.job_specs) > 0:
        # Run n jobs in parallel
        from scripts.run_njobs import main_njobs
        main_njobs(job_specs=args.job_specs, njobs=args.njobs)
    else:
        # Run a single job
        main(args)
