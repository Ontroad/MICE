from omnisafe.envs.core import make, support_envs
import time
import torch
import numpy as np
import torch.nn.functional as F
from omnisafe.utils.config import Config


def estimate_true_value(
    agent,
    env_id: str,
    num_envs: int,
    seed: int,
    cfgs: Config,
    discount: float,
    eval_episodes=1000,
):
    """Estimates true Q-value via launching given policy from sampled state until
    the end of an episode."""

    eval_env = make(env_id, num_envs=num_envs, device=cfgs.train_cfgs.device)

    true_cvalues = []
    estimate_cvalues = []
    for _ in range(eval_episodes):
        obs0, _ = eval_env.reset()

        _, _, estimate_cvalue, _ = agent.step(obs0)

        obs = obs0

        true_cvalue = 0.0
        step = 0
        while True:
            act, _, _, _ = agent.step(obs)
            next_obs, _, c, termniated, truncated, info = eval_env.step(act)
            true_cvalue += c * (discount**step)

            step += 1
            obs = next_obs

            if termniated or truncated:
                break
        true_cvalues.append(true_cvalue)
        estimate_cvalues.append(estimate_cvalue)

        print("Estimation took: ", step)

    return torch.mean(torch.stack(true_cvalues)), torch.mean(
        torch.stack(estimate_cvalues)
    )


class RandomProjection(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RandomProjection, self).__init__()
        self.linear_projection = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear_projection(x)
