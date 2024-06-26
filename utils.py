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

    
class RandomProjection:
    def __init__(self, input_dim, output_dim):
        """
        Args:
            input_dim (int): The dimension of the input features.
            output_dim (int): The dimension of the output features (projected space).
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.projection_matrix = self._generate_projection_matrix()

    def _generate_projection_matrix(self):
        """
        Generates a random Gaussian projection matrix.
        """
        return torch.randn(self.output_dim, self.input_dim) / torch.sqrt(torch.tensor(self.output_dim, dtype=torch.float32))

    def project(self, trajectory):
        """
        Projects the input trajectory to the lower-dimensional space using the random projection matrix.

        Args:
            trajectory (torch.Tensor): The input trajectory data of shape (n_samples, input_dim).

        Returns:
            torch.Tensor: The projected trajectory of shape (n_samples, output_dim).
        """
        return torch.matmul(trajectory, self.projection_matrix.T)

    
class IntrinsicGenerator(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation=torch.nn.ReLU):
    
        super(IntrinsicGenerator, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.activation = activation

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        self.layers.append(torch.nn.Linear(prev_dim, output_dim))

    def forward(self, x):
        
        for layer in self.layers[:-1]:
            x = self.activation()(layer(x))
        x = self.layers[-1](x)
        return x
