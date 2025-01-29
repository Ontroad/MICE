import time
from typing import Dict, Tuple, Optional, Union
import torch
import numpy as np
import torch.nn.functional as F

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.ppo import PPO
from omnisafe.utils import distributed
from torch.utils.data import DataLoader, TensorDataset
from rich.progress import track
from collections import deque
from omnisafe.adapter import OnPolicyAdapter
from omnisafe.common.buffer import VectorOnPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.utils.config import Config
from omnisafe.typing import AdvatageEstimator, OmnisafeSpace
from omnisafe.common.buffer.onpolicy_buffer import OnPolicyBuffer
from omnisafe.utils.math import discount_cumsum
from omnisafe.models.critic.critic_builder import CriticBuilder
from gymnasium.spaces import Box




    
class FlashBulbMemory:
    def __init__(  # pylint: disable=too-many-arguments
        self,
        size: int,
        num_envs: int,
    ):
        self.unsafe_states = [deque(maxlen=size) for _ in range(num_envs)]



class MICEBuffer(OnPolicyBuffer):
    def __init__(  # pylint: disable=too-many-arguments
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        size: int,
        gamma: float,
        lam: float,
        lam_c: float,
        advantage_estimator: AdvatageEstimator,
        penalty_coefficient: float = 0,
        standardized_adv_r: bool = False,
        standardized_adv_c: bool = False,
        device: torch.device = torch.device('cpu'),
    ):
        super().__init__(
            obs_space,
            act_space,
            size,
            gamma,
            lam,
            lam_c,
            advantage_estimator,
            penalty_coefficient,
            standardized_adv_r,
            standardized_adv_c,
            device,
        )
        self.data['intrinsic_costs'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['ep_discount_ci'] = torch.zeros((size,), dtype=torch.float32, device=device)
        self.data['time_step'] = torch.zeros((size,), dtype=torch.float32, device=device)


    def get(self) -> Dict[str, torch.Tensor]:
        """Get the data in the buffer."""
        self.ptr, self.path_start_idx = 0, 0

        data = {
            'obs': self.data['obs'],
            'act': self.data['act'],
            'target_value_r': self.data['target_value_r'],
            'adv_r': self.data['adv_r'],
            'logp': self.data['logp'],
            'discounted_ret': self.data['discounted_ret'],
            'adv_c': self.data['adv_c'],
            'target_value_c': self.data['target_value_c'],
            'cost': self.data['cost'],  
            'ep_discount_ci': self.data['ep_discount_ci'], 
            'value_c': self.data['value_c'],  
            'intrinsic_costs': self.data['intrinsic_costs'], 
            'time_step': self.data['time_step'],
        }

        adv_mean, adv_std, *_ = distributed.dist_statistics_scalar(data['adv_r'])
        cadv_mean, *_ = distributed.dist_statistics_scalar(data['adv_c'])
        if self._standardized_adv_r:
            data['adv_r'] = (data['adv_r'] - adv_mean) / (adv_std + 1e-8)
        if self._standardized_adv_c:
            data['adv_c'] = data['adv_c'] - cadv_mean

        return data

    def finish_path(
        self,
        last_value_r: torch.Tensor = torch.zeros(1),
        last_value_c: torch.Tensor = torch.zeros(1),
        intrinsic_adv_type: int = 0,
    ) -> None:
        """Finish the current path and calculate the advantages of state-action pairs."""
        path_slice = slice(self.path_start_idx, self.ptr)  
        last_value_r = last_value_r.to(
            self._device
        )  
        last_value_c = last_value_c.to(self._device)
        
        rewards = torch.cat(
            [self.data['reward'][path_slice], last_value_r]
        )  
        values_r = torch.cat([self.data['value_r'][path_slice], last_value_r])
        costs = torch.cat([self.data['cost'][path_slice], last_value_c])
        values_c = torch.cat([self.data['value_c'][path_slice], last_value_c])

        discounted_ret = discount_cumsum(rewards, self._gamma)[
            :-1
        ]  
        self.data['discounted_ret'][path_slice] = discounted_ret
        rewards -= self._penalty_coefficient * costs 
        
        adv_r, target_value_r = self._calculate_adv_and_value_targets(
            values_r, rewards, lam=self._lam
        )
        if intrinsic_adv_type == 0:
            adv_c, target_value_c = self._calculate_adv_and_value_targets(
                values_c, costs, lam=self._lam_c
            )

        intrinsic_costs = self.data['intrinsic_costs'][path_slice]

        adv_c, target_value_c, ep_discount_ci = self._calculate_balancing_intrinsic_adv_and_value_targets(
            values_c, costs, lam=self._lam_c, intrinsic_costs=intrinsic_costs
        )
        self.data['ep_discount_ci'][path_slice] = ep_discount_ci

        self.data['adv_r'][path_slice] = adv_r
        self.data['target_value_r'][path_slice] = target_value_r
        self.data['adv_c'][path_slice] = adv_c
        self.data['target_value_c'][path_slice] = target_value_c

        self.path_start_idx = self.ptr 

    
    def _calculate_balancing_intrinsic_adv_and_value_targets(
        self,
        values_c: torch.Tensor,
        costs: torch.Tensor,
        lam: float,
        intrinsic_costs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self._advantage_estimator == 'gae':
            # GAE formula: A_t = \sum_{k=0}^{n-1} (lam*gamma)^k delta_{t+k}
            deltas_n = (
                costs[:-1] + self.beta * intrinsic_costs + self._gamma * values_c[1:] - values_c[:-1]
            )  
            beta_ =torch.where(
                intrinsic_costs != 0, 
                self.beta + 0.0001 * deltas_n / intrinsic_costs,  
                torch.ones_like(deltas_n) * self.beta  
            ) 
            deltas = (
                costs[:-1] + beta_ * intrinsic_costs + self._gamma * values_c[1:] - values_c[:-1]
            )
            ep_disount_ci = discount_cumsum( beta_ * intrinsic_costs, self._gamma)
            ep_disount_ci[:] = ep_disount_ci[0].clone()
            self.beta = 0.9*self.beta + 0.1*beta_.mean().item() 

            
            adv_c = discount_cumsum(deltas, self._gamma * lam)  
            target_value_c = adv_c + values_c[:-1]  

        else:
            raise NotImplementedError

        return adv_c, target_value_c, ep_disount_ci


class MICEVectorBuffer(VectorOnPolicyBuffer):
    def __init__(  # pylint: disable=super-init-not-called,too-many-arguments
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        size: int,
        gamma: float,
        lam: float,
        lam_c: float,
        advantage_estimator: AdvatageEstimator,
        penalty_coefficient: float,
        standardized_adv_r: bool,
        standardized_adv_c: bool,
        num_envs: int = 1,
        device: torch.device = torch.device('cpu'),
    ):
        self._num_buffers = num_envs
        self._standardized_adv_r = standardized_adv_r
        self._standardized_adv_c = standardized_adv_c
        if num_envs < 1:
            raise ValueError('num_envs must be greater than 0.')
        self.buffers = [
            MICEBuffer(
                obs_space=obs_space,
                act_space=act_space,
                size=size,
                gamma=gamma,
                lam=lam,
                lam_c=lam_c,
                advantage_estimator=advantage_estimator,
                penalty_coefficient=penalty_coefficient,
                device=device,
            )
            for _ in range(num_envs)
        ]

    def finish_path(
        self,
        last_value_r: torch.Tensor = torch.zeros(1),
        last_value_c: torch.Tensor = torch.zeros(1),
        idx: int = 0,
        intrinsic_adv_type: int = 0,
    ) -> None:
        """Finish the path."""
        self.buffers[idx].finish_path(last_value_r, last_value_c, intrinsic_adv_type)


