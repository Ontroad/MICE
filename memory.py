""""buffer class of MICE"""

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
from omnisafe.utils.config import Config
from omnisafe.typing import AdvatageEstimator, OmnisafeSpace
from omnisafe.common.buffer.onpolicy_buffer import OnPolicyBuffer
from omnisafe.utils.math import discount_cumsum
from gymnasium.spaces import Box



class LFFOnPolicyBuffer(OnPolicyBuffer):
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
        self.data['discount_cost'] = torch.zeros((size,), dtype=torch.float32, device=device)
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
            'discount_cost': self.data['discount_cost'],  
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
        intrinsic_cost: torch.Tensor = torch.zeros(1),
    ) -> None:
        path_slice = slice(self.path_start_idx, self.ptr)  
        last_value_r = last_value_r.to(self._device)  
        last_value_c = last_value_c.to(self._device)
        intrinsic_cost = intrinsic_cost.to(self._device)

        rewards = torch.cat(
            [self.data['reward'][path_slice], last_value_r]
        )  
        values_r = torch.cat([self.data['value_r'][path_slice], last_value_r])
        costs = torch.cat([self.data['cost'][path_slice], last_value_c])
        values_c = torch.cat([self.data['value_c'][path_slice], last_value_c])

        discounted_ret = discount_cumsum(rewards, self._gamma)[:-1]  
        self.data['discounted_ret'][path_slice] = discounted_ret
        rewards -= self._penalty_coefficient * costs  
        discounted_cost = discount_cumsum(costs, self._gamma)[
            :-1
        ]  
        self.data['discount_cost'][path_slice] = discounted_cost

        
        self.data['intrinsic_costs'][path_slice] = intrinsic_cost

        adv_r, target_value_r = self._calculate_adv_and_value_targets(
            values_r, rewards, lam=self._lam
        )
        adv_c, target_value_c = self._calculate_adv_and_value_targets(
            values_c, costs, lam=self._lam_c
        )

        self.data['adv_r'][path_slice] = adv_r
        self.data['target_value_r'][path_slice] = target_value_r
        self.data['adv_c'][path_slice] = adv_c
        self.data['target_value_c'][path_slice] = target_value_c

        self.path_start_idx = self.ptr  


class LFFVectorOnPolicyBuffer(VectorOnPolicyBuffer):
    def __init__(  
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
            LFFOnPolicyBuffer(
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
        intrinsic_cost: torch.Tensor = torch.zeros(1),
        idx: int = 0,
    ) -> None:
        """Finish the path."""
        self.buffers[idx].finish_path(last_value_r, last_value_c, intrinsic_cost)



class FailOnPolicyBuffer:
    def __init__(  # pylint: disable=too-many-arguments
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        size: int,
        device: torch.device = torch.device('cpu'),
    ):
        if isinstance(obs_space, Box):
            obs_buf = torch.zeros((size, *obs_space.shape), dtype=torch.float32, device=device)
        else:
            raise NotImplementedError
        if isinstance(act_space, Box):
            act_buf = torch.zeros((size, *act_space.shape), dtype=torch.float32, device=device)
        else:
            raise NotImplementedError

        self.data: Dict[str, torch.Tensor] = {
            'obs': obs_buf,
            'act': act_buf,
        }
        self._size = size
        self._device = device
        self.ptr: int = 0
        self.max_size = size

    def store(self, idx, **data: torch.Tensor) -> None:  
        """Store data into the buffer."""
        if self.ptr >= self.max_size: 
            self.ptr = 0
        for key, value_list in data.items():
            for value in value_list:
                self.data[key][self.ptr] = value[idx]
        self.ptr += 1

    def get(self) -> Dict[str, torch.Tensor]:
        """Get the data in the buffer."""
        data = {
            'obs': self.data['obs'],
            'act': self.data['act'],
        }
        return data


class FailVectorOnPolicyBuffer(FailOnPolicyBuffer):
    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        size: int,
        num_envs: int = 1,
        device: torch.device = torch.device('cpu'),
    ):
        self._num_buffers = num_envs
        if num_envs < 1:
            raise ValueError('num_envs must be greater than 0.')
        self.buffers = [
            FailOnPolicyBuffer(
                obs_space=obs_space,
                act_space=act_space,
                size=size,
                device=device,
            )
            for _ in range(num_envs)
        ]

    def store(self, idx, **data: torch.Tensor) -> None:

        self.buffers[idx].store(idx, **{k: v for k, v in data.items()})

    def get(self) -> Dict[str, torch.Tensor]:
        """Get the data from the buffer."""
        data_pre = {
            k: [v] for k, v in self.buffers[0].get().items()
        } 
        for buffer in self.buffers[1:]:  
            for k, v in buffer.get().items():
                data_pre[k].append(v)  
        data = {
            k: torch.cat(v, dim=0) for k, v in data_pre.items()
        }  

        return data


class FailBuffer:
    def __init__(  # pylint: disable=too-many-arguments
        self,
        size: int,
    ):
        self.ep_state = deque(maxlen=size)
        self.ep_action = deque(maxlen=size)
        self.ep_cost = deque(maxlen=size)


class LFFMBuffer(LFFOnPolicyBuffer):
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

    def finish_path(
        self,
        last_value_r: torch.Tensor = torch.zeros(1),
        last_value_c: torch.Tensor = torch.zeros(1),
    ) -> None:
        """Finish the current path and calculate the advantages of state-action pairs."""
        path_slice = slice(self.path_start_idx, self.ptr) 
        last_value_r = last_value_r.to(self._device)  
        last_value_c = last_value_c.to(self._device)
        
        rewards = torch.cat([self.data['reward'][path_slice], last_value_r])  
        values_r = torch.cat([self.data['value_r'][path_slice], last_value_r])
        costs = torch.cat([self.data['cost'][path_slice], last_value_c])
        values_c = torch.cat([self.data['value_c'][path_slice], last_value_c])

        discounted_ret = discount_cumsum(rewards, self._gamma)[:-1]  
        self.data['discounted_ret'][path_slice] = discounted_ret
        rewards -= self._penalty_coefficient * costs 
        discounted_cost = discount_cumsum(costs, self._gamma)[
            :-1
        ] 
        self.data['discount_cost'][path_slice] = discounted_cost

        adv_r, target_value_r = self._calculate_adv_and_value_targets(
            values_r, rewards, lam=self._lam
        )
        adv_c, target_value_c = self._calculate_adv_and_value_targets(
            values_c, costs, lam=self._lam_c
        )

        self.data['adv_r'][path_slice] = adv_r
        self.data['target_value_r'][path_slice] = target_value_r
        self.data['adv_c'][path_slice] = adv_c
        self.data['target_value_c'][path_slice] = target_value_c

        self.path_start_idx = self.ptr  


class LFFMVectorBuffer(LFFVectorOnPolicyBuffer):
    def __init__(  
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
            LFFMBuffer(
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
    ) -> None:
        self.buffers[idx].finish_path(last_value_r, last_value_c)



class LFFM2Buffer(LFFOnPolicyBuffer):
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

    def finish_path(
        self,
        last_value_r: torch.Tensor = torch.zeros(1),
        last_value_c: torch.Tensor = torch.zeros(1),
    ) -> None:
        """Finish the current path and calculate the advantages of state-action pairs."""
        path_slice = slice(self.path_start_idx, self.ptr)  
        last_value_r = last_value_r.to(self._device)  
        last_value_c = last_value_c.to(self._device)

        rewards = torch.cat([self.data['reward'][path_slice], last_value_r])  
        values_r = torch.cat([self.data['value_r'][path_slice], last_value_r])
        costs = torch.cat([self.data['cost'][path_slice], last_value_c])
        values_c = torch.cat([self.data['value_c'][path_slice], last_value_c])

        discounted_ret = discount_cumsum(rewards, self._gamma)[:-1]  
        self.data['discounted_ret'][path_slice] = discounted_ret
        rewards -= self._penalty_coefficient * costs  
        discounted_cost = discount_cumsum(costs, self._gamma)[
            :-1
        ]  
        self.data['discount_cost'][path_slice] = discounted_cost

        adv_r, target_value_r = self._calculate_adv_and_value_targets(
            values_r, rewards, lam=self._lam
        )
        intrinsic_costs = self.data['intrinsic_costs'][path_slice] 
        adv_c, target_value_c = self._calculate_adv_and_value_intrinsic_targets(
            values_c, costs, lam=self._lam_c, intrinsic_costs=intrinsic_costs
        )

        self.data['adv_r'][path_slice] = adv_r
        self.data['target_value_r'][path_slice] = target_value_r
        self.data['adv_c'][path_slice] = adv_c
        self.data['target_value_c'][path_slice] = target_value_c

        self.path_start_idx = self.ptr  

    def _calculate_adv_and_value_intrinsic_targets(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        lam: float,
        intrinsic_costs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self._advantage_estimator == 'gae':
            
            deltas = rewards[:-1] + self._gamma * values[1:] - values[:-1]  
            adv = discount_cumsum(deltas, self._gamma * lam) + intrinsic_costs 
            target_value = adv + values[:-1]

        else:
            raise NotImplementedError

        return adv, target_value


class LFFM2VectorBuffer(LFFVectorOnPolicyBuffer):
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
            LFFM2Buffer(
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
    ) -> None:
        """Finish the path."""
        self.buffers[idx].finish_path(last_value_r, last_value_c)
