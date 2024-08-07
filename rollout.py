import time
from typing import Dict, Tuple, Optional, Union
import torch
import numpy as np
import torch.nn.functional as F
import heapq

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
from memory import (
    LFFVectorOnPolicyBuffer,
)
import os


class MICEAdapter(OnPolicyAdapter):
    def __init__(self, env_id: str, num_envs: int, seed: int, cfgs: Config):
        super().__init__(env_id, num_envs, seed, cfgs)

    def roll_out(  # pylint: disable=too-many-locals
        self,
        steps_per_epoch: int,
        agent: ConstraintActorCritic,
        buffer: LFFVectorOnPolicyBuffer,  
        flashbulb_memory,
        logger: Logger,
        rpnet,
        intrinsicG,
        epoch,
    ) -> None:
        self._reset_log()

        obs, _ = self.reset()
        roll_state = torch.zeros_like(obs)
        roll_cost = torch.zeros(obs.shape[0], 1, device=obs.device)
        for step in track(
            range(steps_per_epoch),
            description=f'Processing rollout for epoch: {logger.current_epoch}...',
        ):
            act, value_r, value_c, logp = agent.step(obs)  
            next_obs, reward, cost, terminated, truncated, info = self.step(
                act
            ) 

            self._log_value(
                reward=reward, cost=cost, info=info
            )  
            ep_cost = self._ep_cost.clone()

            if self._cfgs.algo_cfgs.use_cost:
                logger.store(**{'Value/cost': value_c})
            logger.store(**{'Value/reward': value_r})

            roll_state = torch.cat((roll_state, obs), dim=1)
            roll_cost = torch.cat((roll_cost, info.get('original_cost', cost).unsqueeze(1)), dim=1)

            intrinsic_costs = torch.zeros_like(cost)
            if len(flashbulb_memory.ep_state) > 0:
                for idx, roll_s in enumerate(roll_state):  
                    for (fail_t, fail_c) in zip(flashbulb_memory.ep_state, flashbulb_memory.ep_cost):
                        
                        if ep_cost[idx] < 1:
                            intrinsic_costs[idx] = torch.tensor(0.0, device=obs.device)
                            break
                        else:
                            t_cost_weight = roll_cost[idx].repeat_interleave(obs.shape[1])
                            t_cost_weight[t_cost_weight == 0] = self._cfgs.algo_cfgs.cost_weight
                            mask_len = min(roll_s.shape[0], fail_t.shape[0])
                            similarity = (
                                torch.dist(
                                    rpnet(fail_t[: mask_len] * t_cost_weight[: mask_len]),
                                    rpnet(roll_s[: mask_len] * t_cost_weight[: mask_len]),
                                )
                                / ep_cost[idx]
                            )
                            intrinsic_costs[idx] += self._cfgs.algo_cfgs.risk_factor / (1 + torch.exp(similarity))
                        
                    intrinsic_costs[idx] = (self._cfgs.algo_cfgs.risk_gamma**epoch) *  intrinsic_costs[idx] / len(flashbulb_memory.ep_state)
                   
                    for _ in range(self._cfgs.algo_cfgs.num_epochs):
                        self._cfgs.algo_cfgs.optimizer.zero_grad()
                        rp_traj = rpnet(roll_s)
                        ci_pre = intrinsicG(rp_traj)
                        loss = self._cfgs.algo_cfgs.criterion(ci_pre, intrinsic_costs[idx])
                        loss.backward()
                        self._cfgs.algo_cfgs.optimizer.step()
                    intrinsic_costs[idx] = intrinsicG(rpnet(roll_s))
                    if epoch % 10 == 0 and idx == 6 and ep_cost[idx] > 0:
                        with open(os.path.join(logger._log_dir, 'risk_cost.txt'), 'a') as file:
                            file.write(f'intrinsic_cost: {intrinsic_costs[idx].item()}\n')
                            file.write(f'ep_cost: {ep_cost[idx].item()}\n\n')
                    
            buffer.store(
                obs=obs,
                act=act,
                reward=reward,
                cost=cost,
                value_r=value_r,
                value_c=value_c,
                logp=logp,
                discount_cost=self._discount_cost,
                intrinsic_costs=intrinsic_costs, 
                time_step=self._ep_len - 1,
            )  
            obs = next_obs
            epoch_end = step >= steps_per_epoch - 1  
            for idx, (done, time_out) in enumerate(
                zip(terminated, truncated)
            ):  
                if epoch_end or done or time_out:
                    if not done:
                        if epoch_end:  
                            logger.log(
                                f'Warning: trajectory cut off when rollout by epoch at {self._ep_len[idx]} steps.'
                            )
                            _, last_value_r, last_value_c, _ = agent.step(obs[idx])
                        if time_out:  
                            _, last_value_r, last_value_c, _ = agent.step(
                                info['final_observation'][idx]
                            )
                        last_value_r = last_value_r.unsqueeze(
                            0
                        )  
                        last_value_c = last_value_c.unsqueeze(0)
                    else:  
                        last_value_r = torch.zeros(1)
                        last_value_c = torch.zeros(1)

                    if done or time_out:  
                        self._log_metrics(
                            logger, idx
                        )  
                        if self._ep_cost[idx] > self._cfgs.algo_cfgs.cost_limit:
                            if len(flashbulb_memory.ep_state) == self._cfgs.algo_cfgs.fail_buf_size:
                                if len(flashbulb_memory.ep_cost) >= self._cfgs.algo_cfgs.len_sort:
                                    smallest_costs = heapq.nsmallest(1, flashbulb_memory.ep_cost[:self._cfgs.algo_cfgs.len_sort])
                                    smallest_cost = smallest_costs[0]
                                    flashbulb_memory.ep_cost.remove(smallest_cost)
                                    flashbulb_memory.ep_state.remove(flashbulb_memory.ep_state[flashbulb_memory.ep_cost.index(smallest_cost)])
                            flashbulb_memory.ep_state.append(roll_state[idx])  
                            flashbulb_memory.ep_cost.append(roll_cost[idx])
                        self._reset_log(idx)

                        self._ep_ret[idx] = 0.0
                        self._ep_cost[idx] = 0.0
                        self._ep_len[idx] = 0.0
                    buffer.finish_path(last_value_r, last_value_c, idx) 
        return flashbulb_memory


    def _log_value(
            self,
            reward: torch.Tensor,
            cost: torch.Tensor,
            info: Dict,
            **kwargs,  # pylint: disable=unused-argument
        ) -> None:
            """Log value."""
            self._ep_ret += info.get(
                'original_reward', reward
            ).cpu() 
            self._ep_cost += info.get('original_cost', cost).cpu()
            self._discount_cost += (
                info.get('original_cost', cost).cpu() * self._cfgs.algo_cfgs.cost_gamma**self._ep_len
            )
            self._ep_len += 1

    def _reset_log(self, idx: Optional[int] = None) -> None:
        """Reset log."""
        if idx is None:
            self._ep_ret = torch.zeros(self._env.num_envs)
            self._ep_cost = torch.zeros(self._env.num_envs)
            self._discount_cost = torch.zeros(self._env.num_envs)
            self._ep_len = torch.zeros(self._env.num_envs)
        else:
            self._ep_ret[idx] = 0.0
            self._ep_cost[idx] = 0.0
            self._discount_cost = 0.0
            self._ep_len[idx] = 0.00
