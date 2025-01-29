"""Implement of MICE with CPO"""

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
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.utils.config import Config
from omnisafe.typing import AdvatageEstimator, OmnisafeSpace
from omnisafe.utils.math import discount_cumsum
from omnisafe.models.critic.critic_builder import CriticBuilder
from omnisafe.algorithms.on_policy.second_order.cpo import CPO
from omnisafe.utils import distributed
from omnisafe.utils.math import conjugate_gradients
from omnisafe.utils.tools import (
    get_flat_gradients_from,
    get_flat_params_from,
    set_param_values_to_model,
)
from omnisafe.algorithms.on_policy.mice.mice_rollout import MICEAdapter
from omnisafe.algorithms.on_policy.mice.mice_buffer import FailBuffer,FlashBulbMemory, MICEVectorBuffer
import omnisafe.algorithms.on_policy.mice.utils as utl
import os
import csv
import torch.nn as nn


class RandomProjection(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RandomProjection, self).__init__()
        self.linear_projection = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear_projection(x)





@registry.register
class MICE(CPO):
    def _init_env(self) -> None:
        self._env = MICEAdapter(
            self._env_id, self._cfgs.train_cfgs.vector_env_nums, self._seed, self._cfgs
        )
        assert (self._cfgs.algo_cfgs.steps_per_epoch) % (
            distributed.world_size() * self._cfgs.train_cfgs.vector_env_nums
        ) == 0, ('The number of steps per epoch is not divisible by the number of ' 'environments.')
        self._steps_per_epoch = (
            self._cfgs.algo_cfgs.steps_per_epoch
            // distributed.world_size()
            // self._cfgs.train_cfgs.vector_env_nums
        )

    def _init(self) -> None:
        self._buf = MICEVectorBuffer(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            size=self._steps_per_epoch,
            gamma=self._cfgs.algo_cfgs.gamma,
            lam=self._cfgs.algo_cfgs.lam,
            lam_c=self._cfgs.algo_cfgs.lam_c,
            advantage_estimator=self._cfgs.algo_cfgs.adv_estimation_method,
            standardized_adv_r=self._cfgs.algo_cfgs.standardized_rew_adv,
            standardized_adv_c=self._cfgs.algo_cfgs.standardized_cost_adv,
            penalty_coefficient=self._cfgs.algo_cfgs.penalty_coef,
            num_envs=self._cfgs.train_cfgs.vector_env_nums,
            device=self._device,
        )

        self._RPNet = RandomProjection(self._env.observation_space.shape[0], self._emb_obs_dim).to(
            self._device
        )
        self._maxlen_fail = self._cfgs.algo_cfgs.fail_buf_size

        self._flashbulb_memory = FlashBulbMemory(size=self._maxlen_fail, num_envs=self._cfgs.train_cfgs.vector_env_nums)
       
       

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Train/intrinsic_costs')
        self._logger.register_key('Train/discount_ci')
        self._logger.register_key('Train/epcost_factor')
        self._logger.register_key('Train/intrinsic_factor')
        self._logger.register_key('Train/risk_gamma')
        self._logger.register_key('Value/Adv_c')
        self._logger.register_key('Eval/true_value_c')
        self._logger.register_key('Eval/estimate_value_c')
        

    def learn(self) -> Tuple[Union[int, float], ...]:  
        start_time = time.time()
        self._logger.log('INFO: Start training')

        for epoch in range(self._cfgs.train_cfgs.epochs): 
            epoch_time = time.time()

            rollout_time = time.time()
            
   
            
            self._flashbulb_memory, self._ep_discount_ci = (
                self._env.rollout_ngu(  
                    steps_per_epoch=self._steps_per_epoch,
                    agent=self._actor_critic,
                    buffer=self._buf,
                    flashbulb_memory=self._flashbulb_memory,
                    logger=self._logger,
                    rpnet=self._RPNet,
                    epoch=epoch,
                )
                )

            
            self._logger.store({'Time/Rollout': time.time() - rollout_time})

            update_time = time.time()
            self._update()
            self._logger.store({'Time/Update': time.time() - update_time})

            if self._cfgs.model_cfgs.exploration_noise_anneal:
                self._actor_critic.annealing(epoch)

            if self._cfgs.model_cfgs.actor.lr is not None:
                self._actor_critic.actor_scheduler.step()

            self._logger.store(
                {
                    'TotalEnvSteps': (epoch + 1) * self._cfgs.algo_cfgs.steps_per_epoch,
                    'Time/FPS': self._cfgs.algo_cfgs.steps_per_epoch / (time.time() - epoch_time),
                    'Time/Total': (time.time() - start_time),
                    'Time/Epoch': (time.time() - epoch_time),
                    'Train/Epoch': epoch,
                    'Train/LR': (
                        0.0
                        if self._cfgs.model_cfgs.actor.lr is None
                        else self._actor_critic.actor_scheduler.get_last_lr()[0]
                    ),
                },
            )

            self._logger.dump_tabular()

            if (epoch + 1) % self._cfgs.logger_cfgs.save_model_freq == 0 or (
                epoch + 1
            ) == self._cfgs.train_cfgs.epochs:
                self._logger.torch_save()

        ep_ret = self._logger.get_stats('Metrics/EpRet')[0]
        ep_cost = self._logger.get_stats('Metrics/EpCost')[0]
        ep_len = self._logger.get_stats('Metrics/EpLen')[0]
        self._logger.close()
        self._env.close()

        return ep_ret, ep_cost, ep_len

    def _update(self) -> None:
        data = self._buf.get()
        (
            obs,
            act,
            logp,
            target_value_r,
            target_value_c,
            adv_r,
            adv_c,
            intrinsic_costs,
            balancing_ep_dicount_ci, 
            
        ) = (
            data['obs'],
            data['act'],
            data['logp'],
            data['target_value_r'],
            data['target_value_c'],
            data['adv_r'],
            data['adv_c'],
            data['intrinsic_costs'], 
            data['ep_discount_ci'], 
        )
        self._update_actor(obs, act, logp, adv_r, adv_c, intrinsic_costs, balancing_ep_dicount_ci)

        dataloader = DataLoader(
            dataset=TensorDataset(obs, target_value_r, target_value_c),
            batch_size=self._cfgs.algo_cfgs.batch_size,
            shuffle=True,
        )

        for _ in range(self._cfgs.algo_cfgs.update_iters):
            for (
                obs,
                target_value_r,
                target_value_c,
            ) in dataloader:
                self._update_reward_critic(obs, target_value_r)
                if self._cfgs.algo_cfgs.use_cost:
                    self._update_cost_critic(obs, target_value_c)

        self._logger.store(
            **{
                'Train/StopIter': self._cfgs.algo_cfgs.update_iters,
                'Value/Adv': adv_r.mean().item(),
                'Value/Adv_c': adv_c.mean().item(),
            }
        )

    # pylint: disable=invalid-name, too-many-arguments, too-many-locals
    def _update_actor(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
        intrinsic_costs: torch.Tensor,
        balancing_ep_dicount_ci: torch.Tensor,
    ) -> None:
        self._fvp_obs = obs[:: self._cfgs.algo_cfgs.fvp_sample_freq]
        theta_old = get_flat_params_from(self._actor_critic.actor)  
        self._actor_critic.actor.zero_grad() 
        loss_reward = self._loss_pi(obs, act, logp, adv_r)  
        loss_reward_before = distributed.dist_avg(loss_reward).item()  
        p_dist = self._actor_critic.actor(obs)  

        loss_reward.backward()  
        distributed.avg_grads(self._actor_critic.actor)  

        grads = -get_flat_gradients_from(self._actor_critic.actor)
        x = conjugate_gradients(self._fvp, grads, self._cfgs.algo_cfgs.cg_iters)  
        assert torch.isfinite(x).all(), 'x is not finite'
        xHx = x.dot(self._fvp(x))  #
        assert xHx.item() >= 0, 'xHx is negative'
        alpha = torch.sqrt(2 * self._cfgs.algo_cfgs.target_kl / (xHx + 1e-8))  

        self._actor_critic.zero_grad()
        loss_cost, ratio = self._loss_pi_cost(obs, act, logp, adv_c, intrinsic_costs)
        loss_cost_before = distributed.dist_avg(loss_cost).item()

        loss_cost.backward()
        distributed.avg_grads(self._actor_critic.actor)

        b_grads = get_flat_gradients_from(self._actor_critic.actor)
        self.ep_costs = self._logger.get_stats('Metrics/EpCost')[0] - self._cfgs.algo_cfgs.cost_limit

       

        ep_discount_ci = balancing_ep_dicount_ci.mean().item()

        self._logger.store({'Train/discount_ci': ep_discount_ci, 
                            'Train/epcost_factor': self._epcost_factor, 
                            'Train/intrinsic_factor': self._cfgs.algo_cfgs.intrinsic_factor,
                            'Train/risk_gamma': self._cfgs.algo_cfgs.risk_gamma,})
        self.ep_costs += ep_discount_ci
        

        

        p = conjugate_gradients(self._fvp, b_grads, self._cfgs.algo_cfgs.cg_iters)  # H^-1*b
        q = xHx
        r = grads.dot(p)  # g^T*H^-1*b
        s = b_grads.dot(p)  # b^T*H^-1*b

        # ----------determine optim_case
        optim_case, A, B = self._determine_case(
            b_grads=b_grads,
            ep_costs=self.ep_costs,
            q=q,
            r=r,
            s=s,
        )

        # ----------step direction
        step_direction, lambda_star, nu_star = self._step_direction(
            optim_case=optim_case,
            xHx=xHx,
            x=x,
            A=A,
            B=B,
            q=q,
            p=p,
            r=r,
            s=s,
            ep_costs=self.ep_costs,
        )

        # ----------line search 
        step_direction, accept_step = self._cpo_search_step(
            step_direction=step_direction,
            grads=grads,
            p_dist=p_dist,
            obs=obs,
            act=act,
            logp=logp,
            adv_r=adv_r,
            adv_c=adv_c,
            intrinsic_costs=intrinsic_costs,
            loss_reward_before=loss_reward_before,
            loss_cost_before=loss_cost_before,
            total_steps=20,
            violation_c=self.ep_costs,
            optim_case=optim_case,
        )

        theta_new = theta_old + step_direction 
        set_param_values_to_model(self._actor_critic.actor, theta_new)

        with torch.no_grad():
            loss_reward = self._loss_pi(obs, act, logp, adv_r)
            loss_cost, _ = self._loss_pi_cost(obs, act, logp, adv_c, intrinsic_costs)
            loss = loss_reward + loss_cost

        self._logger.store(
            **{
                'Loss/Loss_pi': loss.item(),
                'Train/intrinsic_costs': intrinsic_costs.mean().item(),
                'Misc/AcceptanceStep': accept_step,
                'Misc/Alpha': alpha.item(),
                'Misc/FinalStepNorm': step_direction.norm().mean().item(),
                'Misc/xHx': xHx.mean().item(),
                'Misc/H_inv_g': x.norm().item(),  # H^-1 g
                'Misc/gradient_norm': torch.norm(grads).mean().item(),
                'Misc/cost_gradient_norm': torch.norm(b_grads).mean().item(),
                'Misc/Lambda_star': lambda_star.item(),
                'Misc/Nu_star': nu_star.item(),
                'Misc/OptimCase': int(optim_case),
                'Misc/A': A.item(),
                'Misc/B': B.item(),
                'Misc/q': q.item(),
                'Misc/r': r.item(),
                'Misc/s': s.item(),
            }
        )

    def _cpo_search_step(
        self,
        step_direction: torch.Tensor,
        grads: torch.Tensor,
        p_dist: torch.distributions.Distribution,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
        intrinsic_costs: torch.Tensor,
        loss_reward_before: float,
        loss_cost_before: float,
        total_steps: int = 15,
        decay: float = 0.8,
        violation_c: int = 0,
        optim_case: int = 0,
    ) -> Tuple[torch.Tensor, int]:
        # get distance each time theta goes towards certain direction
        step_frac = 1.0
        # get and flatten parameters from pi-net
        theta_old = get_flat_params_from(self._actor_critic.actor)
        # reward improvement, g-flat as gradient of reward
        expected_reward_improve = grads.dot(step_direction)

        kl = torch.zeros(1)
        # while not within_trust_region and not finish all steps:
        for step in range(total_steps):
            # get new theta
            new_theta = theta_old + step_frac * step_direction
            # set new theta as new actor parameters
            set_param_values_to_model(self._actor_critic.actor, new_theta)
            # the last acceptance steps to next step
            acceptance_step = step + 1

            with torch.no_grad():
                try:
                    # loss of policy reward from target/expected reward
                    loss_reward = self._loss_pi(obs=obs, act=act, logp=logp, adv=adv_r)
                except ValueError:
                    step_frac *= decay
                    continue
                # loss of cost of policy cost from real/expected reward
                loss_cost, _ = self._loss_pi_cost(
                    obs=obs, act=act, logp=logp, adv_c=adv_c, intrinsic_costs=intrinsic_costs
                )
                # compute KL distance between new and old policy
                q_dist = self._actor_critic.actor(obs)
                kl = torch.distributions.kl.kl_divergence(p_dist, q_dist).mean()
            # compute improvement of reward
            loss_reward_improve = loss_reward_before - loss_reward
            # compute difference of cost
            loss_cost_diff = loss_cost - loss_cost_before

            # average across MPI processes...
            kl = distributed.dist_avg(kl)
            # pi_average of torch_kl above
            loss_reward_improve = distributed.dist_avg(loss_reward_improve)
            loss_cost_diff = distributed.dist_avg(loss_cost_diff)
            self._logger.log(
                f'Expected Improvement: {expected_reward_improve} Actual: {loss_reward_improve}',
            )
            # check whether there are nan.
            if not torch.isfinite(loss_reward) and not torch.isfinite(loss_cost):
                self._logger.log('WARNING: loss_pi not finite')
            if not torch.isfinite(kl):
                self._logger.log('WARNING: KL not finite')
                continue
            if loss_reward_improve < 0 if optim_case > 1 else False:
                self._logger.log('INFO: did not improve improve <0')
            # change of cost's range
            elif loss_cost_diff > max(-violation_c, 0):
                self._logger.log(f'INFO: no improve {loss_cost_diff} > {max(-violation_c, 0)}')
            # check KL-distance to avoid too far gap
            elif kl > self._cfgs.algo_cfgs.target_kl:
                self._logger.log(f'INFO: violated KL constraint {kl} at step {step + 1}.')
            else:
                # step only if surrogate is improved and we are
                # within the trust region
                self._logger.log(f'Accept step at i={step + 1}')
                break
            step_frac *= decay
        else:
            # if didn't find a step satisfy those conditions
            self._logger.log('INFO: no suitable step found...')
            step_direction = torch.zeros_like(step_direction)
            acceptance_step = 0

        self._logger.store(
            {
                'Train/KL': kl,
            },
        )

        set_param_values_to_model(self._actor_critic.actor, theta_old)
        return step_frac * step_direction, acceptance_step

    def _loss_pi_cost(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_c: torch.Tensor,
        intrinsic_costs: torch.Tensor,
    ) -> torch.Tensor:
        self._actor_critic.actor(obs)
        logp_ = self._actor_critic.actor.log_prob(act)
        ratio = torch.exp(logp_ - logp)
        cost_loss = (ratio * adv_c).mean()  
        
        return cost_loss, ratio
