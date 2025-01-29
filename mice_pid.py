"""Implement of MICE with PIDLag"""

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

from omnisafe.utils.config import Config
from omnisafe.typing import AdvatageEstimator, OmnisafeSpace
from omnisafe.utils.math import discount_cumsum

from omnisafe.algorithms.on_policy.pid_lagrange.cppo_pid import CPPOPID
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
from omnisafe.common.pid_lagrange import PIDLagrangian




@registry.register
class MICEPID(CPPOPID):
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
        self._lagrange: PIDLagrangian = PIDLagrangian(**self._cfgs.lagrange_cfgs)
        self._intrinsic_obj_type = self._cfgs.algo_cfgs.intrinsic_obj_type
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

        self._emb_obs_dim = 8
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

            # save model to disk
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

        # note that logger already uses MPI statistics across all processes.
        Jc = self._logger.get_stats('Metrics/EpCost')[0]

 
        ep_discount_ci = balancing_ep_dicount_ci.mean().item()

        self._logger.store({'Train/discount_ci': ep_discount_ci, 
                            'Train/epcost_factor': self._epcost_factor, 
                            'Train/intrinsic_factor': self._cfgs.algo_cfgs.intrinsic_factor,
                            'Train/risk_gamma': self._cfgs.algo_cfgs.risk_gamma,})
        Jc += ep_discount_ci

        self._lagrange.pid_update(Jc)


        original_obs = obs
        old_distribution = self._actor_critic.actor(obs)

        dataloader = DataLoader(
            dataset=TensorDataset(obs, act, logp, target_value_r, target_value_c, adv_r, adv_c),
            batch_size=self._cfgs.algo_cfgs.batch_size,
            shuffle=True,
        )

        update_counts = 0
        final_kl = 0.0

        for i in track(range(self._cfgs.algo_cfgs.update_iters), description='Updating...'):
            for (
                obs,
                act,
                logp,
                target_value_r,
                target_value_c,
                adv_r,
                adv_c,
            ) in dataloader:
                self._update_reward_critic(obs, target_value_r)
                if self._cfgs.algo_cfgs.use_cost:
                    self._update_cost_critic(obs, target_value_c)
                self._update_actor(obs, act, logp, adv_r, adv_c)

            new_distribution = self._actor_critic.actor(original_obs)

            kl = (
                torch.distributions.kl.kl_divergence(old_distribution, new_distribution)
                .sum(-1, keepdim=True)
                .mean()
            )
            kl = distributed.dist_avg(kl)

            final_kl = kl.item()
            update_counts += 1

            if self._cfgs.algo_cfgs.kl_early_stop and kl.item() > self._cfgs.algo_cfgs.target_kl:
                self._logger.log(f'Early stopping at iter {i + 1} due to reaching max kl')
                break

        self._logger.store(
            {
                'Train/StopIter': update_counts,  # pylint: disable=undefined-loop-variable
                'Value/Adv': adv_r.mean().item(),
                'Train/KL': final_kl,
            },
        )
        self._logger.store({'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier})


   