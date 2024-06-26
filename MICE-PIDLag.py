"""Implementation of the Lagrange version of the MICE-PID algorithm."""

import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.ppo import PPO
from omnisafe.common.pid_lagrange import PIDLagrangian
from omnisafe.common.lagrange import Lagrange
from omnisafe.utils import distributed
from rollout import MICEAdapter
from memory import LFFMVectorBuffer, LFFM2VectorBuffer, FailBuffer
import os
import time
from typing import Dict, Tuple, Optional, Union
import utils as utl
from rich.progress import track
from torch.utils.data import DataLoader, TensorDataset


@registry.register
class MICE_PIDLag(PPO):

    def _init_env(self) -> None:
        self._env = MICEAdapter(
            self._env_id, self._cfgs.train_cfgs.vector_env_nums, self._seed, self._cfgs
        )
        assert (self._cfgs.algo_cfgs.update_cycle) % (
            distributed.world_size() * self._cfgs.train_cfgs.vector_env_nums
        ) == 0, ('The number of steps per epoch is not divisible by the number of ' 'environments.')
        self._steps_per_epoch = (
            self._cfgs.algo_cfgs.update_cycle
            // distributed.world_size()
            // self._cfgs.train_cfgs.vector_env_nums
        )

    def _init(self) -> None:
        self._optimization_type = self._cfgs.algo_cfgs.risk_penalty_type 

        if self._optimization_type == 0:
            self._buf = LFFM2VectorBuffer(
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
        elif self._optimization_type in [1, 2]:
            self._buf = LFFMVectorBuffer(
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
        self.RPNet = utl.RandomProjection(self._cfgs.algo_cfgs.input_dim, self._cfgs.algo_cfgs.output_dim).to(self._device)
        self._IntrinsicG = utl.IntrinsicGenerator(self._cfgs.algo_cfgs.output_dim, self._cfgs.algo_cfgs.hidden_dim, self._cfgs.algo_cfgs.cost_dim).to(self._device)
        self._maxlen_fail = self._cfgs.algo_cfgs.fail_buf_size 
        self._flashbulb_memory = FailBuffer(size=self._maxlen_fail)
         
        self._pidlag = PIDLagrangian(**self._cfgs.pidlag_cfgs)

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Train/intrinsic_costs')
        self._logger.register_key('Value/Adv_c')
        self._logger.register_key('Eval/true_value_c')
        self._logger.register_key('Eval/estimate_value_c')
        self._logger.register_key('Metrics/PIDLagrangeMultiplier')

    def learn(self) -> Tuple[Union[int, float], ...]: 
        start_time = time.time()
        self._logger.log('INFO: Start training')

        for epoch in range(self._cfgs.train_cfgs.epochs):
            epoch_time = time.time()

            roll_out_time = time.time()
            self._flashbulb_memory = self._env.roll_out( 
                steps_per_epoch=self._steps_per_epoch,
                agent=self._actor_critic,
                buffer=self._buf,
                flashbulb_memory=self._flashbulb_memory,  
                logger=self._logger,
                rpnet=self.RPNet,
                intrinsicG=self._IntrinsicG,  
                epoch=epoch,
            )
            self._logger.store(**{'Train/Epoch': epoch})


            if self._cfgs.algo_cfgs.test_estimate:
                true_value_c, estimate_value_c = utl.estimate_true_value(
                    agent=self._actor_critic,
                    env_id=self._env_id,
                    num_envs=1,
                    seed=self._seed,
                    cfgs=self._cfgs,
                    discount=self._cfgs.algo_cfgs.cost_gamma,
                    eval_episodes=1,
                )
                self._logger.store(**{'Eval/true_value_c': true_value_c})
                self._logger.store(**{'Eval/estimate_value_c': estimate_value_c})
                with open(os.path.join(self._logger._log_dir, 'overestimation.txt'), 'a') as file:
                    file.write(f'true_value_c: {true_value_c.item()}\n')
                    file.write(f'estimate_value_c: {estimate_value_c.item()}\n\n')

            
            update_time = time.time()
            self._logger.store(**{'Time/Rollout': time.time() - roll_out_time})
            self._update()  
            self._logger.store(**{'Time/Update': time.time() - update_time})

            if self._cfgs.model_cfgs.exploration_noise_anneal:
                self._actor_critic.annealing(epoch)  

            if self._cfgs.model_cfgs.actor.lr != 'None':
                self._actor_critic.actor_scheduler.step() 

            self._logger.store(
                **{
                    'TotalEnvSteps': (epoch + 1) * self._cfgs.algo_cfgs.update_cycle,
                    'Time/FPS': self._cfgs.algo_cfgs.update_cycle / (time.time() - epoch_time),
                    'Time/Total': (time.time() - start_time),
                    'Time/Epoch': (time.time() - epoch_time),
                    'Train/LR': 0.0
                    if self._cfgs.model_cfgs.actor.lr == 'None'
                    else self._actor_critic.actor_scheduler.get_last_lr()[0],
                }
            )

            self._logger.dump_tabular()

            
            if (epoch + 1) % self._cfgs.logger_cfgs.save_model_freq == 0:
                self._logger.torch_save()

        ep_ret = self._logger.get_stats('Metrics/EpRet')[0]
        ep_cost = self._logger.get_stats('Metrics/EpCost')[0]
        ep_len = self._logger.get_stats('Metrics/EpLen')[0]
        self._logger.close()

        return ep_ret, ep_cost, ep_len

    def _update(self) -> None:
        data = self._buf.get() 
        obs, act, logp, target_value_r, target_value_c, adv_r, adv_c, intrinsic_costs = (
            data['obs'],
            data['act'],
            data['logp'],
            data['target_value_r'], 
            data['target_value_c'], 
            data['adv_r'],
            data['adv_c'],
            data['intrinsic_costs'],   
        )

        Jc = self._logger.get_stats('Metrics/EpCost')[0]  
        if self._optimization_type == 1:
            Jc -= intrinsic_costs.mean()

        self._pidlag.pid_update(Jc)  

        original_obs = obs
        old_distribution = self._actor_critic.actor(obs) 

        dataloader = DataLoader(
            dataset=TensorDataset(obs, act, logp, target_value_r, target_value_c, adv_r, adv_c),
            batch_size=self._cfgs.algo_cfgs.batch_size,
            shuffle=True,
        )                                                  

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
                .item()
            )
            kl = distributed.dist_avg(kl)  

            if self._cfgs.algo_cfgs.kl_early_stop and kl > self._cfgs.algo_cfgs.target_kl:
                self._logger.log(f'Early stopping at iter {i + 1} due to reaching max kl')
                break

        self._logger.store(
            **{
                'Train/StopIter': i + 1, 
                'Value/Adv': adv_r.mean().item(),
                'Value/Adv_c': adv_c.mean().item(),
                'Train/KL': kl,
                'Train/intrinsic_costs': intrinsic_costs.mean().item(),
            }
        )

        self._logger.store(**{'Metrics/PIDLagrangeMultiplier': self._pidlag.cost_penalty})


    def _update_actor(  
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> None:
        adv = self._compute_adv_surrogate(adv_r, adv_c)  
        loss, info = self._loss_pi(obs, act, logp, adv)
        self._actor_critic.actor_optimizer.zero_grad()
        loss.backward()
        if self._cfgs.algo_cfgs.use_max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self._actor_critic.actor.parameters(), self._cfgs.algo_cfgs.max_grad_norm
            )
        distributed.avg_grads(self._actor_critic.actor)
        self._actor_critic.actor_optimizer.step()
        self._logger.store(
            **{
                'Train/Entropy': info['entropy'],
                'Train/PolicyRatio': info['ratio'],
                'Train/PolicyStd': info['std'],
                'Loss/Loss_pi': loss.mean().item(),
            }
        )

    def _loss_pi(
        self, obs: torch.Tensor, act: torch.Tensor, logp: torch.Tensor, adv: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        distribution = self._actor_critic.actor(obs)  
        logp_ = self._actor_critic.actor.log_prob(act)  
        std = self._actor_critic.actor.std 
        ratio = torch.exp(logp_ - logp)  
        ratio_cliped = torch.clamp(
            ratio, 1 - self._cfgs.algo_cfgs.clip, 1 + self._cfgs.algo_cfgs.clip
        )
        loss = -torch.min(ratio * adv, ratio_cliped * adv).mean() 
        loss -= self._cfgs.algo_cfgs.entropy_coef * distribution.entropy().mean()  
        entropy = distribution.entropy().mean().item()
        info = {'entropy': entropy, 'ratio': ratio.mean().item(), 'std': std}
        return loss, info


    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        penalty = self._pidlag.cost_penalty
        return (adv_r - penalty * adv_c) / (1 + penalty)  
    

