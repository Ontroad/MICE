"""Implement of MICE-CPO"""

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
from omnisafe.algorithms.on_policy.second_order.cpo import CPO
from omnisafe.utils import distributed
from omnisafe.utils.math import conjugate_gradients
from omnisafe.utils.tools import (
    get_flat_gradients_from,
    get_flat_params_from,
    set_param_values_to_model,
)
from rollout import MICEAdapter
from buffer import LFFMVectorBuffer, LFFM2VectorBuffer, FailBuffer
import utils as utl
import os



@registry.register
class MICECPO(CPO):
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
        self._risk_penalty_type = self._cfgs.algo_cfgs.risk_penalty_type 
        if self._risk_penalty_type == 0:
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
        elif self._risk_penalty_type in [1, 2]:
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
        self._maxlen_fail = self._cfgs.algo_cfgs.fail_buf_size 
        self._failure_buf = FailBuffer(size=self._maxlen_fail)


    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Train/risk_metrics')
        self._logger.register_key('Value/Adv_c')
        self._logger.register_key('Eval/true_value_c')
        self._logger.register_key('Eval/estimate_value_c')

    def learn(self) -> Tuple[Union[int, float], ...]:  
        start_time = time.time()
        self._logger.log('INFO: Start training')

        for epoch in range(self._cfgs.train_cfgs.epochs):  
            epoch_time = time.time()

            roll_out_time = time.time()
            self._failure_buf = self._env.roll_out( 
                steps_per_epoch=self._steps_per_epoch,
                agent=self._actor_critic,
                buffer=self._buf,
                failure_buffer=self._failure_buf,  
                logger=self._logger,
                rpnet=self.RPNet, 
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
        (
            obs,
            act,
            logp,
            target_value_r,
            target_value_c,
            adv_r,
            adv_c,
            risk_metrics,
        ) = (
            data['obs'],
            data['act'],
            data['logp'],
            data['target_value_r'],
            data['target_value_c'],
            data['adv_r'],
            data['adv_c'],
            data['risk_metrics'],  
        )
        self._update_actor(obs, act, logp, adv_r, adv_c, risk_metrics)

        dataloader = DataLoader(
            dataset=TensorDataset(obs, act, logp, target_value_r, target_value_c, adv_r, adv_c),
            batch_size=self._cfgs.algo_cfgs.batch_size,
            shuffle=True,
        )

        for i in range(self._cfgs.algo_cfgs.update_iters):
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

        self._logger.store(
            **{
                'Train/StopIter': i + 1,
                'Value/Adv': adv_r.mean().item(),
                'Value/Adv_c': adv_c.mean().item(),
            }
        )

   
    def _update_actor(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
        risk_metrics: torch.Tensor,
    ) -> None:
        self._fvp_obs = obs[:: self._cfgs.algo_cfgs.fvp_sample_freq]
        theta_old = get_flat_params_from(self._actor_critic.actor)  
        self._actor_critic.actor.zero_grad()  
        loss_reward, info = self._loss_pi(obs, act, logp, adv_r)  
        loss_reward_before = distributed.dist_avg(loss_reward).item() 
        p_dist = self._actor_critic.actor(obs)  

        loss_reward.backward() 
        distributed.avg_grads(self._actor_critic.actor) 

        grad = -get_flat_gradients_from(
            self._actor_critic.actor
        )  
        x = conjugate_gradients(self._fvp, grad, self._cfgs.algo_cfgs.cg_iters)  
        assert torch.isfinite(x).all(), 'x is not finite'
        xHx = x.dot(self._fvp(x))  #
        assert xHx.item() >= 0, 'xHx is negative'
        alpha = torch.sqrt(2 * self._cfgs.algo_cfgs.target_kl / (xHx + 1e-8)) 

        self._actor_critic.zero_grad()
        loss_cost, ratio = self._loss_pi_cost(obs, act, logp, adv_c, risk_metrics)  
        loss_cost_before = distributed.dist_avg(loss_cost).item()

        loss_cost.backward()
        distributed.avg_grads(self._actor_critic.actor)

        b_grad = get_flat_gradients_from(
            self._actor_critic.actor
        )  

        ep_costs = self._logger.get_stats('Metrics/EpCost')[0] - self._cfgs.algo_cfgs.cost_limit
        if self._risk_penalty_type == 1:
            ep_costs = (
                self._logger.get_stats('Metrics/EpCost')[0]
                - self._cfgs.algo_cfgs.cost_limit
                + (ratio * risk_metrics).mean()
            ) 

        p = conjugate_gradients(self._fvp, b_grad, self._cfgs.algo_cfgs.cg_iters)  # H^-1*b
        q = xHx
        r = grad.dot(p)  
        s = b_grad.dot(p)  

        if (
            b_grad.dot(b_grad) <= 1e-6 and ep_costs < 0
        ):  
            A = torch.zeros(1)
            B = torch.zeros(1)
            optim_case = 4
        else:  
            assert torch.isfinite(r).all(), 'r is not finite'
            assert torch.isfinite(s).all(), 's is not finite'

            A = q - r**2 / (s + 1e-8)
            B = 2 * self._cfgs.algo_cfgs.target_kl - ep_costs**2 / (s + 1e-8)

            if ep_costs < 0 and B < 0:  
                optim_case = 3
            elif ep_costs < 0 <= B:  
                optim_case = 2
            elif ep_costs >= 0 and B >= 0:  
                optim_case = 1
                self._logger.log('Alert! Attempting feasible recovery!', 'yellow')
            else: 
                optim_case = 0
                self._logger.log('Alert! Attempting infeasible recovery!', 'red')

        if optim_case in (3, 4):
            alpha = torch.sqrt(2 * self._cfgs.algo_cfgs.target_kl / (xHx + 1e-8))  
            nu_star = torch.zeros(1)
            lambda_star = 1 / (alpha + 1e-8)
            step_direction = alpha * x

        elif optim_case in (1, 2): 

            def project(data: torch.Tensor, low: float, high: float) -> torch.Tensor:
                
                return torch.max(torch.min(data, torch.tensor(high)), torch.tensor(low))

        
            lambda_a = torch.sqrt(A / B)
            lambda_b = torch.sqrt(q / (2 * self._cfgs.algo_cfgs.target_kl))

            r_num = r.item()
            eps_cost = ep_costs + 1e-8
            if ep_costs < 0:
                lambda_a_star = project(lambda_a, 0.0, r_num / eps_cost)
                lambda_b_star = project(lambda_b, r_num / eps_cost, torch.inf)
            else:
                lambda_a_star = project(lambda_a, r_num / eps_cost, torch.inf)
                lambda_b_star = project(lambda_b, 0.0, r_num / eps_cost)

            def f_a(lam):
                return -0.5 * (A / (lam + 1e-8) + B * lam) - r * ep_costs / (s + 1e-8)

            def f_b(lam):
                return -0.5 * (q / (lam + 1e-8) + 2 * self._cfgs.algo_cfgs.target_kl * lam)

            lambda_star = (  
                lambda_a_star if f_a(lambda_a_star) >= f_b(lambda_b_star) else lambda_b_star
            )


            nu_star = torch.clamp(lambda_star * ep_costs - r, min=0) / (s + 1e-8)  # v*
            step_direction = 1.0 / (lambda_star + 1e-8) * (x - nu_star * p)

        else:  
            lambda_star = torch.zeros(1)
            nu_star = torch.sqrt(2 * self._cfgs.algo_cfgs.target_kl / (s + 1e-8))
            step_direction = -nu_star * p

        step_direction, accept_step = self._cpo_search_step(  
            step_direction=step_direction,
            grad=grad,
            p_dist=p_dist,
            obs=obs,
            act=act,
            logp=logp,
            adv_r=adv_r,
            adv_c=adv_c,
            risk_metrics=risk_metrics,
            loss_reward_before=loss_reward_before,
            loss_cost_before=loss_cost_before,
            total_steps=20,
            violation_c=ep_costs,
            optim_case=optim_case,
        )

        theta_new = theta_old + step_direction  
        set_param_values_to_model(self._actor_critic.actor, theta_new)

        with torch.no_grad():
            loss_reward, info = self._loss_pi(obs, act, logp, adv_r)
            loss_cost, _ = self._loss_pi_cost(obs, act, logp, adv_c, risk_metrics)
            loss = loss_reward + loss_cost

        self._logger.store(
            **{
                'Loss/Loss_pi': loss.item(),
                'Train/Entropy': info['entropy'],
                'Train/PolicyRatio': info['ratio'],
                'Train/PolicyStd': info['std'],
                'Train/risk_metrics': risk_metrics.mean().item(),
                'Misc/AcceptanceStep': accept_step,
                'Misc/Alpha': alpha.item(),
                'Misc/FinalStepNorm': step_direction.norm().mean().item(),
                'Misc/xHx': xHx.mean().item(),
                'Misc/H_inv_g': x.norm().item(),  # H^-1 g
                'Misc/gradient_norm': torch.norm(grad).mean().item(),
                'Misc/cost_gradient_norm': torch.norm(b_grad).mean().item(),
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
        grad: torch.Tensor,
        p_dist: torch.distributions.Distribution,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
        risk_metrics: torch.Tensor,
        loss_reward_before: float,
        loss_cost_before: float,
        total_steps: int = 15,
        decay: float = 0.8,
        violation_c: int = 0,
        optim_case: int = 0,
    ) -> Tuple[torch.Tensor, int]:
       
        step_frac = 1.0  
        theta_old = get_flat_params_from(self._actor_critic.actor)  
        expected_reward_improve = grad.dot(step_direction)  

        for step in range(total_steps):
            
            new_theta = theta_old + step_frac * step_direction 
            set_param_values_to_model(self._actor_critic.actor, new_theta)
            acceptance_step = step + 1

            with torch.no_grad():
                try:
                    loss_reward, _ = self._loss_pi(
                        obs=obs, act=act, logp=logp, adv=adv_r
                    )  
                except ValueError:
                    step_frac *= decay
                    continue
                
                loss_cost, _ = self._loss_pi_cost(
                    obs=obs, act=act, logp=logp, adv_c=adv_c, risk_metrics=risk_metrics
                )  
                
                q_dist = self._actor_critic.actor(obs)  
                kl = torch.distributions.kl.kl_divergence(p_dist, q_dist).mean()  
            
            loss_reward_improve = loss_reward_before - loss_reward.item()  
            
            loss_cost_diff = loss_cost.item() - loss_cost_before  

            
            kl = distributed.dist_avg(kl)
            
            loss_reward_improve = distributed.dist_avg(loss_reward_improve)
            loss_cost_diff = distributed.dist_avg(loss_cost_diff)
            self._logger.log(
                f'Expected Improvement: {expected_reward_improve} Actual: {loss_reward_improve}'
            )
           
            if not torch.isfinite(loss_reward) and not torch.isfinite(loss_cost):
                self._logger.log('WARNING: loss_pi not finite')
            if not torch.isfinite(kl):
                self._logger.log('WARNING: KL not finite')
                continue
            if loss_reward_improve < 0 if optim_case > 1 else False:
                self._logger.log('INFO: did not improve improve <0')
            
            elif loss_cost_diff > max(-violation_c, 0):
                self._logger.log(f'INFO: no improve {loss_cost_diff} > {max(-violation_c, 0)}')
            
            elif kl > self._cfgs.algo_cfgs.target_kl:
                self._logger.log(f'INFO: violated KL constraint {kl} at step {step + 1}.')
            else:
                
                self._logger.log(f'Accept step at i={step + 1}')
                break
            step_frac *= decay
        else:
            
            self._logger.log('INFO: no suitable step found...')
            step_direction = torch.zeros_like(step_direction)
            acceptance_step = 0

        self._logger.store(
            **{
                'Train/KL': kl,
            }
        )

        set_param_values_to_model(self._actor_critic.actor, theta_old)
        return step_frac * step_direction, acceptance_step

    def _loss_pi_cost(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_c: torch.Tensor,
        risk_metrics: torch.Tensor,
    ) -> torch.Tensor:
        self._actor_critic.actor(obs)
        logp_ = self._actor_critic.actor.log_prob(act)
        ratio = torch.exp(logp_ - logp)
        cost_loss = (ratio * adv_c).mean() 
        if self._risk_penalty_type == 2:  
            risk_metric = (ratio * risk_metrics).mean()  
            cost_loss += risk_metric
        return cost_loss, ratio

