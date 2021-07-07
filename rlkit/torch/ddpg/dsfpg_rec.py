from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer


class DSFPGTrainer_Rec(TorchTrainer):
    """
    Deep Deterministic Policy Gradient
    """
    def __init__(
            self,
            phi,
            sf,
            target_sf,
            qf,
            target_qf,
            policy,
            target_policy,

            discount=0.99,
            reward_scale=1.0,

            phi_learning_rate=1e-4,
            policy_learning_rate=3e-4,
            qf_learning_rate=1e-3,
            sf_learning_rate=1e-3,
            sf_weight_decay=0,
            target_hard_update_period=1000,
            tau=1e-2,
            use_soft_update=False,
            qf_criterion=None,
            sf_criterion=None,
            general_criterion=None,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,

            min_q_value=-np.inf,
            max_q_value=np.inf,

            pre_training=False,
    ):
        super().__init__()
        if qf_criterion is None:
            qf_criterion = nn.MSELoss()
        if sf_criterion is None:
            sf_criterion = nn.MSELoss()
        if general_criterion is None:
            general_criterion = nn.MSELoss()


        self.phi = phi
        self.sf = sf
        self.target_sf = target_sf
        self.qf = qf
        self.target_qf = target_qf
        self.policy = policy
        self.target_policy = target_policy

        self.discount = discount
        self.reward_scale = reward_scale

        self.phi_learning_rate = phi_learning_rate
        self.policy_learning_rate = policy_learning_rate
        self.qf_learning_rate = qf_learning_rate
        self.sf_learning_rate = sf_learning_rate
        self.sf_weight_decay = sf_weight_decay
        self.target_hard_update_period = target_hard_update_period
        self.tau = tau
        self.use_soft_update = use_soft_update
        self.qf_criterion = qf_criterion
        self.sf_criterion = sf_criterion
        self.general_criterion = general_criterion
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.min_q_value = min_q_value
        self.max_q_value = max_q_value

        # qf might be a linear approximator
        self.phi_optimizer = optimizer_class(
            self.phi.parameters(),
            lr=self.phi_learning_rate,
        )
        self.sf_optimizer = optimizer_class(
            self.sf.parameters(),
            lr=self.sf_learning_rate,
        )
        self.qf_optimizer = optimizer_class(
            self.qf.parameters(),
            lr=self.qf_learning_rate,
        )
        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=self.policy_learning_rate,
        )

        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def train_from_torch(self, batch):
        # torch.autograd.set_detect_anomaly(True)
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        '''
        Phi operations
        '''
        next_state_prediction, action_rec, reward_prediction = self.phi(obs, actions)
        phi_loss = self.general_criterion(next_state_prediction, next_obs) + \
                    0.1 * self.general_criterion(reward_prediction, rewards) + \
                        self.general_criterion(action_rec, actions)
        
        self.phi_optimizer.zero_grad()
        phi_loss.backward()
        self.phi_optimizer.step()
        """
        Policy operations.
        """

        if self.policy_pre_activation_weight > 0:
            policy_actions, pre_tanh_value = self.policy(
                obs, return_preactivations=True,
            )
            pre_activation_policy_loss = (
                (pre_tanh_value**2).sum(dim=1).mean()
            )
            q_output = self.qf(obs, policy_actions)
            raw_policy_loss = - q_output.mean()
            policy_loss = (
                    raw_policy_loss +
                    pre_activation_policy_loss * self.policy_pre_activation_weight
            )
        else:
            policy_actions = self.policy(obs)
            sf_output = self.sf(obs, policy_actions)
            q_output = self.qf(sf_output)
            raw_policy_loss = policy_loss = - q_output.mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        """
        Successor Feature operations.
        """
        next_actions = self.target_policy(next_obs)
        latent = self.phi.latent(obs, actions)
        # speed up computation by not backpropping these gradients
        next_actions.detach()
        latent.detach()
        target_sf_values = self.target_sf(
            next_obs,
            next_actions,
        )
        sf_target = latent + (1. - terminals) * self.discount * target_sf_values

        sf_target = sf_target.detach()
        sf_target = torch.clamp(sf_target, self.min_q_value, self.max_q_value)
        sf_pred = self.sf(obs, actions)
        bellman_errors = (sf_pred - sf_target) ** 2
        raw_sf_loss = self.sf_criterion(sf_pred, sf_target)

        if self.sf_weight_decay > 0:
            reg_loss = self.sf_weight_decay * sum(
                torch.sum(param ** 2)
                for param in self.sf.regularizable_parameters()
            )
            sf_loss = raw_sf_loss + reg_loss
        else:
            sf_loss = raw_sf_loss

        self.sf_optimizer.zero_grad()
        sf_loss.backward()
        self.sf_optimizer.step()

        """
        Critic operations
        """
        latent = self.phi.latent(obs, actions)
        # latent.detach()
        reward_pred = self.qf(latent)
        qf_loss = self.qf_criterion(reward_pred, rewards)

        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        """
        Update Networks
        """

        self._update_target_networks()

        """
        Save some statistics for eval using just one batch.
        """
        q_pred = self.qf(sf_pred.detach())
        q_target = self.qf(sf_target.detach())
        bellman_errors = (q_pred - q_target) ** 2

        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics['Raw Policy Loss'] = np.mean(ptu.get_numpy(
                raw_policy_loss
            ))
            self.eval_statistics['Preactivation Policy Loss'] = (
                    self.eval_statistics['Policy Loss'] -
                    self.eval_statistics['Raw Policy Loss']
            )
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Bellman Errors',
                ptu.get_numpy(bellman_errors),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy Action',
                ptu.get_numpy(policy_actions),
            ))
        self._n_train_steps_total += 1

    def _update_target_networks(self):
        if self.use_soft_update:
            ptu.soft_update_from_to(self.policy, self.target_policy, self.tau)
            ptu.soft_update_from_to(self.qf, self.target_qf, self.tau)
            ptu.soft_update_from_to(self.sf, self.target_sf, self.tau)
        else:
            if self._n_train_steps_total % self.target_hard_update_period == 0:
                ptu.copy_model_params_from_to(self.qf, self.target_qf)
                ptu.copy_model_params_from_to(self.policy, self.target_policy)
                ptu.copy_model_params_from_to(self.sf, self.target_sf)

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf,
            self.target_policy,
            self.target_qf,
        ]

    def get_epoch_snapshot(self):
        return dict(
            qf=self.qf,
            target_qf=self.target_qf,
            trained_policy=self.policy,
            target_policy=self.target_policy,
        )
