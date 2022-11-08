import time
import numpy as np
import copy

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn as nn

from .off_rl_algo import OffRLAlgo

class ARSAC_Embed_withNorm_v4(OffRLAlgo):
    """
    SAC v4 for Action Representation
    """

    def __init__(
        self,
        pf_state,pf_action,
        embedding,embedding4q,
        qf1, qf2,
        plr, qlr,
        task_nums = 1,
        optimizer_class=optim.Adam,

        policy_std_reg_weight=1e-3,
        policy_mean_reg_weight=1e-3,

        reparameterization=True,
        automatic_entropy_tuning=True,
        target_entropy=None,
        **kwargs
    ):
        super(ARSAC_Embed_withNorm_v4,self).__init__(**kwargs)
        self.pf_state=pf_state
        self.pf_action=pf_action
        self.embedding=embedding
        self.embedding4q=embedding4q
        self.qf1=qf1
        self.qf2=qf2

        self.target_qf1 = copy.deepcopy(qf1)
        self.target_qf2 = copy.deepcopy(qf2)

        self.to(self.device)

        self.plr = plr
        self.qlr = qlr

        self.optimizer_class = optimizer_class

        # self.qf1_optimizer = optimizer_class(
        #     self.qf1.parameters(),
        #     lr=self.qlr,
        # )

        # self.qf2_optimizer = optimizer_class(
        #     self.qf2.parameters(),
        #     lr=self.qlr,
        # )

        self.embedding_optimizer = optimizer_class(
            [self.embedding],
            lr=self.plr,
        )

        self.embedding4q_optimizer = optimizer_class(
            [self.embedding4q],
            lr=self.plr,
        )

        self.automatic_entropy_tuning = automatic_entropy_tuning
        if self.automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()  # from rlkit
            self.log_alpha = torch.zeros(1).to(self.device)
            self.log_alpha.requires_grad_()
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=self.plr,
            )
        self.sample_key = ["obs", "next_obs", "acts", "rewards", "terminals",  "task_idxs", "task_inputs", "embeddings", "embeddings4q"]
        self.qf_criterion = nn.MSELoss()

        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_mean_reg_weight = policy_mean_reg_weight

        self.reparameterization = reparameterization

    def update(self, batch):
            self.training_update_num += 1
            obs       = batch['obs']
            actions   = batch['acts']
            next_obs  = batch['next_obs']
            rewards   = batch['rewards']
            terminals = batch['terminals']
            task_inputs = batch["task_inputs"]
            task_idx    = batch['task_idxs']
            example_embeddings = batch['embeddings']
            example_embeddings4q = batch['embeddings4q']
          
            rewards   = torch.Tensor(rewards).to( self.device )
            terminals = torch.Tensor(terminals).to( self.device )
            obs       = torch.Tensor(obs).to( self.device )
            actions   = torch.Tensor(actions).to( self.device )
            next_obs  = torch.Tensor(next_obs).to( self.device )
            task_inputs = torch.Tensor(task_inputs).to(self.device).long()
            task_idx    = torch.Tensor(task_idx).to( self.device ).long()
            example_embeddings = torch.Tensor(example_embeddings)
            example_embeddings4q = torch.Tensor(example_embeddings4q)
            
       
            embedding = copy.deepcopy(self.embedding)
            embedding_ = 5 * F.normalize(embedding.unsqueeze(0))
            embeddings = embedding_.expand_as(example_embeddings).to(self.device)
            embedding4q = copy.deepcopy(self.embedding4q)
            embedding4q_ = 5 * F.normalize(embedding4q.unsqueeze(0))
            embeddings4q = embedding4q_.expand_as(example_embeddings4q).to(self.device)
        
            
            

            
            

            """
            Policy operations.
            """
            representations=self.pf_state.forward(obs)
            # print(representations.size())
            # print(embeddings)
            # print(representations)
            sample_info = self.pf_action.explore(representations, embeddings, return_log_probs=True )

            mean        = sample_info["mean"]
            log_std     = sample_info["log_std"]
            new_actions = sample_info["action"]
            log_probs   = sample_info["log_prob"]

            q1_pred = self.qf1([obs, actions,embeddings4q])
            q2_pred = self.qf2([obs, actions,embeddings4q])

            if self.automatic_entropy_tuning:
                """
                Alpha Loss
                """
                alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                alpha = self.log_alpha.exp().detach()
            else:
                alpha = 1
                alpha_loss = 0

            with torch.no_grad():
                representations = self.pf_state.forward(next_obs)

                target_sample_info = self.pf_action.explore(representations, embeddings, return_log_probs=True )

                target_actions   = target_sample_info["action"]
                target_log_probs = target_sample_info["log_prob"]

                target_q1_pred = self.target_qf1([next_obs, target_actions,embeddings4q.detach()])
                target_q2_pred = self.target_qf2([next_obs, target_actions,embeddings4q.detach()])
                min_target_q = torch.min(target_q1_pred, target_q2_pred)
                target_v_values = min_target_q - alpha * target_log_probs
            """
            QF Loss
            """
            q_target = rewards + (1. - terminals) * self.discount * target_v_values
            qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
            qf2_loss = self.qf_criterion(q2_pred, q_target.detach())
            qf_loss = qf1_loss + qf2_loss
            assert q1_pred.shape == q_target.shape
            assert q2_pred.shape == q_target.shape
            # qf1_loss = (0.5 * ( q1_pred - q_target.detach() ) ** 2).mean()
            # qf2_loss = (0.5 * ( q2_pred - q_target.detach() ) ** 2).mean()

            q_new_actions = torch.min(
                self.qf1([obs, new_actions, embeddings4q.detach()]),
                self.qf2([obs, new_actions, embeddings4q.detach()]))
            """
            Policy Loss
            """
            if not self.reparameterization:
                raise NotImplementedError
            else:
                assert log_probs.shape == q_new_actions.shape
                policy_loss = ( alpha * log_probs - q_new_actions).mean()

            std_reg_loss = self.policy_std_reg_weight * (log_std**2).mean()
            mean_reg_loss = self.policy_mean_reg_weight * (mean**2).mean()
            policy_loss += std_reg_loss + mean_reg_loss
            
            """
            Update Networks
            """

            self.embedding_optimizer.zero_grad()
            policy_loss.backward()
            self.embedding.grad = embedding.grad
            # embedding_norm = torch.nn.utils.clip_grad_norm_([self.embedding], 10)         
            self.embedding_optimizer.step()
            
            self.embedding4q_optimizer.zero_grad()
            qf_loss.backward()
            self.embedding4q.grad = embedding4q.grad
            # embedding4q_norm = torch.nn.utils.clip_grad_norm_([self.embedding4q], 10)         
            self.embedding4q_optimizer.step()
         
            # self.embedding4q_optimizer.zero_grad()
            # qf2_loss.backward()
            # self.embedding4q.grad = embedding4q.grad
            # # embedding4q_norm = torch.nn.utils.clip_grad_norm_([self.embedding4q], 10)         
            # self.embedding4q_optimizer.step()

            # self.qf1_optimizer.zero_grad()
            # qf1_loss.backward()
            # qf1_norm = torch.nn.utils.clip_grad_norm_(self.qf1.parameters(), 10)
            # self.qf1_optimizer.step()

            # self.qf2_optimizer.zero_grad()
            # qf2_loss.backward()
            # qf2_norm = torch.nn.utils.clip_grad_norm_(self.qf2.parameters(), 10)
            # self.qf2_optimizer.step()

            # self._update_target_networks()

            # Information For Logger
            info = {}
            info['Reward_Mean'] = rewards.mean().item()

            if self.automatic_entropy_tuning:
                info["Alpha"] = alpha.item()
                info["Alpha_loss"] = alpha_loss.item()
            info['Training/policy_loss'] = policy_loss.item()
            info['Training/qf1_loss'] = qf1_loss.item()
            info['Training/qf2_loss'] = qf2_loss.item()

            
            # info['Training/embedding_norm'] = embedding_norm.item()
            # info['Training/embedding4q_norm'] = embedding4q_norm.item()
            # info['Training/qf1_norm'] = qf1_norm.item()
            # info['Training/qf2_norm'] = qf2_norm.item()

            info['log_std/mean'] = log_std.mean().item()
            info['log_std/std'] = log_std.std().item()
            info['log_std/max'] = log_std.max().item()
            info['log_std/min'] = log_std.min().item()

            info['log_probs/mean'] = log_probs.mean().item()
            info['log_probs/std'] = log_probs.std().item()
            info['log_probs/max'] = log_probs.max().item()
            info['log_probs/min'] = log_probs.min().item()

            info['mean/mean'] = mean.mean().item()
            info['mean/std'] = mean.std().item()
            info['mean/max'] = mean.max().item()
            info['mean/min'] = mean.min().item()

            return info

    @property
    def networks(self):
        return [
            self.pf_state,
            self.pf_action,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2
        ]
        
    @property
    def snapshot_networks(self):
        return [
            # ["qf1", self.qf1],
            # ["qf2", self.qf2],
        ]

    @property
    def target_networks(self):
        return [
            # ( self.qf1, self.target_qf1 ),
            # ( self.qf2, self.target_qf2 )
        ]
