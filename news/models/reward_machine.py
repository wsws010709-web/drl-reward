from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Sequence, Dict, Any

import numpy as np
import torch
import torch.optim as optim

from .reward_model import Reward, Critic


@dataclass
class RewardStep:
    state: np.ndarray
    action: np.ndarray
    sparse_reward: float
    candidate_actions: Optional[np.ndarray] = None
    overline_V: float = 0.0


class RewardMachine:
    """
    Upper-level learned reward module.

    Real reward: terminal sparse reward from the environment.
    Learned reward: dense reward R_omega(s, a) used to train PPO.

    The implementation follows the paper's simplified upper-level update:
        max E[ A_R(s,a) * A_Romega(s,a) ]
    where A_R is approximated by terminal sparse return minus a learned value
    baseline, and A_Romega is approximated by the chosen edge reward minus the
    average reward over feasible candidate edges in the same state.
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 device: torch.device,
                 hidden_dim: int = 128,
                 encode_dim: int = 128,
                 reward_lr: float = 3e-4,
                 value_lr: float = 3e-4,
                 gamma: float = 0.99,
                 reward_buffer_size: int = 256,
                 batch_size: int = 512,
                 l2_coef: float = 1e-4,
                 activation_function=torch.relu,
                 last_activation=None):
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.l2_coef = l2_coef

        self.value_function = Critic(
            input_dim=state_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            layer_num=2,
            activation_function=activation_function,
            last_activation=None,
        ).to(self.device)
        self.reward_function = Reward(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            encode_dim=encode_dim,
            output_dim=1,
            activation_function=activation_function,
            last_activation=last_activation,
        ).to(self.device)

        self.value_function_optimizer = optim.Adam(self.value_function.parameters(), lr=value_lr)
        self.reward_function_optimizer = optim.Adam(self.reward_function.parameters(), lr=reward_lr)
        self.D_xi: deque[List[RewardStep]] = deque(maxlen=reward_buffer_size)

    def __len__(self) -> int:
        return sum(len(traj) for traj in self.D_xi)

    def state_dict(self) -> Dict[str, Any]:
        return {
            'reward_function': self.reward_function.state_dict(),
            'value_function': self.value_function.state_dict(),
            'reward_optimizer': self.reward_function_optimizer.state_dict(),
            'value_optimizer': self.value_function_optimizer.state_dict(),
            'buffer': list(self.D_xi),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.reward_function.load_state_dict(state_dict['reward_function'])
        self.value_function.load_state_dict(state_dict['value_function'])
        if 'reward_optimizer' in state_dict:
            self.reward_function_optimizer.load_state_dict(state_dict['reward_optimizer'])
        if 'value_optimizer' in state_dict:
            self.value_function_optimizer.load_state_dict(state_dict['value_optimizer'])
        if 'buffer' in state_dict:
            self.D_xi = deque(state_dict['buffer'], maxlen=self.D_xi.maxlen)

    def observe_reward(self, state: np.ndarray, action: np.ndarray, next_state=None) -> float:
        """Dense reward R_omega(s, a) used by PPO."""
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_t = torch.as_tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        reward = self.reward_function(state_t, action_t).detach().cpu().item()
        return float(reward)

    def store_V(self, epidata: Sequence[RewardStep]) -> List[RewardStep]:
        """Back-propagate sparse terminal reward into discounted return overline_V."""
        new_epidata: List[RewardStep] = []
        overline_V = 0.0
        for step in reversed(epidata):
            overline_V = float(step.sparse_reward) + self.gamma * overline_V
            updated_step = RewardStep(
                state=np.asarray(step.state, dtype=np.float32),
                action=np.asarray(step.action, dtype=np.float32),
                sparse_reward=float(step.sparse_reward),
                candidate_actions=None if step.candidate_actions is None else np.asarray(step.candidate_actions, dtype=np.float32),
                overline_V=float(overline_V),
            )
            new_epidata.insert(0, updated_step)
        return new_epidata

    def append_trajectory(self, epidata: Sequence[RewardStep]) -> None:
        if len(epidata) == 0:
            return
        self.D_xi.append(self.store_V(epidata))

    def optimize_value_function(self, states_batch: np.ndarray, overline_V_batch: np.ndarray) -> float:
        states_t = torch.as_tensor(states_batch, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(overline_V_batch, dtype=torch.float32, device=self.device).unsqueeze(-1)
        pred_batch = self.value_function(states_t)
        loss = torch.nn.functional.smooth_l1_loss(pred_batch, returns_t)
        self.value_function_optimizer.zero_grad()
        loss.backward()
        self.value_function_optimizer.step()
        return float(loss.detach().cpu().item())

    def _reward_advantage(self, step: RewardStep) -> torch.Tensor:
        state_t = torch.as_tensor(step.state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_t = torch.as_tensor(step.action, dtype=torch.float32, device=self.device).unsqueeze(0)
        reward_hat = self.reward_function(state_t, action_t).squeeze()

        candidate_actions = step.candidate_actions
        if candidate_actions is None or len(candidate_actions) == 0:
            reward_center = torch.zeros_like(reward_hat)
        else:
            cand_t = torch.as_tensor(candidate_actions, dtype=torch.float32, device=self.device)
            state_rep = state_t.repeat(cand_t.shape[0], 1)
            reward_all = self.reward_function(state_rep, cand_t).squeeze(-1)
            reward_center = reward_all.mean()
        return reward_hat - reward_center

    def optimize_reward(self) -> Dict[str, float]:
        if len(self.D_xi) == 0:
            return {'reward_loss': 0.0, 'value_loss': 0.0, 'align_mean': 0.0}

        D_new = [step for traj in self.D_xi for step in traj]
        if len(D_new) == 0:
            return {'reward_loss': 0.0, 'value_loss': 0.0, 'align_mean': 0.0}

        np.random.shuffle(D_new)
        if self.batch_size and len(D_new) > self.batch_size:
            D_new = D_new[:self.batch_size]

        states_batch = np.stack([step.state for step in D_new], axis=0)
        returns_batch = np.asarray([step.overline_V for step in D_new], dtype=np.float32)
        value_loss = self.optimize_value_function(states_batch, returns_batch)

        align_terms = []
        reward_reg = []
        for step in D_new:
            state_t = torch.as_tensor(step.state, dtype=torch.float32, device=self.device).unsqueeze(0)
            V_s = self.value_function(state_t).squeeze()
            A_sparse = torch.as_tensor(step.overline_V, dtype=torch.float32, device=self.device) - V_s
            A_learned = self._reward_advantage(step)
            align_terms.append(A_sparse.detach() * A_learned)
            reward_reg.append(A_learned.pow(2))

        align_tensor = torch.stack(align_terms)
        reg_tensor = torch.stack(reward_reg).mean()
        reward_loss = -align_tensor.mean() + self.l2_coef * reg_tensor

        self.reward_function_optimizer.zero_grad()
        reward_loss.backward()
        self.reward_function_optimizer.step()

        return {
            'reward_loss': float(reward_loss.detach().cpu().item()),
            'value_loss': float(value_loss),
            'align_mean': float(align_tensor.detach().mean().cpu().item()),
        }
