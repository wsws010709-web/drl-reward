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
    candidate_probs: Optional[np.ndarray] = None
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
    policy-weighted expected reward over feasible candidate edges in the same
    state.
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
                 stratified_sampling: bool = True,
                 activation_function=torch.relu,
                 last_activation=None):
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.l2_coef = l2_coef
        self.stratified_sampling = stratified_sampling

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

    @staticmethod
    def _zero_optimize_stats() -> Dict[str, float]:
        return {
            'reward_loss': 0.0,
            'value_loss': 0.0,
            'align_mean': 0.0,
            'sign_match': 0.0,
            'reward_hat_mean': 0.0,
            'reward_hat_std': 0.0,
            'reward_center_mean': 0.0,
            'reward_center_std': 0.0,
            'A_sparse_mean': 0.0,
            'A_sparse_std': 0.0,
            'A_sparse_raw_mean': 0.0,
            'A_sparse_raw_std': 0.0,
            'A_learned_mean': 0.0,
            'A_learned_std': 0.0,
            'episode_corr': 0.0,
        }

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
                candidate_probs=None if getattr(step, 'candidate_probs', None) is None else np.asarray(getattr(step, 'candidate_probs', None), dtype=np.float32),
                overline_V=float(overline_V),
            )
            new_epidata.insert(0, updated_step)
        return new_epidata

    def append_trajectory(self, epidata: Sequence[RewardStep]) -> None:
        if len(epidata) == 0:
            return
        self.D_xi.append(self.store_V(epidata))

    def clear_buffer(self) -> None:
        self.D_xi.clear()

    def num_trajectories(self) -> int:
        return len(self.D_xi)

    def optimize_value_function(self, states_batch: np.ndarray, overline_V_batch: np.ndarray) -> float:
        states_t = torch.as_tensor(states_batch, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(overline_V_batch, dtype=torch.float32, device=self.device).unsqueeze(-1)
        pred_batch = self.value_function(states_t)
        loss = torch.nn.functional.smooth_l1_loss(pred_batch, returns_t)
        self.value_function_optimizer.zero_grad()
        loss.backward()
        self.value_function_optimizer.step()
        return float(loss.detach().cpu().item())

    def _reward_terms(self, step: RewardStep):
        state_t = torch.as_tensor(step.state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_t = torch.as_tensor(step.action, dtype=torch.float32, device=self.device).unsqueeze(0)
        reward_hat = self.reward_function(state_t, action_t).squeeze()

        candidate_actions = step.candidate_actions
        candidate_probs = getattr(step, 'candidate_probs', None)
        if candidate_actions is None or len(candidate_actions) == 0:
            reward_center = torch.zeros_like(reward_hat)
        else:
            cand_t = torch.as_tensor(candidate_actions, dtype=torch.float32, device=self.device)
            state_rep = state_t.repeat(cand_t.shape[0], 1)
            reward_all = self.reward_function(state_rep, cand_t).squeeze(-1)
            if candidate_probs is not None and len(candidate_probs) == cand_t.shape[0]:
                probs_t = torch.as_tensor(candidate_probs, dtype=torch.float32, device=self.device)
                probs_sum = probs_t.sum()
                if probs_sum.item() > 1e-8:
                    probs_t = probs_t / probs_sum
                    reward_center = (probs_t * reward_all).sum()
                else:
                    reward_center = reward_all.mean()
            else:
                reward_center = reward_all.mean()
        return reward_hat, reward_center, reward_hat - reward_center

    def _reward_advantage(self, step: RewardStep) -> torch.Tensor:
        return self._reward_terms(step)[2]

    def _flatten_steps_with_episode(self):
        return [(ep_idx, step) for ep_idx, traj in enumerate(self.D_xi) for step in traj]

    def _flatten_steps(self) -> List[RewardStep]:
        return [step for _, step in self._flatten_steps_with_episode()]

    def _sample_steps_stratified(self, target_steps: int):
        trajectories: List[List[RewardStep]] = [traj for traj in self.D_xi if len(traj) > 0]
        if len(trajectories) == 0:
            return []

        total_steps = sum(len(traj) for traj in trajectories)
        target_steps = min(max(int(target_steps), 1), total_steps)
        if target_steps >= total_steps:
            all_items = self._flatten_steps_with_episode()
            np.random.shuffle(all_items)
            return all_items

        ep_order = np.arange(len(trajectories))
        np.random.shuffle(ep_order)
        base_quota = target_steps // len(trajectories)
        remainder = target_steps % len(trajectories)

        selected_steps = []
        selected_indices = {}
        for order_idx, ep_idx in enumerate(ep_order):
            episode = trajectories[ep_idx]
            quota = base_quota + (1 if order_idx < remainder else 0)
            quota = min(quota, len(episode))
            if quota <= 0:
                continue
            idx = np.random.choice(len(episode), size=quota, replace=False)
            selected_indices[ep_idx] = set(idx.tolist())
            for i in idx:
                selected_steps.append((int(ep_idx), episode[int(i)]))

        deficit = target_steps - len(selected_steps)
        if deficit > 0:
            backup_pool = []
            for ep_idx, episode in enumerate(trajectories):
                used = selected_indices.get(ep_idx, set())
                for step_idx, step in enumerate(episode):
                    if step_idx not in used:
                        backup_pool.append((int(ep_idx), step))
            if len(backup_pool) > 0:
                np.random.shuffle(backup_pool)
                selected_steps.extend(backup_pool[:deficit])

        np.random.shuffle(selected_steps)
        return selected_steps

    def optimize_reward(self) -> Dict[str, float]:
        if len(self.D_xi) == 0:
            return self._zero_optimize_stats()

        if self.stratified_sampling:
            sampled_items = self._sample_steps_stratified(self.batch_size)
        else:
            sampled_items = self._flatten_steps_with_episode()
            np.random.shuffle(sampled_items)
            if self.batch_size and len(sampled_items) > self.batch_size:
                sampled_items = sampled_items[:self.batch_size]

        if len(sampled_items) == 0:
            return self._zero_optimize_stats()

        D_new = [step for _, step in sampled_items]

        states_batch = np.stack([step.state for step in D_new], axis=0)
        returns_batch = np.asarray([step.overline_V for step in D_new], dtype=np.float32)

        # Freeze sparse advantages with the pre-update value baseline. The
        # value update below should not erase the signal used by this reward
        # update.
        with torch.no_grad():
            states_t = torch.as_tensor(states_batch, dtype=torch.float32, device=self.device)
            old_values = self.value_function(states_t).squeeze(-1).detach().cpu().numpy()
        raw_a_sparse_values = returns_batch - old_values.astype(np.float32)
        raw_a_sparse_mean = float(raw_a_sparse_values.mean())
        raw_a_sparse_std = float(raw_a_sparse_values.std())
        if raw_a_sparse_std > 1e-8:
            fixed_a_sparse_values = (
                raw_a_sparse_values - raw_a_sparse_mean
            ) / (raw_a_sparse_std + 1e-8)
        else:
            fixed_a_sparse_values = raw_a_sparse_values - raw_a_sparse_mean
        fixed_a_sparse_values = fixed_a_sparse_values.astype(np.float32)

        value_loss = self.optimize_value_function(states_batch, returns_batch)

        align_terms = []
        reward_reg = []
        reward_hat_values = []
        reward_center_values = []
        a_learned_values = []
        a_sparse_values = []
        episode_sparse = {}
        episode_learned = {}

        for (ep_idx, step), a_sparse_scalar in zip(sampled_items, fixed_a_sparse_values):
            A_sparse = torch.as_tensor(a_sparse_scalar, dtype=torch.float32, device=self.device)
            reward_hat, reward_center, A_learned = self._reward_terms(step)
            align_terms.append(A_sparse.detach() * A_learned)
            reward_reg.append(A_learned.pow(2))

            a_sparse_scalar = float(a_sparse_scalar)
            a_learned_scalar = float(A_learned.detach().cpu().item())
            reward_hat_scalar = float(reward_hat.detach().cpu().item())
            reward_center_scalar = float(reward_center.detach().cpu().item())

            a_sparse_values.append(a_sparse_scalar)
            a_learned_values.append(a_learned_scalar)
            reward_hat_values.append(reward_hat_scalar)
            reward_center_values.append(reward_center_scalar)
            episode_sparse.setdefault(ep_idx, []).append(a_sparse_scalar)
            episode_learned.setdefault(ep_idx, []).append(a_learned_scalar)

        align_tensor = torch.stack(align_terms)
        reg_tensor = torch.stack(reward_reg).mean()
        reward_loss = -align_tensor.mean() + self.l2_coef * reg_tensor

        self.reward_function_optimizer.zero_grad()
        reward_loss.backward()
        self.reward_function_optimizer.step()

        stats = self._zero_optimize_stats()
        stats['reward_loss'] = float(reward_loss.detach().cpu().item())
        stats['value_loss'] = float(value_loss)
        stats['align_mean'] = float(align_tensor.detach().mean().cpu().item())

        reward_hat_arr = np.asarray(reward_hat_values, dtype=np.float64)
        reward_center_arr = np.asarray(reward_center_values, dtype=np.float64)
        a_learned_arr = np.asarray(a_learned_values, dtype=np.float64)
        a_sparse_arr = np.asarray(a_sparse_values, dtype=np.float64)

        stats['A_sparse_raw_mean'] = float(raw_a_sparse_mean)
        stats['A_sparse_raw_std'] = float(raw_a_sparse_std)
        if reward_hat_arr.size > 0:
            stats['reward_hat_mean'] = float(reward_hat_arr.mean())
            stats['reward_hat_std'] = float(reward_hat_arr.std())
        if reward_center_arr.size > 0:
            stats['reward_center_mean'] = float(reward_center_arr.mean())
            stats['reward_center_std'] = float(reward_center_arr.std())
        if a_learned_arr.size > 0:
            stats['A_learned_mean'] = float(a_learned_arr.mean())
            stats['A_learned_std'] = float(a_learned_arr.std())
        if a_sparse_arr.size > 0:
            stats['A_sparse_mean'] = float(a_sparse_arr.mean())
            stats['A_sparse_std'] = float(a_sparse_arr.std())
        if a_sparse_arr.size > 0 and a_learned_arr.size > 0:
            stats['sign_match'] = float((np.sign(a_sparse_arr) == np.sign(a_learned_arr)).mean())

        episode_ids = sorted(set(episode_sparse.keys()) & set(episode_learned.keys()))
        if len(episode_ids) >= 2:
            sparse_episode_mean = [
                float(np.mean(episode_sparse[ep_id])) for ep_id in episode_ids
            ]
            learned_episode_mean = [
                float(np.mean(episode_learned[ep_id])) for ep_id in episode_ids
            ]

            # Use a scalar Pearson implementation to avoid numpy.corrcoef/cov path,
            # which can trigger OpenMP runtime conflicts on some Windows env mixes.
            n = len(sparse_episode_mean)
            x_mean = sum(sparse_episode_mean) / n
            y_mean = sum(learned_episode_mean) / n
            num = 0.0
            den_x = 0.0
            den_y = 0.0
            for x_val, y_val in zip(sparse_episode_mean, learned_episode_mean):
                dx = x_val - x_mean
                dy = y_val - y_mean
                num += dx * dy
                den_x += dx * dx
                den_y += dy * dy
            if den_x > 1e-16 and den_y > 1e-16:
                corr = num / ((den_x ** 0.5) * (den_y ** 0.5))
                if np.isfinite(corr):
                    stats['episode_corr'] = float(max(-1.0, min(1.0, corr)))

        return stats
