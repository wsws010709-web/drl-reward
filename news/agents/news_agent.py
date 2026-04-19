import math
import pickle, json
import time
import os
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from news.models.reward_machine import RewardMachine, RewardStep

from khrylib.utils import *
from khrylib.utils.torch import *
from khrylib.rl.agents import AgentPPO
from khrylib.rl.core import estimate_advantages
from torch.utils.tensorboard import SummaryWriter
from news.envs import NewsEnv
from news.models.model import create_RL_model, create_GNN1_model, create_GNN2_model, create_GNN3_model, ActorCritic
from news.models.baseline import RandomPolicy, GreedyPolicy, NullModel
from news.utils.tools import TrajBatchDisc
from news.utils.config import Config
from news.utils.logger_rl import LoggerRL


def tensorfy(np_list, device=torch.device('cpu')):
    if isinstance(np_list[0], list):
        return [[torch.tensor(x).to(device) for x in y] for y in np_list]
    else:
        return [torch.tensor(y).to(device) for y in np_list]


class NewsExpansionAgent(AgentPPO):

    def __init__(self,
                 cfg: Config,
                 dtype: torch.dtype,
                 device: torch.device,
                 num_threads: int,
                 training: bool = True,
                 checkpoint: Union[int, Text] = 0,
                 restore_best_rewards: bool = True):
        self.cfg = cfg
        self.training = training
        self.device = device
        self.loss_iter = 0
        self.use_learned_reward = bool(cfg.use_learned_reward) and cfg.agent in ['rl','rl-gnn1','rl-gnn2','rl-gnn3']
        if self.use_learned_reward and num_threads != 1:
            raise ValueError('learned reward mode currently requires num_threads=1, because the upper-level reward buffer is maintained in-process.')
        self.setup_logger(num_threads)
        self.setup_env()
        self.setup_model()
        self.setup_reward_machine()
        self.setup_optimizer()
        if checkpoint != 0:
            self.start_iteration = self.load_checkpoint(
                checkpoint, restore_best_rewards)
        else:
            self.start_iteration = 0
        super().__init__(env=self.env,
                         dtype=dtype,
                         device=device,
                         logger_cls=LoggerRL,
                         traj_cls=TrajBatchDisc,
                         num_threads=num_threads,
                         policy_net=self.policy_net,
                         value_net=self.value_net,
                         optimizer=self.optimizer,
                         opt_num_epochs=cfg.num_optim_epoch,
                         gamma=cfg.gamma,
                         tau=cfg.tau,
                         clip_epsilon=cfg.clip_epsilon,
                         value_pred_coef=cfg.value_pred_coef,
                         entropy_coef=cfg.entropy_coef,
                         policy_grad_clip=[(self.policy_net.parameters(), 1),
                                           (self.value_net.parameters(), 1)],
                         mini_batch_size=cfg.mini_batch_size)


    def sample_worker(self, pid, queue, num_samples, mean_action):
        self.seed_worker(pid)
        memory = Memory()
        logger = self.logger_cls(**self.logger_kwargs)

        while logger.num_steps < num_samples:
            state = self.env.reset()

            last_info = dict()
            episode_success = False
            logger_messages = []
            memory_messages = []
            reward_traj = []
            for t in range(10000):
                state_var = tensorfy([state])
                use_mean_action = mean_action or torch.bernoulli(
                    torch.tensor([1 - self.noise_rate])).item()
                action = self.policy_net.select_action(
                    state_var, use_mean_action).numpy().squeeze(0)

                learned_reward = None
                if self.use_learned_reward and self.reward_machine is not None:
                    rm_state, rm_action, rm_candidates, rm_candidate_probs = self._extract_reward_features(state, action)
                    learned_reward = self.reward_machine.observe_reward(rm_state, rm_action)
                    reward_traj.append(
                        RewardStep(
                            state=rm_state,
                            action=rm_action,
                            sparse_reward=0.0,
                            candidate_actions=rm_candidates,
                            candidate_probs=rm_candidate_probs,
                        )
                    )

                next_state, reward, done, info = self.env.step(
                    action, self.thread_loggers[pid])
                logger_messages.append([reward, info])

                mask = 0 if done else 1
                exp = 1 - use_mean_action
                rollout_reward = learned_reward if learned_reward is not None else reward
                memory_messages.append(
                    [state, action, mask, next_state, rollout_reward, exp])

                if done:
                    episode_success = (reward != self.env.FAILURE_REWARD) and (
                        reward != self.env.INTERMEDIATE_REWARD)
                    if self.use_learned_reward and self.reward_machine is not None and len(reward_traj) > 0:
                        reward_traj[-1].sparse_reward = float(reward)
                    last_info = info
                    break
                state = next_state

            if episode_success:
                logger.start_episode(self.env)
                for var in range(len(logger_messages)):
                    logger.step(self.env, *logger_messages[var])
                    self.push_memory(memory, *memory_messages[var])
                if self.use_learned_reward and self.reward_machine is not None and len(reward_traj) > 0:
                    self.reward_machine.append_trajectory(reward_traj)
                logger.end_episode(last_info)

        if queue is not None:
            queue.put([pid, memory, logger])
        else:
            return memory, logger
    def setup_env(self):
            self.env = env = NewsEnv(self.cfg)
            self.numerical_feature_size = env.get_numerical_feature_size()
            self.node_dim = env.get_node_dim()
            self.edge_dim = env.get_edge_dim()
            self.max_num_nodes = env.get_max_node_num()
            self.max_num_edges = env.get_max_edge_num()

            # with open(self.cfg.log_dir + '/network.json', 'w') as f:
            #     env_info_dict = self.env.get_env_info_dict()
            #     json.dump(env_info_dict, f, indent=4)
    def setup_logger(self, num_threads):
        cfg = self.cfg
        phase = "train" if self.training else "eval"
        run_id = getattr(cfg, 'run_id', None)
        run_suffix = f'_{run_id}' if run_id else ''
        main_log_path = os.path.join(cfg.log_dir, f'log_{phase}{run_suffix}.txt')
        self.tb_logger = SummaryWriter(cfg.tb_dir) if self.training else None
        self.logger = create_logger(main_log_path, file_handle=True)
        latest_log_path = os.path.join(cfg.log_dir, f'latest_{phase}_log.txt')
        with open(latest_log_path, 'w', encoding='utf-8') as f:
            f.write(main_log_path)
        self.reward_offset = 0.0
        self.best_rewards = -1000.0
        self.best_plans = []
        self.current_rewards = -1000.0
        self.current_plans = []
        self.save_best_flag = False
        cfg.log(self.logger, self.tb_logger)

        self.thread_loggers = []
        for i in range(num_threads):
            thread_log_path = os.path.join(
                cfg.log_dir, f'log_{phase}{run_suffix}_{i}.txt')
            self.thread_loggers.append(
                create_logger(thread_log_path, file_handle=True))

    def setup_model(self):
        cfg = self.cfg
        if cfg.agent == 'rl':
            self.policy_net, self.value_net = create_RL_model(cfg, self)
            self.actor_critic_net = ActorCritic(self.policy_net,
                                                self.value_net)
            to_device(self.device, self.actor_critic_net)

        elif cfg.agent == 'rl-gnn1':
            self.policy_net, self.value_net = create_GNN1_model(cfg, self)
            self.actor_critic_net = ActorCritic(self.policy_net,
                                                self.value_net)
            to_device(self.device, self.actor_critic_net)
        elif cfg.agent == 'rl-gnn2':
            self.policy_net, self.value_net = create_GNN2_model(cfg, self)
            self.actor_critic_net = ActorCritic(self.policy_net,
                                                self.value_net)
            to_device(self.device, self.actor_critic_net)
        elif cfg.agent == 'rl-gnn3':
            self.policy_net, self.value_net = create_GNN3_model(cfg, self)
            self.actor_critic_net = ActorCritic(self.policy_net,
                                                self.value_net)
            to_device(self.device, self.actor_critic_net)
        elif cfg.agent == 'greedy':
            self.policy_net = GreedyPolicy()
            self.value_net = NullModel()

        elif cfg.agent == 'random':
            self.policy_net = RandomPolicy()
            self.value_net = NullModel()
        else:
            raise NotImplementedError()


    def setup_reward_machine(self):
        self.reward_machine = None
        if not self.use_learned_reward:
            return
        state_dim = int(self.policy_net.shared_net.output_value_size)
        action_dim = int(self.policy_net.shared_net.output_policy_road_size)
        self.reward_machine = RewardMachine(
            state_dim=state_dim,
            action_dim=action_dim,
            device=self.device,
            hidden_dim=int(self.cfg.reward_hidden_dim),
            encode_dim=int(self.cfg.reward_encode_dim),
            reward_lr=float(self.cfg.reward_lr),
            value_lr=float(self.cfg.reward_value_lr),
            gamma=self.cfg.gamma,
            reward_buffer_size=int(self.cfg.reward_buffer_size),
            batch_size=int(self.cfg.reward_batch_size),
            l2_coef=float(self.cfg.reward_l2_coef),
            stratified_sampling=bool(self.cfg.reward_stratified_sampling),
        )

    def _extract_reward_features(self, state, action):
        state_var = tensorfy([state])
        with torch.no_grad():
            policy_latent, value_latent, mask, stage = self.policy_net.shared_net(state_var)
            logits = self.policy_net.policy_road_head(policy_latent)
            masked_logits = logits.masked_fill(~mask.bool(), -1e9)
            probs = torch.softmax(masked_logits, dim=1)
        action_idx = int(action)
        action_idx = max(0, min(action_idx, policy_latent.shape[1] - 1))
        action_vec = policy_latent[0, action_idx].detach().cpu().numpy().astype(np.float32)
        feasible = mask[0].bool().detach().cpu().numpy()
        candidate_actions = policy_latent[0, feasible].detach().cpu().numpy().astype(np.float32)
        candidate_probs = probs[0, feasible].detach().cpu().numpy().astype(np.float32)
        probs_sum = float(candidate_probs.sum())
        if probs_sum > 1e-8:
            candidate_probs = candidate_probs / probs_sum
        elif len(candidate_probs) > 0:
            candidate_probs = np.ones_like(candidate_probs, dtype=np.float32) / float(len(candidate_probs))
        state_vec = value_latent[0].detach().cpu().numpy().astype(np.float32)
        return state_vec, action_vec, candidate_actions, candidate_probs

    def setup_optimizer(self):
            cfg = self.cfg
            if cfg.agent in ['rl','rl-gnn1','rl-gnn2','rl-gnn3']:
                self.optimizer = torch.optim.Adam(
                    self.actor_critic_net.parameters(),
                    lr=cfg.lr,
                    eps=cfg.eps,
                    weight_decay=cfg.weightdecay)
            else:
                self.optimizer = None

    def _to_serializable_hparam(self, value):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, dict):
            return {
                str(k): self._to_serializable_hparam(v)
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [self._to_serializable_hparam(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return str(value)

    def _build_hyper_params_snapshot(self):
        snapshot = {}
        for key, value in vars(self.cfg).items():
            snapshot[key] = self._to_serializable_hparam(value)
        return snapshot

    def load_checkpoint(self, checkpoint, restore_best_rewards):
        cfg = self.cfg
        if isinstance(checkpoint, int):
            cp_path = '%s/iteration_%04d.p' % (cfg.model_dir, checkpoint)
        else:
            assert isinstance(checkpoint, str)
            cp_path = '%s/%s.p' % (cfg.model_dir, checkpoint)
        self.logger.info('loading model from checkpoint: %s' % cp_path)
        model_cp = pickle.load(open(cp_path, "rb"))
        self.actor_critic_net.load_state_dict(model_cp['actor_critic_dict'])
        if self.optimizer is not None and 'ppo_optimizer_state' in model_cp and model_cp['ppo_optimizer_state'] is not None:
            self.optimizer.load_state_dict(model_cp['ppo_optimizer_state'])
        if self.use_learned_reward and self.reward_machine is not None and 'reward_machine' in model_cp:
            self.reward_machine.load_state_dict(model_cp['reward_machine'])
        if self.use_learned_reward and self.reward_machine is not None:
            if 'reward_optimizer_state' in model_cp and model_cp['reward_optimizer_state'] is not None:
                self.reward_machine.reward_function_optimizer.load_state_dict(model_cp['reward_optimizer_state'])
            if 'reward_value_optimizer_state' in model_cp and model_cp['reward_value_optimizer_state'] is not None:
                self.reward_machine.value_function_optimizer.load_state_dict(model_cp['reward_value_optimizer_state'])
        if 'hyper_params_snapshot' in model_cp:
            cp_hparams = model_cp['hyper_params_snapshot']
            cur_hparams = self._build_hyper_params_snapshot()
            diff_keys = sorted([
                k for k in cp_hparams.keys()
                if k in cur_hparams and cp_hparams[k] != cur_hparams[k]
            ])
            if len(diff_keys) == 0:
                self.logger.info('checkpoint contains hyper_params_snapshot (matched current config).')
            else:
                preview = ', '.join(diff_keys[:10])
                self.logger.info(
                    f'checkpoint contains hyper_params_snapshot (config mismatch keys={len(diff_keys)}; '
                    f'first: {preview})'
                )
        self.loss_iter = model_cp['loss_iter']
        if restore_best_rewards:
            self.best_rewards = model_cp.get('best_rewards', self.best_rewards)
            self.best_plans = model_cp.get('best_plans', self.best_plans)
        self.current_rewards = model_cp.get('current_rewards',
                                            self.current_rewards)
        self.current_plans = model_cp.get('current_plans', self.current_plans)
        start_iteration = model_cp['iteration'] + 1
        return start_iteration

    def save_checkpoint(self, iteration):

        def save(cp_path):
            with to_cpu(self.policy_net, self.value_net):
                model_cp = {
                    'checkpoint_version': 2,
                    'actor_critic_dict': self.actor_critic_net.state_dict(),
                    'ppo_optimizer_state': self.optimizer.state_dict() if self.optimizer is not None else None,
                    'loss_iter': self.loss_iter,
                    'best_rewards': self.best_rewards,
                    'best_plans': self.best_plans,
                    'current_rewards': self.current_rewards,
                    'current_plans': self.current_plans,
                    'iteration': iteration,
                    'hyper_params_snapshot': self._build_hyper_params_snapshot(),
                }
                if self.use_learned_reward and self.reward_machine is not None:
                    model_cp['reward_machine'] = self.reward_machine.state_dict()
                    model_cp['reward_optimizer_state'] = self.reward_machine.reward_function_optimizer.state_dict()
                    model_cp['reward_value_optimizer_state'] = self.reward_machine.value_function_optimizer.state_dict()
                pickle.dump(model_cp, open(cp_path, 'wb'))

        cfg = self.cfg

        if cfg.save_model_interval > 0 and (iteration +
                                            1) % cfg.save_model_interval == 0:
            self.tb_logger.flush()
            save('{}/iteration_{:04d}.p'.format(cfg.model_dir, iteration + 1))
        if self.save_best_flag:
            self.tb_logger.add_scalar('best_reward/best_reward',
                                      self.best_rewards, iteration)
            self.tb_logger.flush()
            self.logger.info(
                f'save best checkpoint with rewards {self.best_rewards:.2f}!')
            save('{}/best.p'.format(cfg.model_dir))
            save('{}/best_reward{:.2f}_iteration_{:04d}.p'.format(
                cfg.model_dir, self.best_rewards, iteration + 1))

    def save_plan(self, log_eval: LoggerRL) -> None:
        """
        Save the current plan to file.

        Args:
            log_eval: LoggerRL object.
        """
        cfg = self.cfg
        self.logger.info(f'save plan to file: {cfg.plan_dir}/plan.p')
        with open(f'{cfg.plan_dir}/plan.p', 'wb') as f:
            pickle.dump(log_eval.plans, f)

    def optimize(self, iteration):
        info = self.optimize_policy(iteration)
        self.log_optimize_policy(iteration, info)

    def optimize_policy(self, iteration):
        """generate multiple trajectories that reach the minimum batch_size"""
        t0 = time.time()
        if self.use_learned_reward and self.reward_machine is not None and self.cfg.reward_clear_buffer_each_iteration:
            self.reward_machine.clear_buffer()

        num_samples = self.cfg.num_episodes_per_iteration * self.cfg.max_sequence_length
        batch, log = self.sample(num_samples)
        """update networks"""
        t1 = time.time()
        if num_samples > 1:
            self.update_params(batch, iteration)
        t2 = time.time()
        """evaluate policy"""
        log_eval, eval_result = self.eval_agent(
            num_samples=self.cfg.train_eval_num_samples,
            mean_action=True,
            return_eval_result=True,
        )
        eval_three_layer = self._summarize_three_layer_metrics(eval_result)
        t3 = time.time()

        info = {
            'log': log,
            'log_eval': log_eval,
            'eval_three_layer': eval_three_layer,
            'T_sample': t1 - t0,
            'T_update': t2 - t1,
            'T_eval': t3 - t2,
            'T_total': t3 - t0
        }
        return info

    def update_params(self, batch, iteration):
        t0 = time.time()
        to_train(*self.update_modules)
        states = batch.states
        actions = torch.from_numpy(batch.actions).to(self.dtype)
        rewards = torch.from_numpy(batch.rewards).to(self.dtype)
        masks = torch.from_numpy(batch.masks).to(self.dtype)
        exps = torch.from_numpy(batch.exps).to(self.dtype)
        with to_test(*self.update_modules):
            with torch.no_grad():
                values = []
                chunk = self.cfg.mini_batch_size
                for i in range(0, len(states), chunk):
                    states_i = tensorfy(states[i:min(i + chunk, len(states))],
                                        self.device)
                    values_i = self.value_net(self.trans_value(states_i))
                    values.append(values_i.cpu())
                values = torch.cat(values)
        """get advantage estimation from the trajectories"""
        advantages, returns = estimate_advantages(rewards, masks, values,
                                                  self.gamma, self.tau)
        self.update_policy(states, actions, returns, advantages, exps,
                           iteration)

        if self.use_learned_reward and self.reward_machine is not None:
            reward_updates = max(1, int(self.cfg.reward_updates_per_iteration))
            reward_stats = {}
            for update_idx in range(reward_updates):
                reward_stats = self.reward_machine.optimize_reward()
                self.logger.info(
                    f'upper_reward_update[{iteration}:{update_idx + 1}/{reward_updates}] '
                    f"reward_loss={reward_stats.get('reward_loss', 0.0):.6f}, "
                    f"value_loss={reward_stats.get('value_loss', 0.0):.6f}, "
                    f"align_mean={reward_stats.get('align_mean', 0.0):.6f}, "
                    f"sign_match={reward_stats.get('sign_match', 0.0):.6f}, "
                    f"reward_hat_mean={reward_stats.get('reward_hat_mean', 0.0):.6f}, "
                    f"reward_hat_std={reward_stats.get('reward_hat_std', 0.0):.6f}, "
                    f"reward_center_mean={reward_stats.get('reward_center_mean', 0.0):.6f}, "
                    f"reward_center_std={reward_stats.get('reward_center_std', 0.0):.6f}, "
                    f"A_learned_mean={reward_stats.get('A_learned_mean', 0.0):.6f}, "
                    f"A_learned_std={reward_stats.get('A_learned_std', 0.0):.6f}, "
                    f"episode_corr={reward_stats.get('episode_corr', 0.0):.6f}"
                )
                if self.tb_logger is not None:
                    for metric_name, metric_value in reward_stats.items():
                        self.tb_logger.add_scalar(
                            f'reward_machine/{metric_name}',
                            float(metric_value),
                            iteration
                        )
            if self.tb_logger is not None:
                self.tb_logger.add_scalar('reward_machine/buffer_trajectories', self.reward_machine.num_trajectories(), iteration)
                self.tb_logger.add_scalar('reward_machine/buffer_steps', len(self.reward_machine), iteration)
            if self.cfg.reward_clear_buffer_after_update:
                self.reward_machine.clear_buffer()

        return time.time() - t0

    def get_perm_batch_stage(self, states):
        inds = [[], []]
        for i, x in enumerate(states):
            stage = x[-1]
            inds[stage.argmax()].append(i)
        perm = np.array(inds[0] + inds[1])
        return perm, LongTensor(perm)

    def update_policy(self, states, actions, returns, advantages, exps,
                      iteration):
        """update policy"""
        with to_test(*self.update_modules):
            with torch.no_grad():
                fixed_log_probs = []
                chunk = self.cfg.mini_batch_size
                for i in range(0, len(states), chunk):
                    states_i = tensorfy(states[i:min(i + chunk, len(states))],
                                        self.device)
                    actions_i = actions[i:min(i + chunk, len(states))].to(
                        self.device)
                    fixed_log_probs_i, _ = self.policy_net.get_log_prob_entropy(
                        self.trans_policy(states_i), actions_i)
                    fixed_log_probs.append(fixed_log_probs_i.cpu())
                fixed_log_probs = torch.cat(fixed_log_probs)
        num_state = len(states)

        tb_logger = self.tb_logger
        total_loss = 0.0
        total_value_loss = 0.0
        total_surr_loss = 0.0
        total_entropy_loss = 0.0
        for epoch in range(self.opt_num_epochs):
            epoch_loss = 0.0
            epoch_value_loss = 0.0
            epoch_surr_loss = 0.0
            epoch_entropy_loss = 0.0

            perm_np = np.arange(num_state)
            np.random.shuffle(perm_np)
            perm = LongTensor(perm_np)

            states, actions, returns, advantages, fixed_log_probs, exps = \
                index_select_list(states, perm_np), actions[perm].clone(), returns[perm].clone(), \
                advantages[perm].clone(), fixed_log_probs[perm].clone(), exps[perm].clone()

            if self.cfg.agent_specs.get('batch_stage', False):
                perm_stage_np, perm_stage = self.get_perm_batch_stage(states)
                states, actions, returns, advantages, fixed_log_probs, exps = \
                    index_select_list(states, perm_stage_np), actions[perm_stage].clone(), \
                    returns[perm_stage].clone(), advantages[perm_stage].clone(), \
                    fixed_log_probs[perm_stage].clone(), exps[perm_stage].clone()

            optim_batch_num = int(math.floor(num_state / self.mini_batch_size))
            for i in range(optim_batch_num):
                ind = slice(i * self.mini_batch_size,
                            min((i + 1) * self.mini_batch_size, num_state))
                states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b, exps_b = \
                    states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind], exps[ind]
                ind = exps_b.nonzero(as_tuple=False).squeeze(1)
                states_b = tensorfy(states_b, self.device)
                actions_b, advantages_b, returns_b, fixed_log_probs_b, ind = batch_to(
                    self.device, actions_b, advantages_b, returns_b,
                    fixed_log_probs_b, ind)
                value_loss = self.value_loss(states_b, returns_b)
                surr_loss, entropy_loss = self.ppo_entropy_loss(
                    states_b, actions_b, advantages_b, fixed_log_probs_b, ind)
                loss = surr_loss + self.value_pred_coef * value_loss + self.entropy_coef * entropy_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.clip_policy_grad()
                self.optimizer.step()
                epoch_loss += loss.item()
                epoch_value_loss += value_loss.item()
                epoch_surr_loss += surr_loss.item()
                epoch_entropy_loss += entropy_loss.item()
                tb_logger.add_scalar('loss/loss', loss.item(), self.loss_iter)
                tb_logger.add_scalar('loss/value_loss', value_loss.item(),
                                     self.loss_iter)
                tb_logger.add_scalar('loss/surr_loss', surr_loss.item(),
                                     self.loss_iter)
                tb_logger.add_scalar('loss/entropy_loss', entropy_loss.item(),
                                     self.loss_iter)
                self.loss_iter += 1

            total_loss += epoch_loss
            total_value_loss += epoch_value_loss
            total_surr_loss += epoch_surr_loss
            total_entropy_loss += epoch_entropy_loss
            global_epoch = iteration * self.opt_num_epochs + epoch
            tb_logger.add_scalar('loss/epoch_loss', epoch_loss, global_epoch)
            tb_logger.add_scalar('loss/epoch_value_loss', epoch_value_loss,
                                 global_epoch)
            tb_logger.add_scalar('loss/epoch_surr_loss', epoch_surr_loss,
                                 global_epoch)
            tb_logger.add_scalar('loss/epoch_entropy_loss', epoch_entropy_loss,
                                 global_epoch)

        tb_logger.add_scalar('loss/total_loss',
                             total_loss / self.opt_num_epochs, iteration)
        tb_logger.add_scalar('loss/total_value_loss',
                             total_value_loss / self.opt_num_epochs, iteration)
        tb_logger.add_scalar('loss/total_surr_loss',
                             total_surr_loss / self.opt_num_epochs, iteration)
        tb_logger.add_scalar('loss/total_entropy_loss',
                             total_entropy_loss / self.opt_num_epochs,
                             iteration)

    def ppo_entropy_loss(self, states, actions, advantages, fixed_log_probs,
                         ind):
        log_probs, entropy = self.policy_net.get_log_prob_entropy(
            self.trans_policy(states), actions)
        ratio = torch.exp(log_probs[ind] - fixed_log_probs[ind])
        advantages = advantages[ind]
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon,
                            1.0 + self.clip_epsilon) * advantages
        surr_loss = -torch.min(surr1, surr2).mean()
        entropy_loss = -entropy[ind].mean()
        return surr_loss, entropy_loss

    def log_optimize_policy(self, iteration, info):
        cfg = self.cfg
        log, log_eval = info['log'], info['log_eval']
        eval_three_layer = info.get('eval_three_layer', None)
        logger, tb_logger = self.logger, self.tb_logger
        log_str = f'{iteration}\tT_sample {info["T_sample"]:.2f}\tT_update {info["T_update"]:.2f}\t' \
                  f'T_eval {info["T_eval"]:.2f}\t' \
                  f'ETA {get_eta_str(iteration, cfg.max_num_iterations, info["T_total"])}\t' \
                  f'train_R_eps {log.avg_episode_reward + self.reward_offset:.2f}\t'\
                  f'eval_R_eps {log_eval.avg_episode_reward + self.reward_offset:.2f}\t{cfg.id}'
        logger.info(log_str)

        self.current_rewards = log_eval.avg_episode_reward + self.reward_offset
        self.current_plans = log_eval.plans
        if log_eval.avg_episode_reward + self.reward_offset > self.best_rewards:
            self.best_rewards = log_eval.avg_episode_reward + self.reward_offset
            self.best_plans = log_eval.plans
            self.save_best_flag = True
        else:
            self.save_best_flag = False

        tb_logger.add_scalar('train/train_R_eps_avg',
                             log.avg_episode_reward + self.reward_offset,
                             iteration)
        tb_logger.add_scalar('train/full_total_i_rate', log.avg_episode_full_total_i_rate, iteration)
        tb_logger.add_scalar('train/total_i_rate', log.avg_episode_total_i_rate, iteration)
        tb_logger.add_scalar('train/reduction', log.avg_episode_reduction, iteration)

        tb_logger.add_scalar('eval/eval_R_eps_avg',
                             log_eval.avg_episode_reward + self.reward_offset,
                             iteration)
        tb_logger.add_scalar('eval/full_total_i_rate', log_eval.avg_episode_full_total_i_rate, iteration)
        tb_logger.add_scalar('eval/total_i_rate', log_eval.avg_episode_total_i_rate, iteration)
        tb_logger.add_scalar('eval/reduction', log_eval.avg_episode_reduction, iteration)

        if eval_three_layer is not None:
            logger.info(
                f"train_eval_three_layer[{iteration}] "
                f"effect={eval_three_layer['effect']}, "
                f"cost={eval_three_layer['cost']}, "
                f"explainability={eval_three_layer['explainability']}"
            )
            for group_name in ['effect', 'cost', 'explainability']:
                group_values = eval_three_layer.get(group_name, {})
                for metric_name, value in group_values.items():
                    tb_logger.add_scalar(
                        f'eval/{group_name}/{metric_name}',
                        float(value),
                        iteration
                    )


    def eval_agent(self, num_samples=1, mean_action=True, agent_dict=None, return_eval_result=False):
        t_start = time.time()
        to_test(*self.sample_modules)
        self.env.eval()
        eval_results = []
        with to_cpu(*self.sample_modules):
            with torch.no_grad():
                logger = self.logger_cls(**self.logger_kwargs)
                while logger.num_steps < num_samples:
                    state = self.env.reset(eval=True, agent_dict=agent_dict)
                    info_dict = self.env.get_info()
                    network_id = info_dict['network_id']
                    source = info_dict['source']
                    logger.start_episode(self.env)

                    info_plan = dict()
                    episode_success = False
                    for t in tqdm(range(1, 10000), position=0):
                        state_var = tensorfy([state])
                        action = self.policy_net.select_action(
                            state_var, mean_action).numpy()
                        next_state, reward, done, info = self.env.step(
                            action, self.logger)
                        logger.step(self.env, reward, info)
                        if done:
                            episode_success = (reward != self.env.FAILURE_REWARD) and \
                                              (reward != self.env.INTERMEDIATE_REWARD)
                            info_plan = info
                            break
                        state = next_state

                        # self.logger.info(f'reward:{reward}  step:{t:02d}')
                    self.logger.info(
                        f"full_total_i_rate:{info_plan['full_total_i_rate']}, "
                        f"total_i_rate:{info_plan['total_i_rate']}, "
                        f"reduction:{info_plan['reduction']}"
                    )

                    if return_eval_result:
                        episode_eval_result = self.env.get_eval_result()
                        episode_eval_result['episode_reward'] = float(info_plan.get('reward', reward))
                        episode_eval_result['episode_raw_reward'] = float(
                            info_plan.get('raw_reward', info_plan.get('reward', reward))
                        )
                        eval_results.append(episode_eval_result)

                    logger.add_plan(info_plan)
                    logger.end_episode(info_plan)
                    if not episode_success:
                        self.logger.info('Plan fails during eval.')
                logger = self.logger_cls.merge([logger], **self.logger_kwargs)

        self.env.train()
        logger.sample_time = time.time() - t_start
        if return_eval_result:
            if len(eval_results) == 1:
                return logger, eval_results[0]
            return logger, eval_results
        return logger

    @staticmethod
    def _mean_std(values):
        arr = np.asarray(values, dtype=np.float64)
        return float(arr.mean()), float(arr.std())

    def _build_infer_metrics(self, eval_result):
        full_total_i_rate = float(eval_result.get('full_total_i_rate', 0.0))
        total_i_rate = float(eval_result.get('total_i_rate', 0.0))

        origin_cir_curve = np.asarray(eval_result.get('origin_cir_curve', []), dtype=np.float64)
        cut_cir_curve = np.asarray(eval_result.get('cut_cir_curve', []), dtype=np.float64)

        origin_peak = float(origin_cir_curve.max()) if origin_cir_curve.size > 0 else 0.0
        cut_peak = float(cut_cir_curve.max()) if cut_cir_curve.size > 0 else 0.0
        origin_peak_step = int(origin_cir_curve.argmax()) if origin_cir_curve.size > 0 else 0
        cut_peak_step = int(cut_cir_curve.argmax()) if cut_cir_curve.size > 0 else 0

        auc_cir_origin = float(origin_cir_curve.sum()) if origin_cir_curve.size > 0 else 0.0
        auc_cir_cut = float(cut_cir_curve.sum()) if cut_cir_curve.size > 0 else 0.0

        deleted_edges = eval_result.get('deleted_edges', [])
        affected_nodes = set()
        for edge in deleted_edges:
            if len(edge) == 2:
                affected_nodes.add(edge[0])
                affected_nodes.add(edge[1])
        node_num = max(1, int(eval_result.get('node_num', 1)))
        affected_node_ratio = len(affected_nodes) / node_num

        source_distances = np.asarray(eval_result.get('deleted_edge_source_distances', []), dtype=np.float64)
        source_betweenness = np.asarray(eval_result.get('deleted_edge_source_betweenness', []), dtype=np.float64)

        final_total_reduction = (full_total_i_rate - total_i_rate) / (full_total_i_rate + 1e-8) \
            if full_total_i_rate > 1e-8 else 0.0
        peak_cir_reduction = (origin_peak - cut_peak) / (origin_peak + 1e-8) \
            if origin_peak > 1e-8 else 0.0
        peak_delay = cut_peak_step - origin_peak_step
        auc_cir_reduction = (auc_cir_origin - auc_cir_cut) / (auc_cir_origin + 1e-8) \
            if auc_cir_origin > 1e-8 else 0.0
        curve_gap_cir = float((origin_cir_curve - cut_cir_curve).sum()) \
            if origin_cir_curve.size == cut_cir_curve.size else 0.0

        return {
            'effect': {
                'final_total_reduction': float(final_total_reduction),
                'peak_cir_origin': float(origin_peak),
                'peak_cir_cut': float(cut_peak),
                'peak_cir_reduction': float(peak_cir_reduction),
                'peak_delay': float(peak_delay),
                'auc_cir_origin': float(auc_cir_origin),
                'auc_cir_cut': float(auc_cir_cut),
                'auc_cir_reduction': float(auc_cir_reduction),
                'curve_gap_cir': float(curve_gap_cir),
            },
            'cost': {
                'affected_node_ratio': float(affected_node_ratio),
            },
            'explainability': {
                'avg_deleted_edge_source_distance': float(source_distances.mean()) if source_distances.size > 0 else 0.0,
                'avg_deleted_edge_source_betweenness': float(source_betweenness.mean()) if source_betweenness.size > 0 else 0.0,
            }
        }

    def _aggregate_metric_group(self, metrics_list, key):
        group = {}
        if len(metrics_list) == 0:
            return group
        for metric_name in metrics_list[0][key].keys():
            values = [m[key][metric_name] for m in metrics_list]
            mean, std = self._mean_std(values)
            group[metric_name] = {'mean': mean, 'std': std}
        return group

    def _select_representative_run(self, metrics_list):
        if len(metrics_list) == 0:
            return -1, None, 'final_total_reduction', None

        key = 'final_total_reduction'
        values = np.asarray(
            [float(m['effect'].get(key, 0.0)) for m in metrics_list],
            dtype=np.float64,
        )
        method = str(self.cfg.infer_representative_selection).strip().lower()
        target = None

        if method == 'max':
            rep_idx = int(values.argmax())
            target = float(values[rep_idx])
        elif method == 'min':
            rep_idx = int(values.argmin())
            target = float(values[rep_idx])
        else:
            if method != 'closest_to_mean':
                self.logger.info(
                    f'Unknown infer representative_selection={method}, fallback to closest_to_mean.'
                )
            method = 'closest_to_mean'
            target = float(values.mean())
            rep_idx = int(np.argmin(np.abs(values - target)))

        return rep_idx, method, key, target

    def _summarize_three_layer_metrics(self, eval_result):
        run_eval_results = eval_result if isinstance(eval_result, list) else [eval_result]
        metrics_list = [self._build_infer_metrics(run_result) for run_result in run_eval_results]
        def _single_value_group(group_name):
            group = {}
            if len(metrics_list) == 0:
                return group
            for metric_name in metrics_list[0][group_name].keys():
                values = [m[group_name][metric_name] for m in metrics_list]
                group[metric_name] = float(np.mean(values))
            return group

        return {
            'num_runs': len(metrics_list),
            'effect': _single_value_group('effect'),
            'cost': _single_value_group('cost'),
            'explainability': _single_value_group('explainability'),
        }

    def _log_infer_run_metrics(self, eval_tag, eval_reward, eval_result, run_metrics):
        effect = run_metrics['effect']
        cost = run_metrics['cost']
        explainability = run_metrics['explainability']
        self.logger.info(f"{eval_tag}: {eval_reward}")
        self.logger.info(
            f"network_id={eval_result.get('network_id')}, source={eval_result.get('source')}\n"
            f"  effect: "
            f"final_total_reduction={effect['final_total_reduction']:.6f}, "
            f"peak_cir_origin={effect['peak_cir_origin']:.6f}, "
            f"peak_cir_cut={effect['peak_cir_cut']:.6f}, "
            f"peak_cir_reduction={effect['peak_cir_reduction']:.6f}, "
            f"peak_delay={effect['peak_delay']:.6f}, "
            f"auc_cir_origin={effect['auc_cir_origin']:.6f}, "
            f"auc_cir_cut={effect['auc_cir_cut']:.6f}, "
            f"auc_cir_reduction={effect['auc_cir_reduction']:.6f}, "
            f"curve_gap_cir={effect['curve_gap_cir']:.6f}\n"
            f"  cost: "
            f"affected_node_ratio={cost['affected_node_ratio']:.6f}\n"
            f"  explainability: "
            f"avg_deleted_edge_source_distance={explainability['avg_deleted_edge_source_distance']:.6f}, "
            f"avg_deleted_edge_source_betweenness={explainability['avg_deleted_edge_source_betweenness']:.6f}"
        )

    def _save_representative_curves(self, eval_result, cut_step, timestamp):
        origin_tir_curve = np.asarray(eval_result.get('origin_tir_curve', []), dtype=np.float64)
        cut_tir_curve = np.asarray(eval_result.get('cut_tir_curve', []), dtype=np.float64)
        origin_cir_curve = np.asarray(eval_result.get('origin_cir_curve', []), dtype=np.float64)
        cut_cir_curve = np.asarray(eval_result.get('cut_cir_curve', []), dtype=np.float64)

        network_id = eval_result.get('network_id')
        source = eval_result.get('source')
        title_suffix = f'network {network_id}, source {source}'

        tir_path = os.path.join(self.cfg.plan_dir, f'infer_cut{cut_step}_rep_{int(timestamp)}_tir.png')
        cir_path = os.path.join(self.cfg.plan_dir, f'infer_cut{cut_step}_rep_{int(timestamp)}_cir.png')

        plt.figure(figsize=(8, 4))
        plt.plot(origin_tir_curve, label='Origin TIR', color='red')
        plt.plot(cut_tir_curve, label='Your method TIR', color='blue')
        plt.title(f'TIR Comparison ({title_suffix})')
        plt.xlabel('Step')
        plt.ylabel('TIR')
        plt.legend()
        plt.tight_layout()
        plt.savefig(tir_path, dpi=150)
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.plot(origin_cir_curve, label='Origin CIR', color='red')
        plt.plot(cut_cir_curve, label='Your method CIR', color='blue')
        plt.title(f'CIR Comparison ({title_suffix})')
        plt.xlabel('Step')
        plt.ylabel('CIR')
        plt.legend()
        plt.tight_layout()
        plt.savefig(cir_path, dpi=150)
        plt.close()

        return {'tir_plot_path': tir_path, 'cir_plot_path': cir_path}

    def infer(self,
              num_samples=None,
              mean_action=None,
              save_video=None,
              only_road=None):
        t_start = time.time()

        num_samples = self.cfg.infer_num_samples if num_samples is None else num_samples
        mean_action = self.cfg.infer_mean_action if mean_action is None else mean_action
        save_video = self.cfg.infer_save_video if save_video is None else save_video
        only_road = self.cfg.infer_only_road if only_road is None else only_road
        _ = (save_video, only_road)

        summary_doc = {
            'schema_version': 'infer_summary_v1',
            'meta': {
                'timestamp': int(t_start),
                'cfg_id': self.cfg.id,
                'data_source': self.cfg.data_source,
                'agent': self.cfg.agent,
                'seed': int(self.cfg.seed),
                'num_cut_steps': int(self.cfg.infer_num_cut_steps),
                'num_trials': int(self.cfg.infer_num_trials),
                'num_samples_per_eval': int(num_samples),
                'mean_action': bool(mean_action),
                'curve_eval_mc': int(self.cfg.infer_curve_eval_mc),
                'save_representative_plots': bool(self.cfg.infer_save_representative_plots),
                'representative_selection': str(self.cfg.infer_representative_selection),
                'use_cut_ration_schedule': bool(self.cfg.infer_use_cut_ration_schedule),
                'cut_ration_start': float(self.cfg.infer_cut_ration_start),
                'cut_ration_step': float(self.cfg.infer_cut_ration_step),
                'default_total_cut_ration': float(self.cfg.env_param.get('total_cut_ration', 0.0)),
            },
            'runs_by_cut_step': [],
        }
        for cut_step in range(self.cfg.infer_num_cut_steps):
            eval_results = []
            metrics_list = []
            agent_dict = {}
            cut_ration = float(self.cfg.env_param.get('total_cut_ration', 0.0))
            if self.cfg.infer_use_cut_ration_schedule:
                cut_ration = float(self.cfg.infer_cut_ration_start + self.cfg.infer_cut_ration_step * cut_step)
                agent_dict['cut_ration'] = cut_ration
            if int(self.cfg.infer_curve_eval_mc) > 0:
                agent_dict['eval_simulation_count'] = int(self.cfg.infer_curve_eval_mc)
            if len(agent_dict) == 0:
                agent_dict = None
            for t in range(self.cfg.infer_num_trials):
                _, eval_result = self.eval_agent(
                    num_samples=num_samples,
                    mean_action=mean_action,
                    agent_dict=agent_dict,
                    return_eval_result=True,
                )
                run_eval_results = eval_result if isinstance(eval_result, list) else [eval_result]
                for run_idx, single_eval_result in enumerate(run_eval_results):
                    eval_results.append(single_eval_result)
                    run_metrics = self._build_infer_metrics(single_eval_result)
                    metrics_list.append(run_metrics)
                    eval_reward = float(
                        single_eval_result.get(
                            'episode_raw_reward',
                            single_eval_result.get('episode_reward', 0.0)
                        )
                    )
                    eval_tag = f"eval_{cut_step}_{t}" if len(run_eval_results) == 1 else f"eval_{cut_step}_{t}_{run_idx}"
                    self._log_infer_run_metrics(
                        eval_tag=eval_tag,
                        eval_reward=eval_reward,
                        eval_result=single_eval_result,
                        run_metrics=run_metrics,
                    )

            effect_stats = self._aggregate_metric_group(metrics_list, 'effect')
            cost_stats = self._aggregate_metric_group(metrics_list, 'cost')
            explain_stats = self._aggregate_metric_group(metrics_list, 'explainability')
            rep_idx = -1
            selection_method = None
            selection_key = 'final_total_reduction'
            selection_target = None
            representative_result = None
            representative_metrics = None
            plot_paths = {}
            if len(metrics_list) > 0:
                rep_idx, selection_method, selection_key, selection_target = self._select_representative_run(
                    metrics_list
                )
                representative_result = eval_results[rep_idx]
                representative_metrics = metrics_list[rep_idx]
                if bool(self.cfg.infer_save_representative_plots):
                    plot_paths = self._save_representative_curves(
                        representative_result, cut_step, t_start
                    )

            run_summary = {
                'cut_step': cut_step,
                'cut_ration': cut_ration,
                'num_runs': len(eval_results),
                'effect': effect_stats,
                'cost': cost_stats,
                'explainability': explain_stats,
                'representative_run': {
                    'index': rep_idx,
                    'selection_method': selection_method,
                    'selection_key': selection_key,
                    'selection_target': selection_target,
                    'metrics': representative_metrics,
                    'plots': plot_paths,
                    'eval_result': representative_result,
                },
                'raw_runs': eval_results,
            }
            summary_doc['runs_by_cut_step'].append(run_summary)

            self.logger.info(
                f"infer_cut_step_{cut_step} effect={effect_stats}, cost={cost_stats}, explainability={explain_stats}"
            )
            self.logger.info(
                f"infer_cut_step_{cut_step} representative_run={rep_idx}, "
                f"selection={selection_method}/{selection_key}, plots={plot_paths}"
            )

        summary_path = os.path.join(self.cfg.plan_dir, f'infer_summary_{int(t_start)}.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_doc, f, indent=2, ensure_ascii=False)
        self.logger.info(f'Infer summary saved to {summary_path}')
        return summary_doc
