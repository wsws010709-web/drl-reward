import os
from khrylib.utils import load_yaml
from typing import Text, Dict


class Config:

    def __init__(self, cfg: Text, global_seed: int, tmp: bool, root_dir: Text,
                 agent: Text = 'random', cfg_dict: Dict = None):
        self.id = cfg
        self.seed = global_seed
        if cfg_dict is not None:
            cfg = cfg_dict
        else:
            cwd = os.getcwd()
            file_path = os.path.join(cwd,'news/cfg/{}.yaml'.format(cfg))
            cfg = load_yaml(file_path)
        # create dirs
        self.root_dir = os.path.join(cwd,'tmp') if tmp else root_dir
        self.data_source = cfg.get('data_source')
        self.data_dir = 'data/{}'.format(self.data_source)
        self.cfg_dir = os.path.join(self.root_dir, self.data_source, agent, self.id, str(self.seed))
        self.model_dir = os.path.join(self.cfg_dir, 'models')
        self.log_dir = os.path.join(self.cfg_dir, 'log')
        self.tb_dir = os.path.join(self.cfg_dir, 'tb')
        self.plan_dir = os.path.join(self.cfg_dir, 'plan')
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        os.makedirs(self.plan_dir, exist_ok=True)

        self.agent = agent

        # env
        self.env_param = cfg.get('spread', dict())
        self.terminal_reward_mc = int(
            cfg.get('terminal_reward_mc',
                    self.env_param.get('terminal_reward_mc', 5)))
        self.env_param['terminal_reward_mc'] = self.terminal_reward_mc

        # agent config
        self.agent_specs = cfg.get('agent_specs', dict())

        # training config
        self.gamma = cfg.get('gamma', 0.99)
        self.tau = cfg.get('tau', 0.95)
        self.state_encoder_specs = cfg.get('state_encoder_specs', dict())
        self.policy_specs = cfg.get('policy_specs', dict())
        self.value_specs = cfg.get('value_specs', dict())
        self.lr = cfg.get('lr', 4e-4)
        self.weightdecay = cfg.get('weightdecay', 0.0)
        self.eps = cfg.get('eps', 1e-5)
        self.value_pred_coef = cfg.get('value_pred_coef', 0.5)
        self.entropy_coef = cfg.get('entropy_coef', 0.01)
        self.clip_epsilon = cfg.get('clip_epsilon', 0.2)
        self.max_num_iterations = cfg.get('max_num_iterations', 1000)
        self.num_episodes_per_iteration = cfg.get('num_episodes_per_iteration', 1000)
        self.max_sequence_length = cfg.get('max_sequence_length', 100)
        self.original_max_sequence_length = cfg.get('max_sequence_length', 100)
        self.num_optim_epoch = cfg.get('num_optim_epoch', 4)
        self.mini_batch_size = cfg.get('mini_batch_size', 1024)
        self.save_model_interval = cfg.get('save_model_interval', 10)

        # upper-level reward learning config
        self.use_learned_reward = bool(cfg.get('use_learned_reward', True))
        self.reward_hidden_dim = int(cfg.get('reward_hidden_dim', 128))
        self.reward_encode_dim = int(
            cfg.get('reward_encode_dim', self.reward_hidden_dim))
        self.reward_lr = float(cfg.get('reward_lr', 1e-4))
        self.reward_value_lr = float(cfg.get('reward_value_lr', 1e-4))
        self.reward_buffer_size = int(cfg.get('reward_buffer_size', 16))
        self.reward_batch_size = int(cfg.get('reward_batch_size', 1024))
        self.reward_l2_coef = float(cfg.get('reward_l2_coef', 1e-4))
        self.reward_updates_per_iteration = int(
            cfg.get('reward_updates_per_iteration', 1))
        self.reward_stratified_sampling = bool(
            cfg.get('reward_stratified_sampling', True))
        self.reward_clear_buffer_each_iteration = bool(
            cfg.get('reward_clear_buffer_each_iteration', True))
        self.reward_clear_buffer_after_update = bool(
            cfg.get('reward_clear_buffer_after_update', True))

        # inference config
        infer_cfg = cfg.get('infer', dict())
        self.infer_num_samples = int(infer_cfg.get('num_samples', 1))
        self.infer_mean_action = bool(infer_cfg.get('mean_action', True))
        self.infer_save_video = bool(infer_cfg.get('save_video', False))
        self.infer_only_road = bool(infer_cfg.get('only_road', False))
        self.infer_num_trials = int(infer_cfg.get('num_trials', 20))
        self.infer_curve_eval_mc = int(infer_cfg.get('curve_eval_mc', 0))
        self.infer_num_cut_steps = int(infer_cfg.get('num_cut_steps', 1))
        self.infer_save_representative_plots = bool(
            infer_cfg.get('save_representative_plots', True))
        self.infer_representative_selection = str(
            infer_cfg.get('representative_selection', 'closest_to_mean')).strip().lower()
        self.infer_use_cut_ration_schedule = bool(infer_cfg.get(
            'use_cut_ration_schedule', False)
        )
        self.infer_cut_ration_start = float(
            infer_cfg.get('cut_ration_start', 0.05))
        self.infer_cut_ration_step = float(
            infer_cfg.get('cut_ration_step', 0.01))
        if self.infer_curve_eval_mc > 0:
            self.env_param['curve_eval_simulation_count'] = self.infer_curve_eval_mc

    def train(self) -> None:
        """Train land use only"""
        self.skip_land_use = False
        self.skip_road = True
        self.max_sequence_length = self.original_max_sequence_length // 2

    def finetune(self) -> None:
        """Change to road network only"""
        self.skip_land_use = True
        self.skip_road = False
        self.max_sequence_length = self.original_max_sequence_length // 2

    def log(self, logger, tb_logger):
        """Log cfg to logger and tensorboard."""
        logger.info(f'data_dir:{self.data_dir}')
        logger.info(f'cfg: {self.id}')
        logger.info(f'seed: {self.seed}')
        logger.info(f'agent: {self.agent}')           
        logger.info(f'env_param: {self.env_param}')
        logger.info(f'terminal_reward_mc: {self.terminal_reward_mc}')

        logger.info(f'agent_specs: {self.agent_specs}')
        logger.info(f'gamma: {self.gamma}')
        logger.info(f'tau: {self.tau}')
        logger.info(f'state_encoder_specs: {self.state_encoder_specs}')
        logger.info(f'policy_specs: {self.policy_specs}')
        logger.info(f'value_specs: {self.value_specs}')
        logger.info(f'lr: {self.lr}')
        logger.info(f'weightdecay: {self.weightdecay}')
        logger.info(f'eps: {self.eps}')
        logger.info(f'value_pred_coef: {self.value_pred_coef}')
        logger.info(f'entropy_coef: {self.entropy_coef}')
        logger.info(f'clip_epsilon: {self.clip_epsilon}')
        logger.info(f'max_num_iterations: {self.max_num_iterations}')
        logger.info(f'num_episodes_per_iteration: {self.num_episodes_per_iteration}')
        logger.info(f'max_sequence_length: {self.max_sequence_length}')
        logger.info(f'num_optim_epoch: {self.num_optim_epoch}')
        logger.info(f'mini_batch_size: {self.mini_batch_size}')
        logger.info(f'save_model_interval: {self.save_model_interval}')
        logger.info(f'use_learned_reward: {self.use_learned_reward}')
        logger.info(f'reward_hidden_dim: {self.reward_hidden_dim}')
        logger.info(f'reward_encode_dim: {self.reward_encode_dim}')
        logger.info(f'reward_lr: {self.reward_lr}')
        logger.info(f'reward_value_lr: {self.reward_value_lr}')
        logger.info(f'reward_buffer_size: {self.reward_buffer_size}')
        logger.info(f'reward_batch_size: {self.reward_batch_size}')
        logger.info(f'reward_l2_coef: {self.reward_l2_coef}')
        logger.info(f'reward_updates_per_iteration: {self.reward_updates_per_iteration}')
        logger.info(f'reward_stratified_sampling: {self.reward_stratified_sampling}')
        logger.info(f'reward_clear_buffer_each_iteration: {self.reward_clear_buffer_each_iteration}')
        logger.info(f'reward_clear_buffer_after_update: {self.reward_clear_buffer_after_update}')
        logger.info(
            f'infer_specs: {{'
            f'"num_samples": {self.infer_num_samples}, '
            f'"mean_action": {self.infer_mean_action}, '
            f'"save_video": {self.infer_save_video}, '
            f'"only_road": {self.infer_only_road}, '
            f'"num_trials": {self.infer_num_trials}, '
            f'"curve_eval_mc": {self.infer_curve_eval_mc}, '
            f'"num_cut_steps": {self.infer_num_cut_steps}, '
            f'"save_representative_plots": {self.infer_save_representative_plots}, '
            f'"representative_selection": "{self.infer_representative_selection}", '
            f'"use_cut_ration_schedule": {self.infer_use_cut_ration_schedule}, '
            f'"cut_ration_start": {self.infer_cut_ration_start}, '
            f'"cut_ration_step": {self.infer_cut_ration_step}'
            f'}}')

        if tb_logger is not None:
            tb_logger.add_hparams(
                hparam_dict={
                    'id': self.id,
                    'seed': self.seed,
                    'agent': self.agent,
                    'env_param': str(self.env_param),
                    'terminal_reward_mc': self.terminal_reward_mc,
                    'agent_specs': str(self.agent_specs),
                    'gamma': self.gamma,
                    'tau': self.tau,
                    'state_encoder_specs': str(self.state_encoder_specs),
                    'policy_specs': str(self.policy_specs),
                    'value_specs': str(self.value_specs),
                    'lr': self.lr,
                    'weightdecay': self.weightdecay,
                    'eps': self.eps,
                    'value_pred_coef': self.value_pred_coef,
                    'entropy_coef': self.entropy_coef,
                    'clip_epsilon': self.clip_epsilon,
                    'max_num_iterations': self.max_num_iterations,
                    'num_episodes_per_iteration': self.num_episodes_per_iteration,
                    'max_sequence_length': self.max_sequence_length,
                    'num_optim_epoch': self.num_optim_epoch,
                    'mini_batch_size': self.mini_batch_size,
                    'save_model_interval': self.save_model_interval,
                    'use_learned_reward': str(self.use_learned_reward),
                    'reward_hidden_dim': self.reward_hidden_dim,
                    'reward_encode_dim': self.reward_encode_dim,
                    'reward_lr': self.reward_lr,
                    'reward_value_lr': self.reward_value_lr,
                    'reward_buffer_size': self.reward_buffer_size,
                    'reward_batch_size': self.reward_batch_size,
                    'reward_l2_coef': self.reward_l2_coef,
                    'reward_updates_per_iteration': self.reward_updates_per_iteration,
                    'reward_stratified_sampling': str(self.reward_stratified_sampling),
                    'reward_clear_buffer_each_iteration': str(self.reward_clear_buffer_each_iteration),
                    'reward_clear_buffer_after_update': str(self.reward_clear_buffer_after_update),
                    'infer_num_samples': self.infer_num_samples,
                    'infer_mean_action': str(self.infer_mean_action),
                    'infer_save_video': str(self.infer_save_video),
                    'infer_only_road': str(self.infer_only_road),
                    'infer_num_trials': self.infer_num_trials,
                    'infer_curve_eval_mc': self.infer_curve_eval_mc,
                    'infer_num_cut_steps': self.infer_num_cut_steps,
                    'infer_save_representative_plots': str(self.infer_save_representative_plots),
                    'infer_representative_selection': self.infer_representative_selection,
                    'infer_use_cut_ration_schedule': str(self.infer_use_cut_ration_schedule),
                    'infer_cut_ration_start': self.infer_cut_ration_start,
                    'infer_cut_ration_step': self.infer_cut_ration_step},
                metric_dict={'hparam/placeholder': 0.0})
