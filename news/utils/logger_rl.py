import itertools
from news.utils.stats_logger import StatsLogger


class LoggerRL:
    """Project-owned logger for rumor mitigation RL pipelines."""

    DEFAULT_STATS_NAMES = [
        'episode_len',
        'reward',
        'episode_reward',
        'episode_raw_reward',
        'episode_total_i_rate',
        'episode_full_total_i_rate',
        'episode_reduction',
    ]

    def __init__(self, init_stats_logger=True, stats_names=None):
        self.num_steps = 0
        self.num_episodes = 0
        self.sample_time = 0
        self.stats_names = list(self.DEFAULT_STATS_NAMES if stats_names is None else stats_names)
        if init_stats_logger:
            self.stats_loggers = {
                name: StatsLogger(is_nparray=False) for name in self.stats_names
            }
        self.plans = []

    def _log_if_exists(self, name, value):
        if name in self.stats_loggers:
            self.stats_loggers[name].log(value)

    def start_episode(self, env):
        self.episode_len = 0
        self.episode_reward = 0.0

    def step(self, env, reward, info):
        self.episode_len += 1
        self.episode_reward += reward
        self._log_if_exists('reward', reward)

    def end_episode(self, info):
        self.num_steps += self.episode_len
        self.num_episodes += 1
        self._log_if_exists('episode_len', self.episode_len)
        self._log_if_exists('episode_reward', info.get('reward', self.episode_reward))
        self._log_if_exists(
            'episode_raw_reward',
            info.get('raw_reward', info.get('reward', self.episode_reward)),
        )
        self._log_if_exists('episode_total_i_rate', info.get('total_i_rate', 0.0))
        self._log_if_exists('episode_full_total_i_rate', info.get('full_total_i_rate', 0.0))
        self._log_if_exists('episode_reduction', info.get('reduction', 0.0))

    def add_plan(self, info_plan):
        self.plans.append(info_plan)

    @classmethod
    def merge(cls, logger_list, **kwargs):
        if len(logger_list) == 0:
            return cls(**kwargs)

        logger = cls(init_stats_logger=False, **kwargs)
        logger.num_episodes = sum(item.num_episodes for item in logger_list)
        logger.num_steps = sum(item.num_steps for item in logger_list)

        logger.stats_loggers = {}
        for stats_name in logger.stats_names:
            logger.stats_loggers[stats_name] = StatsLogger.merge(
                [item.stats_loggers[stats_name] for item in logger_list]
            )

        def _avg(stats_name):
            if stats_name not in logger.stats_loggers:
                return 0.0
            return logger.stats_loggers[stats_name].avg()

        logger.total_reward = logger.stats_loggers['reward'].total() if 'reward' in logger.stats_loggers else 0.0
        logger.avg_episode_len = _avg('episode_len')
        logger.avg_episode_reward = (
            logger.total_reward / logger.num_episodes if logger.num_episodes > 0 else 0.0
        )
        logger.avg_episode_raw_reward = _avg('episode_raw_reward')
        logger.avg_episode_total_i_rate = _avg('episode_total_i_rate')
        logger.avg_episode_full_total_i_rate = _avg('episode_full_total_i_rate')
        logger.avg_episode_reduction = _avg('episode_reduction')
        logger.plans = list(itertools.chain(*[item.plans for item in logger_list]))
        return logger
