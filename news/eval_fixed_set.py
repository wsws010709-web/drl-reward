import argparse
import copy
import csv
import json
import logging
import math
import random
import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from news.agents.news_agent import NewsExpansionAgent, tensorfy
from news.envs import NewsEnv
from news.envs.news import News
from news.models.baseline import RandomPolicy
from news.utils.config import Config


LOGGER = logging.getLogger(__name__)


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_test_set(path):
    with open(path, 'r', encoding='utf-8') as f:
        doc = json.load(f)
    if isinstance(doc, list):
        return doc
    return doc['items']


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def build_fixed_test_set(cfg, num_graphs, test_seed):
    rng = np.random.RandomState(test_seed)
    spread_param = copy.deepcopy(cfg.env_param)
    spread_param['seed'] = cfg.seed
    base_news = News(cfg.data_source, spread_param)
    valid_network = list(getattr(base_news, 'valid_network', []))
    if len(valid_network) == 0:
        raise RuntimeError('No valid networks were found for this config.')

    items = []
    seen_network_ids = set()
    attempts = 0
    max_attempts = max(num_graphs * 200, 1000)
    while len(items) < num_graphs and attempts < max_attempts:
        attempts += 1
        network_id = int(rng.choice(valid_network))
        if network_id in seen_network_ids:
            continue

        temp_param = copy.deepcopy(cfg.env_param)
        temp_param['seed'] = cfg.seed
        temp_news = News(cfg.data_source, temp_param)
        temp_news.graph_construct(agent_dict={'network_id': network_id})
        actual_network_id = int(temp_news.network_id)
        if actual_network_id in seen_network_ids:
            continue
        if getattr(temp_news, 'vailde_source', None):
            source = int(rng.choice(temp_news.vailde_source))
        else:
            source = int(temp_news.source)

        items.append({
            'index': len(items),
            'network_id': actual_network_id,
            'source': source,
            'node_num': int(len(temp_news.node_list)),
            'edge_num': int(len(temp_news.G.edges())),
        })
        seen_network_ids.add(actual_network_id)

    if len(items) < num_graphs:
        raise RuntimeError(
            f'Only built {len(items)} test items after {attempts} attempts; '
            f'requested {num_graphs}.'
        )
    return items


def build_agent(cfg_name, global_seed, root_dir, run_name, agent_name,
                checkpoint, restore_best_rewards=True):
    cfg = Config(
        cfg_name,
        global_seed,
        tmp=False,
        root_dir=root_dir,
        agent=agent_name,
        run_name=run_name,
    )
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    device = torch.device('cpu')
    return NewsExpansionAgent(
        cfg=cfg,
        dtype=dtype,
        device=device,
        num_threads=1,
        training=False,
        checkpoint=checkpoint,
        restore_best_rewards=restore_best_rewards,
    )


def build_env_cfg(cfg_name, global_seed, root_dir, agent_name, run_name):
    return Config(
        cfg_name,
        global_seed,
        tmp=False,
        root_dir=root_dir,
        agent=agent_name,
        run_name=run_name,
    )


def make_episode_agent_dict(item, curve_eval_mc):
    agent_dict = {
        'network_id': int(item['network_id']),
        'source': int(item['source']),
    }
    if int(curve_eval_mc) > 0:
        agent_dict['eval_simulation_count'] = int(curve_eval_mc)
    return agent_dict


def select_rl_action(agent, state):
    state_var = tensorfy([state])
    action = agent.policy_net.select_action(state_var, mean_action=True)
    return int(action.numpy().squeeze())


def select_random_action(policy, state):
    state_var = tensorfy([state])
    action = policy.select_action(state_var, mean_action=False)
    return int(action.numpy().squeeze())


def select_edge_betweenness_action(env):
    news = env._news
    mask = news.get_mask()
    valid_indices = [
        idx for idx, is_valid in enumerate(mask)
        if is_valid and idx < len(news.edge_list)
    ]
    if len(valid_indices) == 0:
        return 0

    edge_scores = nx.edge_betweenness_centrality(news.G, normalized=True)
    best_idx = max(
        valid_indices,
        key=lambda idx: (
            edge_scores.get(news.edge_list[idx], 0.0),
            news.init_node_successor_num.get(news.edge_list[idx][0], 0),
            -idx,
        ),
    )
    return int(best_idx)


def run_episode(env, action_fn, item, curve_eval_mc, logger, episode_seed):
    set_all_seeds(episode_seed)
    state = env.reset(eval=True, agent_dict=make_episode_agent_dict(item, curve_eval_mc))
    info_plan = {}
    reward = 0.0
    done = False
    for _ in range(10000):
        action = action_fn(state, env)
        state, reward, done, info_plan = env.step(action, logger)
        if done:
            break
    if not done:
        raise RuntimeError(
            f'Episode did not finish for network_id={item["network_id"]}, '
            f'source={item["source"]}.'
        )
    result = env.get_eval_result()
    result['episode_reward'] = float(info_plan.get('reward', reward))
    result['episode_raw_reward'] = float(
        info_plan.get('raw_reward', info_plan.get('reward', reward))
    )
    return result


def compute_metrics(eval_result):
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

    affected_nodes = set()
    for edge in eval_result.get('deleted_edges', []):
        if len(edge) == 2:
            affected_nodes.add(edge[0])
            affected_nodes.add(edge[1])
    node_num = max(1, int(eval_result.get('node_num', 1)))

    source_distances = np.asarray(
        eval_result.get('deleted_edge_source_distances', []),
        dtype=np.float64,
    )
    source_betweenness = np.asarray(
        eval_result.get('deleted_edge_source_betweenness', []),
        dtype=np.float64,
    )

    return {
        'effect': {
            'final_total_reduction': (
                (full_total_i_rate - total_i_rate) / (full_total_i_rate + 1e-8)
                if full_total_i_rate > 1e-8 else 0.0
            ),
            'peak_cir_origin': origin_peak,
            'peak_cir_cut': cut_peak,
            'peak_cir_reduction': (
                (origin_peak - cut_peak) / (origin_peak + 1e-8)
                if origin_peak > 1e-8 else 0.0
            ),
            'peak_delay': float(cut_peak_step - origin_peak_step),
            'auc_cir_origin': auc_cir_origin,
            'auc_cir_cut': auc_cir_cut,
            'auc_cir_reduction': (
                (auc_cir_origin - auc_cir_cut) / (auc_cir_origin + 1e-8)
                if auc_cir_origin > 1e-8 else 0.0
            ),
            'curve_gap_cir': (
                float((origin_cir_curve - cut_cir_curve).sum())
                if origin_cir_curve.size == cut_cir_curve.size else 0.0
            ),
        },
        'cost': {
            'affected_node_ratio': float(len(affected_nodes) / node_num),
        },
        'explainability': {
            'avg_deleted_edge_source_distance': (
                float(source_distances.mean()) if source_distances.size > 0 else 0.0
            ),
            'avg_deleted_edge_source_betweenness': (
                float(source_betweenness.mean()) if source_betweenness.size > 0 else 0.0
            ),
        },
    }


def aggregate_group(metrics_list, group_name):
    if len(metrics_list) == 0:
        return {}
    output = {}
    for key in metrics_list[0][group_name].keys():
        values = np.asarray([m[group_name][key] for m in metrics_list], dtype=np.float64)
        output[key] = {
            'mean': float(values.mean()),
            'std': float(values.std()),
            'min': float(values.min()),
            'max': float(values.max()),
        }
    return output


def aggregate_method(raw_runs):
    metrics = [run['metrics'] for run in raw_runs]
    return {
        'num_runs': len(raw_runs),
        'effect': aggregate_group(metrics, 'effect'),
        'cost': aggregate_group(metrics, 'cost'),
        'explainability': aggregate_group(metrics, 'explainability'),
    }


def metric_value(run, group, metric):
    return float(run['metrics'][group][metric])


def build_sample_means(raw_runs):
    grouped = {}
    for run in raw_runs:
        grouped.setdefault(int(run['sample_index']), []).append(run)
    means = {}
    for sample_index, runs in grouped.items():
        metric_names = runs[0]['metrics']['effect'].keys()
        means[sample_index] = {
            'effect': {
                name: float(np.mean([metric_value(r, 'effect', name) for r in runs]))
                for name in metric_names
            },
            'cost': {
                name: float(np.mean([metric_value(r, 'cost', name) for r in runs]))
                for name in runs[0]['metrics']['cost'].keys()
            },
            'explainability': {
                name: float(np.mean([metric_value(r, 'explainability', name) for r in runs]))
                for name in runs[0]['metrics']['explainability'].keys()
            },
        }
    return means


def build_paired_comparison(method_runs, reference='rl_best'):
    if reference not in method_runs:
        return {}
    ref_means = build_sample_means(method_runs[reference])
    paired = {}
    for method, runs in method_runs.items():
        if method == reference:
            continue
        method_means = build_sample_means(runs)
        shared_samples = sorted(set(ref_means.keys()) & set(method_means.keys()))
        if len(shared_samples) == 0:
            continue
        diffs = {}
        for group in ['effect', 'cost', 'explainability']:
            diffs[group] = {}
            for metric in ref_means[shared_samples[0]][group].keys():
                values = np.asarray([
                    ref_means[i][group][metric] - method_means[i][group][metric]
                    for i in shared_samples
                ], dtype=np.float64)
                stderr = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
                diffs[group][metric] = {
                    'mean_diff': float(values.mean()),
                    'std_diff': float(values.std()),
                    'ci95_low': float(values.mean() - 1.96 * stderr),
                    'ci95_high': float(values.mean() + 1.96 * stderr),
                }
        paired[f'{reference}_minus_{method}'] = {
            'num_pairs': len(shared_samples),
            'diff': diffs,
        }
    return paired


def write_flat_csv(path, raw_runs):
    fields = [
        'method', 'sample_index', 'repeat_index', 'network_id', 'source', 'node_num',
        'full_total_i_rate', 'total_i_rate',
        'final_total_reduction', 'peak_cir_reduction', 'auc_cir_reduction',
        'peak_delay', 'curve_gap_cir', 'affected_node_ratio',
        'avg_deleted_edge_source_distance', 'avg_deleted_edge_source_betweenness',
    ]
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for method, runs in raw_runs.items():
            for run in runs:
                metrics = run['metrics']
                eval_result = run['eval_result']
                writer.writerow({
                    'method': method,
                    'sample_index': run['sample_index'],
                    'repeat_index': run['repeat_index'],
                    'network_id': eval_result['network_id'],
                    'source': eval_result['source'],
                    'node_num': eval_result['node_num'],
                    'full_total_i_rate': eval_result['full_total_i_rate'],
                    'total_i_rate': eval_result['total_i_rate'],
                    'final_total_reduction': metrics['effect']['final_total_reduction'],
                    'peak_cir_reduction': metrics['effect']['peak_cir_reduction'],
                    'auc_cir_reduction': metrics['effect']['auc_cir_reduction'],
                    'peak_delay': metrics['effect']['peak_delay'],
                    'curve_gap_cir': metrics['effect']['curve_gap_cir'],
                    'affected_node_ratio': metrics['cost']['affected_node_ratio'],
                    'avg_deleted_edge_source_distance': metrics['explainability']['avg_deleted_edge_source_distance'],
                    'avg_deleted_edge_source_betweenness': metrics['explainability']['avg_deleted_edge_source_betweenness'],
                })


def write_mean_std_csv(path, comparison):
    fields = ['method', 'group', 'metric', 'mean', 'std']
    methods = comparison.get('meta', {}).get('methods', list(comparison.get('methods', {}).keys()))
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for method in methods:
            method_summary = comparison.get('methods', {}).get(method, {})
            for group in ['effect', 'cost', 'explainability']:
                for metric, stats in method_summary.get(group, {}).items():
                    writer.writerow({
                        'method': method,
                        'group': group,
                        'metric': metric,
                        'mean': stats.get('mean', 0.0),
                        'std': stats.get('std', 0.0),
                    })


def run_method(method, args, test_items, output_dir, logger):
    raw_runs = []
    curve_eval_mc = int(args.curve_eval_mc)
    if method == 'rl_best':
        agent = build_agent(
            cfg_name=args.cfg,
            global_seed=args.global_seed,
            root_dir=args.root_dir,
            run_name=args.model_run_name,
            agent_name=args.rl_agent,
            checkpoint=args.rl_iteration,
        )
        env = agent.env
        action_fn = lambda state, _env: select_rl_action(agent, state)
        repeats = 1
    elif method == 'random':
        cfg = build_env_cfg(
            cfg_name=args.cfg,
            global_seed=args.global_seed,
            root_dir=args.root_dir,
            agent_name=args.rl_agent,
            run_name=args.model_run_name,
        )
        env = NewsEnv(cfg)
        policy = RandomPolicy()
        action_fn = lambda state, _env: select_random_action(policy, state)
        repeats = int(args.random_repeats)
    elif method == 'edge_betweenness':
        cfg = build_env_cfg(
            cfg_name=args.cfg,
            global_seed=args.global_seed,
            root_dir=args.root_dir,
            agent_name=args.rl_agent,
            run_name=args.model_run_name,
        )
        env = NewsEnv(cfg)
        action_fn = lambda _state, _env: select_edge_betweenness_action(_env)
        repeats = 1
    else:
        raise ValueError(f'Unknown method: {method}')

    for fallback_sample_index, item in enumerate(test_items):
        for repeat_index in range(repeats):
            sample_index = int(item.get('index', fallback_sample_index))
            episode_seed = int(args.eval_seed + sample_index * 1009 + repeat_index * 9176)
            LOGGER.info(
                'method=%s sample=%s repeat=%s network=%s source=%s',
                method,
                sample_index,
                repeat_index,
                item['network_id'],
                item['source'],
            )
            eval_result = run_episode(
                env=env,
                action_fn=action_fn,
                item=item,
                curve_eval_mc=curve_eval_mc,
                logger=logger,
                episode_seed=episode_seed,
            )
            raw_runs.append({
                'method': method,
                'sample_index': sample_index,
                'repeat_index': repeat_index,
                'episode_seed': episode_seed,
                'eval_result': eval_result,
                'metrics': compute_metrics(eval_result),
            })

    save_json(output_dir / f'{method}_raw_runs.json', raw_runs)
    save_json(output_dir / f'{method}_summary.json', aggregate_method(raw_runs))
    return raw_runs


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate RL and non-training baselines on one fixed news test set.'
    )
    parser.add_argument('--cfg', default='twitter_h')
    parser.add_argument('--global_seed', type=int, default=0)
    parser.add_argument('--root_dir', default='./result')
    parser.add_argument('--num_graphs', type=int, default=50)
    parser.add_argument('--test_set', default='')
    parser.add_argument('--test_seed', type=int, default=20260505)
    parser.add_argument('--eval_seed', type=int, default=202605050)
    parser.add_argument('--curve_eval_mc', type=int, default=25)
    parser.add_argument('--methods', default='rl_best,random,edge_betweenness')
    parser.add_argument('--rl_agent', default='rl-gnn3')
    parser.add_argument('--model_run_name', default='exp_20260505_a')
    parser.add_argument('--rl_iteration', default='best')
    parser.add_argument('--random_repeats', type=int, default=5)
    parser.add_argument('--output_name', default='')
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
    args = parse_args()
    if args.output_name:
        output_name = args.output_name
    else:
        output_name = f'fixed_eval_{args.cfg}_n{args.num_graphs}_{int(time.time())}'

    set_all_seeds(args.global_seed)
    base_cfg = build_env_cfg(
        cfg_name=args.cfg,
        global_seed=args.global_seed,
        root_dir=args.root_dir,
        agent_name=args.rl_agent,
        run_name=args.model_run_name,
    )
    output_dir = Path(base_cfg.cfg_dir) / 'fixed_eval' / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.test_set:
        test_items = load_test_set(args.test_set)
        test_set_path = Path(args.test_set)
    else:
        test_items = build_fixed_test_set(base_cfg, args.num_graphs, args.test_seed)
        test_set_path = output_dir / 'test_set.json'
        save_json(test_set_path, {
            'cfg': args.cfg,
            'global_seed': args.global_seed,
            'test_seed': args.test_seed,
            'num_graphs': len(test_items),
            'items': test_items,
        })

    null_logger = logging.getLogger('news.eval_fixed_set.env')
    method_runs = {}
    methods = [method.strip() for method in args.methods.split(',') if method.strip()]
    for method in methods:
        method_runs[method] = run_method(
            method=method,
            args=args,
            test_items=test_items,
            output_dir=output_dir,
            logger=null_logger,
        )

    method_summaries = {
        method: aggregate_method(runs)
        for method, runs in method_runs.items()
    }
    comparison = {
        'meta': {
            'cfg': args.cfg,
            'global_seed': args.global_seed,
            'test_set': str(test_set_path),
            'num_graphs': len(test_items),
            'curve_eval_mc': int(args.curve_eval_mc),
            'methods': methods,
            'random_repeats': int(args.random_repeats),
            'rl_agent': args.rl_agent,
            'model_run_name': args.model_run_name,
            'rl_iteration': args.rl_iteration,
            'output_dir': str(output_dir),
        },
        'methods': method_summaries,
        'paired': build_paired_comparison(method_runs, reference='rl_best'),
    }
    save_json(output_dir / 'comparison_summary.json', comparison)
    write_flat_csv(output_dir / 'comparison_raw_metrics.csv', method_runs)
    write_mean_std_csv(output_dir / 'all_methods_mean_std.csv', comparison)
    LOGGER.info('Wrote fixed-set comparison to %s', output_dir)


if __name__ == '__main__':
    main()
