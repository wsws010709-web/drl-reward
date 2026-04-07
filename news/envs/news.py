import networkx as nx
import numpy as np
import random, pickle
import copy
import logging
from itertools import combinations
# import geopandas as gpd
# import pandas as pd
# from shapely import geometry
# from haversine import haversine, Unit
import matplotlib.pyplot as plt
# import plotly.express as px

import datetime
import time

RESULT = 0
NETWORK_ID = 62985251
SOURCE = None
# COMMUNITY = [90888992, 200559228, 17434613, 115224382, 16503181, 28019653, 17675120, 72720307, 73050189, 546135380, 211277445, 210428550, 57496410, 318505435, 44483734]
COMMUNITY = []
SIZE = 'k'
LOGGER = logging.getLogger(__name__)

class News(object):

    def __init__(self, data_source, spread_param) -> None:
    
        self.data_source = data_source
        self.spread_param = spread_param
        self.graph_construct()
        # self.dynamic_init()
    
    def graph_construct(self):
        if self.data_source == 'twitter':
            if self.spread_param['network_size'] == 'h':
                data = pickle.load(open('./data/t1he.pkl', 'rb'))
            elif self.spread_param['network_size'] == 'k':
                data = pickle.load(open('./data/t1ke.pkl', 'rb'))
            elif self.spread_param['network_size'] == 'w':
                data = pickle.load(open('./data/t1we.pkl', 'rb'))
            else:
                raise ValueError('network_size error')
        elif self.data_source == 'facebook':
            if self.spread_param['network_size'] == 'h':
                data = pickle.load(open('./data/f1he.pkl', 'rb'))
            elif self.spread_param['network_size'] == 'k':
                data = pickle.load(open('./data/f1ke.pkl', 'rb'))
            elif self.spread_param['network_size'] == 'w':
                data = pickle.load(open('./data/f1we.pkl', 'rb'))
            else:
                raise ValueError('network_size error')
                
        valid_network = data[0]
        self.max_node_num = data[1]
        self.max_edge_num = data[2]
        self.network_id = random.choice(valid_network)

        if RESULT:
            self.network_id = NETWORK_ID

        edges_file_path = f"./data/{self.data_source}/{self.network_id}.edges"

        self.G = nx.DiGraph()
            # with open(feat_file_path, 'r') as feat_file:
            #     for line in feat_file:
            #         parts = line.strip().split()
            #         if len(parts) > 0:
            #             node = int(parts[0])
            #             self.G.add_node(node)

        with open(edges_file_path, 'r') as edges_file:
            for line in edges_file:
                parts = line.strip().split()
                if len(parts) == 2:
                    follower = int(parts[0])
                    followee = int(parts[1])
                    if self.data_source == 'twitter':
                        self.G.add_edge(followee, follower)
                    elif self.data_source == 'facebook':
                        self.G.add_edge(followee, follower)
                        self.G.add_edge(follower, followee)

        self.G = nx.DiGraph(self.G.subgraph(max(nx.strongly_connected_components(self.G), key=len)))
        
        self.node_list = [n for n in self.G.nodes()]
        self.node_list_idx = dict(zip(self.node_list, [i for i in range(len(self.node_list))]))
        self.init_node_successor_num = dict(zip(self.node_list, [len(list(self.G.successors(n))) for n in self.node_list]))
        self.node_predecessor_num = dict(zip(self.node_list, [len(list(self.G.predecessors(n))) for n in self.node_list]))
        avg_init_node_successor_num = sum(self.init_node_successor_num.values()) / len(self.node_list)
        self.vailde_source = [n for n in self.node_list if self.init_node_successor_num[n] > avg_init_node_successor_num]
        self.total_cut = len(self.G.edges()) * self.spread_param['total_cut_ration']
        if self.total_cut < 5:
            self.graph_construct()

        if self.spread_param['source'] is None:
            avg_init_node_successor_num = sum(self.init_node_successor_num.values()) / len(self.node_list)
            initial_infected_node = random.choice(self.node_list)
            try_count = 0
            while self.init_node_successor_num[initial_infected_node] < max(avg_init_node_successor_num, 5) or initial_infected_node in COMMUNITY:
                initial_infected_node = random.choice(self.node_list)
                try_count += 1
                if try_count > 100:
                    assert('no initial infected node')
            self.source = initial_infected_node
        else:
            self.source = self.spread_param['source']

        if RESULT and SOURCE is not None:
            self.source = SOURCE

        print(self.network_id, self.source)

        # self.G_deepcopy = copy.deepcopy(self.G)

    def dynamic_init(self):
        self.init_result()
        self.node_cut_successor_num = dict(zip(self.node_list, [0 for i in range(len(self.node_list))]))
        self.node_cut_predecessor_num = dict(zip(self.node_list, [0 for i in range(len(self.node_list))]))
        self.edge_list = [e for e in self.G.edges()]
        self.edge_index = self._cal_edge_index()
        self.G_full = copy.deepcopy(self.G)
        self.deleted_edges = []
        self.cut_num = 0

    def init_result(self):
        for attr in ['result_full', 'result_cut', 'result_full_curve', 'result_cut_curve']:
            if hasattr(self, attr):
                delattr(self, attr)

    def reset(self,eval=False,agent_dict=None):
        self.eval = eval
        self.eval_simulation_count_override = None
        if agent_dict is not None and 'cut_ration' in agent_dict:
            self.spread_param['total_cut_ration'] = agent_dict['cut_ration']
        if eval and agent_dict is not None and 'eval_simulation_count' in agent_dict:
            override = int(agent_dict['eval_simulation_count'])
            if override > 0:
                self.eval_simulation_count_override = override
        self.graph_construct()
        self.dynamic_init()

        base_sim_count = self._resolve_terminal_simulation_count()
        self.propagation_simulation(simulation_count=base_sim_count, save_to='full')
        self.result_cut = None
        self.result_cut_curve = None

    def _resolve_terminal_simulation_count(self):
        if self.eval and getattr(self, 'eval_simulation_count_override', None) is not None:
            return int(max(1, self.eval_simulation_count_override))
        terminal_mc = int(self.spread_param['terminal_reward_mc'])
        return int(self.spread_param['simulation_count'] * max(1, terminal_mc))

    def _resolve_curve_eval_simulation_count(self):
        if self.eval and getattr(self, 'eval_simulation_count_override', None) is not None:
            return int(max(1, self.eval_simulation_count_override))
        default_count = self.spread_param['simulation_count'] * 5
        return int(max(1, self.spread_param.get('curve_eval_simulation_count', default_count)))

    def get_terminal_simulation_count(self):
        return self._resolve_terminal_simulation_count()

    def _simulate_sir_mc(self, graph, simulation_count, total_steps, tag='simulation'):
        if self.spread_param['model'] != 'SIR':
            raise NotImplementedError('Only SIR model is supported.')

        sir_gamma = self.spread_param['gamma']
        sir_beta = self.spread_param['beta']
        simulation_count = int(max(0, simulation_count))
        total_steps = int(max(0, total_steps))
        valid_threshold = max(0.01 * len(self.node_list), 5)

        final_i_sum = 0.0
        total_i_sum = 0.0
        valid_count = 0
        cir_sum = np.zeros(total_steps, dtype=np.float64)
        tir_sum = np.zeros(total_steps, dtype=np.float64)

        for _ in range(simulation_count):
            for node in self.node_list:
                graph.nodes[node]['status'] = 'S'
            graph.nodes[self.source]['status'] = 'I'

            rollout_cir = np.zeros(total_steps, dtype=np.float64)
            rollout_tir = np.zeros(total_steps, dtype=np.float64)

            for t in range(total_steps):
                new_status = {}
                for node in self.node_list:
                    if graph.nodes[node]['status'] == 'I' and node != self.source:
                        new_status[node] = 'R' if random.random() < sir_gamma else 'I'
                    elif graph.nodes[node]['status'] == 'S':
                        sources = list(graph.predecessors(node))
                        infected_successors = sum(1 for s in sources if graph.nodes[s]['status'] == 'I')
                        new_status[node] = 'I' if random.random() < (1 - (1 - sir_beta) ** infected_successors) else 'S'

                for node, status in new_status.items():
                    graph.nodes[node]['status'] = status

                if len(COMMUNITY) > 0:
                    final_statuses = [graph.nodes[node]['status'] for node in COMMUNITY]
                else:
                    final_statuses = [graph.nodes[node]['status'] for node in self.node_list]
                infected_count = final_statuses.count('I')
                recovered_count = final_statuses.count('R')

                rollout_cir[t] = infected_count
                rollout_tir[t] = infected_count + recovered_count

            final_total_infected = rollout_tir[-1] if total_steps > 0 else 0.0
            if final_total_infected > valid_threshold:
                valid_count += 1
                final_i_sum += rollout_cir[-1] if total_steps > 0 else 0.0
                total_i_sum += final_total_infected
                cir_sum += rollout_cir
                tir_sum += rollout_tir

        denom = valid_count * len(self.node_list) + 1e-8
        final_i_rate = final_i_sum / denom
        total_i_rate = total_i_sum / denom
        if valid_count == 0:
            LOGGER.warning(
                'No valid outbreak samples in %s (network_id=%s, source=%s, mc=%s). '
                'CIR/TIR curves are set to zeros.',
                tag,
                self.network_id,
                self.source,
                simulation_count,
            )
            cir_curve = np.zeros(total_steps, dtype=np.float64)
            tir_curve = np.zeros(total_steps, dtype=np.float64)
        else:
            cir_curve = cir_sum / denom
            tir_curve = tir_sum / denom

        return {
            'final_i_rate': float(final_i_rate),
            'total_i_rate': float(total_i_rate),
            'cir_curve': cir_curve,
            'tir_curve': tir_curve,
        }

    def _propagation_eval_on_graph(self, graph, tag='propagation_eval'):
        total_steps = int(self.spread_param['spread_steps'])
        simulation_count = self._resolve_curve_eval_simulation_count()
        stats = self._simulate_sir_mc(graph, simulation_count, total_steps, tag=tag)
        return stats['cir_curve'], stats['tir_curve']

    def propagation_eval(self):
        return self._propagation_eval_on_graph(self.G)


    def propagation_simulation(self, simulation_count=None, save_to='auto'):
        if self.spread_param['model'] != 'SIR':
            raise NotImplementedError('Only SIR model is supported.')

        if simulation_count is None:
            simulation_count = self.spread_param['simulation_count'] * 5 if self.eval else self.spread_param['simulation_count']
        total_steps = int(self.spread_param['spread_steps'])
        stats = self._simulate_sir_mc(
            self.G,
            simulation_count=simulation_count,
            total_steps=total_steps,
            tag=f'propagation_simulation[{save_to}]',
        )
        final_i_rate = stats['final_i_rate']
        total_i_rate = stats['total_i_rate']
        cir_curve = stats['cir_curve']
        tir_curve = stats['tir_curve']

        if save_to == 'full':
            self.result_full = total_i_rate
            self.result_full_curve = (cir_curve, tir_curve)
        elif save_to == 'cut':
            self.result_cut = total_i_rate
            self.result_cut_curve = (cir_curve, tir_curve)
        else:
            if not hasattr(self, 'result_full'):
                self.result_full = total_i_rate
                self.result_full_curve = (cir_curve, tir_curve)
            else:
                self.result_cut = total_i_rate
                self.result_cut_curve = (cir_curve, tir_curve)

        return final_i_rate, total_i_rate

    


    def _cal_node_static(self):

        def _dict_normalize(d):
            avg = sum(d.values()) / len(d)
            if avg == 0:
                return d
            return {k: v / avg for k, v in d.items()}
        
        def _dict_zero(d):
            return {k: 0 for k, v in d.items()}
        
        # t0 = time.time()
        degree_cen = nx.degree_centrality(self.G)
        degree_cen = _dict_normalize(degree_cen)
        # degree_cen = _dict_zero(degree_cen)
        # t1 = time.time()
        betweenness_cen = nx.betweenness_centrality(self.G , normalized = False)
        betweenness_cen = _dict_normalize(betweenness_cen)
        # betweenness_cen = _dict_zero(betweenness_cen)
        # t2 = time.time()
        betweenness_cen_s = nx.betweenness_centrality_subset(self.G, normalized = False, sources=[self.source], targets=self.node_list)
        betweenness_cen_s = _dict_normalize(betweenness_cen_s)
        # betweenness_cen_s = _dict_zero(betweenness_cen_s)
        # t3 = time.time()
        # Avoid scipy/arpack path on Windows, which can trigger OpenMP runtime conflicts.
        try:
            eigenvector_cen = nx.eigenvector_centrality(self.G, max_iter=500, tol=1e-6)
        except Exception:
            try:
                # Directed/disconnected graphs can be numerically harder; retry with looser settings.
                eigenvector_cen = nx.eigenvector_centrality(self.G, max_iter=2000, tol=1e-4)
            except Exception:
                # Keep rollout alive even when eigenvector centrality fails for a specific graph.
                eigenvector_cen = {node: 0.0 for node in self.node_list}
        eigenvector_cen = _dict_normalize(eigenvector_cen)
        # eigenvector_cent = _dict_zero(eigenvector_cen)
        # t4 = time.time()
        closeness_cen = nx.closeness_centrality(self.G)
        closeness_cen = _dict_normalize(closeness_cen)
        # closeness_cent = _dict_zero(closeness_cen)
        # t5 = time.time()
        clustering_cen = nx.clustering(self.G)
        clustering_cen = _dict_normalize(clustering_cen)
        # clustering_cen = _dict_zero(clustering_cen)
        # t6 = time.time()
        shortest_path = dict(nx.shortest_path_length(self.G, source = self.source))
        shortest_path = _dict_normalize(shortest_path)
        # shortest_path = _dict_zero(shortest_path)
        # shortest_path2 = dict(nx.shortest_path_length(self.G, target = self.source))
        # t7 = time.time()


        # print(len(self.node_list))
        # print(len(self.edge_list))
        # print(len(shortest_path))

        # print('degree_cen: ',t1-t0)
        # print('betweenness_cen: ',t2-t1)
        # print('betweenness_cen_s: ',t3-t2)
        # print('eigenvector_cen: ',t4-t3)
        # print('closeness_cen: ',t5-t4)
        # print('clustering_cen: ',t6-t5)
        # print('shortest_path: ',t7-t6)

        node_static = {}
        for node in self.node_list:
            if node not in shortest_path:
                shortest_path[node] = self.max_node_num - 1

            successors = len(list(self.G.successors(node)))
            presuccessor = len(list(self.G.predecessors(node)))
                
            node_static[node] = [degree_cen[node], betweenness_cen[node], betweenness_cen_s[node], eigenvector_cen[node], closeness_cen[node], clustering_cen[node], successors, presuccessor, shortest_path[node]]

        return node_static

    def _cal_edge_static(self):
        def _dict_normalize(d):
            avg = sum(d.values()) / len(d)
            return {k: v / avg for k, v in d.items()}

        def _dict_zero(d):
            return {k: 0 for k, v in d.items()}
        
        e_betweenness_cen_s = nx.edge_betweenness_centrality_subset(self.G, normalized = False, sources=[self.source], targets=self.node_list)
        e_betweenness_cen_s = _dict_normalize(e_betweenness_cen_s)
        # e_betweenness_cen_s = _dict_zero(e_betweenness_cen_s)


        edge_static = {}
        for e in self.edge_list:
            successors_followee = list(self.G.successors(e[0]))
            successors_follower = list(self.G.successors(e[1]))
            successors_difference_len = len(set(successors_followee).symmetric_difference(set(successors_follower)))
            # successors_difference_len = 0
            presuccessor_followee = list(self.G.predecessors(e[0]))
            presuccessor_follower = list(self.G.predecessors(e[1]))
            presuccessor_difference_len = len(set(presuccessor_followee).symmetric_difference(set(presuccessor_follower)))
            # presuccessor_difference_len = 0

            edge_static[e] = [e_betweenness_cen_s[e], successors_difference_len, presuccessor_difference_len]

        return edge_static
    
    def _cal_node_all(self):
        node_static = self._cal_node_static()
        node_all = {}
        for n in self.node_list:
            node_all[n] = node_static[n] + [1 if n == self.source else 0]

        return node_all
    
    def _cal_edge_all(self):
        # edge_static = self._cal_edge_static()
        # edge_all = {}
        # for e in self.edge_list:
        #     edge_all[e] = edge_static[e]

        return self._cal_edge_static()
            
    def get_numerical_dim(self):
        return 4
    
    def get_node_dim(self):
        return 10
    
    def get_edge_dim(self):
        return 3

    def _cal_edge_index(self):
        edge_index = []

        for e in self.edge_list:
            idx1 = self.node_list_idx[e[0]]
            idx2 = self.node_list_idx[e[1]]
            edge_index.append([idx1, idx2])

        edge_index = edge_index + [[self.max_node_num-1, self.max_node_num-1] for i in range(self.max_edge_num - len(self.edge_list))]
        
        return edge_index
    
    
    def _get_numerical(self):
        numerical = [len(self.node_list) / 1000, len(self.edge_list) / 1000, \
                     self.total_cut / len(self.edge_list), self.cut_num / self.total_cut]

        return numerical

            
    def get_obs(self):
        numerical = self._get_numerical()
        
        node_all = self._cal_node_all()
        node_feature = np.concatenate([[node_all[n] for n in self.node_list], np.zeros(((self.max_node_num - len(self.node_list)), self.get_node_dim()))], axis=0)
        edghe_all = self._cal_edge_all()
        edge_feature = np.concatenate([[edghe_all[e] for e in self.edge_list], np.zeros(((self.max_edge_num - len(self.edge_list)), self.get_edge_dim()))], axis=0)
        mask = self.get_mask()

        return numerical, node_feature, edge_feature, self.edge_index, mask


    def cut_edge_from_action(self,action):
        try:
            cut_edge = self.edge_list[action]
        except:
            print((self._cal_mask())[action])
            raise ValueError('action error')

        self.G.remove_edge(cut_edge[0],cut_edge[1])
        self.deleted_edges.append((cut_edge[0], cut_edge[1]))
        self.node_cut_successor_num[cut_edge[0]] += 1
        self.node_cut_predecessor_num[cut_edge[1]] += 1
        self.cut_num += 1

        self.edge_list.remove(cut_edge)
        self.edge_index.remove([self.node_list_idx[cut_edge[0]],self.node_list_idx[cut_edge[1]]])
        self.edge_index.append([self.max_node_num-1, self.max_node_num-1])


    def _cal_mask(self):

        mask = [False for _ in range(self.max_edge_num)]
        for idx in range(len(self.edge_list)):
            edge = self.edge_list[idx]
            # (self.node_cut_successor_num[edge[1]] + 1) <= self.init_node_successor_num[edge[1]] * self.spread_param['max_node_cut_ration'] and \
            # (self.node_cut_predecessor_num[edge[0]] + 1) <= self.node_predecessor_num[edge[0]] * self.spread_param['max_node_cut_ration'] and \
            if (self.node_cut_successor_num[edge[0]] + 1) <= self.init_node_successor_num[edge[0]] * self.spread_param['max_node_cut_ration'] and \
            (self.node_cut_predecessor_num[edge[1]] + 1) <= self.node_predecessor_num[edge[1]] * self.spread_param['max_node_cut_ration']:
                mask[idx] = True
            if len(COMMUNITY) > 0:
                if edge[0] in COMMUNITY or edge[1] in COMMUNITY:
                    mask[idx] = False
        
        if np.array(mask).sum() == 0:
            self.done = 1
        else:
            self.done = 0

        return mask


    def get_cut_num(self):
        return self.cut_num
    
    def get_total_cut_num(self):
        return self.total_cut

    def get_mask(self):
        return self._cal_mask()
    
    def get_edge_index(self):
        return self._cal_edge_index()

    def get_reward(self):
        if self.spread_param['model'] == 'SIR':
            if (not hasattr(self, 'result_full')) or self.result_full is None or self.result_full < 1e-8:
                return 0.0
            if (not hasattr(self, 'result_cut')) or self.result_cut is None:
                return 0.0
            r2 = (self.result_full - self.result_cut) / (self.result_full + 1e-8)
            return r2

    def get_total_i_rate(self):
        if (not hasattr(self, 'result_cut')) or self.result_cut is None:
            return self.result_full if hasattr(self, 'result_full') else 0.0
        return self.result_cut

    def get_full_total_i_rate(self):
        return self.result_full if hasattr(self, 'result_full') else 0.0
    
    def get_done(self):
        self._cal_mask()
        return self.done
            
    def get_env_info_dict(self):
        return {'network_id': self.network_id, 'source': self.source, 'node_num': len(self.node_list), 'edge_num': len(self.edge_list)}

    def get_eval_result(self):
        graph_full = self.G_full if hasattr(self, 'G_full') else copy.deepcopy(self.G)

        if hasattr(self, 'result_full_curve') and self.result_full_curve is not None:
            origin_cir_curve, origin_tir_curve = self.result_full_curve
        else:
            origin_cir_curve, origin_tir_curve = self._propagation_eval_on_graph(graph_full, tag='eval_full_fallback')

        if hasattr(self, 'result_cut_curve') and self.result_cut_curve is not None:
            cut_cir_curve, cut_tir_curve = self.result_cut_curve
        else:
            cut_cir_curve, cut_tir_curve = self._propagation_eval_on_graph(self.G, tag='eval_cut_fallback')

        full_total_i_rate = self.get_full_total_i_rate()
        total_i_rate = self.get_total_i_rate()
        reduction = (full_total_i_rate - total_i_rate) / (full_total_i_rate + 1e-8) \
            if full_total_i_rate > 1e-8 else 0.0

        deleted_edges = [[int(u), int(v)] for u, v in getattr(self, 'deleted_edges', [])]
        edge_source_distances = []
        edge_source_betweenness = []
        if len(deleted_edges) > 0:
            source_shortest_path = dict(nx.shortest_path_length(graph_full, source=self.source))
            e_betweenness = nx.edge_betweenness_centrality_subset(
                graph_full,
                normalized=False,
                sources=[self.source],
                targets=self.node_list
            )
            for u, v in deleted_edges:
                dist_u = source_shortest_path.get(u, self.max_node_num)
                dist_v = source_shortest_path.get(v, self.max_node_num)
                edge_source_distances.append(float(min(dist_u, dist_v)))
                edge_source_betweenness.append(float(e_betweenness.get((u, v), 0.0)))

        return {
            'network_id': int(self.network_id),
            'source': int(self.source),
            'node_num': int(len(self.node_list)),
            'full_total_i_rate': float(full_total_i_rate),
            'total_i_rate': float(total_i_rate),
            'reduction': float(reduction),
            'origin_cir_curve': origin_cir_curve.astype(float).tolist(),
            'origin_tir_curve': origin_tir_curve.astype(float).tolist(),
            'cut_cir_curve': cut_cir_curve.astype(float).tolist(),
            'cut_tir_curve': cut_tir_curve.astype(float).tolist(),
            'deleted_edges': deleted_edges,
            'deleted_edge_source_distances': edge_source_distances,
            'deleted_edge_source_betweenness': edge_source_betweenness,
        }
    
    def get_max_edge_num(self):
        return self.max_edge_num
    
    def get_max_node_num(self):
        return self.max_node_num
    
    def plot(self):
        had_result_full = hasattr(self, 'result_full')
        had_result_cut = hasattr(self, 'result_cut')
        result_full_snapshot = self.result_full if had_result_full else None
        result_cut_snapshot = self.result_cut if had_result_cut else None

        try:
            cir_curve, tir_curve = self.propagation_eval()
            steps = range(len(cir_curve))
            plt.plot(steps, tir_curve, label='TIR', color='red')
            plt.plot(steps, cir_curve, label='CIR', color='blue')
        finally:
            if had_result_full:
                self.result_full = result_full_snapshot
            elif hasattr(self, 'result_full'):
                del self.result_full

            if had_result_cut:
                self.result_cut = result_cut_snapshot
            elif hasattr(self, 'result_cut'):
                del self.result_cut

        # plt.legend()
