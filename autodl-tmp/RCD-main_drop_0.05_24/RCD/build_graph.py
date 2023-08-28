# -*- coding: utf-8 -*-

import dgl
import torch
import networkx as nx
import matplotlib.pyplot as plt
#from LightGCN import create_adjacency_matrix,compute_degree_matrix,load_edges_from_file,normalized_laplacian
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
from GraphLayer import GraphLayer
import dgl.nn as dglnn


def remove_random_edges(g, fraction=0.05):
    # 获取图的边列表
    edges = g.edges()
    
    # 计算需要删除的边的数量
    num_edges = len(edges[0])
    num_remove = int(num_edges * fraction)
    
    # 随机选择一部分边进行删除
    remove_indices = np.random.choice(num_edges, num_remove, replace=False)    
    # 创建一个新的图，不包含被删除的边
    new_g = dgl.remove_edges(g, remove_indices)
    
    return new_g

def build_graph(type, node,args):
    g = dgl.DGLGraph()
    # add 34 nodes into the graph; nodes are labeled from 0~33
    g.add_nodes(node)
    edge_list = []
    if type == 'direct':
        with open('../data/ASSIST/graph/K_Directed.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))

        # add edges two lists of nodes: src and dst
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        g = remove_random_edges(g,args.dropout)
        
        # edges are directional in DGL; make them bi-directional
        # g.add_edges(dst, src)
        return g
    elif type == 'undirect':
        with open('../data/ASSIST/graph/K_Undirected.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        # add edges two lists of nodes: src and dst
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        # edges are directional in DGL; make them bi-directional
        g.add_edges(dst, src)
        return g
    elif type == 'k_from_e':
        with open('../data/ASSIST/graph/k_from_e.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        # add edges two lists of nodes: src and dst
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g
    elif type == 'e_from_k':
        with open('../data/ASSIST/graph/e_from_k.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        # add edges two lists of nodes: src and dst
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g
    elif type == 'u_from_e':
        with open('../data/ASSIST/graph/u_from_e.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        # add edges two lists of nodes: src and dst
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g
    
    # elif type == 'u_from_e':
    #     u_from_e_edges = load_edges_from_file('../data/ASSIST/graph/u_from_e.txt')

    #     all_edges = u_from_e_edges
    #     # Assuming you know the total number of nodes
    #     num_nodes = 20737  # Replace this with the actual number of nodes
    #     adj_matrix = create_adjacency_matrix(all_edges, num_nodes)
    #     degree_matrix = compute_degree_matrix(adj_matrix)
    #     normalized_laplacian_matrix = normalized_laplacian(adj_matrix, degree_matrix)
    #     return normalized_laplacian_matrix
        
    elif type == 'e_from_u':
        with open('../data/ASSIST/graph/e_from_u.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        # add edges two lists of nodes: src and dst
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g