import numpy as np
import random
import copy


def read_feature(filename):
    with open(filename) as f:
        first_line = f.readline()
        n, m = first_line.strip('\n').split(' ')
        n = int(n)
        m = int(m)
        lines = f.readlines()
        feature = np.zeros((n, m))
        for i in range(len(lines)):
            line = lines[i]
            items = line.strip('\n').split(' ')
            for j in range(len(items)):
                feature[i, j] = float(items[j])
    return feature, n, m


def read_label(filename):
    with open(filename) as f:
        lines = f.readlines()
        N = len(lines)
        label = np.zeros((N, ), dtype=int)
        for line in lines:
            items = line.strip('\n').split(' ')
            a = int(items[0])
            b = int(items[1])
            label[a] = b
    return label


def read_edgelist(filename, n):
    adj = np.zeros((n, n), dtype=np.float32)
    edge_dict = {}
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            items = line.strip('\n').split(' ')
            a = int(items[0])
            b = int(items[1])
            adj[a][b] = 1
            adj[b][a] = 1
            if a in edge_dict:
                edge_dict[a].append(b)
            else:
                edge_dict[a] = [b]
            if b in edge_dict:
                edge_dict[b].append(a)
            else:
                edge_dict[b] = [a]
    A = copy.deepcopy(adj)
    for i in range(n):
        A[i][i] = 1
    A /= np.tile(np.mat(np.sum(A, axis=1)), (n, 1)).T
    return A, adj, edge_dict


def read_test_lp_data(filename):
    node1s = []
    node2s = []
    labels = []
    with open(filename) as fr:
        lines = fr.readlines()
        for line in lines:
            items = line.strip().split(' ')
            a = int(items[0])
            b = int(items[1])
            l = int(items[2])
            node1s.append(a)
            node2s.append(b)
            labels.append(l)
    return node1s, node2s, labels

