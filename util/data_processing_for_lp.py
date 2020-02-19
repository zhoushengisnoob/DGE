import numpy as np
from util import load_data
import random


# prefix = '../datasets/citeseer/citeseer'
prefix = '../datasets/cora/cora'


def read_edges(filename):
    edge_flag = {}
    n = 0
    with open(filename) as fr:
        lines = fr.readlines()
        for line in lines:
            items = line.strip().split(' ')
            a = int(items[0])
            b = int(items[1])
            n = max(a, n)
            n = max(b, n)
            edge_flag[(a, b)] = True
    return edge_flag, n


def sample_edges(edge_flag, n, ratio=0.5):
    cur = 0
    neg_flag = {}
    edge_len = len(edge_flag.keys())
    sample_len = int(ratio * edge_len)
    while cur < sample_len:
        a = random.randint(0, n)
        b = random.randint(0, n)
        if a != b and (a, b) not in edge_flag and (b, a) not in edge_flag and (a, b) not in neg_flag and (b, a) not in neg_flag:
            neg_flag[(a, b)] = True
            cur += 1
    neg_edges = list(neg_flag.keys())
    pos_edges = list(edge_flag.keys())
    random.shuffle(pos_edges)
    test_pos_edges = pos_edges[0: sample_len]
    train_pos_edges = pos_edges[sample_len:]
    test_data = []
    for e in neg_edges:
        test_data.append(str(e[0]) + ' ' + str(e[1]) + ' ' + '0' + '\n')
    for e in test_pos_edges:
        test_data.append(str(e[0]) + ' ' + str(e[1]) + ' ' + '1' + '\n')
    train_data = []
    for e in train_pos_edges:
        train_data.append(str(e[0]) + ' ' + str(e[1]) + '\n')
    with open(prefix + '_lp.edgelist', 'w') as fw:
        fw.writelines(train_data)
    with open(prefix + '_lp_test', 'w') as fw:
        fw.writelines(test_data)


if __name__ == '__main__':
    edge_flag, n = read_edges(prefix + '.edgelist')
    sample_edges(edge_flag, n)