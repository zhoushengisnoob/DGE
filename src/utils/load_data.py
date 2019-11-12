import numpy as np
import networkx as nx
# from sklearn.preprocessing import normalize
import scipy.sparse as sp
import torch
import time


def load_network(dataset, self_edge=True):
    edge_path = '../data/' + dataset + '/' + dataset + '.edgelist'
    label_path = '../data/' + dataset + '/' + dataset + '.label'
    feature_path = '../data/' + dataset + '/' + dataset + '.feature'

    feature_mat = np.loadtxt(feature_path, dtype=np.float32)
    label_mat = np.loadtxt(label_path, dtype=int)
    edge_mat = np.loadtxt(edge_path, dtype=int)
    adj = sp.coo_matrix((np.ones(edge_mat.shape[0]), (edge_mat[:, 0], edge_mat[:, 1])),
                        shape=(np.max(edge_mat) + 1, np.max(edge_mat) + 1), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + np.eye(adj.shape[0], dtype=int)
    GT_mat = np.loadtxt(label_path, dtype=int)
    GT = GT_mat[GT_mat[:, 0].argsort()][:, 1]
    return adj, feature_mat, GT


def load_network_sparse(dataset, self_edge=True):
    edge_path = '../data/' + dataset + '/' + dataset + '.edgelist'
    label_path = '../data/' + dataset + '/' + dataset + '.label'
    feature_path = '../data/' + dataset + '/' + dataset + '.feature'
    feature_mat = sp.csr_matrix(np.loadtxt(feature_path, dtype=np.float32))
    label_mat = np.loadtxt(label_path, dtype=int)
    edge_mat = np.loadtxt(edge_path, dtype=int)
    adj = sp.coo_matrix((np.ones(edge_mat.shape[0]), (edge_mat[:, 0], edge_mat[:, 1])),
                        shape=(np.max(edge_mat) + 1, np.max(edge_mat) + 1), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = normalize(adj + sp.eye(adj.shape[0]))
    fea = normalize(feature_mat)

    adj = sparse_mx_to_torch_sparse_tensor(adj)
    fea = sparse_mx_to_torch_sparse_tensor(fea)
    return adj, fea, label_mat


def normalize(mx):
    # 邻接矩阵按行进行normalize
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    # scipy稀疏矩阵转化为稀疏tensor
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def graph_distance_matrix(adj_mat):
    G = nx.from_numpy_matrix(adj_mat)
    length = dict(nx.all_pairs_shortest_path_length(G))
    node_num = G.number_of_nodes()
    mat = np.zeros((node_num, node_num))
    for i in range(node_num):
        for j in length[i]:
            mat[i, j] = 1.0 / (length[i][j] + 1)
    return mat


def load_embedding(path):
    embedding_res = np.loadtxt(path, dtype=float)
    return embedding_res


def matrix2sparsetensor(A):
    B = np.nonzero(A)
    i = torch.LongTensor(np.concatenate((B[0].reshape(-1, 1), B[1].reshape(-1, 1)), axis=1))
    v = torch.FloatTensor(A[B])
    return torch.sparse.FloatTensor(i.t(), v, torch.Size([A.shape[0], A.shape[1]]))


def index_mask(adj, index):
    index2num = {}
    num2index = {}
    mask_num = 0
    for n in index:
        index2num[n] = mask_num
        num2index[mask_num] = n
        mask_num += 1
    sampled_adj = torch.zeros(mask_num, mask_num)
    edge_mat = torch.nonzero(adj[index])
    for edge in edge_mat:
        if adj[num2index[edge[0].item()], edge[1]] == 1:
            sampled_adj[edge[0], index2num[edge[1].item()]] = 1
    return sampled_adj


def make_input(adj, fea, batch_size, k):
    '''
    Make two matrix for edge and non-edge
    '''
    t0=time.time()
    node_num = adj.shape[0]
    fea_dim = fea.shape[1]
    x1, y1 = np.nonzero(adj)
    x0, y0 = np.nonzero(1 - adj)
    index1 = list(range(len(x1)))
    index0 = list(range(len(x0)))
    t1=time.time()
    #np.random.shuffle(index1)
    #np.random.shuffle(index0)
    s_index1 = np.random.choice(len(x1),batch_size)
    s_index0 = np.random.choice(len(x0),batch_size*k)
    t2=time.time()
    s_x1 = x1[s_index1]
    s_y1 = y1[s_index1]
    s_x0 = x0[s_index0]
    s_y0 = y0[s_index0]
    M1 = np.concatenate((fea[s_x1], fea[s_y1], adj[s_x1], adj[s_y1],np.ones((batch_size,1))),axis=1)
    M0 = np.concatenate((fea[s_x0], fea[s_y0], adj[s_x0], adj[s_y0],np.ones((batch_size*k,1))),axis=1)
    t3=time.time()
    print(t1-t0,t2-t1,t3-t2)
    return M1,M0
