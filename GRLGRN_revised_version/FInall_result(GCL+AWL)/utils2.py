import torch
import numpy as np
import pandas as Pd
import torch.nn.functional as F
from collections import defaultdict
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler

class G_vocab():
    def __init__(self, node, type, index):
        super().__init__()
        self.node = node
        self.index = index
        self.type = type


def get_type_genes_dict(exp_path, TF_list_file):  # get different type of gene
    type_gene_dict = {}
    exp = Pd.read_csv(exp_path, index_col=0).values
    nodes_num = np.arange(len(exp))
    All_TF_list = Pd.read_csv(TF_list_file, index_col=0)["index"].values  # Regulator gene index
    Target_only = np.setdiff1d(nodes_num, All_TF_list) #Target gene index
    assert len(nodes_num) == len(Target_only) + len(All_TF_list)
    type_gene_dict[0] = All_TF_list  # 0: Regulator gene
    type_gene_dict[1] = Target_only  # 1:Target gene
    return type_gene_dict, nodes_num

def get_node2vocab(type_gene_dict):  # 修改
    gene_type_dict = {}
    for type, _ in type_gene_dict.items():
        for node in type_gene_dict[type]:
            gene_type_dict[node] = type
    node2vocab = {}
    for gene_node, type in gene_type_dict.items():
        node2vocab[gene_node] = G_vocab(node=gene_node, type=gene_type_dict[gene_node],
                                        index=gene_node)
    return node2vocab


def get_Network_expression(exp_data_file, type=None):
    exp_data = Pd.read_csv(exp_data_file, index_col=0).values
    if type == 'all':
        exp_data = data_normalize(exp_data)
        exp_data_normal = expression_normal(exp_data)
    if type == 'l2':
        exp_data_normal = expression_normal(exp_data)
    if type == 'normal':
        exp_data_normal = data_normalize(exp_data)
    return exp_data_normal, len(exp_data_normal)

def data_normalize(data):
    standard = StandardScaler()
    epr = standard.fit_transform(data.T)
    return epr.T

def expression_normal(exp_data):
    expression_data = F.normalize(torch.tensor(exp_data), p=2, dim=1)
    return np.array(expression_data, dtype=np.float32)


def get_train_exp_adj_pairs(type_gene_dict, train_set_file, exp_data_file, type=None,
                            directed=True):
    # each gene type class
    all_nodes_vocab = get_node2vocab(type_gene_dict)
    training_nodes_exp, net_work_nodes_num = get_Network_expression(exp_data_file, type=type)
    training_nodes_pairs_index = Pd.read_csv(train_set_file, index_col=0).values

    # decouple graph
    self_loop = np.tile(np.arange(net_work_nodes_num), (2, 1)).T
    TF_TF_row, TF_TF_col = [], []  # 0-0
    TF_Target_row, TF_Target_col = [], []  # 0-1
        # prior directed GRN based on train_sets
    training_original_edges = np.array([edge[:2] for edge in training_nodes_pairs_index if edge[2] == 1], dtype=np.int32)  # the link in train set as prior graph
        # decouple graph
    for edge in training_original_edges:
        if all_nodes_vocab[edge[0]].type == 0 and all_nodes_vocab[edge[1]].type == 0:  # 0-0
            TF_TF_row.append(all_nodes_vocab[edge[0]].index)
            TF_TF_col.append(all_nodes_vocab[edge[1]].index)

        if all_nodes_vocab[edge[0]].type == 0 and all_nodes_vocab[edge[1]].type == 1:  # 0-1
            TF_Target_row.append(all_nodes_vocab[edge[0]].index)
            TF_Target_col.append(all_nodes_vocab[edge[1]].index)
    TF_TF_index = np.vstack((np.array(TF_TF_row), np.array(TF_TF_col)))  # TF_TF directed graph
    TF_Target_index = np.vstack((np.array(TF_Target_row), np.array(TF_Target_col))) # TF_Target directed graph

    #  索引转化为稀疏矩阵存储格式
    if directed == True:
        TF_TF_sqarse_numpy = sp.coo_matrix(
            (np.ones(TF_TF_index.shape[1]), (TF_TF_index[0, :], TF_TF_index[1, :])),
            shape=(net_work_nodes_num, net_work_nodes_num))
        TF_TF_sqarse_tensor = torch.sparse.FloatTensor(
            torch.LongTensor([TF_TF_sqarse_numpy.row, TF_TF_sqarse_numpy.col]),
            torch.FloatTensor(TF_TF_sqarse_numpy.data),
            torch.Size(TF_TF_sqarse_numpy.shape))

        TF_Target_sqarse_numpy = sp.coo_matrix(
            (np.ones(TF_Target_index.shape[1]), (TF_Target_index[0, :], TF_Target_index[1, :])),
            shape=(net_work_nodes_num, net_work_nodes_num))
        TF_Target_sqarse_tensor = torch.sparse.FloatTensor(
            torch.LongTensor([TF_Target_sqarse_numpy.row, TF_Target_sqarse_numpy.col]),
            torch.FloatTensor(TF_Target_sqarse_numpy.data),
            torch.Size(TF_Target_sqarse_numpy.shape))
        training_adj_edges = [TF_TF_sqarse_tensor, TF_Target_sqarse_tensor]
        return training_nodes_exp, training_original_edges, training_adj_edges, training_nodes_pairs_index
    else:
        # Prior GRN graph adj for explicit embedding
        training_undirected_edge = np.concatenate((np.stack((training_original_edges[:, 1],
                                                             training_original_edges[:, 0]), axis=1),
                                                   training_original_edges), axis=0)  #
        training_undirected_edge = np.unique(np.concatenate((training_undirected_edge,
                                                             self_loop), axis=0), axis=0)
        # sub-graph
        TF_TF_Transpose_index = TF_TF_index[[1, 0], :]  # The reverse graph of TF_TF directed graph
        TF_Target_Transpose_index = TF_Target_index[[1, 0], :]  #  The reverse graph of TF_Target directed graph
        # adj based on index
        TF_TF_sqarse_numpy = sp.coo_matrix(
            (np.ones(TF_TF_index.shape[1]), (TF_TF_index[0, :], TF_TF_index[1, :])),
            shape=(net_work_nodes_num, net_work_nodes_num))
        TF_TF_sqarse_tensor = torch.sparse.FloatTensor(
            torch.LongTensor([TF_TF_sqarse_numpy.row, TF_TF_sqarse_numpy.col]),
            torch.FloatTensor(TF_TF_sqarse_numpy.data),
            torch.Size(TF_TF_sqarse_numpy.shape))

        TF_Target_sqarse_numpy = sp.coo_matrix(
            (np.ones(TF_Target_index.shape[1]), (TF_Target_index[0, :], TF_Target_index[1, :])),
            shape=(net_work_nodes_num, net_work_nodes_num))
        TF_Target_sqarse_tensor = torch.sparse.FloatTensor(
            torch.LongTensor([TF_Target_sqarse_numpy.row, TF_Target_sqarse_numpy.col]),
            torch.FloatTensor(TF_Target_sqarse_numpy.data),
            torch.Size(TF_Target_sqarse_numpy.shape))

        TF_TF_Transpose_sqarse_numpy = sp.coo_matrix(
            (np.ones(TF_TF_Transpose_index.shape[1]), (TF_TF_Transpose_index[0, :], TF_TF_Transpose_index[1, :])),
            shape=(net_work_nodes_num, net_work_nodes_num))
        TF_TF_Transpose_sqarse_tensor = torch.sparse.FloatTensor(
            torch.LongTensor([TF_TF_Transpose_sqarse_numpy.row, TF_TF_Transpose_sqarse_numpy.col]),
            torch.FloatTensor(TF_TF_Transpose_sqarse_numpy.data),
            torch.Size(TF_TF_Transpose_sqarse_numpy.shape))

        TF_Target_Transpose_sqarse_numpy = sp.coo_matrix(
            (np.ones(TF_Target_Transpose_index.shape[1]),
             (TF_Target_Transpose_index[0, :], TF_Target_Transpose_index[1, :])),
            shape=(net_work_nodes_num, net_work_nodes_num))
        TF_Target_Transpose_sqarse_tensor = torch.sparse.FloatTensor(
            torch.LongTensor([TF_Target_Transpose_sqarse_numpy.row, TF_Target_Transpose_sqarse_numpy.col]),
            torch.FloatTensor(TF_Target_Transpose_sqarse_numpy.data),
            torch.Size(TF_Target_Transpose_sqarse_numpy.shape))
        training_adj_edges = [TF_TF_sqarse_tensor, TF_Target_sqarse_tensor, TF_TF_Transpose_sqarse_tensor,
                              TF_Target_Transpose_sqarse_tensor]
        return training_nodes_exp, training_undirected_edge, training_adj_edges, training_nodes_pairs_index


def load_positive_negative_pair(data_pair, directed=False):
    data_original_positive_edges = np.array([edge[:2] for edge in data_pair if edge[2] == 1], dtype=np.int32)
    data_original_negative_edges = np.array([edge[:2] for edge in data_pair if edge[2] == 0], dtype=np.int32)  # 会出现相同的索引
    if not directed:  # not为取反操作-输入为False时才会执行
        data_original_positive_edges = np.concatenate((data_original_positive_edges,
                                                       data_original_positive_edges[:, [1, 0]]), axis=0)
        data_original_negative_edges = np.concatenate((data_original_negative_edges,
                                                       data_original_negative_edges[:, [1, 0]]), axis=0)
    return data_original_positive_edges, data_original_negative_edges


def load_data_csv(exp_data_file, train_set_file, validate_set_file, test_data_file, directed=True, self_loop=None):
    gene_exp, gene_net_work_num = get_Network_expression(exp_data_file)
    training_pair = Pd.read_csv(train_set_file, index_col=0).values
    validate_pair = Pd.read_csv(validate_set_file, index_col=0).values
    test_pair = Pd.read_csv(test_data_file, index_col=0).values
    training_pos_edges, training_neg_edges = load_positive_negative_pair(data_pair=training_pair, directed=directed)  # 因为要生成label所以只能是有向的
    validate_pos_edges, validate_neg_edges = load_positive_negative_pair(data_pair=validate_pair, directed=directed)
    test_pos_edges, test_neg_edges = load_positive_negative_pair(data_pair=test_pair, directed=directed)

    all_edges = np.concatenate([training_pos_edges, validate_pos_edges, test_pos_edges], axis=0)
    value = np.ones(all_edges.shape[0])
    gene_network_adj = sp.csr_matrix((value, (all_edges[:, 0], all_edges[:, 1])),
                                     shape=(gene_net_work_num, gene_net_work_num))

    train_pos_values = np.ones(training_pos_edges.shape[0])  # 训练集中正样本的标签
    train_pos_adj = sp.csr_matrix((train_pos_values, (training_pos_edges[:, 0], training_pos_edges[:, 1])),
                                  shape=(gene_net_work_num, gene_net_work_num))  # 训练集正样本组成的adj

    train_neg_values = np.ones(training_neg_edges.shape[0])  # 训练集中正样本的标签
    train_neg_adj = sp.csr_matrix((train_neg_values, (training_neg_edges[:, 0], training_neg_edges[:, 1])),
                                  shape=(gene_net_work_num, gene_net_work_num))  # 训练集正样本组成的adj

    return gene_exp, gene_network_adj, train_pos_adj, train_neg_adj, training_pos_edges, training_neg_edges, \
        validate_pos_edges, validate_neg_edges, \
        test_pos_edges, test_neg_edges






























