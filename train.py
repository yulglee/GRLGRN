import torch
import random
import numpy as np
import pandas as Pd
import argparse
import torch.nn.functional as F
from utils2 import get_train_exp_adj_pairs, get_type_genes_dict
from GRLGRN_layer import GRLGRN
from compute_metrics import evaluate_model
from parser import get_train_parameter
from parser import parser_add_main_args
from utils import get_batches
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
fix_seed(3047)
parser = argparse.ArgumentParser()
parser_add_main_args(parser)
args = parser.parse_args()


Benchmark_root = 'Dataset/'
Ground_truth_network = 'STRING Dataset'
Cells_type_name = 'mESC'
Gene_num = 'TFs+500'

args.lr, args.alpha_loss, args.beta_loss, contra_learning,  args.epoch = get_train_parameter(Ground_truth_network)
if Ground_truth_network == 'STRING Dataset':
    operation = 'l2'
else:
    operation = 'all'
# Dara path
exp_data_file = Benchmark_root + '/' + Ground_truth_network + '/' + Cells_type_name + '/' + Gene_num + '/BL--ExpressionData.csv'
Gene_file = Benchmark_root + '/' + Ground_truth_network + '/' + Cells_type_name + '/' + Gene_num + '/Target.csv'
TF_list_file = Benchmark_root + '/' + Ground_truth_network + '/' + Cells_type_name + '/' + Gene_num + '/TF.csv'
train_set_file = Benchmark_root + '/' + Ground_truth_network + '/' + Cells_type_name + '/' + Gene_num + '/Train_set.csv'
validata_set_file = Benchmark_root + '/' + Ground_truth_network + '/' + Cells_type_name + '/' + Gene_num + '/Validation_set.csv'
test_set_file = Benchmark_root + '/' + Ground_truth_network + '/' + Cells_type_name + '/' + Gene_num + '/Test_set.csv'

type_gene_dict, _ = get_type_genes_dict(exp_data_file, TF_list_file)
Gene_exp, training_adj, training_adj_list, training_pairs_m = get_train_exp_adj_pairs(type_gene_dict,
                                                                                      train_set_file,
                                                                                      exp_data_file, type=operation,
                                                                                      directed=False)
validata_pairs_m = Pd.read_csv(validata_set_file, index_col=0).values
test_pairs_m = Pd.read_csv(test_set_file, index_col=0).values
num_nodes = Gene_exp.shape[0]
feature_dim = Gene_exp.shape[1]
gene_nums = num_nodes
for i, edge in enumerate(training_adj_list):
    if i == 0:
        A = edge.to_dense().type(torch.FloatTensor).unsqueeze(0)
    else:
        A = torch.cat([A, edge.to_dense().type(torch.FloatTensor).unsqueeze(0)], dim=0)
A = torch.cat([A, torch.eye(num_nodes).type(torch.FloatTensor).unsqueeze(0)], dim=0)
training_adjs_m = A.to(args.device)
# 存储路径
node_embedding = torch.from_numpy(Gene_exp).type(torch.FloatTensor).to(args.device)
model = GRLGRN(num_gene=len(Gene_exp), args=args, num_implicit_edge_types=training_adjs_m.size()[0],
               num_implicit_links=args.num_implicit_links,
               input_dim=feature_dim,
               output_dim=args.output_dim,
               device=args.device).to(args.device)
node_feature = node_embedding.detach()
optim_function = torch.optim.Adam(model.parameters(), lr=args.lr) #args.lr
Validata_auc = []
for epoch in range(1, args.epoch+1):  # args.epoch
    running_loss = 0.0
    np.random.shuffle(training_pairs_m)
    labeled_training_interation = get_batches(training_pairs_m, 1024)
    for i, data in enumerate(labeled_training_interation):
        training_batch_edges = torch.from_numpy(data[0]).to(args.device)
        training_batch_labels = torch.from_numpy(data[1]).to(args.device)
        model.train()
        labeled_pri_pre, contra_loss, Ws, embedding = model(X=node_feature, A=training_adjs_m,
                                                            gene_pairs=training_batch_edges,
                                                            adj_tensor=torch.LongTensor(training_adj).to(args.device),
                                                            test=False, use_con_learning=contra_learning)
        pre_cov_loss = F.binary_cross_entropy(labeled_pri_pre, training_batch_labels.to(torch.float32).view(-1, 1))
        loss = args.alpha_loss * pre_cov_loss + args.beta_loss * contra_loss
        running_loss += loss.item()
        model.zero_grad()
        loss.backward()
        optim_function.step()
    metrics = evaluate_model(model, gene_exp=node_feature, gtn_adj=training_adjs_m,
                             adj=torch.LongTensor(training_adj).to(args.device),
                             nums=gene_nums, gene_pair=validata_pairs_m, device=args.device, save_path=None)
    print('Epoch:{}'.format(epoch),
          'train loss:{}'.format(running_loss), 'pre_conv_loss:{}'.format(pre_cov_loss),
          'contra_loss:{}'.format(contra_loss),
          "AUROC: {:.4f},AUPRC: {:.4f}".format(metrics['AUROC'], metrics['AUPRC']))
metrics = evaluate_model(model, gene_exp=node_feature, gtn_adj=training_adjs_m,
                                     adj=torch.LongTensor(training_adj).to(args.device),
                                     nums=gene_nums, gene_pair=test_pairs_m , device=args.device, save_path=None)
print("The test set AUROC: {:.4f}, AUPRC: {:.4f}\n".format(metrics['AUROC'], metrics['AUPRC']))

