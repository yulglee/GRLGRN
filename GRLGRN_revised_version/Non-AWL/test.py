import pandas as Pd
import argparse
from utils2 import get_train_exp_adj_pairs, get_type_genes_dict
from GRLGRN_layer import *
from compute_metrics import evaluate_model
# from compute_metrics_vision import evaluate_model
from parser import get_train_parameter
from parser import parser_add_main_args
import os
def fix_seed(seed): # Make sure that all random number generators are affected by the same seed throughout the code, so that the experimental results are reproducible.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser_add_main_args(parser)
args = parser.parse_args()

fix_seed(args.seed)
Benchmark_root = 'Dataset'
Ground_truth_network = 'Non-Specific Dataset'
Cells_type_name = 'mESC'
Gene_num = 'TFs+500'


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
Target_file = Benchmark_root + '/' + Ground_truth_network + '/' + Cells_type_name + '/' + Gene_num + '/Target.csv'
ground_network_path = Benchmark_root + '/' + Ground_truth_network + '/' + Cells_type_name + '/' + Gene_num + '/Label.csv'
parameter_path =  'model_parameter' + '/' + Ground_truth_network + '/' + Cells_type_name + '/' + Gene_num + '/model_parameter.pth'
# Read Data
type_gene_dict, _ = get_type_genes_dict(exp_data_file, TF_list_file)
Gene_exp, training_adj, training_adj_list, training_pairs_m = get_train_exp_adj_pairs(type_gene_dict,
                                                                                      train_set_file,
                                                                                      exp_data_file,
                                                                                      type=operation,
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
node_embedding = torch.from_numpy(Gene_exp).type(torch.FloatTensor).to(args.device)
# Define model
model = GRLGRN(num_gene=len(Gene_exp), args=args, num_implicit_edge_types=training_adjs_m.size()[0],
               num_implicit_links=args.num_implicit_links,
               input_dim=feature_dim,
               output_dim=args.output_dim,
               device=args.device).to(args.device)
node_feature = node_embedding.detach()
# Load parameter
checkpoint = torch.load(parameter_path)
model.load_state_dict(checkpoint)
metrics = evaluate_model(model, gene_exp=node_feature, gtn_adj=training_adjs_m,
                         adj=torch.LongTensor(training_adj).to(args.device),
                         nums=gene_nums, gene_pair=test_pairs_m, device=args.device, save_path=None)
print("The test set AUROC: {:.4f}, AUPRC: {:.4f}\n".format(metrics['AUROC'], metrics['AUPRC']))
