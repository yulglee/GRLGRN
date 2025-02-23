import torch
import random
import numpy as np
import pandas as Pd
import argparse
import os
import torch.nn.functional as F
from utils2 import get_train_exp_adj_pairs, get_type_genes_dict
from GRLGRN_layer import GRLGRN
from compute_metrics import evaluate_model, get_data_point_plot, auc_of_epoch
from parser import get_train_parameter
from parser import parser_add_main_args
from utils import get_batches

def fix_seed(seed):  # 确保在整个代码中，所有的随机数生成器都受到相同的种子的影响，从而使实验结果可重复。
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
Ground_truth_network = [
    # 'STRING Dataset',
    # 'Specific Dataset',
    'Non-Specific Dataset'
]
Cells_type_name = [
                    'hESC',
                   'hHEP',
                   'mDC',
                   'mESC',
                   'mHSC-E',
                   'mHSC-GM',
                   'mHSC-L'
                   ]
Gene_num = [
            'TFs+500',
            'TFs+1000'
            ]

AUROC_result_500 = Pd.DataFrame()
AUROC_result_1000 = Pd.DataFrame()
AUPRC_result_500 = Pd.DataFrame()
AUPRC_result_1000 = Pd.DataFrame()

for _, network_name in enumerate(Ground_truth_network):
    for _, cells_name in enumerate(Cells_type_name):
        if network_name == 'Lofgof Dataset':
            cells_name = 'mESC'
        cells_root = '/' + cells_name
        for count, gene_name in enumerate(Gene_num):
            gene_root = '/' + gene_name
            # 加载文件路径
            print('The training process is {}'.format(network_name + cells_root + gene_root))
            if network_name =='STRING Dataset':
                operation = 'l2'
            else:
                operation = 'all'
            # model parameter
            args.lr, args.alpha_loss, args.beta_loss, contra_learning, args.epoch = get_train_parameter(network_name)
            # file path
            exp_data_file = '/data/stu1/lyl_pycharm_project/A_GRLGRN_github/'+ Benchmark_root + network_name + cells_root + gene_root + '/BL--ExpressionData.csv'
            Gene_file = '/data/stu1/lyl_pycharm_project/A_GRLGRN_github/'+  Benchmark_root + network_name + cells_root + gene_root + '/Target.csv'
            TF_list_file = '/data/stu1/lyl_pycharm_project/A_GRLGRN_github/'+  Benchmark_root + network_name + cells_root + gene_root + '/TF.csv'
            train_set_file = '/data/stu1/lyl_pycharm_project/A_GRLGRN_github/'+  Benchmark_root + network_name + cells_root + gene_root + '/Train_set.csv'
            validata_set_file = '/data/stu1/lyl_pycharm_project/A_GRLGRN_github/'+  Benchmark_root + network_name + cells_root + gene_root + '/Validation_set.csv'
            test_set_file = '/data/stu1/lyl_pycharm_project/A_GRLGRN_github/'+  Benchmark_root + network_name + cells_root + gene_root + '/Test_set.csv'
            parameter_path = '/data/stu1/lyl_pycharm_project/A_GRLGRN_github/Use_aw_training/Use_contrastive_learning/' + 'model_parameter/' + network_name +  cells_root + gene_root + '/model_parameter.pth'
            # /data/stu1/lyl_pycharm_project/A_GRLGRN_github/Use_aw_training/Use_contrastive_learning/model_parameter/Non-Specific Dataset/hESC/TFs+500/model_parameter.pth
                # gene type
            type_gene_dict, _ = get_type_genes_dict(exp_data_file, TF_list_file)
                # load input data
            Gene_exp, training_adj, training_adj_list, training_pairs_m = get_train_exp_adj_pairs(type_gene_dict,
                                                                                                  train_set_file,
                                                                                                  exp_data_file,type=operation,
                                                                                                  directed=False)
            test_pairs_m = Pd.read_csv(test_set_file, index_col=0).values
            num_nodes = Gene_exp.shape[0]
            feature_dim = Gene_exp.shape[1]
            gene_nums = num_nodes
            self_loop = False
            if self_loop == False:
                for i, edge in enumerate(training_adj_list):
                    if i == 0:
                        A = edge.to_dense().type(torch.FloatTensor).unsqueeze(0)
                    else:
                        A = torch.cat([A, edge.to_dense().type(torch.FloatTensor).unsqueeze(0)], dim=0)

            A = torch.cat([A, torch.eye(num_nodes).type(torch.FloatTensor).unsqueeze(0)], dim=0)
            training_adjs_m = A.to(args.device)
            # save model parameter path
            Result_root = 'Result/'
            node_embedding = torch.from_numpy(Gene_exp).type(torch.FloatTensor).to(args.device)
            model = GRLGRN(num_gene=len(Gene_exp), args=args, num_implicit_edge_types=training_adjs_m.size()[0],
                            num_implicit_links=args.num_implicit_links,
                            input_dim=feature_dim,
                            output_dim=args.output_dim,
                            device=args.device).to(args.device)
            node_feature = node_embedding.detach()
            checkpoint = torch.load(parameter_path)
            model.load_state_dict(checkpoint)
            metrics = evaluate_model(model, gene_exp=node_feature, gtn_adj=training_adjs_m,
                                     adj=torch.LongTensor(training_adj).to(args.device),
                                     nums=gene_nums, gene_pair= test_pairs_m, device=args.device, save_path=None)
            print("The test set AUROC: {:.4f}, AUPRC: {:.4f}\n".format(metrics['AUROC'], metrics['AUPRC']))
            if count == 0:
                AUROC_result_500.loc[cells_name, network_name] = metrics['AUROC']
                AUPRC_result_500.loc[cells_name, network_name] = metrics['AUPRC']

            if count == 1:
                AUROC_result_1000.loc[cells_name, network_name] = metrics['AUROC']
                AUPRC_result_1000.loc[cells_name, network_name] = metrics['AUPRC']

        if network_name == 'Lofgof Dataset' and cells_name == 'mESC':
            break
