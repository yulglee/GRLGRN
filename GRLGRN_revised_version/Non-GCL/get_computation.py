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
import time
import subprocess
import time

def get_gpu_memory():
    try:
        # 执行 nvidia-smi 查询命令，获取 GPU 内存使用情况
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,nounits,noheader'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # 解析返回的内存信息
        memory_info = result.stdout.strip().split('\n')[0].split(', ')
        total_memory = int(memory_info[0])  # 总内存（MB）
        used_memory = int(memory_info[1])  # 已用内存（MB）
        free_memory = int(memory_info[2])  # 空闲内存（MB）

        return total_memory, used_memory, free_memory

    except Exception as e:
        print(f"Error retrieving GPU memory: {e}")
        return None, None, None
def fix_seed(seed):  # 确保在整个代码中，所有的随机数生成器都受到相同的种子的影响，从而使实验结果可重复。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
fix_seed(3047)
parser = argparse.ArgumentParser()
parser_add_main_args(parser)
args = parser.parse_args()


Benchmark_root = '/data/stu1/lyl_pycharm_project/A_GRLGRN_github/Dataset/'
Ground_truth_network = [
    # 'STRING Dataset',
    # 'Specific Dataset',
    'Non-Specific Dataset',
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
            # 'TFs+500',
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
            exp_data_file = Benchmark_root + network_name + cells_root + gene_root + '/BL--ExpressionData.csv'
            Gene_file = Benchmark_root + network_name + cells_root + gene_root + '/Target.csv'
            TF_list_file = Benchmark_root + network_name + cells_root + gene_root + '/TF.csv'
            train_set_file = Benchmark_root + network_name + cells_root + gene_root + '/Train_set.csv'
            validata_set_file = Benchmark_root + network_name + cells_root + gene_root + '/Validation_set.csv'
            test_set_file = Benchmark_root + network_name + cells_root + gene_root + '/Test_set.csv'
            # load data
                # gene type
            type_gene_dict, _ = get_type_genes_dict(exp_data_file, TF_list_file)
                # load input data
            Gene_exp, training_adj, training_adj_list, training_pairs_m = get_train_exp_adj_pairs(type_gene_dict,
                                                                                                  train_set_file,
                                                                                                  exp_data_file,type=operation,
                                                                                                  directed=False)
            validata_pairs_m = Pd.read_csv(validata_set_file, index_col=0).values
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
            curve_path = 'Curve/' + network_name + cells_root + gene_root
            parameter_root = 'model_parameter/' + network_name + cells_root + gene_root
            Result_root = 'Result/'
            node_embedding = torch.from_numpy(Gene_exp).type(torch.FloatTensor).to(args.device)
            model = GRLGRN(num_gene=len(Gene_exp), args=args, num_implicit_edge_types=training_adjs_m.size()[0],
                            num_implicit_links=args.num_implicit_links,
                            input_dim=feature_dim,
                            output_dim=args.output_dim,
                            device=args.device).to(args.device)
            node_feature = node_embedding.detach()
            optim_function = torch.optim.Adam(model.parameters(), lr=args.lr) #args.lr
            Validata_auc = []
            # contra_learning = False
            # 记录训练开始时间
            start_time = time.time()

            # 获取初始的GPU内存信息
            total_memory_start, used_memory_start, free_memory_start = get_gpu_memory()

            # 记录训练前的GPU占用
            gpu_memory_start = used_memory_start

            max_memory = 0  # 记录训练过程中最大占用

            for epoch in range(1, args.epoch + 1):
                running_loss = 0.0
                np.random.shuffle(training_pairs_m)
                labeled_training_interation = get_batches(training_pairs_m, 1024)

                for i, data in enumerate(labeled_training_interation):
                    training_batch_edges = torch.from_numpy(data[0]).to(args.device)
                    training_batch_labels = torch.from_numpy(data[1]).to(args.device)
                    model.train()

                    labeled_pri_pre, contra_loss, Ws, embedding = model(
                        X=node_feature, A=training_adjs_m,
                        gene_pairs=training_batch_edges,
                        adj_tensor=torch.LongTensor(training_adj).to(args.device),
                        test=False, use_con_learning=contra_learning
                    )

                    pre_cov_loss = F.binary_cross_entropy(
                        labeled_pri_pre, training_batch_labels.to(torch.float32).view(-1, 1)
                    )
                    loss = args.alpha_loss * pre_cov_loss + args.beta_loss * contra_loss
                    running_loss += loss.item()

                    model.zero_grad()
                    loss.backward()
                    optim_function.step()

                    # 获取当前 GPU 内存使用情况
                    _, used_memory_current, _ = get_gpu_memory()
                    max_memory = max(max_memory, used_memory_current)


                print(
                    f'Epoch:{epoch}', f'train loss:{running_loss}', f'pre_conv_loss:{pre_cov_loss}',
                    f'contra_loss:{contra_loss}'
                )

            # 训练结束后统计总时间和GPU占用
                end_time = time.time()
                total_time = end_time - start_time

                # 获取训练结束后的GPU内存信息
                total_memory_end, used_memory_end, _ = get_gpu_memory()
                #  model test_evaluate
                if not os.path.exists(parameter_root):
                    # 如果路径不存在，则创建文件夹
                    os.makedirs(parameter_root)
                log_path = parameter_root + "/training_log.txt"
                with open(log_path, "w") as f:
                    f.write(f"Total training time: {total_time:.2f} seconds\n")
                    f.write(f"Initial GPU memory usage: {gpu_memory_start} MB\n")
                    f.write(f"Final GPU memory usage: {used_memory_end} MB\n")
                    f.write(f"Peak GPU memory usage: {max_memory} MB\n")

                print(f"Training log saved to {log_path}")
                # 将训练信息写入txt文件



