import numpy
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
import pickle
import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances

#--------------------------------------------


def evaluate_model(model, gene_exp, gtn_adj, adj, nums, gene_pair, device, save_path=None):
    gene_interaction = torch.from_numpy(gene_pair[:, :-1]).to(device)
    labels = gene_pair[:, -1]
    model.eval()
    with torch.no_grad():  # 在这个上下文中，不计算梯度
        score, embedding = model(X=gene_exp, A=gtn_adj, gene_pairs=gene_interaction,
                                 adj_tensor=adj, use_con_learning=False, test=True)
    embedding = embedding.detach().cpu().numpy()
    if save_path is not None:
        if not os.path.exists(save_path):
            # 如果路径不存在，则创建文件夹
            os.makedirs(save_path)
            np.save(save_path + '/epoch_200-embedding.npy', embedding)
        else:
            np.save(save_path + '/epoch_200-embedding.npy', embedding)
    score = score.detach().cpu().numpy()
    metrics = compute_metrics(y_true=labels, y_pred=score, save_path=save_path)
    return metrics

def auc_of_epoch(num_epochs, validation_metrics, save_path):
    if save_path is not None:
        if not os.path.exists(save_path):
            # 如果路径不存在，则创建文件夹
            os.makedirs(save_path)
    plt.plot(range(1, num_epochs + 1), validation_metrics, label='Validation-epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Validation_metrics')
    plt.title('Validation AUC over Epochs')
    plt.legend()
    plt.savefig(save_path + '/Validation AUC over Epochs.png')
    # torch.save(validation_metrics, save_path+'/Validation AUC_list over Epochs.pkl')
    with open(save_path+'/Validation AUC_list over Epochs.pkl', 'wb') as file:
        pickle.dump(validation_metrics, file)
    # plt.show()

def compute_metrics(y_true, y_pred, save_path):

    y_p = y_pred
    y_p = y_p.flatten()
    y_t = y_true.flatten().astype(int)
    # 计算AUROC
    fpr, tpr, _ = roc_curve(y_t, y_p)
    roc_auc = auc(fpr, tpr)

    # 计算 AUPRC
    precision, recall, _ = precision_recall_curve(y_t, y_p)
    prc_auc = auc(recall, precision)
    if save_path is not None:
        if not os.path.exists(save_path):
            # 如果路径不存在，则创建文件夹
            os.makedirs(save_path)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(save_path)
        plt.legend(loc="lower right")
        plt.savefig(save_path + '/ROC_cuve.png')
        # plt.show()

    return {
        'AUROC': roc_auc,
        'AUPRC': prc_auc
    }



def get_data_point_plot(data1, epoch, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if epoch == 1:
        feature = data1.detach().cpu().numpy()
        np.save(save_path + '/epoch_{}-embedding.npy'.format(epoch), feature)
    curve_path = os.path.join(save_path, 'epoch{}_scatter_plot.png'.format(epoch))
    data1_np = data1.detach().cpu().numpy()

    # 使用PCA进行降维
    pca = PCA(n_components=2)

    merged_data_pca1 = pca.fit_transform(data1_np)  # 蓝色

    # 提取降维后的数据的 x 和 y 坐标
    x = merged_data_pca1[:, 0]
    y = merged_data_pca1[:, 1]
    #
    # # 绘制散点图
    plt.scatter(x, y, label='Gene embedding', s=10)  # 数据1的散点图
    # 添加标题和标签

    # plt.title('epoch{}-AUROC:{:.3f}'.format(epoch, AUROC))
    plt.title('Feature plot_lr')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='upper right')  # 显示图例
    plt.savefig(curve_path)
    # plt.show()
    plt.close('all')