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
    with torch.no_grad():
        score, embedding = model(X=gene_exp, A=gtn_adj, gene_pairs=gene_interaction,
                                 adj_tensor=adj, use_con_learning=False, test=True)
    embedding = embedding.detach().cpu().numpy()
    if save_path is not None:
        if not os.path.exists(save_path):
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

def compute_metrics(y_true, y_pred, save_path):

    y_p = y_pred
    y_p = y_p.flatten()
    y_t = y_true.flatten().astype(int)
    fpr, tpr, _ = roc_curve(y_t, y_p)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_t, y_p)
    prc_auc = auc(recall, precision)
    if save_path is not None:
        if not os.path.exists(save_path):
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


