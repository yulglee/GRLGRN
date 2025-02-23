import torch
import random
import torch.nn as nn
from torch_geometric.utils import dropout_adj
from torch_scatter import scatter_mean, scatter_add, scatter_std
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
class Augement_frature_layer(nn.Module):
    def __init__(self, device: str = 'cuda'):
        super(Augement_frature_layer, self).__init__()

        self.device = device
    def aug_feature(self, x, drop_prob=0.2, dropout=False):
        torch.manual_seed(42)
        random.seed(42)
        if dropout == True:
            drop_mask = torch.empty(
                (x.size(1),),
                dtype=torch.float32,
                device=x.device).uniform_(0, 1) < drop_prob  # 只有二成的概率被丢弃
            x = x.clone()
            x[:, drop_mask] = 0  # 只有二成的概率被丢弃
            return x
        else:
            return x
    def aug_link(self, edge_index:torch.Tensor, drop_prob=0.2):
        edge_index = dropout_adj(edge_index, p=drop_prob)[0]
        return edge_index

    def forward(self, x: torch.Tensor, edge_idx: torch.Tensor, drop_rate: list, drop_out=True):
        aug_edge_index = dropout_adj(edge_index=edge_idx, p=drop_rate[0])[0]
        aug_feature = self.aug_feature(x=x, drop_prob=drop_rate[1], dropout=drop_out)
        return aug_feature, aug_edge_index

class Contrast_loss(nn.Module):
    def __init__(self, args):
        super(Contrast_loss, self).__init__()
        self.hidden_dim = args.output_dim
        self.hidden_channel = args.num_implicit_links
        self.feature_dim = self.hidden_channel * self.hidden_dim
        self.output_dim = args.output_dim
        self.tau = args.tau
        self.augment_feature = nn.Linear(self.feature_dim, self.output_dim)
        self.fc1 = nn.Linear(self.output_dim, self.output_dim)
        self.fc2 = nn.Linear(self.output_dim, self.output_dim)

    def forward(self, pri_embedding: torch.Tensor, aux_embedding: torch.Tensor,
                mean: bool = True):
        h1 = self.projection(pri_embedding)
        h2 = self.projection(aux_embedding)  # 将基因的特征映射到节点同一空间中
        l1 = self.semi_loss(h1, h2)  # 计算对比损失函数第一项
        l2 = self.semi_loss(h2, h1)  # 计算对比损失函数第二项

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()
        return ret
    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)  # lamba定义匿名函数，输入参数为x，后续程序为冒号之后的
        refl_sim = f(self.sim(z1, z1))  # self.sim(z1, z1)：同一个试图内所有基因对之间的相似度
        between_sim = f(self.sim(z1, z2))  # self.sim(z1, z2)：同一个试图内不同基因对之间的相似度

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

