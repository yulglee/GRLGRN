import math
from CBAM import *
from contrastive_learning import *
class GRLGRN(nn.Module):
    def __init__(self, num_gene, args, input_dim, output_dim, num_implicit_edge_types, num_implicit_links, device):
        super(GRLGRN, self).__init__()
        self.out_channels = num_implicit_links 
        self.input_dim = input_dim  
        self.out_dim = output_dim 
        self.args = args
        self.batch_normal = nn.BatchNorm1d(input_dim)
        self.normal_layer = nn.LayerNorm(output_dim)
        self.device = device
        layers = []
        layers.append(Iner_pro_Layer(num_implicit_edge_types, num_implicit_links))
        self.layers = nn.ModuleList(layers)[0]
        self.weight = nn.Parameter(torch.Tensor(self.input_dim, self.out_dim))
        self.CBAM = CBAMBlock(channel=num_gene, reduction=10, kernel_size=int(np.sqrt(self.out_dim)))
        self.Pri_predictor = nn.Sequential(
            nn.Linear(self.out_dim, int(self.out_dim/2), bias=True),
            nn.ReLU(),
            nn.Linear(int(self.out_dim/2), 1))
        self.contra_layer = Augement_frature_layer(device=self.device)
        self.contra_loss = Contrast_loss(self.args)
        self.device = device
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
    def gcn_conv(self,X,H):
        X = torch.mm(X, self.weight)
        H = self.adj_norm(H, add=True)
        return torch.mm(H, X)
    def adj_norm(self, H, add):
        H = H.t()
        if add == False:
            H = H*((torch.eye(H.shape[0])==0).type(torch.FloatTensor)).to(self.device)
        else:
            H = H*((torch.eye(H.shape[0])==0).type(torch.FloatTensor)).to(self.device) + torch.eye(H.shape[0]).type(torch.FloatTensor).to(self.device)
        deg = torch.sum(H, dim=1)
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0 
        deg_inv = deg_inv.to(self.device)*torch.eye(H.shape[0]).type(torch.FloatTensor).to(self.device)
        H = torch.mm(deg_inv, H)
        H = H.t()
        return H

    def GRN_link_pre(self, h, gene_pairs): 
        Regulator_embedding = h[gene_pairs[:, 0]]
        Target_embedding = h[gene_pairs[:, 1]]
        dual_interation_embedding = F.relu(Regulator_embedding * Target_embedding)
        pre = self.Pri_predictor(dual_interation_embedding)
        return pre

    def forward(self, X, A, gene_pairs, adj_tensor,  use_con_learning=True, test=False):
        A = A.unsqueeze(0)
        Ws = []
        H, W = self.layers(A)
        Ws.append(W)
        X_list = []
        for i in range(self.out_channels):
            if i == 0:
                X_ = self.normal_layer(F.relu(self.gcn_conv(X, H[i])))
                X_list.append(X_)
            else:
                X_tmp = self.normal_layer(F.relu(self.gcn_conv(X, H[i])))
                X_list.append(X_tmp)
        # mix_attention
        X_ = torch.stack(X_list, dim=0)
        gene_mix_embedding = self.CBAM(X_)
        pre = self.GRN_link_pre(gene_mix_embedding, gene_pairs)
        labeled_pri_pre = torch.sigmoid(pre)  # labeled_pre
        if use_con_learning == True:
            aux_feature, edge_index = self.contra_layer(x=X, edge_idx=adj_tensor.T, drop_rate=[0.2, 0.4], drop_out=True)
            adj_size = len(aux_feature)
            connection_matrix = torch.zeros(adj_size, adj_size).to(self.device)
            connection_matrix[edge_index[0, :], edge_index[1, :]] = 1
            aux_embedding = self.gcn_conv(X=aux_feature, H=connection_matrix)
            contra_loss = self.contra_loss(gene_mix_embedding, aux_embedding)
        else:
            contra_loss = 0
        if test == False:
            return labeled_pri_pre,  contra_loss,  Ws, gene_mix_embedding
        else:
            return labeled_pri_pre, gene_mix_embedding

class Iner_pro_Layer(nn.Module):
    def __init__(self, basic_links_types, implicit_link_channel):
        super(Iner_pro_Layer, self).__init__()
        self.conv1 = basic_links_parameterization(basic_links_types, implicit_link_channel)
        self.conv2 = basic_links_parameterization(basic_links_types, implicit_link_channel)
    def forward(self, A):
        a = self.conv1(A)
        b = self.conv2(A)
        H = torch.bmm(a, b)
        W = [(F.softmax(self.conv1.weight, dim=1)).detach(), (F.softmax(self.conv2.weight, dim=1)).detach()]
        return H, W

class basic_links_parameterization(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(basic_links_parameterization, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels,1,1))
        self.bias = None
        self.scale = nn.Parameter(torch.Tensor([0.1]), requires_grad=False)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.constant_(self.weight, 0.1)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, A):
        A = torch.sum(A*F.softmax(self.weight, dim=1), dim=1) 
        return A
