import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.spmm(input, self.weight)
        output = torch.spmm(adj,support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class VAE_GCN(nn.Module):
    def __init__(self, feature_dim, node_num,fea_red_dim,adj_red_dim):
        super(VAE_GCN, self).__init__()
        self.feature_dim = feature_dim
        self.node_num = node_num
        self.activation = torch.sigmoid
        self.fea_red_dim = fea_red_dim
        self.adj_red_dim = adj_red_dim
        self.embed_dim = 128
        self.gcn_encoder = GraphConvolution(self.feature_dim, self.fea_red_dim)
        self.mu_encoder = GraphConvolution(self.fea_red_dim, self.embed_dim)
        self.logvar_encoder = GraphConvolution(self.fea_red_dim, self.embed_dim)
        self.fea_decode_layer = nn.Linear(self.embed_dim, self.fea_red_dim)
        self.fea_decode_layer2 = nn.Linear(self.fea_red_dim, self.feature_dim)
        torch.nn.init.xavier_normal_(self.fea_decode_layer.weight)
        torch.nn.init.xavier_normal_(self.fea_decode_layer2.weight)

    def encoder(self, fea, adj):
        x = self.activation(self.gcn_encoder(fea, adj))
        mu = self.mu_encoder(x, adj)
        logvar = self.logvar_encoder(x, adj)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        return z

    def fea_decoder(self, z):
        h = self.activation(self.fea_decode_layer(z))
        recon_fea = self.fea_decode_layer2(h)
        return recon_fea

    def adj_decoder(self, z):
        recon_adj = z.matmul(z.t())
        return recon_adj

    def run(self, fea, adj):
        mu, logvar = self.encoder(fea, adj)
        z = self.reparameterize(mu, logvar)
        recon_fea = self.fea_decoder(z)
        recon_adj = self.adj_decoder(z)
        return recon_fea, recon_adj, mu, logvar

    def forward(self,fea, fea_adj, adj,global_weight):
        adj_weight = adj * global_weight
        recon_fea, recon_adj, mu, logvar = self.run(fea, adj)
        adj_BCE_loss = F.binary_cross_entropy_with_logits(recon_adj, adj, reduction='sum', weight=adj_weight)
        fea_BCE_loss = F.binary_cross_entropy_with_logits(recon_fea, fea, reduction='sum')
        KLD_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return adj_BCE_loss + fea_BCE_loss + KLD_loss