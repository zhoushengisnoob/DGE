import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

class VAE(nn.Module):
    def __init__(self,feature_dim,node_num,fea_red_dim,adj_red_dim,feature_bernoulli):
        super(VAE, self).__init__()
        self.feature_bernoulli=feature_bernoulli
        self.feature_dim = feature_dim
        self.node_num = node_num
        self.activation = torch.sigmoid
        self.fea_red_dim = fea_red_dim
        self.adj_red_dim = adj_red_dim
        self.embed_dim = 128
        self.fea_dim_reduce = nn.Linear(self.feature_dim, self.fea_red_dim)
        self.adj_dim_reduce = nn.Linear(self.node_num, self.adj_red_dim)
        self.mu_layer = nn.Linear(self.fea_red_dim + self.adj_red_dim, self.embed_dim)
        self.logvar_layer = nn.Linear(self.fea_red_dim + self.adj_red_dim, self.embed_dim)
        self.fea_decode_layer = nn.Linear(self.embed_dim, self.fea_red_dim)
        self.fea_decode_layer2 = nn.Linear(self.fea_red_dim, self.feature_dim)

        torch.nn.init.xavier_normal_(self.fea_dim_reduce.weight)
        torch.nn.init.xavier_normal_(self.adj_dim_reduce.weight)
        torch.nn.init.xavier_normal_(self.mu_layer.weight)
        torch.nn.init.xavier_normal_(self.logvar_layer.weight)
        torch.nn.init.xavier_normal_(self.fea_decode_layer.weight)
        torch.nn.init.xavier_normal_(self.fea_decode_layer2.weight)

    def encoder(self, fea, adj):
        h_fea = self.activation(self.fea_dim_reduce(fea))
        h_adj = self.activation(self.adj_dim_reduce(adj))
        h_fea_adj = torch.cat((h_fea, h_adj), dim=1)
        mu = self.mu_layer(h_fea_adj)
        logvar = self.logvar_layer(h_fea_adj)
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


    def forward(self,fea, fea_adj,adj,global_weight):
        adj_weight = adj * global_weight
        recon_fea, recon_adj, mu, logvar = self.run(fea, fea_adj)
        adj_BCE_loss = F.binary_cross_entropy_with_logits(recon_adj, adj, reduction='sum', weight=adj_weight)
        if self.feature_bernoulli:
            fea_BCE_loss = F.binary_cross_entropy_with_logits(recon_fea, fea, reduction='sum')
        else:
            fea_BCE_loss = F.mse_loss(recon_fea,F.normalize(fea),reduction='sum')
        KLD_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return adj_BCE_loss,fea_BCE_loss,KLD_loss