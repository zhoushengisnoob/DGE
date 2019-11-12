import torch.nn as nn
import torch
import torch.nn.functional as F

class VAE_SAGE(nn.Module):
    def __init__(self,feature_dim,node_num):
        super(VAE_SAGE, self).__init__()
        self.eps=5e-6
        self.feature_dim = feature_dim
        self.node_num = node_num
        self.hidden_dim = 256
        self.embed_dim = 128
        self.fea2hidden = nn.Linear(self.feature_dim, self.hidden_dim)
        self.sage_layer = nn.Linear(self.hidden_dim*2,self.hidden_dim)
        self.mu_layer = nn.Linear(self.hidden_dim, self.embed_dim)
        self.logvar_layer = nn.Linear(self.hidden_dim, self.embed_dim)
        self.fea_decode_layer = nn.Linear(self.embed_dim, self.hidden_dim)
        self.fea_decode_layer2 = nn.Linear(self.hidden_dim, self.feature_dim)

        torch.nn.init.normal_(self.fea2hidden.weight)
        torch.nn.init.normal_(self.sage_layer.weight)
        torch.nn.init.normal_(self.mu_layer.weight)
        torch.nn.init.normal_(self.logvar_layer.weight)
        torch.nn.init.normal_(self.fea_decode_layer.weight)

    def encoder(self, fea, adj):
        h_fea = self.fea2hidden(fea)
        aggre_fea = F.normalize(adj,p=1).mm(h_fea)# mean aggregator of the nbr feature
        sage = self.sage_layer(torch.cat((aggre_fea,h_fea),dim=1))
        sage = torch.sigmoid(sage)
        sage = F.normalize(sage)
        mu = self.mu_layer(sage)
        logvar = self.logvar_layer(sage)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        return z

    def fea_decoder(self, z):
        h = F.relu(self.fea_decode_layer(z))
        recon_fea = torch.sigmoid(self.fea_decode_layer2(h))
        return recon_fea

    def adj_decoder(self, z):
        recon_adj = torch.sigmoid(z.matmul(z.t()))
        return recon_adj

    def forward(self, fea, adj):
        mu, logvar = self.encoder(fea, adj)
        z = self.reparameterize(mu, logvar)
        recon_fea = self.fea_decoder(z)
        recon_adj = self.adj_decoder(z)
        return recon_fea, recon_adj, mu, logvar