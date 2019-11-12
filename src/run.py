import warnings
import torch
from torch import optim
import numpy as np
from utils import load_data, evaluation
#from layers2.VAE_adj import VAE_adj
#from layers2.VAE_feature import VAE_feature
from layers.VAE import VAE
#from layers2.VAE_two_Z import VAE_two_Z
#from layers2.VAE_batch import VAE_batch
import argparse
import os
from tqdm import tqdm




if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

warnings.filterwarnings("ignore")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate')
    parser.add_argument('--decay', type=float, default=0.1, help='Weighted Decay')
    parser.add_argument('--epoch', type=int, default=2000, help='Number of epoch')
    parser.add_argument('--fea_red_dim', type=int, default=256, help='Feature Reduction Dimension')
    parser.add_argument('--adj_red_dim', type=int, default=256, help='Adjacent Reduction Dimension')
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset name')
    parser.add_argument('--batch', type=int, default=0, help='Batch Size')
    parser.add_argument('--layer',type=str,default='VAE',help='layer for VAE')
    parser.add_argument('--round',type=int,default=1,help='round of mean f1 score')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID for use')
    parser.add_argument('--l_adj', type=float, default=1, help='Lambda of adj')
    parser.add_argument('--l_fea', type=float, default=1, help='Lambda of fea')
    args = parser.parse_args()
    lambda_adj = args.l_adj
    lambda_fea = args.l_fea
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.dataset=='pubmed':
        feature_bernoulli=False
    else:
        feature_bernoulli = True
    dataset_name = args.dataset
    print('Loading ',dataset_name,' data')
    adj_mat, feature_mat, GT= load_data.load_network(dataset_name)
    adj_mat = np.array(adj_mat)
    feature_dim = feature_mat.shape[1]
    node_num = adj_mat.shape[0]

    if args.batch==0:
        batch_size = node_num
    else:
        batch_size = args.batch

    adj_mat = adj_mat.dot(adj_mat.T)
    adj_mat[adj_mat > 0] = 1
    fea = torch.FloatTensor(feature_mat).to(device)
    adj = torch.FloatTensor(adj_mat).to(device)
    global_weight = (node_num * node_num - np.sum(adj_mat)) / np.sum(adj_mat)
    print('Global Weight:',global_weight)
    mean_best=[]
    for i in range(args.round):
        if args.layer == 'adj':
            model = VAE_adj(feature_dim,node_num,args.fea_red_dim,args.adj_red_dim).to(device)
        elif args.layer == 'feature':
            model = VAE_feature(feature_dim,node_num,args.fea_red_dim,args.adj_red_dim).to(device)
        elif args.layer == 'GCN':
            model =VAE_GCN(feature_dim,node_num,args.fea_red_dim,args.adj_red_dim).to(device)
        elif args.layer =='SAGE':
            model = VAE_SAGE(feature_dim,node_num,args.fea_red_dim,args.adj_red_dim).to(device)
        elif args.layer =='VAE':
            model = VAE(feature_dim,node_num,args.fea_red_dim,args.adj_red_dim,feature_bernoulli).to(device)
        elif args.layer == 'VAE_two_Z':
            model = VAE_two_Z(feature_dim, node_num, args.fea_red_dim, args.adj_red_dim,feature_bernoulli).to(device)
        else:
            print('Wrong layer setting')
            exit()

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
        print('Start Training:')
        best_micro=0
        index=list(range(node_num))
        for epoch in tqdm(range(args.epoch)):
            np.random.shuffle(index)
            index_s = index[:batch_size]
            fea_s = fea[index_s,:]
            fea_adj = adj[index_s,:]
            adj_s = fea_adj[:,index_s]
            model.train()
            optimizer.zero_grad()
            adj_BCE_loss,fea_BCE_loss,KLD_loss = model(fea_s,fea_adj,adj_s, global_weight)
            loss = lambda_adj*adj_BCE_loss+lambda_fea*fea_BCE_loss+KLD_loss
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                model.eval()
                mu, logvar = model.encoder(fea,adj)
                macro, micro, accuracy = evaluation.eval_classification(mu.cpu().detach().numpy(), GT)
                best_micro = max(micro, best_micro)
                print('Loss:',loss.item(),'Micro:',micro,'Macro',macro)
        mean_best.append(best_micro)
    # recon_fea, recon_adj, mu, logvar = model(torch.FloatTensor(feature_mat).to(device),torch.FloatTensor(adj_mat).to(device))
    # mu = mu.cpu().detach().numpy()
    # logvar = logvar.cpu().detach().numpy()
    # recon_fea = torch.sigmoid(recon_fea.cpu().detach()).numpy()
    # np.savetxt('./results/' + dataset_name + '_both_mean.txt', mu)
    # np.savetxt('./results/' + dataset_name + '_both_variance.txt', logvar)
    # np.savetxt('./results/' + dataset_name + '_both_recon_fea.txt', recon_fea)
