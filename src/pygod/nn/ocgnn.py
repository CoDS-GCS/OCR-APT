import torch
import torch.nn as nn
from torch_geometric.nn import GCN
torch.use_deterministic_algorithms(True)
import matplotlib.pyplot as plt
from numpy import sin, cos, pi, linspace
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
class OCGNNBase(nn.Module):
    """
    One-Class Graph Neural Networks for Anomaly Detection in
    Attributed Networks

    OCGNN is an anomaly detector that measures the
    distance of anomaly to the centroid, in a similar fashion to the
    support vector machine, but in the embedding space after feeding
    towards several layers of GCN.

    See :cite:`wang2021one` for details.

    Parameters
    ----------
    in_dim : int
        Input dimension of model.
    hid_dim :  int, optional
        Hidden dimension of model. Default: ``64``.
    num_layers : int, optional
        Total number of layers in model. Default: ``2``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    backbone : torch.nn.Module
        The backbone of the deep detector implemented in PyG.
        Default: ``torch_geometric.nn.GCN``.
    beta : float, optional
        The weight between the reconstruction loss and radius.
        Default: ``0.5``.
    warmup : int, optional
        The number of epochs for warm-up training. Default: ``2``.
    eps : float, optional
        The slack variable. Default: ``0.001``.
    **kwargs
        Other parameters for the backbone model.
    """

    def __init__(self,
                 in_dim,
                 hid_dim,
                 num_layers=2,
                 dropout=0.,
                 act=torch.nn.functional.relu,
                 backbone=GCN,
                 beta=0.5,
                 warmup=2,
                 eps=0.001,
                 patience=5,
                 max_delta=0.2,
                 visualize=False,
                 **kwargs):
        super(OCGNNBase, self).__init__()

        self.beta = beta
        self.warmup = warmup
        self.eps = eps

        self.first_epoch = True
        self.patience = patience
        self.max_validation_auc = 0
        self.max_validation_f1_score = 0
        self.counter = 0
        self.max_validation_TNR = 0
        self.max_validation_TPR = 0
        self.TPR_baseline = 0.5
        self.TNR_baseline = 0.5
        self.early_stop = 0
        self.max_delta = max_delta

        self.gnn = backbone(in_channels=in_dim,
                            hidden_channels=hid_dim,
                            num_layers=num_layers,
                            out_channels=hid_dim,
                            dropout=dropout,
                            act=act,
                            **kwargs)
        # self.reset_layer_parameters(self.gnn)
        self.r = 0
        self.c = torch.zeros(hid_dim)

        self.emb = None
    def reset_layer_parameters(self,gnn):
        for layer in gnn.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        gnn.reset_parameters()
    def forward(self, x, edge_index):
        """
        Forward computation.

        Parameters
        ----------
        x : torch.Tensor
            Input attribute embeddings.
        edge_index : torch.Tensor
            Edge index.

        Returns
        -------
        emb : torch.Tensor
            Output embeddings.
        """

        self.emb = self.gnn(x, edge_index)
        return self.emb
    # def loss_func_val(self, emb, label=None, fig_title=None, visualize=False):
    #     early_stop =False
    #     label = label.reshape(-1)
    #     emb_benign = emb[~label]
    #     dist_benign = torch.sqrt(torch.sum(torch.pow(emb_benign - self.c, 2), 1))
    #     avg_dist_benign = torch.mean(torch.relu(dist_benign))
    #     emb_malicious = emb[label]
    #     dist_malicious = torch.sqrt(torch.sum(torch.pow(emb_malicious - self.c, 2), 1))
    #     avg_dist_malicious = torch.mean(torch.relu(dist_malicious))
    #     valid_loss = avg_dist_malicious - avg_dist_benign
    #     print("Validation loss:", valid_loss.item())
    #     print("Average Benign Distance",avg_dist_benign.item()," with STD: ", torch.std(torch.relu(dist_benign)).item())
    #     print("Average Malicious Distance", avg_dist_malicious.item(), " with STD: ", torch.std(torch.relu(dist_malicious)).item())
    #     if visualize:
    #         self.visualize(emb, label, fig_title=fig_title)
    #     return early_stop


    def early_stop_f1_auc_max(self, validation_auc, validation_f1_score, validation_TNR):
        # set early stopping technique f1_score & AUC & TNR
        if self.malicious_percentage > 0.05:
            # Optimize on F1-Score when relatively balanced data
            if validation_f1_score > self.max_validation_f1_score:
                self.max_validation_f1_score = validation_f1_score
                self.counter = 0
            # elif validation_f1_score < self.f1_baseline:
            elif (validation_f1_score < self.f1_baseline) or (validation_f1_score < (self.max_validation_f1_score - self.max_delta)):
                self.counter = 0
            elif validation_f1_score <= self.max_validation_f1_score:
                self.counter += 1
                if self.counter >= self.patience:
                    return 1
        else:
            # # set early stopping technique TPR & TNR
            if self.malicious_percentage > 0:
                # Optimize on auc when imbalanced data
                if validation_auc > self.max_validation_auc:
                    self.max_validation_auc = validation_auc
                    self.counter = 0
                # elif validation_auc <= 0.5:
                elif (validation_auc <= 0.5) or (validation_auc < (self.max_validation_auc - self.max_delta)):
                    self.counter = 0
                elif validation_auc <= self.max_validation_auc:
                    self.counter += 1
                    if self.counter >= self.patience:
                        return 1
            else:
                # Optimize on TNR when no malicious samples
                if validation_TNR > self.max_validation_TNR:
                    self.counter = 0
                    self.max_validation_TNR = validation_TNR
                # elif validation_TNR <= 0.01:
                elif (validation_TNR <= 0.01) or (validation_TNR < self.max_validation_TNR - self.max_delta):
                    self.counter = 0
                elif validation_TNR <= self.max_validation_TNR:
                    self.counter += 1
                    if self.counter >= self.patience:
                        return 1
        return 0

    # def early_stop_f1_auc_max(self, validation_auc, validation_f1_score, validation_TNR):
    #     # set early stopping technique f1_score & AUC & TNR
    #     if self.malicious_percentage > 0.05:
    #         # Optimize on F1-Score when relatively balanced data
    #         if validation_f1_score > self.max_validation_f1_score:
    #             self.max_validation_f1_score = validation_f1_score
    #             self.counter = 0
    #         # elif (validation_f1_score < self.f1_baseline) or (validation_f1_score < (self.max_validation_f1_score - self.max_delta)):
    #         elif validation_f1_score < self.f1_baseline:
    #             self.counter = 0
    #         elif validation_f1_score <= self.max_validation_f1_score:
    #             self.counter += 1
    #             if self.counter >= self.patience:
    #                 return 1
    #     else:
    #         # # set early stopping technique TPR & TNR
    #         if validation_auc:
    #             # Optimize on auc when imbalanced data
    #             if validation_auc > self.max_validation_auc:
    #                 self.max_validation_auc = validation_auc
    #                 self.counter = 0
    #             elif validation_auc <= 0.5:
    #                 self.counter = 0
    #             elif validation_auc <= self.max_validation_auc:
    #                 self.counter += 1
    #                 if self.counter >= self.patience:
    #                     return 1
    #         else:
    #             # Optimize on TNR when no malicious samples
    #             if validation_TNR > self.max_validation_TNR:
    #                 self.counter = 0
    #                 self.max_validation_TNR = validation_TNR
    #             elif validation_TNR <= 0.01:
    #                 self.counter = 0
    #             elif validation_TNR <= self.max_validation_TNR:
    #                 self.counter += 1
    #                 if self.counter >= self.patience:
    #                     return 1
    #     return 0
    def man_confusion_matrix(self,y_true, y_pred):
        TP,FP,TN,FN = 0,0,0,0
        TP = len([i for i in range(len(y_pred)) if (y_true[i] == y_pred[i] == 1) ])
        FP = len([i for i in range(len(y_pred)) if (y_pred[i] == 1 and y_true[i] != y_pred[i])])
        TN = len([i for i in range(len(y_pred)) if (y_true[i] == y_pred[i] == 0)])
        FN = len([i for i in range(len(y_pred)) if (y_pred[i] == 0 and y_true[i] != y_pred[i])])
        return (TP, FP, TN, FN)

    def loss_func_val(self, emb, label, threshold, fig_title=None, visualize=False):
        if self.first_epoch:
            self.malicious_percentage = sum(label) / len(label)
            self.f1_baseline = (self.malicious_percentage * 2) / (self.malicious_percentage + 1)
            print("Malicious Percentage: ", self.malicious_percentage)
            print("F1-Score baseline is: ", self.f1_baseline)
        validation_dist = torch.sum(torch.pow(emb - self.c, 2), 1)
        validation_score = validation_dist - self.r ** 2
        if sum(label) > 0:
            validation_auc = roc_auc_score(label, validation_score.detach())
            print("Validation AUC is:", validation_auc)
        else:
            validation_auc = None
        validation_pred = (validation_score.detach() > threshold).long()
        validation_pred = validation_pred.reshape(len(validation_pred), 1)
        validation_f1_score = f1_score(y_true=label, y_pred=validation_pred, zero_division=0)
        print('Validation F1-score:', round(validation_f1_score.item(), 5))
        # TN, FP, FN, TP = confusion_matrix(y_true= label.int(),y_pred= validation_pred,labels= [0, 1]).ravel()
        TP, FP, TN, FN = self.man_confusion_matrix(y_true=label.int(), y_pred=validation_pred)
        if (TP + FN) == 0:
            validation_TPR = None
        else:
            validation_TPR = TP / (TP + FN)
            print("Validation TPR (sensitivity):", round(validation_TPR, 5))
        if (TN + FP) == 0:
            validation_TNR = None
        else:
            validation_TNR = TN / (TN + FP)
            print("Validation TNR (selectivity):", round(validation_TNR, 5))
            validation_FPR = FP / (FP + TN)
            print('Validation FPR:', round(validation_FPR, 5))
        if visualize:
            self.visualize(emb, label, fig_title=fig_title)

        self.early_stop = self.early_stop_f1_auc_max(validation_auc, validation_f1_score, validation_TNR)
        # if early_stop == 1:
        #     if validation_TPR is not None:
        #         if validation_TPR <= self.TPR_baseline:
        #             self.counter = 0
        #             early_stop = 2
        #     if validation_TNR is not None:
        #         if validation_TNR <= self.TNR_baseline:
        #             self.counter = 0
        #             early_stop = 3
        self.first_epoch = False
        del validation_pred, validation_dist, validation_score
        return self.early_stop

    def sample_per_dist(self, dist, emb2D, n_bins, n_sample):
        df_dist = pd.DataFrame(dist.detach().numpy(), columns=['dist'])
        df_dist['dist_cat'] = pd.cut(df_dist['dist'], n_bins, labels=list(range(0, n_bins)), retbins=True)[0]
        sampled_df_dist = df_dist.groupby('dist_cat', group_keys=False,observed=True).apply(
            lambda x: x.sample(n_sample) if len(x) > n_sample else x)
        sample_emb2D = emb2D[sampled_df_dist.index]
        return sample_emb2D
    def visualize(self, emb, label=None, train=True , fig_title=None):
        dist = torch.sum(torch.pow(emb - self.c, 2), 1)
        transformTo2D = PCA(n_components=2)
        emb2D = transformTo2D.fit_transform(emb.detach().numpy())
        if train:
            self.c2D = transformTo2D.transform(self.c.reshape(1, -1)).reshape(-1)
        # plt.xlim((self.c2D[0] - self.r) * 2, (self.c2D[0] + self.r) * 2)
        # plt.ylim((self.c2D[1] - self.r) * 2, (self.c2D[1] + self.r) * 2)
        if torch.is_tensor(label):
            label = label.reshape(-1)
            benign_emb2D = emb2D[~label]
            if len(benign_emb2D) > 0:
                benign_dist = dist[~label]
                benign_emb2D = self.sample_per_dist(benign_dist, benign_emb2D, n_bins=50, n_sample=200)
                plt.plot(benign_emb2D[:, 0], benign_emb2D[:, 1], '.', color='blue')
                del benign_emb2D, benign_dist
            malicious_emb2D = emb2D[label]
            if len(malicious_emb2D) > 0:
                malicious_dist = dist[label]
                malicious_emb2D = self.sample_per_dist(malicious_dist, malicious_emb2D, n_bins=50, n_sample=200)
                plt.plot(malicious_emb2D[:, 0], malicious_emb2D[:, 1], '.', color='red')
                del malicious_emb2D, malicious_dist
        else:
            emb2D = self.sample_per_dist(dist, emb2D, n_bins=50, n_sample=200)
            plt.plot(emb2D[:, 0], emb2D[:, 1], '.', color='blue')

        plt.plot(self.c2D[0], self.c2D[1], color='orange', marker='o')
        hypershpere_circle = plt.Circle((self.c2D[0], self.c2D[1]), self.r, color='orange', linestyle='--',
                                        fill=False)
        plt.yscale("symlog")
        plt.xscale("symlog")
        fig = plt.gcf()
        ax = fig.gca()
        ax.add_patch(hypershpere_circle)
        ax.set_title(fig_title.split("/")[-1].replace(".png", ""))
        # fig.show()
        fig.savefig(fig_title)
        fig.clf()
        plt.cla()
        ax.cla()
        del emb2D
    def loss_func(self, emb,train=True,visualize=False,label=None,fig_title=None):
        """
        Loss function for OCGNN

        Parameters
        ----------
        emb : torch.Tensor
            Embeddings.

        Returns
        -------
        loss : torch.Tensor
            Loss value.
        score : torch.Tensor
            Outlier scores of shape :math:`N` with gradients.
        """

        # ############# modified by amer ##############
        if train:
            with torch.no_grad():
                self.c = torch.mean(emb, 0)
                self.c[(abs(self.c) < self.eps) & (self.c < 0)] = -self.eps
                self.c[(abs(self.c) < self.eps) & (self.c > 0)] = self.eps
        # ##########################################

        dist = torch.sum(torch.pow(emb - self.c, 2), 1)
        score = dist - self.r ** 2
        loss = self.r ** 2 + 1 / self.beta * torch.mean(torch.relu(score))

        ############# modified by amer ##############
        if train:
            with torch.no_grad():
                self.r = torch.quantile(torch.sqrt(dist), 1 - self.beta)

        if visualize:
            self.visualize(emb, label, train, fig_title)
            ### debug #####
            # print("Center of the hypersphere is", self.c)
            print("Radious of the hypersphere is", self.r)
            # print("Current Mean Embedding is (current center of data)", torch.mean(emb, 0))
            # print("Score", score)
            # print("Distance:", dist)
            ################
        #########################################

        return loss, score
