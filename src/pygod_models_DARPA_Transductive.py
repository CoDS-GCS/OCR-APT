from copy import deepcopy
import argparse
import shutil
import pandas as pd
from tqdm import tqdm
import datetime
import torch
from torch_geometric.nn import RGCNConv, RGATConv
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, ParameterDict, Parameter
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.utils.hetero import group_hetero_graph
from torch_geometric.nn import MessagePassing
from torch_geometric.seed import seed_everything
from statistics import mean
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import index_sort
import sys
import os
import psutil
from pathlib import Path
import glob
import time
from pygod.utils import load_data
from pygod.detector import OCGNN, DOMINANT , AnomalyDAE, CoLA , CONAD, GAE, GUIDE, OCRGCN
from pygod.metric import eval_roc_auc , eval_average_precision, eval_precision_at_k, eval_recall_at_k, eval_f1
from sklearn.decomposition import PCA , TruncatedSVD
from sklearn import metrics
import numpy as np
import random
import statistics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import traceback
import sys

from dataset_pyg_custom import PygNodePropPredDataset_custom
from evaluate_hsh import Evaluator
from resource import *
from logger import Logger
import faulthandler
torch.use_deterministic_algorithms(True)
import pickle
faulthandler.enable()

pd.set_option('display.max_columns', 50)

parser = argparse.ArgumentParser(description='OCR-APT')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--num-layers', type=int, default=3)
parser.add_argument('--input-layer', type=int, default=64)
parser.add_argument('--hidden-channels', type=int, default=64)
parser.add_argument('--adjust-hidden-channels', action="store_true", default=False)
parser.add_argument('--n-classes', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--beta', type=float, default=0.5)
parser.add_argument('--warmup', type=int, default=2)
parser.add_argument('--visualize-training', action="store_true", default=False)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--contamination', type=float, default=0.001)
parser.add_argument('--runs', type=int, default=3)
parser.add_argument('--batch-size', type=int, default=0)
parser.add_argument('--batch-size-percentage', type=float, default=None)
parser.add_argument('--walk-length', type=int, default=2)
parser.add_argument('--num-steps', type=int, default=30)
parser.add_argument('--flexable-rate', type=float, default=0.001)
parser.add_argument('--max-contamination', type=float, default=0.05)
parser.add_argument('--top-k', type=int, default=1000)
parser.add_argument('--random-features', action="store_true", default=False)
parser.add_argument('--dataset', type=str,required=True)
parser.add_argument('--host', type=str,required=True)
parser.add_argument('--exp-name', type=str,required=True)
parser.add_argument('--root-path', type=str,required=True)
parser.add_argument('--load-model', type=str, default=None)
parser.add_argument('--consider-related-nodes', action="store_true", default=False)
parser.add_argument('--store-embedding', action="store_true", default=False)
parser.add_argument('--save-model', type=str, default=None)
parser.add_argument('--save-emb', action="store_true", default=False)
parser.add_argument('--backbone', type=str, default=None)
parser.add_argument('--detector', type=str, default="OCGNN")
parser.add_argument('--multiple-models', action="store_true", default=False)
parser.add_argument('--ensemble-models', action="store_true", default=False)
parser.add_argument('--features-per-node-type', action="store_true", default=False)
parser.add_argument('--dynamic-contamination', action="store_true", default=False)
parser.add_argument('--dynamic-contamination-val', action="store_true", default=False)
parser.add_argument('--feature-reduction-PCA', action="store_true", default=False)
parser.add_argument('--feature-reduction-TruncatedSVD', action="store_true", default=False)
parser.add_argument('--reduction-target-node', action="store_true", default=False)
parser.add_argument('--standard-scaler', action="store_true", default=False)
parser.add_argument('--feature-reduction-NoSelection', action="store_true", default=False)
parser.add_argument('--mask-edge-types', action="store_true", default=False)
# parser.add_argument('--consider-direction', action="store_true", default=False)
parser.add_argument('--edge-weight', action="store_true", default=False)
parser.add_argument('--debug-one-subject', action="store_true", default=False)
parser.add_argument('--debug-subjects', type=int, default=-1)
parser.add_argument('--correlate-subgraphs', action="store_true", default=False)


init_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
args = parser.parse_args()
assert args.dataset in ['tc3', 'optc', 'nodlink']
assert args.host in ['cadets', 'trace', 'theia', 'fivedirections','SysClient0051','SysClient0501','SysClient0201','SimulatedUbuntu','SimulatedW10','SimulatedWS12']
dataset_numofClasses = str(args.n_classes)

root_path = args.root_path
process = psutil.Process(os.getpid())

def checkpoint(data, file_path):
    ensure_dir(file_path)
    torch.save(data, file_path)
    return
def print_memory_usage(message=None):
    print(message)
    print("Memory usage (ru_maxrss) : ",getrusage(RUSAGE_SELF).ru_maxrss/1024," MB")
    print("Memory usage (psutil) : ", psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2), "MB")
    print('The CPU usage is (per process): ', psutil.Process(os.getpid()).cpu_percent(4))
    load1, load5, load15 = psutil.getloadavg()
    cpu_usage = (load15 / os.cpu_count()) * 100
    print("The CPU usage is : ", cpu_usage)
    print('used virtual memory GB:', psutil.virtual_memory().used / (1024.0 ** 3), " percent",
          psutil.virtual_memory().percent)
    return

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def delete_folder(dir_path):
    ###################################Delete Folder if exist #############################
    try:
        shutil.rmtree(dir_path)
        print("Folder Deleted")
    except OSError as e:
        print("Error Deleting : %s : %s" % (dir_path, e.strerror))
    ####################
    return

def man_confusion_matrix(y_true, y_pred):
    TP,FP,TN,FN = 0,0,0,0
    TP = len([i for i in range(len(y_pred)) if (y_true[i] == y_pred[i] == 1) ])
    FP = len([i for i in range(len(y_pred)) if (y_pred[i] == 1 and y_true[i] != y_pred[i])])
    TN = len([i for i in range(len(y_pred)) if (y_true[i] == y_pred[i] == 0)])
    FN = len([i for i in range(len(y_pred)) if (y_pred[i] == 0 and y_true[i] != y_pred[i])])
    return (TP, FP, TN, FN)
def print_predict_evaluation_metrics(tp,tn,fp,fn,message=None):
    if (tp + fp) == 0:
        precision = None
    else:
        precision = tp / (tp + fp)
    if (tp + fn) == 0:
        recall = None
        tpr = None
    else:
        recall = tp / (tp + fn)
        tpr = tp / (tp + fn)
    if (fp + tn) == 0:
        fpr = None
    else:
        fpr = fp / (fp + tn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    if not precision or not recall or (precision + recall) == 0:
        f_measure = None
    else:
        f_measure = 2 * (precision * recall) / (precision + recall)
    print("Evaluation on Anomaly Detection task")
    if message:
        print(message)
    print("TP: {}\tTN: {}\tFP: {}\tFN: {}".format(tp, tn, fp, fn))
    print("Accuracy: {}\tPrecision: {}\tRecall: {}\tF-1: {}".format(accuracy, precision, recall, f_measure))
    print("TPR: {}\tFPR: {}".format(tpr, fpr))
    anomaly_results_df = pd.DataFrame({'accuracy': [accuracy], 'precision': [precision],'recall': [recall],
                                       'f_measure':[f_measure],'tp':[tp],'tn': [tn],'fp': [fp],'fn': [fn]})
    anomaly_results_df['tpr'] = tpr
    anomaly_results_df['fpr'] = fpr
    return anomaly_results_df

def decide_hidden_channel(feature_size):
    if feature_size >=128 :
        hidden_layer = 128
    elif feature_size >=92 :
        hidden_layer = 92
    elif feature_size >=64 :
        hidden_layer = 64
    elif feature_size >=32 :
        hidden_layer = 32
    else:
        hidden_layer = 16
    return hidden_layer

def pyGod(run,seed):
    def train(detector,homo_data,fig_title=None):
        start_train_time = time.time()
        homo_data.active_mask = deepcopy(homo_data.train_mask)
        if args.detector in ["OCGNN","OCRGCN"]:
            fig_title += "_training_"
            detector.fit(homo_data,fig_title=fig_title)
        else:
            detector.fit(homo_data)
        ########## Debug ########
        print("The model Threshold is:", detector.threshold_)
        #########################
        train_time = time.time() - start_train_time
        print("Training time: ", train_time , "seconds.")
        print_memory_usage()
        return detector,train_time

    def validate(detector,homo_data,fig_title=None):
        validate_time = time.time()
        try:
            homo_data.active_mask = deepcopy(homo_data.train_mask)
            if args.visualize_training:
                y_pred_train = detector.predict(homo_data, return_pred=True,label=homo_data.y[homo_data.active_mask],fig_title=str(fig_title+"_evaluate_training_set.png"))
            else:
                y_pred_train = detector.predict(homo_data, return_pred=True)
            homo_data.active_mask = deepcopy(homo_data.val_mask)
            # if args.batch_size_percentage:
            #     batch_size = int(len(homo_data.y[homo_data.active_mask]) * args.batch_size_percentage)
            # else:
            #     batch_size = args.batch_size
            # print("Used validate Batch size is: ", batch_size)
            if args.visualize_training:
                y_pred_val = detector.predict(homo_data, return_pred=True,label=homo_data.y[homo_data.active_mask],fig_title=str(fig_title+"_evaluate_validating_set.png"))
            else:
                y_pred_val = detector.predict(homo_data, return_pred=True)
        except Exception as e:
            print(e)
            if args.save_model:
                print("Saving the model to", args.save_model)
                model_path = root_path + "models/" +args.exp_name +'/'+ args.save_model
                ensure_dir(model_path)
                torch.save(detector, model_path)
            return
        y_pred_train = y_pred_train.reshape(len(y_pred_train), 1)
        y_pred_val = y_pred_val.reshape(len(y_pred_val), 1)
        train_acc = evaluator.eval({
            'y_true': y_true[homo_data.train_mask],
            'y_pred': y_pred_train,
        })['acc']
        valid_acc = evaluator.eval({
            'y_true': y_true[homo_data.val_mask],
            'y_pred': y_pred_val,
        })['acc']
        valid_cm = evaluator_cm.eval({
            'y_true': y_true[homo_data.val_mask],
            'y_pred': y_pred_val,
            'labels': [0, 1]
        })['cm']
        print('validating set CM:\n', valid_cm)
        # tn, tp, fp, fn = 0, 0, 0, 0
        # tn = valid_cm[0][0]
        # tp = valid_cm[1][1]
        # fp = valid_cm[0][1]
        # fn = valid_cm[1][0]
        tp, fp, tn, fn = man_confusion_matrix(y_true=y_true[homo_data.val_mask], y_pred=y_pred_val)
        print_predict_evaluation_metrics(tp, tn, fp, fn)
        result = train_acc, valid_acc
        print("Validation time: ", time.time() - validate_time, "seconds.")
        return result
    def data_statistics_contamination(homo_data):
        n_train = sum(homo_data.train_mask).item()
        n_val = sum(homo_data.val_mask).item()
        n_test = sum(homo_data.test_mask).item()
        print("Total Training Samples:", n_train)
        print("Number of malicious samples:", sum(homo_data.y[homo_data.train_mask]).item())
        print("Number of benign samples:", n_train - sum(homo_data.y[homo_data.train_mask]).item())
        print("Total Validating Samples:", n_val)
        print("Number of malicious samples:", sum(homo_data.y[homo_data.val_mask]).item())
        print("Number of benign samples:", n_val - sum(homo_data.y[homo_data.val_mask]).item())
        if args.dynamic_contamination_val:
            contamination = round(sum(homo_data.y[homo_data.val_mask]).item() / (n_val),3)
            print("The contamination based on validating data is:", contamination)
        else:
            contamination = round(sum(homo_data.y[homo_data.val_mask]).item() / (n_train + n_val),3)
            print("The contamination based on training and validating data is:", contamination)
            contamination = (contamination * 2) + args.flexable_rate
            print("Use the second formula, (con * 2) + flex_rate")
        print("The contamination has set to be between",args.flexable_rate," and",args.max_contamination)
        contamination = max(contamination,args.flexable_rate)
        contamination = min(contamination, args.max_contamination)
        print("Total Testing Samples:", n_test)
        print("Number of malicious samples:", sum(homo_data.y[homo_data.test_mask]).item())
        print("Number of benign samples:", n_test - sum(homo_data.y[homo_data.test_mask]).item())
        return contamination
    def split_per_node_type(homo_data,target_node):
        homo_data_per_type = deepcopy(homo_data)
        homo_data_per_type.train_mask = torch.zeros((node_type.size(0)), dtype=torch.bool)
        homo_data_per_type.val_mask = torch.zeros((node_type.size(0)), dtype=torch.bool)
        homo_data_per_type.test_mask = torch.zeros((node_type.size(0)), dtype=torch.bool)
        homo_data_per_type.train_mask[local2global[target_node][split_idx['train'][target_node]]] = True
        homo_data_per_type.val_mask[local2global[target_node][split_idx['valid'][target_node]]] = True
        homo_data_per_type.test_mask[local2global[target_node][split_idx['test'][target_node]]] = True
        print("Remove ", sum(homo_data_per_type.y[homo_data_per_type.train_mask]).item(), " malicious samples from training set:")
        homo_data_per_type.train_mask[homo_data_per_type.y.view(-1) == 1] = False
        contamination = data_statistics_contamination(homo_data_per_type)
        if not args.dynamic_contamination:
            contamination = args.contamination
        if args.features_per_node_type:
            # get different features set for each node type
            target_node_feat = load_features(target_node)
            non_empty_mask = target_node_feat.abs().sum(dim=0).bool()
            target_node_feat = target_node_feat[:, non_empty_mask]
            feat = torch.zeros((node_type.size(0)), target_node_feat.size(1))
            if args.random_features:
                print("randomly initialize other node types")
                torch.nn.init.xavier_uniform_(feat)
                feat[local2global[target_node]] = target_node_feat
            else:
                for subject_node in subject_nodes:
                    if args.feature_reduction_PCA or args.feature_reduction_TruncatedSVD:
                        if subject_node != target_node or args.feature_reduction_NoSelection:
                            if args.feature_reduction_PCA:
                                reduction_alg = PCA(n_components=target_node_feat.size(1))
                            elif args.feature_reduction_TruncatedSVD and args.reduction_target_node:
                                reduction_alg = TruncatedSVD(n_components=target_node_feat.size(1),algorithm='randomized')
                            elif args.feature_reduction_TruncatedSVD:
                                reduction_alg = TruncatedSVD(n_components=target_node_feat.size(1))
                            temp_feat = load_features(subject_node)
                            if target_node_feat.size(1) > min(temp_feat.shape):
                                print("Number of samples of the node",subject_node ," is:",min(temp_feat.shape) ,",which is less the reduction feature size")
                                print("Use feature selection based on target node feature size instead of feature reduction")
                                temp_feat = temp_feat[:, non_empty_mask]
                                # print("Use the full feature for the target node",target_node)
                                # print("Target node features size is", homo_data.x.size(1))
                                # return homo_data_per_type, contamination
                            else:
                                temp_feat = reduction_alg.fit_transform(temp_feat)
                                temp_feat = torch.from_numpy(temp_feat).float()
                            feat[local2global[subject_node]] = temp_feat
                        else:
                            if args.reduction_target_node:
                                if args.feature_reduction_PCA:
                                    reduction_alg = PCA(n_components=target_node_feat.size(1))
                                elif args.feature_reduction_TruncatedSVD:
                                    reduction_alg = TruncatedSVD(n_components=target_node_feat.size(1),algorithm='randomized')
                                target_node_feat = reduction_alg.fit_transform(target_node_feat)
                                target_node_feat = torch.from_numpy(target_node_feat).float()
                            feat[local2global[target_node]] = target_node_feat
                            print("Target node features size is",target_node_feat.size(1))
                    else:
                        print("load other node types features and fill na with 0")
                        temp_feat = load_features(subject_node)
                        temp_feat = temp_feat[:, non_empty_mask]
                        feat[local2global[subject_node]] = temp_feat
            print("The `model` for ", target_node, "nodes has initial features of size:", temp_feat.size(1))
            homo_data_per_type.x = feat
        if args.mask_edge_types:
            # Replace edges not related to the target node with "other" edge for each (subject,object) pair
            ignore_dic = {}
            for subject, predicate, object in key_lst:
                if target_node != subject and target_node != object and 'inv_' not in predicate:
                    if (subject,'OTHER',object) in ignore_dic.keys():
                        ignore_dic[(subject,'OTHER',object)].append(key2int[(subject, predicate, object)])
                        # ignore_dic[(subject, 'OTHER', object)].append(key2int[(object, "inv_" + predicate, subject)])
                        ignore_dic[(object,'inv_OTHER',subject)].append(key2int[(object, "inv_" + predicate, subject)])
                    else:
                        ignore_dic[(subject, 'OTHER', object)] = [key2int[(subject, predicate, object)]]
                        # ignore_dic[(subject, 'OTHER', object)].append(key2int[(object, "inv_" + predicate, subject)])
                        ignore_dic[(object,'inv_OTHER',subject)] = [key2int[(object, "inv_" + predicate, subject)]]

            new_id = len(homo_data_per_type.edge_attr.unique())
            for key,ignore_lst in ignore_dic.items():
                edge_ignore_mask = np.isin(homo_data_per_type.edge_attr, ignore_lst)
                homo_data_per_type.edge_attr[edge_ignore_mask] = new_id
                del edge_ignore_mask
                new_id +=1
            # Resort the edge_type and edge_index tensors
            old_edge_attr = deepcopy(homo_data_per_type.edge_attr)
            old_edge_index = deepcopy(homo_data_per_type.edge_index)
            new_edge_indices, new_edge_types = [], []
            for new_edge_id, old_edge_id in enumerate(old_edge_attr.unique()):
                # print("Convert edge id:",old_edge_id.item(),"to:", new_edge_id)
                current_mask = [old_edge_attr==old_edge_id.item()]
                new_edge_indices.append(torch.stack((old_edge_index[0][current_mask],old_edge_index[1][current_mask])))
                new_edge_types.append(torch.full(old_edge_attr[current_mask].size(),new_edge_id))
                del current_mask
            homo_data_per_type.edge_index = torch.cat(new_edge_indices, dim=-1)
            homo_data_per_type.edge_attr = torch.cat(new_edge_types, dim=0)
            print("Number of edges:",homo_data_per_type.edge_index.size(1))
            del old_edge_attr, old_edge_index, new_edge_indices, new_edge_types
        return homo_data_per_type , contamination
    def load_features(node_type):
        feat_path = root_path + args.exp_name + "/features/" + node_type + "/node-features.pt"
        feat = torch.load(feat_path)
        if args.standard_scaler:
            # apply StandardScaler
            scalar = StandardScaler()
            feat = scalar.fit_transform(feat)
            feat = torch.from_numpy(feat).float()
        return feat

    def inintailize_model(contamination,num_relations,batch_size):
        print("Used contamination is ", contamination)
        if args.adjust_hidden_channels :
            hidden_channels = decide_hidden_channel(homo_data.x.size(1))
            print("The used hidden layer is: ",hidden_channels)
        else:
            hidden_channels = args.hidden_channels
        if args.detector == "OCGNN":
            detector = OCGNN(hid_dim=hidden_channels,num_layers=args.num_layers,
                                            epoch=args.epochs, batch_size=batch_size, dropout=args.dropout,
                                            lr=args.lr, contamination=contamination,save_emb = args.save_emb,beta=args.beta,warmup=args.warmup,visualize=args.visualize_training)
        elif args.detector == "OCRGCN":
            if args.backbone == "RGATConv":
                detector = OCRGCN(num_relations=num_relations, hid_dim=hidden_channels,num_layers=args.num_layers,
                                  epoch=args.epochs, batch_size=batch_size, dropout=args.dropout,
                                  lr=args.lr, contamination=contamination,backbone=RGATConv,save_emb = args.save_emb,beta=args.beta,warmup=args.warmup,visualize=args.visualize_training)
            else:
                detector = OCRGCN(num_relations=num_relations,hid_dim=hidden_channels,num_layers=args.num_layers,
                                            epoch=args.epochs, batch_size=batch_size, dropout=args.dropout,
                                            lr=args.lr, contamination=contamination,save_emb = args.save_emb,beta=args.beta,warmup=args.warmup,visualize=args.visualize_training)
        elif args.detector == "DOMINANT":
            detector = DOMINANT(hid_dim=hidden_channels,num_layers=args.num_layers, epoch=args.epochs,
                                               batch_size=batch_size, dropout=args.dropout, lr=args.lr,
                                contamination=contamination,save_emb = args.save_emb,beta=args.beta,warmup=args.warmup,visualize=args.visualize_training)
        elif args.detector == "AnomalyDAE":
            detector = AnomalyDAE(emb_dim=hidden_channels, hid_dim=hidden_channels,epoch=args.epochs,
                                                 batch_size=batch_size, dropout=args.dropout, lr=args.lr,
                                  contamination=contamination,save_emb = args.save_emb,beta=args.beta,warmup=args.warmup,visualize=args.visualize_training)
        elif args.detector == "GAE":
            detector = GAE(hid_dim=hidden_channels,epoch=args.epochs,num_layers=args.num_layers,
                                  batch_size=batch_size, dropout=args.dropout, lr=args.lr,
                           contamination=contamination,save_emb = args.save_emb,beta=args.beta,warmup=args.warmup,visualize=args.visualize_training)
        elif args.detector == "CONAD":
            detector = CONAD( hid_dim=hidden_channels,epoch=args.epochs,num_layers=args.num_layers,
                                  batch_size=batch_size, dropout=args.dropout, lr=args.lr,
                              contamination=contamination, save_emb = args.save_emb,beta=args.beta,warmup=args.warmup,visualize=args.visualize_training)
        elif args.detector == "CoLA":
            detector = CoLA(hid_dim=hidden_channels,epoch=args.epochs,num_layers=args.num_layers,
                                  batch_size=batch_size, dropout=args.dropout, lr=args.lr,
                            contamination=contamination, save_emb = args.save_emb,beta=args.beta,warmup=args.warmup,visualize=args.visualize_training)
        elif args.detector == "GUIDE":
            detector = GUIDE(hid_a=hidden_channels, epoch=args.epochs, num_layers=args.num_layers,
                            batch_size=batch_size, dropout=args.dropout, lr=args.lr,
                             contamination=contamination,save_emb = args.save_emb,beta=args.beta,warmup=args.warmup,visualize=args.visualize_training)
        else:
            print("Not Defined detector:", args.detector)
            return None
        return detector
    def log_raised_alarms(homo_data, y_pred, score, y_prob,target_node=None,alerts_2hop=None,run=0):
        model_prediction_df = pd.DataFrame({'node_id': range(homo_data.num_nodes)})[homo_data.test_mask.tolist()]
        model_prediction_df['Label'] = homo_data.y.view(-1)[homo_data.test_mask.tolist()]
        model_prediction_df['Model_prediction'] = y_pred.view(-1).bool()
        model_prediction_df['Anomaly_score'] = score
        model_prediction_df['Prediction_probability'] = y_prob
        mapping_nodes_df = pd.DataFrame()
        if target_node:
            file_path = root_path + args.exp_name + "/mapping/" + target_node + "_entidx2name.csv"
            mapping_nodes_df = pd.read_csv(file_path, header=None, skiprows=1, names=["node_id", "node_uuid"])
            mapping_nodes_df["node_id"] = local2global[target_node][mapping_nodes_df["node_id"]]
        else:
            for subject_node in subject_nodes:
                file_path = root_path + args.exp_name + "/mapping/" + subject_node + "_entidx2name.csv"
                mapping_nodes_df_tmp = pd.read_csv(file_path, header=None, skiprows=1, names=["node_id", "node_uuid"])
                mapping_nodes_df_tmp["node_id"] = local2global[subject_node][mapping_nodes_df_tmp["node_id"]]
                mapping_nodes_df_tmp["node_type"] = subject_node
                mapping_nodes_df = pd.concat([mapping_nodes_df, mapping_nodes_df_tmp])
        model_prediction_df = pd.merge(model_prediction_df,mapping_nodes_df, on="node_id")
        del mapping_nodes_df
        if target_node:
            model_prediction_df = model_prediction_df[["node_uuid", "Label", "Model_prediction", "Anomaly_score", "Prediction_probability"]]
        else:
            model_prediction_df = model_prediction_df[
                ["node_uuid", "node_type" , "Label", "Model_prediction", "Anomaly_score", "Prediction_probability"]]
        model_prediction_df = model_prediction_df.sort_values(by=['Anomaly_score','Label'], ascending=[False, False]).reset_index(drop=True)

        if alerts_2hop:
            model_prediction_df_2hop = model_prediction_df.copy()
            model_prediction_df_2hop["Model_prediction"] = model_prediction_df_2hop["node_uuid"].isin(alerts_2hop)
            raised_alarms_2hop = model_prediction_df_2hop[model_prediction_df_2hop["Model_prediction"] == True]
            missed_anomalies_2hop = model_prediction_df_2hop[model_prediction_df_2hop["Model_prediction"] == False]
            missed_anomalies_2hop = missed_anomalies_2hop[missed_anomalies_2hop["Label"] == True]
        raised_alarms = model_prediction_df[model_prediction_df["Model_prediction"]==True]
        missed_anomalies = model_prediction_df[model_prediction_df["Model_prediction"] == False]
        missed_anomalies = missed_anomalies[missed_anomalies["Label"] == True]

        if args.save_model:
            out_path = root_path + "results/" + args.exp_name + "/" + args.save_model.replace(".model","") +"/run_" + str(run) + "_raised_alarms.csv"
        else:
            out_path = root_path + "results/" + args.exp_name + "/" + args.load_model.replace(".model","") + "/run_" + str(run) + "_raised_alarms.csv"
        if target_node:
            out_path = out_path.replace(".csv",str("_" + target_node + ".csv"))
        ensure_dir(out_path)
        raised_alarms.to_csv(out_path, index=None)
        del raised_alarms
        out_path = out_path.replace("_raised_alarms","_missed_anomalies")
        ensure_dir(out_path)
        missed_anomalies.to_csv(out_path, index=None)
        del missed_anomalies, model_prediction_df
        if alerts_2hop:
            out_path_2hop = out_path.replace("_missed_anomalies","_missed_anomalies_2hop")
            missed_anomalies_2hop.to_csv(out_path_2hop, index=None)
            out_path_2hop = out_path_2hop.replace("_missed_anomalies_2hop","_raised_alarms_2hop")
            raised_alarms_2hop.to_csv(out_path_2hop, index=None)
            del raised_alarms_2hop, missed_anomalies_2hop, model_prediction_df_2hop
        return

    def map_save_embedding(homo_data,emb,mapping_nodes_df,target_node=None):
        global final_emb
        with torch.no_grad():
            emb = emb.numpy()
        emb_df = pd.DataFrame(emb)
        emb_df["node_id"] = emb_df.index
        if target_node:
            target_node_emb_df = pd.concat([emb_df[homo_data.train_mask.tolist()],emb_df[homo_data.val_mask.tolist()],emb_df[homo_data.test_mask.tolist()]]).drop_duplicates()
            final_emb_tmp = pd.merge(mapping_nodes_df, target_node_emb_df, on="node_id")
            final_emb = pd.concat([final_emb,final_emb_tmp])
        else:
            final_emb = pd.merge(mapping_nodes_df,emb_df,on="node_id")
        del final_emb_tmp, target_node_emb_df, emb_df
        print("Number of prepared node embeddings", len(final_emb))
        return

    def Get_Adjacent(ids, mapp, edges, hops):
        if hops == 0:
            return set()

        neighbors = set()
        for edge in zip(edges[0], edges[1]):
            if any(mapp[node] in ids for node in edge):
                neighbors.update(mapp[node] for node in edge)

        if hops > 1:
            neighbors = neighbors.union(Get_Adjacent(neighbors, mapp, edges, hops - 1))

        return neighbors

    def calculate_metrics(TP, FP, FN, TN):
        acc = (TP + TN) / (TP + TN+FP + FN)
        FPR = FP / (FP + TN) if FP + TN > 0 else 0
        TPR = TP / (TP + FN) if TP + FN > 0 else 0

        prec = TP / (TP + FP) if TP + FP > 0 else 0
        rec = TP / (TP + FN) if TP + FN > 0 else 0
        fscore = (2 * prec * rec) / (prec + rec) if prec + rec > 0 else 0

        return acc, prec, rec, fscore, FPR, TPR
    def helper(MP, all_pids, GP, edges, mapping_dict):
        print("Initial number of ground truth labels", len(GP))
        print("Initial number of identified anomolous nodes", len(MP))
        TP = MP.intersection(GP)
        FP = MP - GP
        FN = GP - MP
        TN = all_pids - (GP | MP)

        print("Debugging, initial results without two hops anomolous nodes")
        acc, prec, rec, fscore, FPR, TPR = calculate_metrics(len(TP), len(FP), len(FN), len(TN))
        print(f"TP: {len(TP)}, FP: {len(FP)}, FN: {len(FN)}, TN: {len(TN)}")
        print(f"Precision: {round(prec, 3)}, Recall: {round(rec, 3)}, Fscore: {round(fscore, 3)}")

        two_hop_gp = Get_Adjacent(GP, mapping_dict, edges, 2)
        print("Number of nodes that connect to ground truth malicious nodes with 2-hops", len(two_hop_gp))
        two_hop_tp = Get_Adjacent(TP, mapping_dict, edges, 2)
        print("Number of nodes that connect to true anomalous identified nodes with 2-hops", len(two_hop_tp))

        FPL = FP - two_hop_gp
        TPL = TP.union(FN.intersection(two_hop_tp))
        FN = FN - two_hop_tp

        TP, FP, FN, TN = len(TPL), len(FPL), len(FN), len(TN)

        acc,prec, rec, fscore, FPR, TPR = calculate_metrics(TP, FP, FN, TN)
        print("\n*******************************************")
        print("Debugging, results with two hops identified anomalous nodes (FLASH and THREATRACE Logic)")
        print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
        print(f"Precision: {round(prec, 3)}, Recall: {round(rec, 3)}, Fscore: {round(fscore, 3)}")
        anomaly_2hop_results_df = pd.DataFrame({'accuracy':[acc],'precision': [prec], 'recall': [rec],'f_measure': [fscore],
                                                'tp': [TP], 'tn': [TN], 'fp': [FP], 'fn': [FN],'tpr': [TPR], 'fpr': [FPR]})
        del two_hop_gp, two_hop_tp
        alerts_2hop = TPL.union(FPL)
        return alerts_2hop , anomaly_2hop_results_df
    def predict_with_related_nodes(homo_data,y_true,y_pred):
        start_compute_related_nodes = time.time()
        print("Result with post-processing evaluation (Considering 2-hops related nodes of anomalous nodes as true anomalies)")
        # confirm mapping, get only testing set , fix getAdjacency function
        mapping_nodes_df = pd.DataFrame()
        for subject_node in subject_nodes:
            file_path = root_path + args.exp_name + "/mapping/" + subject_node + "_entidx2name.csv"
            mapping_nodes_df_tmp = pd.read_csv(file_path, header=None, skiprows=1, names=["node_id", "node_uuid"])
            mapping_nodes_df_tmp["node_id"] = local2global[subject_node][mapping_nodes_df_tmp["node_id"]]
            mapping_nodes_df = pd.concat([mapping_nodes_df, mapping_nodes_df_tmp])
        mapping_dict = mapping_nodes_df.set_index("node_id")["node_uuid"].to_dict()
        del mapping_nodes_df
        testing_idx = set(np.where(homo_data.test_mask)[0])
        testing_uuid = {mapping_dict[idx] for idx in testing_idx}
        identified_uuid = {mapping_dict[idx] for idx in testing_idx if y_pred[idx] == 1}
        gt_nodes_uuid = {mapping_dict[idx] for idx in testing_idx if y_true[idx] == 1}
        alerts_2hop , anomaly_2hop_results_df = helper(identified_uuid, testing_uuid, gt_nodes_uuid, homo_data.edge_index.tolist(),mapping_dict)

        # if args.save_model:
        #     out_path = root_path + "results/" + args.exp_name + "/" + args.save_model.replace(".model","") + "_raised_related_alarms.pt"
        # else:
        #     out_path = root_path + "results/" + args.exp_name + "/" + args.load_model.replace(".model","") + "_raised_related_alarms.pt"
        # checkpoint(alerts_2hop,out_path)
        del gt_nodes_uuid, identified_uuid, testing_uuid, testing_idx
        # print_predict_evaluation_metrics(tp, tn, fp, fn, "Consider 2-hops of related nodes")
        print("Computed related nodes in ", time.time() - start_compute_related_nodes)
        return alerts_2hop , anomaly_2hop_results_df


    def test(detector,homo_data,analyze_per_type=True,fig_title=None):
        predict_time = time.time()
        global final_y_pred_testing,final_y_prob_testing,final_score_testing
        try:
            homo_data.active_mask = deepcopy(homo_data.test_mask)
            if args.visualize_training:
                y_pred, score, y_prob = detector.predict(homo_data, return_pred=True, return_score=True, return_prob=True,label=homo_data.y[homo_data.active_mask],fig_title=str(fig_title+"_evaluate_testing_set.png"))
            else:
                detector.visualize = False
                y_pred, score, y_prob = detector.predict(homo_data, return_pred=True, return_score=True,return_prob=True)
        except Exception as e:
            print(e)
            if args.save_model:
                print("Saving the model to", args.save_model)
                model_path = root_path + "models/" +args.exp_name +'/'+ args.save_model
                ensure_dir(model_path)
                torch.save(detector, model_path)
            return
        y_pred = y_pred.reshape(len(y_pred), 1)
        final_y_pred_testing[homo_data.test_mask] = y_pred
        final_y_prob_testing[homo_data.test_mask] = y_prob
        final_score_testing[homo_data.test_mask] = score
        test_acc = evaluator.eval({
            'y_true': y_true[homo_data.test_mask],
            'y_pred': y_pred,
        })['acc']
        y_true_testing = y_true[homo_data.test_mask]
        y_pred_testing = y_pred
        node_type_testing = node_type[homo_data.test_mask]
        test_cm = evaluator_cm.eval({
            'y_true': y_true_testing,
            'y_pred': y_pred_testing,
            'labels': [0, 1]
        })['cm']
        print("\n*****************************************")
        print('Testing set CM:\n', test_cm)
        tp, fp, tn, fn = man_confusion_matrix(y_true=y_true_testing, y_pred=y_pred_testing)
        anomaly_results_df = print_predict_evaluation_metrics(tp, tn, fp, fn)
        if len(y_true_testing.unique()) > 1:
            anomaly_results_df['auc'] = eval_roc_auc(y_true_testing, score)
            print('AUC Score:', anomaly_results_df['auc'].item())
            anomaly_results_df['avg_precision'] = eval_average_precision(y_true_testing, score)
            print('Average precision:', anomaly_results_df['avg_precision'].item())
            top_k = min(args.top_k,len(y_true_testing))
            anomaly_results_df['precision_at_k'] = eval_precision_at_k(y_true_testing, score,top_k)
            print("precision at",top_k,"is",anomaly_results_df['precision_at_k'].item())
            anomaly_results_df['recall_at_k'] = eval_recall_at_k(y_true_testing, score, top_k)
            print("recall at", top_k, "is", anomaly_results_df['recall_at_k'].item())
        if analyze_per_type:
            for subject_node in subject_nodes:
                print("\n****************************************")
                y_true_per_type = y_true_testing[node_type_testing == key2int[subject_node]]
                y_pred_per_type = y_pred_testing[node_type_testing == key2int[subject_node]]
                test_cm_per_type = evaluator_cm.eval({
                    'y_true': y_true_per_type,
                    'y_pred': y_pred_per_type,
                    'labels': [0, 1]
                })['cm']
                tp, fp, tn, fn = man_confusion_matrix(y_true=y_true_per_type, y_pred=y_pred_per_type)
                message = "Profiling results of node type:" + subject_node
                print_predict_evaluation_metrics(tp, tn, fp, fn, message)
                del y_true_per_type, y_pred_per_type, test_cm_per_type
        # if args.consider_related_nodes:
        #     predict_with_related_nodes(y_true_testing,y_pred)
        detection_time = time.time() - predict_time
        print("Detection time: ", detection_time , "seconds.")
        return anomaly_results_df, y_pred, score, y_prob,detection_time, test_acc

    def test_ensemble_models(y_true_testing,y_pred_lst,score_lst,threshold_lst):
        # Ensemble models with majority voting
        all_y_pred = torch.stack(y_pred_lst)
        mode_y_pred = torch.mode(all_y_pred, dim=0)
        test_cm = evaluator_cm.eval({
            'y_true': y_true_testing,
            'y_pred': mode_y_pred[0],
            'labels': [0, 1]
        })['cm']
        print('Test_cm:\n', test_cm)
        # tn = test_cm[0][0]
        # tp = test_cm[1][1]
        # fp = test_cm[0][1]
        # fn = test_cm[1][0]
        tp, fp, tn, fn = man_confusion_matrix(y_true=y_true_testing, y_pred=mode_y_pred[0])
        print_predict_evaluation_metrics(tp, tn, fp, fn, "Ensemble models with majority voting")

        # Ensemle models with score averaging
        all_score = torch.stack(score_lst)
        avg_score = torch.mean(all_score, dim=0)
        avg_threshold = mean(threshold_lst)
        avg_y_pred = (avg_score > avg_threshold).long()
        test_cm = evaluator_cm.eval({
            'y_true': y_true_testing,
            'y_pred': avg_y_pred.view(y_true_testing.shape),
            'labels': [0, 1]
        })['cm']
        print('Test_cm:\n', test_cm)
        tp, fp, tn, fn = man_confusion_matrix(y_true=y_true_testing, y_pred=avg_y_pred.view(y_true_testing.shape))
        anomaly_results_df = print_predict_evaluation_metrics(tp, tn, fp, fn, "Ensemble models with score averaging")
        anomaly_results_df['auc'] = eval_roc_auc(y_true_testing, avg_score)
        print('AUC Score:', anomaly_results_df['auc'].item())
        max_score = torch.max(all_score, dim=0)
        max_threshold = max(threshold_lst)
        max_y_pred = (max_score[0] > max_threshold).long()
        test_cm = evaluator_cm.eval({
            'y_true': y_true_testing,
            'y_pred': max_y_pred.view(y_true_testing.shape),
            'labels': [0, 1]
        })['cm']
        print('Test_cm:\n', test_cm)
        tp, fp, tn, fn = man_confusion_matrix(y_true=y_true_testing, y_pred=max_y_pred.view(y_true_testing.shape))
        anomaly_results_df = print_predict_evaluation_metrics(tp, tn, fp, fn, "Ensemble models with maximum score")
        anomaly_results_df['auc'] = eval_roc_auc(y_true_testing, max_score[0])
        print('AUC Score:', anomaly_results_df['auc'].item())
        # Ensemle models with maximum score
        return


    to_remove_pedicates = []
    to_remove_subject_object = []
    # to_keep_edge_idx_map = []
    # GA_Index = 0
    print(args)
    # gsaint_start_t = datetime.datetime.now()
    ###################################Delete Folder if exist #############################
    #### DEBUG try without deleting #########
    dir_path = root_path + args.exp_name
    delete_folder(dir_path)
    ##########################################

    dataset = PygNodePropPredDataset_custom(name=args.exp_name, root=root_path,
                                            numofClasses=dataset_numofClasses)

    print(getrusage(RUSAGE_SELF))
    start_t = datetime.datetime.now()
    data = dataset[0]
    global subject_node
    subject_nodes = list(data.y_dict.keys())
    split_idx = dataset.get_idx_split('node_type')
    end_t = datetime.datetime.now()
    print("dataset init time=", end_t - start_t, " sec.")
    evaluator = Evaluator(name='ogbn-mag', p_eval_metric='acc')
    evaluator_cm = Evaluator(name='ogbn-mag', p_eval_metric='cm')


    # We do not consider those attributes for now.
    data.node_year_dict = None
    data.edge_reltype_dict = None

    to_remove_rels = []
    for keys, (row, col) in data.edge_index_dict.items():
        if (keys[2] in to_remove_subject_object) or (keys[0] in to_remove_subject_object):
            # print("to remove keys=",keys)
            to_remove_rels.append(keys)

    for keys, (row, col) in data.edge_index_dict.items():
        if (keys[1] in to_remove_pedicates):
            # print("to remove keys=",keys)
            to_remove_rels.append(keys)
            to_remove_rels.append((keys[2], '_inv_' + keys[1], keys[0]))

    for elem in to_remove_rels:
        data.edge_index_dict.pop(elem, None)
        data.edge_reltype.pop(elem, None)

    for key in to_remove_subject_object:
        data.num_nodes_dict.pop(key, None)

    # dic_results[dataset_name]["data"] = str(data)
    edge_index_dict = data.edge_index_dict

    key_lst = list(edge_index_dict.keys())
    # if not args.consider_direction:
    ##############add inverse edges ###################
    for key in key_lst:
        r, c = edge_index_dict[(key[0], key[1], key[2])]
        edge_index_dict[(key[2], 'inv_' + key[1], key[0])] = torch.stack([c, r])

    out = group_hetero_graph(data.edge_index_dict, data.num_nodes_dict)
    edge_index, edge_type, node_type, local_node_idx, local2global,key2int  = out
    ######### remove empty edges ##############
    num_relations = len(edge_type.unique())
    print("Number of relations", num_relations)
    # edge_weight = None
    # if args.edge_weight:
    #     weight_lst = ([1] * int(num_relations/2)) + ([0.2] * int(num_relations/2))
    #     edge_weight = torch.tensor(weight_lst, dtype=torch.float)
    # mask = (edge_index[0][:] != -1)
    # edge_index = torch.stack((edge_index[0][mask], edge_index[1][mask]))
    # edge_type = edge_type[mask]

    homo_data = Data(edge_index=edge_index, edge_attr=edge_type,
                     node_type=node_type, local_node_idx=local_node_idx,
                     num_nodes=node_type.size(0))

    homo_data.y = node_type.new_full((node_type.size(0), 1), -1)
    for subject_node in subject_nodes:
        homo_data.y[local2global[subject_node]] = data.y_dict[subject_node]

    homo_data.active_mask = torch.zeros((node_type.size(0)), dtype=torch.bool)
    homo_data.active_mask[:] = True
    homo_data.train_mask = torch.zeros((node_type.size(0)), dtype=torch.bool)
    homo_data.val_mask = torch.zeros((node_type.size(0)), dtype=torch.bool)
    homo_data.test_mask = torch.zeros((node_type.size(0)), dtype=torch.bool)
    subject_nodes_num,subject_nodes_mal = {},{}
    for subject_node in subject_nodes:
        homo_data.train_mask[local2global[subject_node][split_idx['train'][subject_node]]] = True
        homo_data.val_mask[local2global[subject_node][split_idx['valid'][subject_node]]] = True
        homo_data.test_mask[local2global[subject_node][split_idx['test'][subject_node]]] = True
        subject_nodes_num[subject_node] = local2global[subject_node][split_idx['valid'][subject_node]].size()[0]
        subject_nodes_mal[subject_node] = sum(homo_data.y[local2global[subject_node]][split_idx['valid'][subject_node]])
    subject_nodes_num = dict(sorted(subject_nodes_num.items(), key=lambda item: item[1],reverse=True))
    subject_nodes_mal = dict(sorted(subject_nodes_mal.items(), key=lambda item: item[1],reverse=True))
    subject_nodes = list(subject_nodes_mal.keys())
    #remove malicious samples from training
    print("Remove ", sum(homo_data.y[homo_data.train_mask]).item(), " malicious samples from training set")
    homo_data.train_mask[homo_data.y.view(-1) == 1] = False
    contamination = data_statistics_contamination(homo_data)
    if not args.dynamic_contamination:
        contamination = args.contamination
    #######################intialize features ###############################
    if not args.features_per_node_type:
        if args.random_features:
            input_layer = args.input_layer
            feat = torch.zeros((node_type.size(0)), input_layer)
            torch.nn.init.xavier_uniform_(feat)
        else:
            temp_feat = load_features(subject_nodes[0])
            input_layer = temp_feat.size(1)
            feat = torch.zeros((node_type.size(0)), input_layer)
            del temp_feat
            for subject_node in subject_nodes:
                temp_feat = load_features(subject_node)
                feat[local2global[subject_node]] = temp_feat
        print("Size of input layers:", input_layer)
        homo_data.x = deepcopy(feat)

    print("Data:", homo_data)
    print(key2int)
    # num_nodes_dict = {}
    # for key, N in data.num_nodes_dict.items():
    #     num_nodes_dict[key2int[key]] = N

    # Start Training the PyGOD model
    y_true = deepcopy(homo_data.y)
    homo_data.y = homo_data.y.bool()
    debug_subjects = args.debug_subjects
    # del homo_data.y
    global final_y_pred_testing,final_y_prob_testing,final_score_testing
    global final_emb
    final_emb = pd.DataFrame()
    final_y_pred_testing = torch.zeros((y_true.size()), dtype=int)
    final_y_prob_testing = torch.zeros((y_true.size(0)))
    final_score_testing = torch.zeros((y_true.size(0)))
    if args.save_emb and run == 0:
        # final_emb = pd.DataFrame({'node_id': range(homo_data.num_nodes)})
        mapping_nodes_df = pd.DataFrame()
        for subject_node in subject_nodes:
            file_path = root_path + args.exp_name + "/mapping/" + subject_node + "_entidx2name.csv"
            mapping_nodes_df_tmp = pd.read_csv(file_path, header=None, skiprows=1, names=["node_id", "node_uuid"])
            mapping_nodes_df_tmp["node_id"] = local2global[subject_node][mapping_nodes_df_tmp["node_id"]]
            # mapping_nodes_df_tmp["node_type"] = subject_node
            mapping_nodes_df = pd.concat([mapping_nodes_df, mapping_nodes_df_tmp])
        # final_emb = pd.merge(final_emb, mapping_nodes_df, on="node_id")
    if args.load_model:
        out_path = root_path + "figures/" + args.exp_name + "/" + args.load_model.replace(".model", "")+"/"
        ensure_dir(out_path)
        if args.multiple_models:
            all_tp, all_tn, all_fp, all_fn = 0,0,0,0
            detection_time_run = 0
            auc_dic,avg_precision,precision_at_k,recall_at_k = {},[],[],[]
            for subject_node in subject_nodes:
                print("****************************************")
                print("Testing with model for node type:", subject_node)
                model_path = root_path + "models/" +args.exp_name + "/" + subject_node + "_"+ args.load_model
                detector = torch.load(model_path)
                homo_data_per_type,contamination = split_per_node_type(homo_data,subject_node)
                anomaly_results_df_subject, y_pred, score, y_prob,detection_time_subject,test_acc = test(detector, homo_data_per_type,analyze_per_type=False,fig_title=str(out_path+"_"+subject_node))
                # if run == 0:
                #     log_raised_alarms(homo_data_per_type, y_pred, score, y_prob, subject_node)
                log_raised_alarms(homo_data_per_type, y_pred, score, y_prob, target_node=subject_node,run=run)
                all_tp += anomaly_results_df_subject['tp'].item()
                all_tn += anomaly_results_df_subject['tn'].item()
                all_fp += anomaly_results_df_subject['fp'].item()
                all_fn += anomaly_results_df_subject['fn'].item()
                detection_time_run += detection_time_subject
                if 'auc' in anomaly_results_df_subject.columns:
                    auc_dic[subject_node] = anomaly_results_df_subject['auc'].item()
                if 'avg_precision' in anomaly_results_df_subject.columns:
                    avg_precision.append(anomaly_results_df_subject['avg_precision'].item())
                if 'precision_at_k' in anomaly_results_df_subject.columns:
                    precision_at_k.append(anomaly_results_df_subject['precision_at_k'].item())
                if 'recall_at_k' in anomaly_results_df_subject.columns:
                    recall_at_k.append(anomaly_results_df_subject['recall_at_k'].item())
                if args.save_emb and run == 0:
                    map_save_embedding(homo_data_per_type, detector.emb, mapping_nodes_df, subject_node)
                del detector, homo_data_per_type, anomaly_results_df_subject, y_pred, score, y_prob,detection_time_subject, test_acc
            print("****************************************")
            anomaly_results_df_run = print_predict_evaluation_metrics(all_tp, all_tn, all_fp, all_fn,"Overall Evaluation results")
            anomaly_results_df_run['avg_auc'] = mean(auc_dic.values())
            print("Average AUC:", mean(auc_dic.values()))
            print("Average avg_precision:", mean(avg_precision))
            print("Average precision at :",args.top_k,"is", mean(precision_at_k))
            print("Average recall at :",args.top_k,"is", mean(recall_at_k))
            print("Total detection time is:",detection_time_run)
        else:
            if args.ensemble_models:
                y_pred_lst,score_lst,threshold_lst = [],[],[]
                detection_time_run = 0
                anomaly_results_df = pd.DataFrame()
                for run in range(args.runs):
                    model_path = root_path + "models/" +args.exp_name + "/" + args.load_model
                    detector = torch.load(model_path)
                    anomaly_results_df_run, y_pred, score, y_prob,detection_time_run_tmp, test_acc = test(detector, homo_data)
                    y_pred_lst.append(y_pred)
                    score_lst.append(score)
                    threshold_lst.append(detector.threshold_)
                    anomaly_results_df = pd.concat([anomaly_results_df, anomaly_results_df_run])
                    detection_time_run += detection_time_run_tmp
                # get average & majority voting
                y_true_testing = y_true[homo_data.test_mask]
                test_ensemble_models(y_true_testing, y_pred_lst, score_lst, threshold_lst)
                print("Average performance across all runs:")
                print(anomaly_results_df.describe())
                print("Total Detection time is:",detection_time_run)
            else:
                model_path = root_path + "models/"+args.exp_name+ "/" + args.load_model
                detector = torch.load(model_path)
                anomaly_results_df_run, y_pred, score, y_prob,detection_time_run, test_acc = test(detector, homo_data,fig_title=out_path)
                # if run == 0:
                #     log_raised_alarms(homo_data, y_pred, score, y_prob)
                log_raised_alarms(homo_data, y_pred, score, y_prob, run=run)
                print("Detection time is:", detection_time_run)
                if args.save_emb and run == 0:
                    map_save_embedding(homo_data, detector.emb, mapping_nodes_df)
    else:
        out_path = root_path + "figures/" + args.exp_name + "/"+ args.save_model.replace(".model","") +"/"
        ensure_dir(out_path)
        if args.multiple_models:
            all_tp, all_tn, all_fp, all_fn = 0, 0, 0, 0
            detection_time_run,train_time_run = 0,0
            auc_dic,avg_precision,precision_at_k,recall_at_k = {},[],[],[]
            for subject_node in subject_nodes:
                seed_everything(seed)
                print("****************************************")
                print("Training a model for node type:",subject_node)
                # if subject_nodes_mal[subject_node].item() == 0 and all_tp ==0:
                #     print("Couldn't detect any true alarms")
                #     break
                homo_data_per_type,contamination = split_per_node_type(homo_data, subject_node)
                num_relations = len(homo_data_per_type.edge_attr.unique())
                if args.batch_size_percentage:
                    batch_size = int(homo_data_per_type.num_nodes * args.batch_size_percentage)
                else:
                    batch_size = args.batch_size
                print("Used Batch size is: ",batch_size)
                print("Number of relations is:", num_relations)
                detector = inintailize_model(contamination,num_relations,batch_size)
                detector,train_time = train(detector, homo_data_per_type,fig_title=str(out_path+"_"+subject_node))
                # Seed after training to minimize affects on testing
                # seed_everything(seed)
                result = validate(detector, homo_data_per_type,fig_title=str(out_path+"_"+subject_node))
                anomaly_results_df_subject, y_pred, score, y_prob,detection_time_subject, test_acc = test(detector, homo_data_per_type,analyze_per_type=False,fig_title=str(out_path+"_"+subject_node))
                result = result + (test_acc,)
                # if run == 0:
                #     log_raised_alarms(homo_data_per_type, y_pred, score, y_prob,subject_node)
                log_raised_alarms(homo_data_per_type, y_pred, score, y_prob, target_node=subject_node,run=run)
                print(f' Train accuracy: {result[0]:.3f}')
                print(f' Valid accuracy: {result[1]:.3f}')
                print(f' Test accuracy: {result[2]:.3f}')
                all_tp += anomaly_results_df_subject['tp'].item()
                all_tn += anomaly_results_df_subject['tn'].item()
                all_fp += anomaly_results_df_subject['fp'].item()
                all_fn += anomaly_results_df_subject['fn'].item()
                detection_time_run += detection_time_subject
                train_time_run += train_time
                if 'auc' in anomaly_results_df_subject.columns:
                    auc_dic[subject_node] =  anomaly_results_df_subject['auc'].item()
                if 'avg_precision' in anomaly_results_df_subject.columns:
                    avg_precision.append(anomaly_results_df_subject['avg_precision'].item())
                if 'precision_at_k' in anomaly_results_df_subject.columns:
                    precision_at_k.append(anomaly_results_df_subject['precision_at_k'].item())
                if 'recall_at_k' in anomaly_results_df_subject.columns:
                    recall_at_k.append(anomaly_results_df_subject['recall_at_k'].item())
                if args.debug_one_subject:
                    del detector
                    break
                if debug_subjects !=-1:
                    debug_subjects -= 1
                    if debug_subjects == 0:
                        del detector
                        break
                if run == 0:
                    model_path = root_path + "models/" + args.exp_name + "/" + subject_node + "_" + args.save_model
                    ensure_dir(model_path)
                    print("Saving the model to", model_path)
                    torch.save(detector, model_path)
                if args.save_emb and run == 0:
                    map_save_embedding(homo_data_per_type,detector.emb,mapping_nodes_df,subject_node)
                del detector, homo_data_per_type, anomaly_results_df_subject, y_pred, score, y_prob,detection_time_subject, test_acc , result


            print("****************************************")
            anomaly_results_df_run = print_predict_evaluation_metrics(all_tp, all_tn, all_fp, all_fn,"Overall Evaluation results")
            anomaly_results_df_run['avg_auc'] = mean(auc_dic.values())
            # anomaly_results_df_run['auc_dic'] = auc_dic
            print("Average AUC:", mean(auc_dic.values()))
            print("Average avg_precision:", mean(avg_precision))
            print("Average precision at :", args.top_k, "is", mean(precision_at_k))
            print("Average recall at :", args.top_k, "is", mean(recall_at_k))
        else:
            seed_everything(seed)
            if args.batch_size_percentage:
                batch_size = int(homo_data.num_nodes * args.batch_size_percentage)
            else:
                batch_size = args.batch_size
            print("Used Batch size is: ", batch_size)
            detector = inintailize_model(contamination,num_relations,batch_size)
            detector,train_time_run = train(detector, homo_data,fig_title=out_path)
            # Seed after training to minimize affects on testing
            # seed_everything(seed)
            result = validate(detector, homo_data,fig_title=out_path)
            anomaly_results_df_run, y_pred, score, y_prob, detection_time_run, test_acc = test(detector, homo_data,fig_title=out_path)
            result = result + (test_acc,)
            # if run == 0:
            #     log_raised_alarms(homo_data, y_pred, score, y_prob)
            log_raised_alarms(homo_data, y_pred, score, y_prob,run=run)
            # if args.ensemble_models:
            #     y_pred_lst.append(y_pred)
            #     score_lst.append(score)
            #     threshold_lst.append(detector.threshold_)
            logger.add_result(run, result)
            logger.print_statistics(run)
            if run == 0:
                model_path = root_path + "models/" + args.exp_name + "/" + args.save_model
                ensure_dir(model_path)
                print("Saving the model to", model_path)
                torch.save(detector, model_path)
            if args.save_emb and run == 0:
                map_save_embedding(homo_data, detector.emb, mapping_nodes_df)
            del detector
    # Confirm final overall results
    test_cm = evaluator_cm.eval({
        'y_true': y_true[homo_data.test_mask],
        'y_pred': final_y_pred_testing[homo_data.test_mask],
        'labels': [0, 1]
    })['cm']
    print("\n*****************************************")
    print('Overall Testing set CM:\n', test_cm)
    tp, fp, tn, fn = man_confusion_matrix(y_true=y_true[homo_data.test_mask], y_pred=final_y_pred_testing[homo_data.test_mask])
    print_predict_evaluation_metrics(tp, tn, fp, fn)
    if args.consider_related_nodes:
        alerts_2hop,anomaly_2hop_results_df = predict_with_related_nodes(homo_data,y_true,final_y_pred_testing)
        anomaly_results_df_run[['accuracy_2hop','precision_2hop','recall_2hop','f_measure_2hop','tp_2hop','tn_2hop','fp_2hop','fn_2hop','tpr_2hop','fpr_2hop']] = anomaly_2hop_results_df[['accuracy','precision','recall','f_measure','tp','tn','fp','fn','tpr','fpr']]
    final_auc = eval_roc_auc(y_true[homo_data.test_mask], final_score_testing[homo_data.test_mask])
    print("Final AUC is:",final_auc)
    # if run == 0:
    #     if args.consider_related_nodes:
    #         log_raised_alarms(homo_data, final_y_pred_testing[homo_data.test_mask],
    #                           final_score_testing[homo_data.test_mask], final_y_prob_testing[homo_data.test_mask],
    #                           target_node=None,alerts_2hop=alerts_2hop)
    #     else:
    #         log_raised_alarms(homo_data, final_y_pred_testing[homo_data.test_mask],
    #                           final_score_testing[homo_data.test_mask], final_y_prob_testing[homo_data.test_mask],
    #                           target_node=None)
    if args.consider_related_nodes:
        log_raised_alarms(homo_data, final_y_pred_testing[homo_data.test_mask],
                          final_score_testing[homo_data.test_mask], final_y_prob_testing[homo_data.test_mask],
                          target_node=None, alerts_2hop=alerts_2hop,run=run)
    else:
        log_raised_alarms(homo_data, final_y_pred_testing[homo_data.test_mask],
                          final_score_testing[homo_data.test_mask], final_y_prob_testing[homo_data.test_mask],
                          target_node=None,run=run)
    if args.correlate_subgraphs:
        pass
    if args.save_emb and run == 0:
        print("Number of prepared node embeddings",len(final_emb))
        del final_emb["node_id"]
        emb_path = root_path + "embedding/" + args.exp_name + "/" + (args.save_model.replace(".model",".csv"))
        ensure_dir(emb_path)
        print("Saving the embedding to", emb_path)
        final_emb.to_csv(emb_path, index=None)

    # get average & majority voting
    # if args.ensemble_models:
    #     y_true_testing = y_true[homo_data.test_mask]
        # test_ensemble_models(y_true_testing,y_pred_lst,score_lst,threshold_lst)
    # print("\n************************************************************")
    # if not args.multiple_models:
    #     logger.print_statistics()
    # print("\n************************************************************")
    # print("Average performance across all runs:")
    # print(anomaly_results_df.describe())
    # log_raised_alarms(homo_data, y_pred, score, y_prob)
    # print("Average detection time", mean(detection_time_lst))
    # print("Average training time",mean(train_time_lst))

    ###################################Delete Folder if exist #############################
    ### DEBUG: try without deleting #############
    dir_path = args.root_path + args.exp_name
    delete_folder(dir_path)

    anomaly_results_df_run["run"] = str(run)
    anomaly_results_df_run["detection_time_run"] = detection_time_run
    if not args.load_model:
        anomaly_results_df_run["train_time_run"] = train_time_run
    anomaly_results_df_run["memory_usage_mb"] = getrusage(RUSAGE_SELF).ru_maxrss / 1024
    del homo_data, data
    return anomaly_results_df_run


if __name__ == '__main__':
    seed = 360
    start_time = time.time()
    logger = Logger(args.runs, args)
    anomaly_results_df = pd.DataFrame()
    for run in range(args.runs):
        print("******************************************")
        print("Run number:", run)
        print("Seed: ", seed)
        anomaly_results_df_run = pyGod(run, seed)
        anomaly_results_df = pd.concat([anomaly_results_df, anomaly_results_df_run])
        print("Total detection time for this run is", anomaly_results_df_run["detection_time_run"])
        if not args.load_model:
            print("Total training time for this run ", anomaly_results_df_run["train_time_run"])
        seed = np.random.randint(0, 1000)
    # if args.save_model:
    print("\n************************************************************")
    if not args.multiple_models and not args.load_model:
        logger.print_statistics()
    print("\n************************************************************")
    print("Average performance across all runs:")
    first_column = anomaly_results_df.pop('run')
    print(anomaly_results_df.describe())
    mean_results = anomaly_results_df.mean(0)
    mean_results['run'] = "avg"
    std_results = anomaly_results_df.std(0)
    std_results['run'] = 'std'
    anomaly_results_df.insert(0, 'run', first_column)
    anomaly_results_df.loc[len(anomaly_results_df.index)] = mean_results
    anomaly_results_df.loc[len(anomaly_results_df.index)] = std_results
    if args.load_model:
        out_path = root_path + "results/" + args.exp_name +"/"+ args.load_model.replace(".model","")+ "/anomaly_results_summary.csv"
    else:
        out_path = root_path + "results/" + args.exp_name + "/" + args.save_model.replace(".model","") + "/anomaly_results_summary.csv"
    ensure_dir(out_path)
    anomaly_results_df.to_csv(out_path, index=None)
    total_running_time = time.time() - start_time
    print("Total time: ", total_running_time, "seconds.")
    io_counters = process.io_counters()
    program_IOPs = (io_counters[0] + io_counters[1]) / total_running_time
    print("program IOPS (over total time): ", program_IOPs)
    print("I/O counters", io_counters)
    print_memory_usage()


