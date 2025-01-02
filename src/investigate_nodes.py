import psutil
from resource import *
import time
from statistics import mean
import torch
import matplotlib.pyplot as plt
import argparse
import io
import networkx as nx
from torch.autograd import detect_anomaly
from torch_geometric.seed import seed_everything
import dgl
import pandas as pd
from torch_geometric.data import Data
import numpy as np
from datetime import datetime , timedelta
import os
import itertools
from IPython.display import display
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('max_colwidth', None)
pd.set_option('display.max_rows', 20)
import logging
from sparql_queries import get_extraction_queries
from SPARQLWrapper import SPARQLWrapper , JSON, POST, BASIC
from layers import AttentionModule
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from dataset_pyg_custom import PygNodePropPredDataset_custom
from torch_geometric.utils.hetero import group_hetero_graph
from sklearn.decomposition import PCA
from database_config import get_attack_time_range
import pytz
my_tz = pytz.timezone('America/Nipigon')
import math
from random import sample, choice
import glob
import json
from networkx.readwrite import json_graph
from database_config import get_subgraphs_attributes
from outliers import smirnov_grubbs as grubbs
from scipy import stats
# import multiprocessing
# from dask.distributed import Client, LocalCluster
# import dask.bag as db
# import gc
# import ctypes
def read_json_graph(filename):
    with open(filename) as f:
        js_graph = json.load(f)
    return json_graph.node_link_graph(js_graph)
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
def checkpoint(data, file_path):
    file_path = rename_path_with_param(file_path)
    ensure_dir(file_path)
    torch.save(data, file_path)
    return
def checkpoint_save_csv(data, file_path):
    file_path = rename_path_with_param(file_path)
    ensure_dir(file_path)
    data.to_csv(file_path, index=False)
    return

def clear_globals():
    global_vars = list(globals().keys())
    for var in global_vars:
        if var not in ['__builtins__', '__name__', '__doc__', '__package__', '__loader__', '__spec__', '__annotations__', '__file__', '__cached__']:
            del globals()[var]

parser = argparse.ArgumentParser(description='OCR-APT')
parser.add_argument('--dataset', type=str,required=True)
parser.add_argument('--host', type=str,required=True)
parser.add_argument('--runs', type=int, default=3)
# parser.add_argument('--graph-optc', action="store_true", default=False)
# parser.add_argument('--graph-nodlink', action="store_true", default=False)
parser.add_argument('--parallel', help='Encode Subgraphs in parallel', action="store_true", default=False)
parser.add_argument('--exp-name', type=str, required=True)
parser.add_argument('--inv-exp-name', type=str, default=None)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--root-path', type=str, required=True)
# parser.add_argument('--source-graph', type=str, default="cadets")
parser.add_argument('--number-of-hops', type=int, default=1)
parser.add_argument('--anomalies-path-only', action="store_true", default=False)
parser.add_argument('--backwards-only', action="store_true", default=False)
parser.add_argument('--anomalies-or-process-only', action="store_true", default=False)
parser.add_argument('--retraverse-disconnected-nodes', action="store_true", default=False)
parser.add_argument('--node-emb-size', type=int, default=64)
parser.add_argument('--tensor-neurons', type=int, default=64)
# parser.add_argument('--num-layers', type=int, default=2)
parser.add_argument('--node-emb-per-subgraph', action="store_true", default=False)
parser.add_argument('--draw-subgraphs', action="store_true", default=False)
parser.add_argument('--get-node-attrs', action="store_true", default=False)
parser.add_argument('--subgraphs-sample', type=int, default=0)
parser.add_argument('--controlled-construction', action="store_true", default=False)
parser.add_argument('--construct-from-anomaly-subgraph', action="store_true", default=False)
parser.add_argument('--remove-duplicated-subgraph', action="store_true", default=False)
parser.add_argument('--correlate-anomalous-once', action="store_true", default=False)
parser.add_argument('--unvisited-seeds', action="store_true", default=False)
parser.add_argument('--process-centric', action="store_true", default=False)
# parser.add_argument('--max-nodes-mult-qg', type=int, help='Maximum number of nodes for subgraphs', default=10)
# parser.add_argument('--max-edges-mult-qg', type=int, help='Maximum number of edges for subgraphs', default=25)
parser.add_argument('--max-edges', type=int, help='Maximum number of edges for subgraphs', default=10000)
parser.add_argument('--top-k', type=int, default=15)
parser.add_argument('--top-k-percent', type=float, default=None)
parser.add_argument('--top-k-anomalous-subgraphs', type=int, default=10)
parser.add_argument('--min-edges', type=int, help='Minimum number of edges for subgraphs', default=3)
parser.add_argument('--min-nodes', type=int, help='Minimum number of nodes for subgraphs', default=3)
parser.add_argument('--cluster-subgraphs', action="store_true", default=False)
parser.add_argument('--consider-related-nodes', action="store_true", default=False)
parser.add_argument('--expand-2-hop', type=str,default="no")

args = parser.parse_args()
assert args.dataset in ['tc3', 'optc', 'nodlink']
assert args.host in ['cadets', 'trace', 'theia', 'fivedirections','SysClient0051','SysClient0501','SysClient0201','SimulatedUbuntu','SimulatedW10','SimulatedWS12']

process = psutil.Process(os.getpid())

if args.dataset == "optc":
    SourceDataset = "DARPA_OPTC"
elif args.dataset == "nodlink":
    SourceDataset = "NODLINK"
else:
    SourceDataset = "DARPA_TC3"

prefix = "https://"+SourceDataset+".graph/" + args.host + "/"
with open("../config.json", "r") as f:
    config = json.load(f)
    if args.dataset == "optc":
        repository_url = config["repository_url_optc"]
    elif args.dataset == "nodlink":
        repository_url = config["repository_url_nodlink"]
    else:
        repository_url = config["repository_url_tc3"]
sparql_queries = get_extraction_queries(args.host,SourceDataset)

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

def analyze_anomaly_score(anomalies_nodes,node_type=None):
    if node_type :
        anomalies_nodes = anomalies_nodes[anomalies_nodes["node_type"] == node_type]
    true_alarms_sorted = np.sort(anomalies_nodes.loc[anomalies_nodes["Label"] == True]["Prediction_probability"])
    false_alarms_sorted = np.sort(anomalies_nodes.loc[anomalies_nodes["Label"] == False]["Prediction_probability"])
    if len(true_alarms_sorted) == 0:
        print("No true alarms detected for node type",node_type)
        return
    if len(false_alarms_sorted) == 0:
        print("No false alarms detected for node type",node_type)
        return
    # Calculate the cumulative probabilities
    true_alarms_cdf = np.linspace(0, 1, len(true_alarms_sorted))
    false_alarms_cdf = np.linspace(0, 1, len(false_alarms_sorted))
    # Plotting
    if len(true_alarms_sorted) == 1:
        plt.scatter(true_alarms_sorted, true_alarms_cdf, label='True Alarms')
        plt.plot(false_alarms_sorted, false_alarms_cdf, label='False Alarms',color='orange')
    elif len(false_alarms_sorted) == 1:
        plt.plot(true_alarms_sorted, true_alarms_cdf, label='False Alarms')
        plt.scatter(false_alarms_sorted, false_alarms_cdf, label='True Alarms', color='orange')
    else:
        plt.plot(true_alarms_sorted, true_alarms_cdf, label='True Alarms')
        plt.plot(false_alarms_sorted, false_alarms_cdf, label='False Alarms')


    # Add labels and legend
    plt.xlabel('Anomaly Score')
    plt.ylabel('CDF')
    if node_type :
        plt.title('CDF of anomaly scores for True Alarms and False Alarms, for node type {}'.format(node_type))
    else:
        plt.title('CDF of anomaly scores for True Alarms and False Alarms')
    plt.legend()
    # Show plot
    plt.grid(True)
    if node_type:
        out_path = args.root_path + "figures/" + args.exp_name + "/" + args.model.replace(".model","") + "/" + node_type + "_CDF_AnomalyScore_ExpParam.pdf"
    else:
        out_path = args.root_path + "figures/" + args.exp_name + "/" + args.model.replace(".model","") + "/CDF_AnomalyScore_ExpParam.pdf"
    out_path = rename_path_with_param(out_path)
    ensure_dir(out_path)
    plt.savefig(out_path)
    # plt.show()
    plt.close()

    return
def read_anomalies_nodes(testing_uuid,run):
    global anomalies_nodes_uuid, malicious_nodes_uuid , malicious_nodes
    anomalies_path = args.root_path + "results/" + args.exp_name + "/" + args.model.replace(".model","") +"/run_" + str(run) + "_raised_alarms.csv"
    anomalies_nodes = pd.read_csv(anomalies_path)
    anomalies_nodes_uuid = anomalies_nodes["node_uuid"].unique().tolist()
    print("Number of anomalies nodes: ", len(anomalies_nodes_uuid))
    ## debug ##
    read_sparql = SPARQLWrapper(repository_url)
    read_sparql.setReturnFormat(JSON)
    read_sparql.setQuery(sparql_queries['get_malicious_nodes'])
    results_df = read_sparql.queryAndConvert()
    malicious_nodes = map_sparql_query(results_df)
    malicious_nodes.loc[:, 'node_type'] = malicious_nodes['node_type'].str.split('/').str[-1]
    malicious_nodes = malicious_nodes[malicious_nodes["node_uuid"].isin(testing_uuid)]
    malicious_nodes_uuid = malicious_nodes['node_uuid'].unique()
    # malicious_nodes_uuid = [node for node in malicious_nodes['node_uuid'].unique() if node in testing_uuid]
    print("malicious nodes per node type:",malicious_nodes["node_type"].value_counts())

    # if args.dataset == "optc":
    #     file_path = args.root_path + "optc_ground_truth.txt"
    #     with open(file_path, 'r') as file:
    #         malicious_nodes_uuid = list(set(file.read().split()))
    #     malicious_nodes_uuid = [node for node in malicious_nodes_uuid if node in testing_uuid]
    #     ############# fix_node_uuid ################
    #     # base_testing_uuid = ["-".join(node_uuid.split("-")[0:-1]) for node_uuid in testing_uuid]
    #     # base_malicious_nodes_uuid = [node for node in malicious_nodes_uuid if node in base_testing_uuid]
    #     # malicious_nodes_uuid = [node for node in testing_uuid if "-".join(node.split("-")[0:-1]) in base_malicious_nodes_uuid]
    #     ############################################
    # else:
    #     ground_path = args.root_path + args.host + "_ground_truth.txt"
    #     malicious_nodes_df = pd.read_csv(ground_path, header=None)
    #     malicious_nodes_uuid = malicious_nodes_df[0].unique().tolist()

    print("Number of benign nodes predicted anomalies: ", len(anomalies_nodes.loc[anomalies_nodes["Label"] == False]))
    print("Number of malicious nodes predicted anomalies: ", len(anomalies_nodes.loc[anomalies_nodes["Label"] == True]))
    print("Number of nodes labelled malicious: ", len(malicious_nodes_uuid))
    analyze_anomaly_score(anomalies_nodes)
    for node_type in anomalies_nodes["node_type"].unique():
        analyze_anomaly_score(anomalies_nodes,node_type)
    return anomalies_nodes


def clear_and_insert_anomalies(anomalies_nodes):
    write_sparql = SPARQLWrapper(repository_url + "/statements")
    # Delete previous Anomalies labels
    write_sparql.method = "POST"
    write_sparql.setReturnFormat('json')
    write_sparql.queryType = "DELETE"
    write_sparql.setQuery(query=sparql_queries['Delete_Anomalies_Labels'])
    results = write_sparql.query()
    # Annotate Anomalies nodes
    turtle_anomalies = anomalies_nodes["node_uuid"].apply(
        lambda uuid: 'node:' + str(uuid) + ' graph:anomalies "True" .')
    for n in range(int(len(turtle_anomalies) / 1000) + 1):
        start_bulk = n * 1000
        end_bulk = start_bulk + 1000
        if end_bulk > len(turtle_anomalies):
            end_bulk = len(turtle_anomalies)
        bulk_data = turtle_anomalies[start_bulk:end_bulk].to_string(index=False)
        write_sparql.setQuery(sparql_queries['Insert_Anomalies_Labels'].replace("<TRIPLES>", bulk_data))
        results = write_sparql.query()
    read_sparql = SPARQLWrapper(repository_url)
    read_sparql.setReturnFormat(JSON)
    read_sparql.setQuery(sparql_queries['Count_Anomalies_Nodes'])
    results = read_sparql.queryAndConvert()
    print("Number of inserted anomalies nodes:", results['results']['bindings'][0]['count_anomalies']['value'])
    return

def convert_timestamp_to_datetime(timestamp):
    # DEBUG: To be verified
    if args.host == "SimulatedW10":
        base_datetime = datetime(2022, 4, 8, 13, 0, 0 ,0)
    elif args.host == "SimulatedWS12":
        base_datetime = datetime(2022, 3, 16, 13, 28, 4, 0)
    this_datetime = base_datetime + timedelta(seconds=((int(float(timestamp))/1000)))
    return this_datetime.strftime("%Y-%m-%d %H:%M:%S")

def result_df_to_subgraph_nx(results_df):
    results_df.loc[:, 'predicate'] = results_df['predicate'].str.split('/').str[-1]
    results_df = results_df.rename(columns={'predicate': 'action'})
    results_df.loc[:, 'subject_type'] = results_df['subject_type'].str.split('/').str[-1]
    results_df.loc[:, 'object_type'] = results_df['object_type'].str.split('/').str[-1]
    nodes_df = get_subgraph_node_df(results_df)
    if args.get_node_attrs:
        nodes_df = get_node_attr_by_query(nodes_df)
    # nodes_df = get_subgraph_node_attr(results_df)
    print("Total number of triples before dropping node attributes", len(results_df))
    if 'subject_attr' in results_df.columns:
        results_df = results_df.drop(columns=['subject_attr'])
    if 'object_attr' in results_df.columns:
        results_df = results_df.drop(columns=['object_attr'])
    results_df = results_df.drop_duplicates()
    print("Total number of triples after dropping node attributes", len(results_df))
    date_format = '%Y-%m-%d %H:%M:%S'
    if args.dataset == "optc":
        results_df["timestamp"] = results_df["timestamp"].apply(lambda x: x[:19])
    elif args.host in ["SimulatedW10","SimulatedWS12"]:
        print("The timestamp format of host", args.host," is not accurate, need to be fixed")
        results_df["timestamp"] = results_df["timestamp"].apply(lambda x:convert_timestamp_to_datetime(x))
    else:
        results_df["timestamp"] = results_df["timestamp"].apply(
            lambda x: datetime.fromtimestamp(int(x) // 1000000000, tz=pytz.timezone("America/Nipigon"))).dt.floor('S')
        results_df['timestamp'] = results_df['timestamp'].apply(lambda t: t.strftime(date_format))
    print("Total number of triples before dropping duplicated actions (within one second)", len(results_df))
    results_df = results_df.drop_duplicates()
    print("Total number of triples after dropping duplicated actions (within one second)", len(results_df))
    subgraph_obj = nx.from_pandas_edgelist(
        results_df,
        source="subject_uuid",
        target="object_uuid",
        edge_attr=["action", "timestamp"],
        create_using=nx.MultiDiGraph()
    )

    return subgraph_obj, nodes_df

def get_subgraph_node_df(results_df):
    results_df = results_df.replace({np.nan: None})
    nodes_df_s = results_df[['subject_uuid', 'subject_type']]
    nodes_df_s.columns = ["node_uuid", "node_type"]
    nodes_df_o = results_df[['object_uuid', 'object_type']]
    nodes_df_o.columns = ["node_uuid", "node_type"]
    nodes_df = pd.concat([nodes_df_s, nodes_df_o], ignore_index=True).drop_duplicates(subset=['node_uuid'])
    nodes_df = pd.concat([nodes_df, anomalies_nodes[["node_uuid", "node_type"]]], ignore_index=True).drop_duplicates(
        subset=['node_uuid'])
    return nodes_df

def get_subgraph_node_attr(results_df):
    results_df = results_df.replace({np.nan: None})
    # get nodes attributes
    if args.get_node_attrs:
        if 'subject_attr' in results_df.columns:
            nodes_df_s = results_df[['subject_uuid', 'subject_type', 'subject_attr']].drop_duplicates(
                subset=['subject_uuid'])
            # DEBUG: why split by ":" !?
            nodes_df_s['subject_attr'] = nodes_df_s['subject_attr'].str.split(':').str[0]
            nodes_df_s.columns = ["node_uuid", "node_type", 'node_attr']
        else:
            nodes_df_s = results_df[['subject_uuid', 'subject_type']].drop_duplicates(subset=['subject_uuid'])
            nodes_df_s.columns = ["node_uuid", "node_type"]
            nodes_df_s["node_attr"] = np.nan
        if 'object_attr' in results_df.columns:
            nodes_df_o = results_df[['object_uuid', 'object_type', 'object_attr']].drop_duplicates(
                subset=['object_uuid'])
            nodes_df_o['object_attr'] = nodes_df_o['object_attr'].str.split(':').str[0]
            nodes_df_o.columns = ["node_uuid", "node_type", 'node_attr']
        else:
            nodes_df_o = results_df[['object_uuid', 'object_type']].drop_duplicates(subset=['object_uuid'])
            nodes_df_o.columns = ["node_uuid", "node_type"]
            nodes_df_o["node_attr"] = np.nan
        ### DEBUG: Here I loose duplicated nodes with different node type, which is wrong, corrected by adjusting UUID ###
        nodes_df = pd.concat([nodes_df_s, nodes_df_o], ignore_index=True).drop_duplicates(subset=['node_uuid'])
        del nodes_df_s, nodes_df_o
        nodes_df = nodes_df.replace({np.nan: None})
    else:
        nodes_df_s = results_df[['subject_uuid', 'subject_type']]
        nodes_df_s.columns = ["node_uuid", "node_type"]
        nodes_df_o = results_df[['object_uuid', 'object_type']]
        nodes_df_o.columns = ["node_uuid", "node_type"]
        nodes_df = pd.concat([nodes_df_s, nodes_df_o], ignore_index=True).drop_duplicates(subset=['node_uuid'])
        nodes_df = pd.concat([nodes_df, anomalies_nodes[["node_uuid", "node_type"]]], ignore_index=True).drop_duplicates(
            subset=['node_uuid'])
    return nodes_df

def map_sparql_query(results_df_tmp):
    results_df_tmp = pd.DataFrame(results_df_tmp['results']['bindings'])
    results_df_tmp = results_df_tmp.map(lambda x: x['value'] if type(x) is dict else x).drop_duplicates()
    return  results_df_tmp
def split_2steps_result(results_df_tmp,base_headers,second_step_headers):
    results_df_tmp = results_df_tmp.map(lambda x: x['value'] if type(x) is dict else x).drop_duplicates()
    results_df_tmp_1 = results_df_tmp[base_headers]
    results_df_tmp_2 = results_df_tmp[second_step_headers]
    results_df_tmp_2.columns = base_headers
    results_df = pd.concat([results_df_tmp_1, results_df_tmp_2], ignore_index=True).drop_duplicates()
    del results_df_tmp, results_df_tmp_1, results_df_tmp_2
    return results_df

def get_node_attr_by_query(nodes_df):
    read_sparql = SPARQLWrapper(repository_url)
    read_sparql.setReturnFormat(JSON)
    read_sparql.setQuery(sparql_queries['get_nodes_attributes'])
    all_testing_node_attrs = read_sparql.queryAndConvert()
    all_testing_node_attrs = pd.DataFrame(all_testing_node_attrs['results']['bindings'])
    all_testing_node_attrs = all_testing_node_attrs.map(lambda x: x['value'] if type(x) is dict else x).drop_duplicates(subset=['node_uuid', 'node_type'])
    # all_testing_node_attrs['node_attr'] = all_testing_node_attrs['node_attr'].str.split(':').str[0]
    all_testing_node_attrs.loc[:, 'node_type'] = all_testing_node_attrs['node_type'].str.split('/').str[-1]
    if len(all_testing_node_attrs) > 0:
        nodes_df = pd.merge(nodes_df, all_testing_node_attrs, on=["node_uuid", "node_type"], how="left")
        print("Number of nodes with attributes", nodes_df['node_attr'].notna().sum())
        nodes_df = nodes_df.replace({np.nan: None})
    del all_testing_node_attrs
    return nodes_df

def query_sparql_triple_subgraph(anomalies_nodes, number_of_hops):
    query_time = time.time()
    read_sparql = SPARQLWrapper(repository_url)
    read_sparql.setReturnFormat(JSON)
    if number_of_hops == 0:
        # Construct with 0 hops
        read_sparql.setQuery(sparql_queries['construct_subgraphs_0_hop'])
        results_df = read_sparql.queryAndConvert()
        results_df = map_sparql_query(results_df)
    elif number_of_hops == 1:
        if args.anomalies_or_process_only:
            read_sparql.setQuery(sparql_queries['construct_subgraphs_1_hop_Backwards_process'])
            results_df = read_sparql.queryAndConvert()
            results_df = map_sparql_query(results_df)
        else:
            read_sparql.setQuery(sparql_queries['construct_subgraphs_1_hop_Backwards'])
            results_df = read_sparql.queryAndConvert()
            results_df = map_sparql_query(results_df)
        if not args.backwards_only:
            if args.anomalies_or_process_only:
                read_sparql.setQuery(sparql_queries['construct_subgraphs_1_hop_Forward_process'])
                results_df_F = read_sparql.queryAndConvert()
                results_df_F = map_sparql_query(results_df_F)
            else:
                read_sparql.setQuery(sparql_queries['construct_subgraphs_1_hop_Forward'])
                results_df_F = read_sparql.queryAndConvert()
                results_df_F = map_sparql_query(results_df_F)
            results_df = pd.concat([results_df, results_df_F], ignore_index=True).drop_duplicates()
    # elif number_of_hops == 2:
    #     base_headers = ["subject_uuid", "subject_type", "predicate", "timestamp", "object_uuid", "object_type"]
    #     results_df = pd.DataFrame(columns=base_headers)
    #     ##### debug  ###
    #     read_sparql.setQuery(sparql_queries['construct_subgraphs_2_hop_BF'])
    #     results_df_tmp = read_sparql.queryAndConvert()
    #     results_df_tmp = map_sparql_query(results_df_tmp)
    #     if len(results_df_tmp) > 0:
    #         second_step_headers = ["subject_uuid", "subject_type" "next_predicate", "next_timestamp", "next_object_uuid", "next_object_type"]
    #         results_df_BF = split_2steps_result(results_df_tmp, base_headers, second_step_headers)
    #         results_df = pd.concat([results_df, results_df_BF], ignore_index=True).drop_duplicates()
    #         del results_df_tmp, results_df_BF
    #     read_sparql.setQuery(sparql_queries['construct_subgraphs_2_hop_BB'])
    #     results_df_tmp = read_sparql.queryAndConvert()
    #     results_df_tmp = map_sparql_query(results_df_tmp)
    #     if len(results_df_tmp) > 0:
    #         second_step_headers = ["next_subject_uuid", "next_subject_type", "next_predicate","next_timestamp", "subject_uuid", "subject_type"]
    #         results_df_BB = split_2steps_result(results_df_tmp, base_headers, second_step_headers)
    #         results_df = pd.concat([results_df, results_df_BB], ignore_index=True).drop_duplicates()
    #         del results_df_tmp, results_df_BB
    #     if not args.backwards_only:
    #         read_sparql.setQuery(sparql_queries['construct_subgraphs_2_hop_FF'])
    #         results_df_tmp = read_sparql.queryAndConvert()
    #         results_df_tmp = map_sparql_query(results_df_tmp)
    #         if len(results_df_tmp) > 0:
    #             second_step_headers = ["object_uuid", "object_type", "next_predicate", "next_timestamp", "next_object_uuid", "next_object_type"]
    #             results_df_FF = split_2steps_result(results_df_tmp, base_headers, second_step_headers)
    #             results_df = pd.concat([results_df, results_df_FF], ignore_index=True).drop_duplicates()
    #             del results_df_tmp,results_df_FF
    #         read_sparql.setQuery(sparql_queries['construct_subgraphs_2_hop_FB'])
    #         results_df_tmp = read_sparql.queryAndConvert()
    #         results_df_tmp = map_sparql_query(results_df_tmp)
    #         if len(results_df_tmp) > 0:
    #             second_step_headers = ["next_subject_uuid","next_subject_type", "next_predicate", "next_timestamp", "object_uuid", "object_type"]
    #             results_df_FB = split_2steps_result(results_df_tmp, base_headers, second_step_headers)
    #             results_df = pd.concat([results_df, results_df_FB], ignore_index=True).drop_duplicates()
    #             del results_df_tmp,results_df_FB

    if len(results_df) == 0:
        print("Couldn't construct subgraphs, (No results returned from construction queries)")
        subgraph_obj = nx.MultiDiGraph()
        subgraph_obj.add_nodes_from(anomalies_nodes["node_uuid"])
        nodes_df = anomalies_nodes[["node_uuid", "node_type"]].drop_duplicates(subset=['node_uuid'])
    else:
        subgraph_obj, nodes_df = result_df_to_subgraph_nx(results_df)

        if args.anomalies_path_only:
            # Keep only anomalies paths
            subgraph_obj = traverse_anomalies_path(subgraph_obj, anomalies_nodes["node_uuid"].tolist())

        # Add other anomalies nodes (disconnected nodes)
        subgraph_obj.add_nodes_from(anomalies_nodes["node_uuid"])

        # if args.get_node_attrs:
        #     read_sparql.setQuery(sparql_queries['get_annomalies_attributes'])
        #     anomalies_attrs = read_sparql.queryAndConvert()
        #     anomalies_attrs_df = pd.DataFrame(anomalies_attrs['results']['bindings'])
        #     anomalies_attrs_df = anomalies_attrs_df.map(lambda x: x['value'] if type(x) is dict else x).drop_duplicates(
        #         subset=['node_uuid'])
        #     if len(anomalies_attrs_df) > 0:
        #         # anomalies_attrs_df['node_type'] = anomalies_attrs_df['node_type'].str.split('/').str[-1]
        #         # ########## Debug ##################
        #         # print("Number of all anomalies ", len(anomalies_nodes["node_uuid"].unique()))
        #         # print("Number of anomalies with node attributes",len(anomalies_attrs_df["node_uuid"].unique()))
        #         # ########################3
        #         anomalies_nodes = pd.merge(anomalies_nodes, anomalies_attrs_df, on="node_uuid", how="outer")
        #         nodes_df = pd.concat([nodes_df, anomalies_nodes[["node_uuid", "node_type", "node_attr"]]], ignore_index=True).drop_duplicates(
        #             subset=['node_uuid'])
        #         nodes_df = nodes_df.replace({np.nan: None})
        #     # nodes_df = get_node_attr_by_query(nodes_df)

    node_attr = nodes_df.set_index('node_uuid').to_dict('index')
    nx.set_node_attributes(subgraph_obj, node_attr)
    del node_attr, nodes_df
    print("Number of nodes in queried subgraph object: {}".format(subgraph_obj.number_of_nodes()))
    print("Number of edges in queried subgraph object: {}".format(subgraph_obj.number_of_edges()))
    if subgraph_obj.number_of_edges() == 0:
        print("Couldn't construct any subgraphs from all annomalies nodes ")
        global run_start_time
        print("Total time for this run: ", time.time() - run_start_time, "seconds.")
        return None
    ######## To Debug ##########
    # One node uuid has two node type (File and DIR)
    ############################
    # node_attr = anomalies_nodes[["node_uuid", "node_type", "Label", "Model_prediction"]].set_index('node_uuid').to_dict('index')
    # nx.set_node_attributes(subgraph_obj, node_attr)
    print("The subgraph query time is: ", time.time() - query_time, "seconds.")
    return subgraph_obj

def calculate_mcc(tp, tn, fp, fn):
    numerator = (tp * tn) - (fp * fn)
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    if denominator == 0:
        return 0  # Handle division by zero case
    mcc = numerator / denominator
    return mcc
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
    mcc = calculate_mcc(tp, tn, fp, fn)
    print("TP: {}\tTN: {}\tFP: {}\tFN: {}".format(tp, tn, fp, fn))
    print("Accuracy: {}\tPrecision: {}\tRecall: {}\tF-1: {}".format(accuracy, precision, recall, f_measure))
    print("TPR: {}\tFPR: {}\tMCC: {}".format(tpr, fpr,mcc))
    anomaly_results_df = pd.DataFrame({'accuracy': [accuracy], 'precision': [precision],'recall': [recall],
                                       'f_measure':[f_measure],'tp':[tp],'tn': [tn],'fp': [fp],'fn': [fn]})
    anomaly_results_df['tpr'] = tpr
    anomaly_results_df['fpr'] = fpr
    anomaly_results_df['mcc'] = mcc
    return anomaly_results_df

def get_outliers_anomaly_score_IQR(anomaly_scores):
    Q1 = anomaly_scores.quantile(0.25)
    Q3 = anomaly_scores.quantile(0.75)

    # Calculate the IQR
    IQR = Q3 - Q1

    # Make a dataset with the upper IQR outliers
    IQR_outliers = anomaly_scores[((anomaly_scores > (Q3 + 1.5 * IQR.item()))).any(axis=1)]
    is_anomaly_scores_outliers = anomaly_scores >= IQR_outliers.min().item()
    return is_anomaly_scores_outliers

def get_outliers_anomaly_score_z_score(anomaly_scores):
    # Calculate the z-scores
    z_scores = stats.zscore(anomaly_scores)

    # Select data points with a z-scores above 2
    is_anomaly_scores_outliers = (z_scores > 2).all(axis=1)

    return is_anomaly_scores_outliers

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
def evaluation_with_2hop_neighbours(MP, all_pids, GP, edges, mapping_dict):
    start_compute_related_nodes = time.time()
    print("Result with post-processing evaluation (Considering 2-hops related nodes of anomalous nodes as true anomalies)")
    print("Initial number of ground truth labels", len(GP))
    print("Initial number of identified anomolous nodes", len(MP))
    TP = MP.intersection(GP)
    FP = MP - GP
    FN = GP - MP
    TN = all_pids - (GP | MP)

    print("Initial results without two hops anomolous nodes")
    acc, prec, rec, fscore, FPR, TPR = calculate_metrics(len(TP), len(FP), len(FN), len(TN))
    mcc = calculate_mcc(len(TP), len(TN), len(FP), len(FN))
    print(f"TP: {len(TP)}, FP: {len(FP)}, FN: {len(FN)}, TN: {len(TN)}")
    print(f"Precision: {round(prec, 3)}, Recall: {round(rec, 3)}, Fscore: {round(fscore, 3)}")
    print(f"MCC: {round(mcc, 3)}")

    two_hop_tp = Get_Adjacent(TP, mapping_dict, edges, 2)
    print("Number of nodes that connect to true anomalous identified nodes with 2-hops", len(two_hop_tp))

    two_hop_gp = Get_Adjacent(GP, mapping_dict, edges, 2)
    print("Number of nodes that connect to ground truth malicious nodes with 2-hops", len(two_hop_gp))


    FPL = FP - two_hop_gp
    TPL = TP.union(FN.intersection(two_hop_tp))
    FN = FN - two_hop_tp
    # DEBUG: update the TN after 2Hop calculation, added by ahmed aly ##
    TN = TN.union((FP-FPL))
    ##################################

    acc,prec, rec, fscore, FPR, TPR = calculate_metrics(len(TPL), len(FPL), len(FN), len(TN))
    mcc = calculate_mcc(len(TPL), len(TN), len(FPL), len(FN))
    print("\n*******************************************")
    print("Results with two hops identified anomalous nodes (FLASH and THREATRACE Logic)")
    print(f"TP: {len(TPL)}, FP: {len(FPL)}, FN: {len(FN)}, TN: {len(TN)}")
    print(f"Precision: {round(prec, 3)}, Recall: {round(rec, 3)}, Fscore: {round(fscore, 3)}")
    print(f"MCC: {round(mcc, 3)}")
    anomaly_2hop_results_df = pd.DataFrame({'accuracy':[acc],'precision': [prec], 'recall': [rec],'f_measure': [fscore],
                                            'tp': [len(TPL)], 'tn': [len(TN)], 'fp': [len(FPL)], 'fn': [len(FN)],'tpr': [TPR], 'fpr': [FPR], 'mcc': [mcc]})
    # anomaly_2hop_results_df = print_predict_evaluation_metrics(TP, TN, FP, FN,
    #                                                            "Evaluate with two hops from identified anomalous nodes (FLASH and THREATRACE Logic)")

    # DEBUG missed nodes ##
    global malicious_nodes, anomalies_nodes
    # print("DEBUG: FN are:", FN)
    print("Initially Detected malicious nodes:",
          malicious_nodes[malicious_nodes["node_uuid"].isin(TP)]["node_type"].value_counts())
    print("All Detected malicious nodes:",
          malicious_nodes[malicious_nodes["node_uuid"].isin(TPL)]["node_type"].value_counts())
    print("Missed malicious nodes per node type:", malicious_nodes[malicious_nodes["node_uuid"].isin(FN)]["node_type"].value_counts())
    # print("anomalies score of Missed malicious nodes per node type:",anomalies_nodes[anomalies_nodes["node_uuid"].isin(FN)][["node_uuid","Prediction_probability"]])
    ###############
    del two_hop_gp, two_hop_tp
    alerts_2hop = TPL.union(FPL)
    print("Computed related nodes in ", time.time() - start_compute_related_nodes)
    print("\n*******************************************")
    return alerts_2hop , anomaly_2hop_results_df


def initial_evaluation(identified_ids, all_ids, gt_malicious_ids):
    TP = identified_ids.intersection(gt_malicious_ids)
    FP = identified_ids - gt_malicious_ids
    FN = gt_malicious_ids - identified_ids
    TN = all_ids - (gt_malicious_ids | identified_ids)
    # print("DEBUG: TP:", list(TP))
    return TP, TN, FP, FN

def re_evaluate_anomaly_detection_v2(detected_nodes):
    global malicious_nodes_uuid
    global subject_nodes, edge_index, mapping_dict, testing_uuid
    construction_TP, construction_TN, construction_FP, construction_FN = initial_evaluation(set(detected_nodes),testing_uuid,set(malicious_nodes_uuid))
    TP, TN, FP, FN = len(construction_TP), len(construction_TN), len(construction_FP), len(construction_FN)
    anomaly_results_df = print_predict_evaluation_metrics(TP, TN, FP, FN,"Evaluate Node anomaly detection with correlated nodes (all nodes in detected subgraphs)")
    if args.consider_related_nodes:
        alerts_2hop, anomaly_2hop_results_df = evaluation_with_2hop_neighbours(set(detected_nodes), testing_uuid, set(malicious_nodes_uuid),edge_index.tolist(), mapping_dict)
        return anomaly_2hop_results_df
    return anomaly_results_df

# def re_evaluate_anomaly_detection(all_correlated_nodes, correlated_anomalies_nodes,disregarded_anomalies_nodes,consider_all_correlated_nodes=False):
#     node_anomalies_results_f = args.root_path + "results/" + args.exp_name + "/" + args.model.replace(".model","") + "/anomaly_results_summary.csv"
#     node_anomalies_results = pd.read_csv(node_anomalies_results_f)
#     Node_anomaly_TP = int(node_anomalies_results[node_anomalies_results["run"] == "0"]["tp"].item())
#     Node_anomaly_FP = int(node_anomalies_results[node_anomalies_results["run"] == "0"]["fp"].item())
#     Node_anomaly_FN = int(node_anomalies_results[node_anomalies_results["run"] == "0"]["fn"].item())
#     Node_anomaly_TN = int(node_anomalies_results[node_anomalies_results["run"] == "0"]["tn"].item())
#     # total_n_nodes = Node_anomaly_TP + Node_anomaly_FP + Node_anomaly_TN + Node_anomaly_FN
#     print_predict_evaluation_metrics(Node_anomaly_TP, Node_anomaly_TN, Node_anomaly_FP, Node_anomaly_FN, "Evaluate Node anomaly detection before subgraph construction")
#
#     # consider only anomalous nodes that appeares in constructed subgraphs as alarms (positive cases). And disregard the normal nodes that appears in the subgraphs
#     print("consider only anomalous nodes that appeares in constructed subgraphs as alarms (positive cases).")
#     construction_TP = len(correlated_anomalies_nodes[correlated_anomalies_nodes["Label"] == True]["node_uuid"].unique())
#     # Needs to confirm whether this is a correct definition of false positive
#     construction_FP = len(correlated_anomalies_nodes[correlated_anomalies_nodes["Label"] == False]["node_uuid"].unique())
#     construction_FN = len(disregarded_anomalies_nodes[disregarded_anomalies_nodes["Label"] == True]["node_uuid"].unique())
#     construction_TN = len(disregarded_anomalies_nodes[disregarded_anomalies_nodes["Label"] == False]["node_uuid"].unique())
#     print("Number of malicious nodes in constructed subgraph anomalies nodes:",construction_TP)
#     print("Number of benign nodes in constructed subgraph anomalies nodes:",construction_FP)
#     print("Number of malicious nodes in disregarded nodes:", construction_FN)
#     print("Number of benign nodes in disregarded nodes:", construction_TN)
#     TP = construction_TP
#     FP = construction_FP
#     FN = construction_FN + Node_anomaly_FN
#     TN = construction_TN + Node_anomaly_TN
#     print_predict_evaluation_metrics(TP, TN, FP, FN, "Evaluate Node anomaly detection after subgraph construction (anomalous nodes in constructed subgraphs)")
#     print("False Positive distribution per node type after subgraph construction")
#     print(correlated_anomalies_nodes[correlated_anomalies_nodes["Label"] == False].groupby("node_type").count().sort_values(by="node_uuid",ascending=False)["node_uuid"])
#     print("True Positive distribution per node type after subgraph construction")
#     print(correlated_anomalies_nodes[correlated_anomalies_nodes["Label"] == True].groupby("node_type").count().sort_values(by="node_uuid",ascending=False)["node_uuid"])
    # if consider_all_correlated_nodes:
    #     global anomalies_nodes_uuid, malicious_nodes_uuid
    #     # Consider all nodes in constructed subgrphs as alarms (positive cases)
    #     print("Consider all nodes in constructed subgrphs as alarms (positive cases)")
    #     correlated_benign_nodes = all_correlated_nodes - set(malicious_nodes_uuid)
    #     correlated_malicious_nodes = all_correlated_nodes - correlated_benign_nodes
    #     correlated_normal_malicious_nodes = correlated_malicious_nodes - set(anomalies_nodes_uuid)
    #     correlated_normal_benign_nodes = correlated_benign_nodes - set(anomalies_nodes_uuid)
    #     TP = len(correlated_malicious_nodes)
    #     FP = len(correlated_benign_nodes)
    #
    #     FN = (Node_anomaly_FN - len(correlated_normal_malicious_nodes)) + construction_FN
    #     TN = (Node_anomaly_TN - len(correlated_normal_benign_nodes)) + construction_TN
    #
    #     print_predict_evaluation_metrics(TP, TN, FP, FN,
    #                                      "Evaluate Node anomaly detection after subgraph construction (all nodes in constructed subgraphs)")
    #     print("False Positive distribution per node type after subgraph construction")
    #     print(correlated_anomalies_nodes[correlated_anomalies_nodes["Label"] == False].groupby(
    #         "node_type").count().sort_values(by="node_uuid", ascending=False)["node_uuid"])
    #     print("True Positive distribution per node type after subgraph construction")
    #     print(correlated_anomalies_nodes[correlated_anomalies_nodes["Label"] == True].groupby(
    #         "node_type").count().sort_values(by="node_uuid", ascending=False)["node_uuid"])
    # print("*******************************************************")
    # return

def get_coorelated_and_disregardes_nodes(subgraphs_lst):
    global anomalies_nodes, testing_uuid
    all_correlated_nodes = {subgraph_nodes for subgraph in subgraphs_lst for subgraph_nodes in list(subgraph.nodes())}
    all_correlated_nodes = all_correlated_nodes.intersection(testing_uuid)
    # global anomalies_nodes_uuid
    # correlated_anomalies_nodes = all_correlated_nodes.intersection(set(anomalies_nodes_uuid))
    # disconnected_anomalies_nodes = set(anomalies_nodes_uuid) - all_correlated_nodes
    disconnected_anomalies_nodes = anomalies_nodes[~anomalies_nodes["node_uuid"].isin(all_correlated_nodes)]
    correlated_anomalies_nodes = anomalies_nodes[anomalies_nodes["node_uuid"].isin(all_correlated_nodes)]
    return all_correlated_nodes, correlated_anomalies_nodes , disconnected_anomalies_nodes

def classify_anomaly_score(score):
    if score <= 1:
        return 'Negligible'
    elif score <= 10:
        return 'Minor'
    elif score <= 100:
        return 'Moderate'
    elif score <= 1000:
        return 'Significant'
    else:
        return 'Critical'
def evaluate_with_outliers(subgraphs_lst,subgraphs_stats_df,outlier_test='Grubbs',least_severity=None,top_k=None,top_k_percentile=None,top_k_percent=None):
    global anomalies_nodes
    if outlier_test in ["top_percent", "top_percentile","top_k"]:
        sorted_subgraphs_stats_df = subgraphs_stats_df[["ID", "subgraph_anomaly_score"]].sort_values(
            by="subgraph_anomaly_score", ascending=False)
        if outlier_test == "top_percentile":
            print("evaluate with top percentile {} anomalous subgraphs".format(top_k_percentile))
            detected_subgraphs_IDs = sorted_subgraphs_stats_df[
                sorted_subgraphs_stats_df["subgraph_anomaly_score"] >= sorted_subgraphs_stats_df[
                    "subgraph_anomaly_score"].quantile(top_k_percentile)]["ID"].tolist()
        elif outlier_test == "top_percent":
            print("evaluate with top percent {} anomalous subgraphs".format(top_k_percent))
            detected_subgraphs_IDs = \
            sorted_subgraphs_stats_df.nlargest(int(math.ceil(top_k_percent * len(sorted_subgraphs_stats_df))),
                                               "subgraph_anomaly_score")["ID"].tolist()
        elif outlier_test == 'top_k':
            print("evaluate with top {} anomalous subgraphs ".format(top_k))
            detected_subgraphs_IDs = subgraphs_stats_df.nlargest(top_k, "subgraph_anomaly_score")["ID"].tolist()
        else:
            print("outlier test:{} is not defined".format(outlier_test))
    elif outlier_test=="severity_levels":
        severity_order = ['Negligible', 'Minor', 'Moderate', 'Significant', 'Critical']
        least_severity_order = severity_order.index(least_severity)
        severity_level_lst = severity_order[least_severity_order:]
        print("evaluate with anomalous subgraphs that has {} severity levels ".format(severity_level_lst))
        subgraphs_stats_df['severity_level'] = subgraphs_stats_df['subgraph_anomaly_score'].apply(classify_anomaly_score)
        detected_subgraphs_IDs = subgraphs_stats_df[subgraphs_stats_df['severity_level'].isin(severity_level_lst)]["ID"].tolist()
        while len(detected_subgraphs_IDs) == 0:
            print("** Couldn't find anomalous subgraphs with {} severity level **".format(least_severity))
            least_severity_order -=1
            if least_severity_order <= 0:
                print("No anomalous subgraphs")
                break
            severity_level_lst = severity_order[least_severity_order:]
            print("evaluate with anomalous subgraphs that has {} severity levels ".format(severity_level_lst))
            detected_subgraphs_IDs = subgraphs_stats_df[subgraphs_stats_df['severity_level'].isin(severity_level_lst)]["ID"].tolist()
    else:
        print("evaluate with anomalous subgraphs that has anomaly score outliers based on {} test".format(outlier_test))
        if outlier_test=='grubbs':
            subgraphs_stats_df["Is_anomaly_score_outliers"] = get_outliers_anomaly_score_grubbs(subgraphs_stats_df["subgraph_anomaly_score"])
        elif outlier_test=='z_score':
            subgraphs_stats_df["Is_anomaly_score_outliers"] = get_outliers_anomaly_score_z_score(subgraphs_stats_df[["subgraph_anomaly_score"]])
        elif outlier_test=='iqr':
            subgraphs_stats_df["Is_anomaly_score_outliers"] = get_outliers_anomaly_score_IQR(subgraphs_stats_df[["subgraph_anomaly_score"]])
        else:
            print("outlier test:{} is not defined".format(outlier_test))
        detected_subgraphs_IDs = subgraphs_stats_df[subgraphs_stats_df["Is_anomaly_score_outliers"]]["ID"].tolist()

    detected_subgraphs_lsl = [subgraphs_lst[i] for i in detected_subgraphs_IDs]
    if len(detected_subgraphs_lsl) > 0:
        print("Number of subgraphs is ", len(detected_subgraphs_lsl))
        all_detected_nodes, detected_subgraphs_anomalies_nodes, disregarded_anomalies_nodes = get_coorelated_and_disregardes_nodes(detected_subgraphs_lsl)
        # re_evaluate_anomaly_detection(all_detected_nodes, detected_subgraphs_anomalies_nodes,disregarded_anomalies_nodes)
        # re_evaluate_anomaly_detection_v2(detected_subgraphs_anomalies_nodes)
        anomaly_results_df_run = re_evaluate_anomaly_detection_v2(all_detected_nodes)
    else:
        print("\n** No raised subgraphs based on {}, will report top K ({}) anomalous subgraphs **".format(outlier_test,top_k))
        evaluate_with_outliers(subgraphs_lst, subgraphs_stats_df, outlier_test='top_k',top_k=args.top_k_anomalous_subgraphs)
    del detected_subgraphs_lsl, subgraphs_stats_df
    return anomaly_results_df_run

def explore_and_draw_subgaphs(subgraphs_lst):
    ##
    # store evaluation results in anomaly_results_df_run
    ##
    all_correlated_nodes, correlated_anomalies_nodes, disconnected_anomalies_nodes = get_coorelated_and_disregardes_nodes(subgraphs_lst)
    disconnected_nodes = disconnected_anomalies_nodes["node_uuid"].unique().tolist()
    print("Number of constructed subgraphs:", len(subgraphs_lst))
    print("Number of correlated anomalies nodes:", len(correlated_anomalies_nodes))
    print("Number of disconnected nodes", len(disconnected_nodes))
    subgraphs_stats_df = explore_all_subgraphs(subgraphs_lst)
    end_construction_time = time.time()
    print("Evaluate Node anomaly detection after subgraph construction (with all anomalous subgraphs) ")

    # anomaly_results_df_run = re_evaluate_anomaly_detection_v2(all_correlated_nodes)

    # print("Evaluate with Significant abnormality")
    # evaluate_with_outliers(subgraphs_lst, subgraphs_stats_df, outlier_test='severity_levels',least_severity='Significant')
    print("Evaluate with Moderate abnormality")
    anomaly_results_df_run = evaluate_with_outliers(subgraphs_lst, subgraphs_stats_df, outlier_test='severity_levels', least_severity='Moderate')
    # print("Evaluate with Minor abnormality")
    # evaluate_with_outliers(subgraphs_lst, subgraphs_stats_df, outlier_test='severity_levels', least_severity='Minor')

    # evaluate_with_outliers(subgraphs_lst, subgraphs_stats_df, outlier_test='top_k',top_k=args.top_k_anomalous_subgraphs)

    # evaluate_with_outliers(subgraphs_lst,subgraphs_stats_df,outlier_test='grubbs')
    # evaluate_with_outliers(subgraphs_lst, subgraphs_stats_df, outlier_test='z_score')
    # evaluate_with_outliers(subgraphs_lst, subgraphs_stats_df, outlier_test='iqr')
    #
    # evaluate_with_outliers(subgraphs_lst, subgraphs_stats_df, outlier_test='top_percentile',top_k_percentile=0.99)
    # evaluate_with_outliers(subgraphs_lst, subgraphs_stats_df, outlier_test='top_percentile', top_k_percentile=0.95)
    # evaluate_with_outliers(subgraphs_lst, subgraphs_stats_df, outlier_test='top_percentile', top_k_percentile=0.90)

    if args.draw_subgraphs:
        print("Drawing Subgraphs")
        for i, sg in enumerate(subgraphs_lst):
            if sg.number_of_edges() <= 10000:
                draw_subgraph(i, sg)
            else:
                print("Skipping drawing subgraph with", sg.number_of_edges(), " edges")

    return disconnected_nodes, subgraphs_stats_df, all_correlated_nodes, end_construction_time, anomaly_results_df_run


def sample_subgraph_per_edge_type(subgraph):
    print("Sampling a subgraph with {} nodes and {} edges (sample from different actions)".format(subgraph.number_of_nodes(),subgraph.number_of_edges()))
    subgraph_df = nx.to_pandas_edgelist(subgraph, edge_key='ekey')
    # preserve actions distribution
    edge_type_weights = {edge_type:int((len(subgraph_df[subgraph_df["action"]==edge_type])/len(subgraph_df)*max_edges) + 1) for edge_type in subgraph_df["action"].unique().tolist()}
    sample_subgraph_df = pd.concat([subgraph_df.loc[subgraph_df["action"] == edge_type].sample(edge_type_weights[edge_type],random_state=0) for edge_type in subgraph_df["action"].unique().tolist()])
    del edge_type_weights,subgraph_df
    ## DEBUG: should I take samples till considering all edges ?##
    # max_edges_per_actions = int(max_edges / len(subgraph_df["action"].unique()))
    # sample_subgraph_df = subgraph_df.groupby('action', group_keys=False).apply(lambda x: x.sample(min(len(x), max_edges_per_actions)))

    sample_edges = [tuple(edge) for edge in sample_subgraph_df[["source", "target", "ekey"]].values]
    #### DEBUG: those edges are not necesserly connected !!! ########3
    subgraph = subgraph.edge_subgraph(sample_edges).copy()
    print("kept only a sample subgraph with {} nodes and {} edges".format(subgraph.number_of_nodes(), subgraph.number_of_edges()))
    del sample_edges, sample_subgraph_df
    return subgraph


def sample_subgraph_per_edge_type_then_get_connected_subgraphs(subgraph):
    global max_edges
    print("Sampling a subgraph with {} nodes and {} edges (sample from different actions)".format(subgraph.number_of_nodes(),subgraph.number_of_edges()))
    subgraph_df = nx.to_pandas_edgelist(subgraph, edge_key='ekey')
    # preserve actions distribution
    edge_type_weights={}
    for edge_type in subgraph_df["action"].unique().tolist():
        if len(subgraph_df[subgraph_df["action"] == edge_type]) > int(max_edges / len(subgraph_df["action"].unique())):
            edge_type_weights[edge_type] = int((len(subgraph_df[subgraph_df["action"] == edge_type]) / len(subgraph_df) * max_edges))
        else:
            edge_type_weights[edge_type] = int(len(subgraph_df[subgraph_df["action"] == edge_type]))
    # edge_type_weights = {edge_type:int((len(subgraph_df[subgraph_df["action"]==edge_type])/len(subgraph_df)*max_edges) + 1) for edge_type in subgraph_df["action"].unique().tolist()}
    sample_subgraph_df = pd.concat([subgraph_df.loc[subgraph_df["action"] == edge_type].sample(edge_type_weights[edge_type],random_state=0) for edge_type in subgraph_df["action"].unique().tolist()])
    del edge_type_weights,subgraph_df
    sample_edges = [tuple(edge) for edge in sample_subgraph_df[["source", "target", "ekey"]].values]
    sampled_subgraph = subgraph.edge_subgraph(sample_edges).copy()

    sampled_subgraph_lst = [sampled_subgraph.subgraph(nodes).copy() for nodes in nx.weakly_connected_components(sampled_subgraph) if len(nodes) > 1]
    for this_sampled_subgraph in sampled_subgraph_lst:
        print("sampled a subgraph with {} nodes and {} edges".format(this_sampled_subgraph.number_of_nodes(), this_sampled_subgraph.number_of_edges()))
    del sample_edges, sample_subgraph_df

    sampled_subgraph_nodes = {subgraph_nodes for subgraph_nodes in list(sampled_subgraph.nodes())}
    subgraph.remove_nodes_from(sampled_subgraph_nodes)
    print("Remains unsampled a subgraph with {} nodes and {} edges (sample from different actions)".format(
        subgraph.number_of_nodes(), subgraph.number_of_edges()))
    connected_subgraphs_lst = [subgraph.subgraph(nodes).copy() for nodes in nx.weakly_connected_components(subgraph) if len(nodes) > 1]
    sampled_subgraph_lst.extend([subgraph for subgraph in connected_subgraphs_lst if 0 < subgraph.number_of_edges() <= max_edges ])
    return sampled_subgraph_lst

def partition_by_sample_per_edge_type_all_anomalies(subgraph):
    global anomalies_nodes_uuid,max_edges
    print("Sampling a subgraph with {} nodes and {} edges (sample from different actions)".format(
        subgraph.number_of_nodes(), subgraph.number_of_edges()))
    all_subgraph_nodes = {subgraph_nodes for subgraph_nodes in list(subgraph.nodes())}
    subgraph_anomalies_node = all_subgraph_nodes.intersection(set(anomalies_nodes_uuid))
    print("The subgraph contain {} anomalies nodes".format(len(subgraph_anomalies_node)))
    sampled_subgraph_lst = []
    while subgraph_anomalies_node:
        subgraph_df = nx.to_pandas_edgelist(subgraph, edge_key='ekey')
        # preserve actions distribution
        edge_type_weights = {edge_type:int((len(subgraph_df[subgraph_df["action"]==edge_type])/len(subgraph_df)*max_edges) + 1) for edge_type in subgraph_df["action"].unique().tolist()}
        sample_subgraph_df = pd.concat([subgraph_df.loc[subgraph_df["action"] == edge_type].sample(edge_type_weights[edge_type],random_state=0) for edge_type in subgraph_df["action"].unique().tolist()])
        del edge_type_weights,subgraph_df
        sampled_edges = [tuple(edge) for edge in sample_subgraph_df[["source", "target", "ekey"]].values]
        sampled_subgraph = subgraph.edge_subgraph(sampled_edges).copy()
        print("kept a sample subgraph with {} nodes and {} edges".format(sampled_subgraph.number_of_nodes(),
                                                                              sampled_subgraph.number_of_edges()))

        sampled_subgraph_lst.append(sampled_subgraph)
        sampled_subgraph_nodes = {subgraph_nodes for subgraph_nodes in list(sampled_subgraph.nodes())}
        # sampled_subgraph_anomalies_node = subgraph_anomalies_node.intersection(sampled_subgraph_nodes)
        subgraph_anomalies_node = subgraph_anomalies_node - sampled_subgraph_nodes

        print("Remains {} anomalies nodes to be sampled".format(len(subgraph_anomalies_node)))

        ###### debug: Should remove sampled nodes not edges , and confirm the unsampled subgraph is connected and has edges ############
        # subgraph.remove_nodes_from(sampled_subgraph_anomalies_node)
        subgraph.remove_edges_from(sampled_edges)
        print("Remains unsampled a subgraph with {} nodes and {} edges (sample from different actions)".format(
                subgraph.number_of_nodes(), subgraph.number_of_edges()))
        if 0 < subgraph.number_of_edges() <= max_edges:
            sampled_subgraph_lst.append(subgraph)
            break
        elif subgraph.number_of_edges() == 0: break
        del sampled_edges, sample_subgraph_df, sampled_subgraph_nodes, sampled_subgraph

    return sampled_subgraph_lst

def partition_subgraph(big_subgraph):
    start_time_partition = time.time()
    global max_edges
    print("partition a graph with {} nodes and {} edges".format(big_subgraph.number_of_nodes(), big_subgraph.number_of_edges()))
    communities = nx.community.louvain_communities(big_subgraph, resolution=1,seed=0)
    # n_communities = int(big_subgraph.number_of_edges() / max_edges) + 1
    # communities = nx.community.greedy_modularity_communities(big_subgraph,cutoff=n_communities,best_n=n_communities)
    # communities_gen = nx.community.girvan_newman(big_subgraph)
    # communities = next(communities_gen)
    subgraphs_lst_tmp = [big_subgraph.subgraph(nodes).copy() for nodes in communities if len(nodes) >= args.min_nodes]
    min_edges = 2
    # setting min_edges to ignore corner cases when the partition algorithm assign a community that doesn't has connected edges #
    subgraphs_lst_tmp = [subgraph for subgraph in subgraphs_lst_tmp if subgraph.number_of_edges() > min_edges]
    if len(subgraphs_lst_tmp) == 0:
        print("Couldn't partition the big subgraph")
        subgraphs_lst_tmp = [big_subgraph]
    for subgraph in subgraphs_lst_tmp:
        print("partitioned a subgraph with {} nodes and {} edges".format(subgraph.number_of_nodes(),
                                                                         subgraph.number_of_edges()))
    subgraphs_lst = []
    for subgraph in subgraphs_lst_tmp:
        if subgraph.number_of_edges() > max_edges:
            # subgraphs_lst.extend(partition_by_sample_per_edge_type_all_anomalies(subgraph))
            subgraphs_lst.extend(sample_subgraph_per_edge_type_then_get_connected_subgraphs(subgraph))
        else:
            subgraphs_lst.append(subgraph)
    # subgraphs_lst = [sample_subgraph_per_edge_type(subgraph) if subgraph.number_of_edges() > max_edges else subgraph for subgraph in subgraphs_lst_tmp]
    print("Partition time: ", time.time() - start_time_partition, "seconds.")
    return subgraphs_lst

# def traverse_connected_nodes(conn_node):
#     global correlated_anomalies_nodes, visited_direct_anomalies_nodes
#     if args.correlate_anomalous_once:
#         anomalies_nodes_1_hop = {neighbour_node for neighbour_node, _ in subgraph_obj.in_edges(conn_node) if
#                                  (neighbour_node in anomalies_nodes_uuid) & (
#                                              neighbour_node not in visited_direct_anomalies_nodes) & (
#                                              neighbour_node not in correlated_anomalies_nodes)}
#         anomalies_nodes_1_hop.update({neighbour_node for _, neighbour_node in subgraph_obj.out_edges(conn_node) if
#                                       (neighbour_node in anomalies_nodes_uuid) & (
#                                                   neighbour_node not in visited_direct_anomalies_nodes) & (
#                                                   neighbour_node not in correlated_anomalies_nodes)})
#     else:
#         anomalies_nodes_1_hop = {neighbour_node for neighbour_node, _ in subgraph_obj.in_edges(conn_node) if
#                                  (neighbour_node in anomalies_nodes_uuid) & (
#                                              neighbour_node not in visited_direct_anomalies_nodes)}
#         anomalies_nodes_1_hop.update({neighbour_node for _, neighbour_node in subgraph_obj.out_edges(conn_node) if
#                                       (neighbour_node in anomalies_nodes_uuid) & (
#                                                   neighbour_node not in visited_direct_anomalies_nodes)})
#     return anomalies_nodes_1_hop

def filter_correlated_nodes(neighbour_nodes,visited_direct_anomalies_nodes,correlated_anomalies_nodes):
    global anomalies_nodes_uuid
    neighbour_nodes = neighbour_nodes.intersection(set(anomalies_nodes_uuid))
    if args.correlate_anomalous_once:
        neighbour_nodes = neighbour_nodes - correlated_anomalies_nodes
    neighbour_nodes = neighbour_nodes - visited_direct_anomalies_nodes
    return neighbour_nodes

def expand_subgraph_in_memory_V2(node):
    global subgraph_obj, anomalies_nodes_uuid
    global anomalies_nodes
    global correlated_anomalies_nodes
    traverse_time = time.time()

    connected_nodes = {neighbour_node for neighbour_node, _ in subgraph_obj.in_edges(node) if neighbour_node != node}
    connected_nodes.update({neighbour_node for _, neighbour_node in subgraph_obj.out_edges(node) if neighbour_node != node})

    if len(connected_nodes) == 0:
        yield None
        return
    subgraph_nodes = {node}
    visited_direct_anomalies_nodes = {node}

    anomalies_connected_nodes = connected_nodes.intersection(set(anomalies_nodes_uuid))
    if args.correlate_anomalous_once:
        anomalies_connected_nodes = anomalies_connected_nodes - correlated_anomalies_nodes
    subgraph_nodes.update(anomalies_connected_nodes)
    del anomalies_connected_nodes
    visited_direct_anomalies_nodes.update(subgraph_nodes)
    # subgraph_nodes_tmp
    # if len(subgraph_nodes) < args.min_nodes:
    if args.number_of_hops >= 1:
        for conn_node in connected_nodes:
            neighbour_nodes = {neighbour_node for neighbour_node, _ in subgraph_obj.in_edges(conn_node)}
            anomalies_nodes_1_hop = filter_correlated_nodes(neighbour_nodes,visited_direct_anomalies_nodes,correlated_anomalies_nodes)
            del neighbour_nodes
            neighbour_nodes = {neighbour_node for _, neighbour_node in subgraph_obj.out_edges(conn_node)}
            anomalies_nodes_1_hop.update(filter_correlated_nodes(neighbour_nodes, visited_direct_anomalies_nodes, correlated_anomalies_nodes))
            if len(anomalies_nodes_1_hop) > 0:
                subgraph_nodes.update(anomalies_nodes_1_hop)
                subgraph_nodes.add(conn_node)
                if len(subgraph_nodes) >= args.min_nodes:
                    subgraph = subgraph_obj.subgraph(subgraph_nodes).copy()
                    if subgraph.number_of_edges() >= max_edges and max_edges != 0:
                        subgraphs_lst = partition_subgraph(subgraph)
                        for subgraph in subgraphs_lst:
                            print("Constructed a subgraph with {} nodes and {} edges".format(subgraph.number_of_nodes(),
                                                                                             subgraph.number_of_edges()))
                            correlated_anomalies_nodes.update(list(subgraph.nodes()))
                            yield subgraph
                        # construct new subgraphs
                        del subgraph_nodes
                        subgraph_nodes = {node}
                    else:
                        # Add more 1_hop anomalies nodes from next connected nodes paths
                        del subgraph
            del anomalies_nodes_1_hop, neighbour_nodes

    del connected_nodes, visited_direct_anomalies_nodes
    if len(subgraph_nodes) >= args.min_nodes:
        subgraph = subgraph_obj.subgraph(subgraph_nodes).copy()
        del subgraph_nodes
        if subgraph.number_of_edges() > max_edges and max_edges != 0:
            subgraphs_lst = partition_subgraph(subgraph)
        else:
            subgraphs_lst = [subgraph]

        for subgraph in subgraphs_lst:
            print("Constructed a subgraph with {} nodes and {} edges".format(subgraph.number_of_nodes(),subgraph.number_of_edges()))
            if args.correlate_anomalous_once:
                correlated_anomalies_nodes.update(list(subgraph.nodes()))
            yield subgraph
    else:
        print("subgraph is less than ", str(args.min_nodes), "nodes")
        del subgraph_nodes
        yield None
    print("The traverse time is: ", time.time() - traverse_time, "seconds.")
    return

def expand_subgraph_in_memory(node):
    global subgraph_obj, anomalies_nodes_uuid
    global anomalies_nodes
    global correlated_anomalies_nodes
    traverse_time = time.time()

    connected_nodes = {neighbour_node for neighbour_node, _ in subgraph_obj.in_edges(node) if neighbour_node != node}
    connected_nodes.update({neighbour_node for _, neighbour_node in subgraph_obj.out_edges(node) if neighbour_node != node})

    if len(connected_nodes) == 0:
        yield None
        return
    subgraph_nodes = {node}
    visited_direct_anomalies_nodes = {node}

    anomalies_connected_nodes = connected_nodes.intersection(set(anomalies_nodes_uuid))
    if args.correlate_anomalous_once:
        anomalies_connected_nodes = anomalies_connected_nodes - correlated_anomalies_nodes
    subgraph_nodes.update(anomalies_connected_nodes)
    del anomalies_connected_nodes
    visited_direct_anomalies_nodes.update(subgraph_nodes)

    # ### Under Development ##########
    # connected_nodes = list(connected_nodes)
    # connected_nodes.sort()
    ###############################

    # if len(subgraph_nodes) < args.min_nodes:
    if args.number_of_hops >= 1 :
        for conn_node in sorted(connected_nodes):
            neighbour_nodes = {neighbour_node for neighbour_node, _ in subgraph_obj.in_edges(conn_node)}
            anomalies_nodes_1_hop = filter_correlated_nodes(neighbour_nodes,visited_direct_anomalies_nodes,correlated_anomalies_nodes)
            del neighbour_nodes
            neighbour_nodes = {neighbour_node for _, neighbour_node in subgraph_obj.out_edges(conn_node)}
            anomalies_nodes_1_hop.update(filter_correlated_nodes(neighbour_nodes, visited_direct_anomalies_nodes, correlated_anomalies_nodes))
            if len(anomalies_nodes_1_hop) > 0:
                subgraph_nodes.update(anomalies_nodes_1_hop)
                subgraph_nodes.add(conn_node)
                ### Under Development ##########
                visited_direct_anomalies_nodes.update(subgraph_nodes)
                ###############################
            del anomalies_nodes_1_hop, neighbour_nodes

    # # ##### DEBUG: expand for 2hop in memory ##################
    if args.expand_2_hop == "upto":
        if len(subgraph_nodes) < args.min_nodes:
            print("expand to 2 hop")
            visited_direct_anomalies_nodes.update(subgraph_nodes)
            second_connected_nodes = set()
            for node in sorted(connected_nodes):
                second_connected_nodes.update({neighbour_node for neighbour_node, _ in subgraph_obj.in_edges(node) if neighbour_node != node})
                second_connected_nodes.update({neighbour_node for _, neighbour_node in subgraph_obj.out_edges(node) if neighbour_node != node})
                second_connected_nodes = second_connected_nodes - connected_nodes
            ### Under Development ##########
            # second_connected_nodes = list(second_connected_nodes)
            # second_connected_nodes.sort()
            ###############################
            for conn_node in sorted(second_connected_nodes):
                neighbour_nodes = {neighbour_node for neighbour_node, _ in subgraph_obj.in_edges(conn_node)}
                anomalies_nodes_2_hop = filter_correlated_nodes(neighbour_nodes,visited_direct_anomalies_nodes,correlated_anomalies_nodes)
                del neighbour_nodes
                neighbour_nodes = {neighbour_node for _, neighbour_node in subgraph_obj.out_edges(conn_node)}
                anomalies_nodes_2_hop.update(filter_correlated_nodes(neighbour_nodes, visited_direct_anomalies_nodes, correlated_anomalies_nodes))
                anomalies_nodes_2_hop = anomalies_nodes_2_hop - connected_nodes
                if len(anomalies_nodes_2_hop) > 0:
                    subgraph_nodes.update(anomalies_nodes_2_hop)
                    subgraph_nodes.add(conn_node)
                    ### Under Development ##########
                    visited_direct_anomalies_nodes.update(subgraph_nodes)
                    ###############################
                del anomalies_nodes_2_hop, neighbour_nodes
    elif args.expand_2_hop == "always":
        visited_direct_anomalies_nodes.update(subgraph_nodes)
        second_connected_nodes = set()
        for node in connected_nodes:
            second_connected_nodes.update({neighbour_node for neighbour_node, _ in subgraph_obj.in_edges(node) if neighbour_node != node})
            second_connected_nodes.update({neighbour_node for _, neighbour_node in subgraph_obj.out_edges(node) if neighbour_node != node})
            second_connected_nodes = second_connected_nodes - connected_nodes
        ### Under Development ##########
        # second_connected_nodes = list(second_connected_nodes)
        # second_connected_nodes.sort()
        ###############################
        for conn_node in sorted(second_connected_nodes):
            neighbour_nodes = {neighbour_node for neighbour_node, _ in subgraph_obj.in_edges(conn_node)}
            anomalies_nodes_2_hop = filter_correlated_nodes(neighbour_nodes, visited_direct_anomalies_nodes,
                                                            correlated_anomalies_nodes)
            del neighbour_nodes
            neighbour_nodes = {neighbour_node for _, neighbour_node in subgraph_obj.out_edges(conn_node)}
            anomalies_nodes_2_hop.update(
                filter_correlated_nodes(neighbour_nodes, visited_direct_anomalies_nodes, correlated_anomalies_nodes))
            anomalies_nodes_2_hop = anomalies_nodes_2_hop - connected_nodes
            if len(anomalies_nodes_2_hop) > 0:
                subgraph_nodes.update(anomalies_nodes_2_hop)
                subgraph_nodes.add(conn_node)
                ### Under Development ##########
                visited_direct_anomalies_nodes.update(subgraph_nodes)
                ###############################
            del anomalies_nodes_2_hop, neighbour_nodes
    # # #################################################

    del connected_nodes, visited_direct_anomalies_nodes
    if len(subgraph_nodes) >= args.min_nodes:
        subgraph = subgraph_obj.subgraph(subgraph_nodes).copy()
        del subgraph_nodes
        # if subgraph.number_of_nodes() > max_nodes and max_edges != 0:
        if subgraph.number_of_edges() > max_edges and max_edges != 0:
            # subgraph = sample_subgraph_per_edge_type(subgraph)
            subgraphs_lst = partition_subgraph(subgraph)
        else:
            subgraphs_lst = [subgraph]

        for subgraph in subgraphs_lst:
            print("Constructed a subgraph with {} nodes and {} edges".format(subgraph.number_of_nodes(),subgraph.number_of_edges()))
            correlated_anomalies_nodes.update(list(subgraph.nodes()))
            yield subgraph
    else:
        print("subgraph is less than ", str(args.min_nodes), "nodes")
        del subgraph_nodes
        yield None
    print("The traverse time is: ", time.time() - traverse_time, "seconds.")
    return
def check_identical(subgraph1, subgraph2):
    # df_equal = graphs_equal(subgraph1, subgraph2)
    subgraph1_df = nx.to_pandas_edgelist(subgraph1, edge_key='ekey')
    subgraph2_df = nx.to_pandas_edgelist(subgraph2, edge_key='ekey')
    df_equal = subgraph1_df.equals(subgraph2_df)
    del subgraph1_df, subgraph2_df
    return df_equal

def remove_identicatl_subgraphs(subgraphs_lst):
    remove_subgraphs = set()
    for i1, subgraph1 in enumerate(subgraphs_lst):
        for i2, subgraph2 in enumerate(subgraphs_lst):
            if i1 != i2 and (i2 not in remove_subgraphs):
                if (subgraph1.number_of_nodes() == subgraph2.number_of_nodes()) & (
                        subgraph1.number_of_edges() == subgraph2.number_of_edges()):
                    if check_identical(subgraph1, subgraph2):
                        print("subgraphs {} and {} are identical".format(i1, i2))
                        print("Remove subgraph",i1)
                        remove_subgraphs.add(i1)
                        break
    filtered_subgraphs_lst = [subgraph for i,subgraph in enumerate(subgraphs_lst) if i not in remove_subgraphs]
    print("Number of removed duplicated subgraphs is", len(remove_subgraphs))
    del subgraphs_lst
    return filtered_subgraphs_lst

def node_type_order(node_type):
    if node_type.lower() in ["flow","net","netflowobject"]:
        type_order = 1
    elif node_type.lower().split("_")[-1] == "process":
        type_order = 2
    elif node_type.lower().split("_")[0] == "file":
        type_order = 3
    else:
        type_order = 4
    return type_order

def construct_subgraphs_by_criteria_from_anomaly_subgraph(run_start_time,number_of_hops=1,k=100,k_percent=None):
    start_construction_time = time.time()
    global anomalies_nodes
    clear_and_insert_anomalies(anomalies_nodes)
    global subgraph_obj
    subgraph_obj = query_sparql_triple_subgraph(anomalies_nodes, number_of_hops)
    if subgraph_obj is None:
        # global run_start_time
        anomaly_results_df_run = pd.DataFrame({'accuracy': [0], 'precision': [0], 'recall': [0], 'f_measure': [0],'tp': [0], 'tn': [0], 'fp': [0], 'fn': [0], 'tpr': [0], 'fpr': [0],'mcc': [0]})
        anomaly_results_df_run["subgraph_detection_time"] =  time.time() - run_start_time
        anomaly_results_df_run["memory_usage_mb"] = getrusage(RUSAGE_SELF).ru_maxrss / 1024
        return None , None , None, None , anomaly_results_df_run
    init_disconnected_nodes = [nodes for nodes in nx.weakly_connected_components(subgraph_obj) if len(nodes) == 1]
    init_disconnected_nodes = list({n.pop() for n in init_disconnected_nodes})
    print("Number of initially disconnected nodes nodes:", len(init_disconnected_nodes))
    # DEBUG: Should I consider all correlated nodes that are anomalies in seed ? Now it's all anomalies nodes that has correlated.
    init_correlated_anomalies_nodes = anomalies_nodes[~anomalies_nodes["node_uuid"].isin(init_disconnected_nodes)]
    print("Number of initially correlated nodes:", len(init_correlated_anomalies_nodes))

    if args.process_centric:
        short_consider_node_types = ["process"]
    else:
        short_consider_node_types = ["process", "flow","net","netflowobject","file","module"]
    consider_node_types = [node_type for node_type in init_correlated_anomalies_nodes["node_type"].unique() if node_type.split("_")[-1].lower() in short_consider_node_types]

    seed_anomalies_nodes = pd.DataFrame()
    for node_type in consider_node_types:
        seed_anomalies_nodes_tmp = init_correlated_anomalies_nodes[init_correlated_anomalies_nodes["node_type"] == node_type]
        print("Total number of {} nodes is {} ".format(node_type, len(seed_anomalies_nodes_tmp)))
        if k_percent:
            k = int(math.ceil(k_percent * len(seed_anomalies_nodes_tmp)))
        seed_anomalies_nodes_tmp = seed_anomalies_nodes_tmp.nlargest(k,"Anomaly_score")
        seed_anomalies_nodes = pd.concat([seed_anomalies_nodes, seed_anomalies_nodes_tmp],ignore_index=True).drop_duplicates(subset=['node_uuid'])
        print("Number of seed nodes for  ", node_type, "is", len(seed_anomalies_nodes_tmp))
        del seed_anomalies_nodes_tmp
    seed_anomalies_nodes['node_type_order'] = seed_anomalies_nodes['node_type'].apply(lambda x: node_type_order(x))
    # seed_anomalies_nodes = seed_anomalies_nodes.sort_values(by=['node_type_order','Prediction_probability'],ascending=[True,False]).drop(columns='node_type_order')
    seed_anomalies_nodes = seed_anomalies_nodes.sort_values(by=['Prediction_probability', 'node_type_order'], ascending=[False, True]).drop(columns='node_type_order')
    print("Total Number of seed nodes:",len(seed_anomalies_nodes["node_uuid"]))

    # Traverse from node through anomalies nodes to get anomalies subgraphs
    all_traverse_time = time.time()
    # Correlate each anomalous nodes into only one subgraph.
    global correlated_anomalies_nodes
    correlated_anomalies_nodes = set()
    unvisited_anomalies_nodes = set(init_correlated_anomalies_nodes["node_uuid"].unique())
    subgraphs_lst = []
    patience = 10
    avoid_stuck = 30
    last_remaining = 0
    for i,seed_node in enumerate(seed_anomalies_nodes["node_uuid"].tolist()):
        print("Debug: seed node is", seed_node)
        subgraphs_gen = expand_subgraph_in_memory(seed_node)
        subgraphs_lst.extend([subgraph for subgraph in iter(subgraphs_gen) if subgraph is not None])
        print("DEBUG: number of correlated_anomalies_nodes", len(correlated_anomalies_nodes))
        unvisited_anomalies_nodes = unvisited_anomalies_nodes - correlated_anomalies_nodes
        print("In seed number {}, number of remaining anomalous nodes is {}".format(i, len(unvisited_anomalies_nodes)))
        if len(unvisited_anomalies_nodes) == 0:
            print("Correlated all seed nodes in seed number {}".format(i))
            break
        elif k_percent:
            if len(unvisited_anomalies_nodes) < last_remaining:
                patience = 10
            patience -= 1
            if patience == 0:
                print("couldn't construct more subgraphs after 10 patience iteration in seed number {}".format(i))
                break
            last_remaining = len(unvisited_anomalies_nodes)
    print("The total traverse time is: ", time.time() - all_traverse_time, "seconds.")
    if args.remove_duplicated_subgraph:
        print("Number of subgraphs before removing duplication", len(subgraphs_lst))
        subgraphs_lst = remove_identicatl_subgraphs(subgraphs_lst)

    if len(subgraphs_lst) > 0:
        disconnected_nodes, subgraphs_stats_df, all_correlated_nodes,end_construction_time, anomaly_results_df_run = explore_and_draw_subgaphs(subgraphs_lst)
    else:
        print("Couldn't construct any subgraph")
        # global run_start_time
        print("Total time: ", time.time() - run_start_time, "seconds.")
        anomaly_results_df_run = pd.DataFrame(
            {'accuracy': [0], 'precision': [0], 'recall': [0], 'f_measure': [0], 'tp': [0], 'tn': [0], 'fp': [0],
             'fn': [0], 'tpr': [0], 'fpr': [0], 'mcc': [0]})
        anomaly_results_df_run["subgraph_detection_time"] = time.time() - run_start_time
        anomaly_results_df_run["memory_usage_mb"] = getrusage(RUSAGE_SELF).ru_maxrss / 1024
        return None , None , None, None, anomaly_results_df_run
    print("The subgraph construction time is: ", end_construction_time - start_construction_time , "seconds.")
    print("The subgraph detection time is: ", end_construction_time - run_start_time, "seconds.")
    anomaly_results_df_run["subgraph_detection_time"] = end_construction_time - run_start_time
    anomaly_results_df_run["memory_usage_mb"] = getrusage(RUSAGE_SELF).ru_maxrss / 1024
    print_memory_usage()
    return subgraphs_lst, disconnected_nodes, subgraphs_stats_df, all_correlated_nodes, anomaly_results_df_run




def construct_subgraphs_by_query(number_of_hops):
    start_construction_time = time.time()
    global anomalies_nodes
    clear_and_insert_anomalies(anomalies_nodes)
    subgraph_obj = query_sparql_triple_subgraph(anomalies_nodes, number_of_hops)

    subgraphs_lst, disconnected_nodes, subgraphs_stats_df = get_connected_sugraphs(subgraph_obj)


    # ########### Debug #########
    # subgraphs_sample = {}
    # for subgraph_id,subgraph in enumerate(subgraphs_lst):
    #     subgraphs_sample[subgraph_id] = subgraph
    # investigate_subgraphs(subgraphs_sample)
    # ###########################

    if args.retraverse_disconnected_nodes and len(disconnected_nodes) > 0 and number_of_hops == 0:
        disconnected_anomalies_nodes = anomalies_nodes[anomalies_nodes["node_uuid"].isin(disconnected_nodes)]
        subgraph_obj_2 = query_sparql_triple_subgraph(disconnected_anomalies_nodes,number_of_hops=1)
        subgraph_obj = nx.compose(subgraph_obj_2,subgraph_obj)
        subgraphs_lst, disconnected_nodes, subgraphs_stats_df = get_connected_sugraphs(subgraph_obj)
    if args.draw_subgraphs:
        print("Drawing Subgraphs")
        for i, sg in enumerate(subgraphs_lst):
            if sg.number_of_edges() < 6000:
                draw_subgraph(i, sg)
            else:
                print("Skipping drawing subgraph with", sg.number_of_edges(), " edges")
    print("The subgraph construction time is: ", time.time() - start_construction_time, "seconds.")
    return subgraphs_lst, disconnected_nodes, subgraphs_stats_df

def prepare_a_query(query,node):
    global max_edges
    if max_edges == 0:
        query = query.replace("<SEED_NODE>", str(node)).replace("LIMIT <LIMIT>", "")
    else:
        query = query.replace("<SEED_NODE>", str(node)).replace("<LIMIT>", str(max_edges))
    return query
def expand_subgraph_by_query(node):
    global max_edges
    base_headers = ["subject_uuid", "subject_type", "predicate", "timestamp", "object_uuid","object_type"]
    results_df = pd.DataFrame(columns=base_headers)

    read_sparql = SPARQLWrapper(repository_url)
    read_sparql.setReturnFormat(JSON)
    read_sparql.setQuery(prepare_a_query(sparql_queries['expand_a_node_Backward_via_anomalies_nodes'], node))
    results = read_sparql.queryAndConvert()
    results_df = pd.DataFrame(results['results']['bindings'])
    results_df = results_df.map(lambda x: x['value'] if type(x) is dict else x).drop_duplicates()

    read_sparql.setQuery(prepare_a_query(sparql_queries['expand_a_node_Forward_via_anomalies_nodes'],node))
    results = read_sparql.queryAndConvert()
    results_df_F = pd.DataFrame(results['results']['bindings'])
    results_df_F = results_df_F.map(lambda x: x['value'] if type(x) is dict else x).drop_duplicates()

    results_df = pd.concat([results_df, results_df_F], ignore_index=True).drop_duplicates()
    if len(results_df) == 0:
        subgraph_nodes = 0
    else:
        subgraph_nodes = len(set(results_df["subject_uuid"].unique().tolist() + results_df["object_uuid"].unique().tolist()))
    del results, results_df_F
    # if len(results_df) < args.min_edges:
    if subgraph_nodes < args.min_nodes:
        print("Traverse Through intermediate nodes")
        read_sparql.setQuery(prepare_a_query(sparql_queries['expand_a_node_Forward_via_intermediate_nodes_RR'],node))
        results = read_sparql.queryAndConvert()
        results_df_tmp = pd.DataFrame(results['results']['bindings'])
        if len(results_df_tmp) > 0:
            results_df_tmp = results_df_tmp.map(lambda x: x['value'] if type(x) is dict else x).drop_duplicates()
            results_df_tmp_1 = results_df_tmp[["subject_uuid","subject_type","predicate", "timestamp", "object_uuid","object_type"]]
            results_df_tmp_2 = results_df_tmp[["object_uuid", "object_type", "next_predicate", "next_timestamp", "next_object_uuid", "next_object_type"]]
            results_df_tmp_2.columns = base_headers
            # results_df_tmp_2.rename(columns={"next_predicate": "predicate","next_timestamp":"timestamp","next_object_uuid":"object_uuid","next_object_type":"object_type","next_object_attr":"object_attr"}, inplace=True)
            results_df_RR = pd.concat([results_df_tmp_1, results_df_tmp_2], ignore_index=True).drop_duplicates()
            results_df = pd.concat([results_df, results_df_RR], ignore_index=True).drop_duplicates()
            del results_df_tmp,results_df_tmp_1, results_df_tmp_2, results_df_RR

        read_sparql.setQuery(prepare_a_query(sparql_queries['expand_a_node_Forward_via_intermediate_nodes_RL'], node))
        results = read_sparql.queryAndConvert()
        results_df_tmp = pd.DataFrame(results['results']['bindings'])
        if len(results_df_tmp) > 0:
            results_df_tmp = results_df_tmp.map(lambda x: x['value'] if type(x) is dict else x).drop_duplicates()
            results_df_tmp_1 = results_df_tmp[["subject_uuid", "subject_type", "predicate", "timestamp", "object_uuid", "object_type"]]
            results_df_tmp_2 = results_df_tmp[["subject_uuid", "subject_type", "next_predicate", "next_timestamp", "next_object_uuid","next_object_type"]]
            # results_df_tmp_2.rename(columns={"next_predicate": "predicate","next_timestamp":"timestamp","next_object_uuid":"object_uuid","next_object_type":"object_type","next_object_attr":"object_attr"}, inplace=True)
            results_df_tmp_2.columns = base_headers
            results_df_RL = pd.concat([results_df_tmp_1, results_df_tmp_2], ignore_index=True).drop_duplicates()
            results_df = pd.concat([results_df, results_df_RL], ignore_index=True).drop_duplicates()
            del results_df_tmp, results_df_tmp_1, results_df_tmp_2, results_df_RL

        read_sparql.setQuery(prepare_a_query(sparql_queries['expand_a_node_Forward_via_intermediate_nodes_LR'], node))
        results = read_sparql.queryAndConvert()
        results_df_tmp = pd.DataFrame(results['results']['bindings'])
        if len(results_df_tmp) > 0:
            results_df_tmp = results_df_tmp.map(lambda x: x['value'] if type(x) is dict else x).drop_duplicates()
            results_df_tmp_1 = results_df_tmp[["subject_uuid", "subject_type", "predicate", "timestamp", "object_uuid", "object_type"]]
            results_df_tmp_2 = results_df_tmp[["next_subject_uuid","next_subject_type", "next_predicate", "next_timestamp", "object_uuid", "object_type"]]
            results_df_tmp_2.columns = base_headers
            results_df_LR = pd.concat([results_df_tmp_1, results_df_tmp_2], ignore_index=True).drop_duplicates()
            results_df = pd.concat([results_df, results_df_LR], ignore_index=True).drop_duplicates()
            del results_df_tmp, results_df_tmp_1, results_df_tmp_2,results_df_LR

        read_sparql.setQuery(prepare_a_query(sparql_queries['expand_a_node_Forward_via_intermediate_nodes_LL'], node))
        results = read_sparql.queryAndConvert()
        results_df_tmp = pd.DataFrame(results['results']['bindings'])
        if len(results_df_tmp) > 0:
            results_df_tmp = results_df_tmp.map(lambda x: x['value'] if type(x) is dict else x).drop_duplicates()
            results_df_tmp_1 = results_df_tmp[["subject_uuid", "subject_type", "predicate", "timestamp", "object_uuid", "object_type"]]
            results_df_tmp_2 = results_df_tmp[["next_subject_uuid","next_subject_type", "next_predicate", "next_timestamp", "subject_uuid", "subject_type"]]
            results_df_tmp_2.columns = base_headers
            results_df_LL = pd.concat([results_df_tmp_1, results_df_tmp_2], ignore_index=True).drop_duplicates()
            results_df = pd.concat([results_df, results_df_LL], ignore_index=True).drop_duplicates()
            del results_df_tmp, results_df_tmp_1, results_df_tmp_2,results_df_LL
        if len(results_df) == 0:
            subgraph_nodes = 0
        else:
            subgraph_nodes = len(set(results_df["subject_uuid"].unique().tolist() + results_df["object_uuid"].unique().tolist()))
        # if len(results_df) < args.min_edges:
        if subgraph_nodes < args.min_nodes:
            print("Subgraph is less than ", str(args.min_nodes),"nodes")
            subgraph = None
        else:
            subgraph, nodes_df = result_df_to_subgraph_nx(results_df)
            # if args.get_node_attrs:
            #     nodes_df = get_node_attr_by_query(nodes_df)
            node_attr = nodes_df.set_index('node_uuid').to_dict('index')
            nx.set_node_attributes(subgraph, node_attr)
            del node_attr, nodes_df
    else:
        subgraph, nodes_df = result_df_to_subgraph_nx(results_df)
        # if args.get_node_attrs:
        #     nodes_df = get_node_attr_by_query(nodes_df)
        node_attr = nodes_df.set_index('node_uuid').to_dict('index')
        nx.set_node_attributes(subgraph, node_attr)
        del node_attr, nodes_df

    return subgraph

def get_outliers_anomaly_score_grubbs(anomaly_scores):
    outliers_anomaly_scores = grubbs.max_test_outliers(anomaly_scores.tolist(),alpha=0.05)
    if len(outliers_anomaly_scores)>0:
        is_anomaly_scores_outliers = anomaly_scores >= min(outliers_anomaly_scores)
    else:
        is_anomaly_scores_outliers = [False for _ in anomaly_scores]
    return is_anomaly_scores_outliers

def explore_all_subgraphs(subgraphs_lst):
    subgraphs_stats = []
    # RaisedBenignNodes, BenignNodes, MaliciousNodes, MissedMaliciousNodes
    global all_predicted_anomalies_benign, all_predicted_anomalies_malicious, all_predicted_normal_benign, all_predicted_normal_malicious
    all_predicted_anomalies_benign, all_predicted_anomalies_malicious, all_predicted_normal_benign, all_predicted_normal_malicious = set(), set(), set(), set()
    for i, sg in enumerate(subgraphs_lst):
        subgraphs_stats.append(explore_subgraph(i, sg))
    print("----------------------------------------\n")
    print("Total Number of Predicted Anomalies Nodes in correlated subgraphs",
          len(all_predicted_anomalies_benign) + len(all_predicted_anomalies_malicious))
    print("Total Number of Predicted Anomalies, Labeled Malicious Nodes in correlated subgraphs",
          len(all_predicted_anomalies_malicious))
    print("Total Number of Predicted Anomalies, Labeled Benign Nodes in correlated subgraphs",
          len(all_predicted_anomalies_benign))
    print("\nTotal Number of Predicted Normal Nodes in correlated subgraphs",
          len(all_predicted_normal_benign) + len(all_predicted_normal_malicious))
    print("Total Number of Predicted Normal, Labeled Malicious Nodes in correlated subgraphs",
          len(all_predicted_normal_malicious))
    print("Total Number of Predicted Normal, Labeled Benign Nodes in correlated subgraphs",
          len(all_predicted_normal_benign))

    subgraphs_stats_df = pd.DataFrame(subgraphs_stats, columns=["ID", "number_of_nodes", "number_of_edges",
                                                                "predicted_anomalies_benign",
                                                                "predicted_anomalies_malicious",
                                                                "predicted_normal_benign", "predicted_normal_malicious", "number_of_malicious_nodes",
                                                                "first_action_datetime", "last_action_datetime",
                                                                "number_of_actions_in_attack_time",
                                                                "percentage_of_actions_in_attack_time","subgraph_anomaly_score"])

    print(subgraphs_stats_df[['number_of_nodes', 'number_of_edges','subgraph_anomaly_score']].describe())
    # display(subgraphs_stats_df)
    return subgraphs_stats_df

def construct_subgraphs_by_criteria(k=100):
    start_construction_time = time.time()
    global anomalies_nodes
    # Select seed anomalies nodes
    short_consider_node_types = ["process","file","flow","net"]
    consider_node_types = [ node_type for node_type in anomalies_nodes["node_type"].unique().tolist() if any(short_node_type in node_type.lower() for short_node_type in short_consider_node_types) ]
    seed_anomalies_nodes = pd.DataFrame()
    for node_type in consider_node_types:
        seed_anomalies_nodes_tmp = anomalies_nodes[anomalies_nodes["node_type"] == node_type].sort_values(by=["Anomaly_score"],ascending=False)[:k]
        seed_anomalies_nodes = pd.concat([seed_anomalies_nodes,seed_anomalies_nodes_tmp], ignore_index=True).drop_duplicates(subset=['node_uuid'])
        print("Number of seed nodes for  ",node_type,"is", len(seed_anomalies_nodes_tmp))
        del seed_anomalies_nodes_tmp
    seed_nodes = set(seed_anomalies_nodes["node_uuid"].unique())
    print("Total number of seed nodes is ", len(seed_nodes))
    clear_and_insert_anomalies(anomalies_nodes)

    # Traverse from node through anomalies nodes to get anomalies subgraphs
    subgraphs_lst = [expand_subgraph_by_query(node) for node in seed_nodes ]
    subgraphs_lst = [subgraph for subgraph in subgraphs_lst if subgraph is not None]
    if args.remove_duplicated_subgraph:
        print("Number of subgraphs before removing duplication", len(subgraphs_lst))
        subgraphs_lst = remove_identicatl_subgraphs(subgraphs_lst)
    disconnected_nodes, subgraphs_stats_df, all_correlated_nodes, end_construction_time = explore_and_draw_subgaphs(subgraphs_lst)

    print("The subgraph detection time is: ", end_construction_time - start_construction_time, "seconds.")
    return subgraphs_lst, disconnected_nodes, subgraphs_stats_df , all_correlated_nodes

def traverse_anomalies_path(subgraph_obj,anomalies_nodes_lst):
    # Not Completed Function
    # filtered_subgraph_obj = subgraph_obj.copy()
    filtered_subgraph_obj = nx.MultiDiGraph()
    # filtered_subgraph_df = pd.DataFrame()
    for subject,object, edge_attr in subgraph_obj.edges(data=True):
        if (subject in anomalies_nodes_lst) and (object in anomalies_nodes_lst):
            filtered_subgraph_obj.add_edge(subject,object,action= edge_attr['action'],timestamp=edge_attr['timestamp'])
        elif subject in anomalies_nodes_lst:
            for _, object2 in subgraph_obj.out_edges(object):
                if object2 in anomalies_nodes_lst:
                    filtered_subgraph_obj.add_edge(subject, object, action=edge_attr['action'],
                                                   timestamp=edge_attr['timestamp'])
                    all_edge_attr2 = subgraph_obj.get_edge_data(object, object2).items()
                    for key, edge_attr2 in all_edge_attr2:
                        filtered_subgraph_obj.add_edge(object, object2, action=edge_attr2['action'],
                                                   timestamp=edge_attr2['timestamp'])
            for subject2, _ in subgraph_obj.in_edges(object):
                if subject2 in anomalies_nodes_lst:
                    filtered_subgraph_obj.add_edge(subject, object, action=edge_attr['action'],
                                                   timestamp=edge_attr['timestamp'])
                    all_edge_attr2 = subgraph_obj.get_edge_data(subject2, object).items()
                    for key,edge_attr2 in all_edge_attr2:
                        filtered_subgraph_obj.add_edge(subject2, object, action=edge_attr2['action'],
                                                   timestamp=edge_attr2['timestamp'])
        elif object in anomalies_nodes_lst:
            for _, object2 in subgraph_obj.out_edges(subject):
                if object2 in anomalies_nodes_lst:
                    filtered_subgraph_obj.add_edge(subject, object, action=edge_attr['action'],
                                                   timestamp=edge_attr['timestamp'])
                    all_edge_attr2 = subgraph_obj.get_edge_data(subject, object2).items()
                    for key, edge_attr2 in all_edge_attr2:
                        filtered_subgraph_obj.add_edge(subject, object2, action=edge_attr2['action'],
                                                   timestamp=edge_attr2['timestamp'])
            for subject2, _ in subgraph_obj.in_edges(subject):
                if subject2 in anomalies_nodes_lst:
                    filtered_subgraph_obj.add_edge(subject, object, action=edge_attr['action'],
                                                   timestamp=edge_attr['timestamp'])
                    all_edge_attr2 = subgraph_obj.get_edge_data(subject2, subject).items()
                    for key, edge_attr2 in all_edge_attr2:
                        filtered_subgraph_obj.add_edge(subject2, subject, action=edge_attr2['action'],
                                                   timestamp=edge_attr2['timestamp'])
    print("Number of Nodes:",filtered_subgraph_obj.number_of_nodes(), ", Number of edges:", filtered_subgraph_obj.number_of_edges())
    return filtered_subgraph_obj
def clustering_graphs(encoded_subgraphs_data_lst,subgraphs_stats_df):
    start_clustering_time = time.time()
    X_graph_emb = np.empty((len(encoded_subgraphs_data_lst), args.tensor_neurons))
    for i, graph_data in enumerate(encoded_subgraphs_data_lst):
        X_graph_emb[i] = graph_data.__getitem__("graph_emb").detach().numpy().flatten()
    scaler = StandardScaler()
    X_graph_emb_scaled = scaler.fit_transform(X_graph_emb)

    # labels = ClusterWithKMeans(X_graph_emb_scaled, n_clusters=4)
    # labels = ClusterWithDBSCAN(X_graph_emb_scaled)
    labels = ClusterWithAgglomerative(X_graph_emb_scaled,subgraphs_stats_df)
    for i, subgraph in enumerate(encoded_subgraphs_data_lst):
        subgraph.cluster_label = labels[i]
    print("The clustering time is: ", time.time() - start_clustering_time, "seconds.")
    return encoded_subgraphs_data_lst

def ClusterWithAgglomerative(X_graph_emb,subgraphs_stats_df,distance_threshold=3,n_clusters=None):
    agg_clustering = AgglomerativeClustering(distance_threshold=distance_threshold, n_clusters=n_clusters)
    # Fit AgglomerativeClustering to the data and get cluster labels
    labels = agg_clustering.fit_predict(X_graph_emb)

    pca = PCA(n_components=2)
    X_graph_emb_draw = pca.fit_transform(X_graph_emb)

    # Plotting the clusters
    plt.figure(figsize=(8, 6))

    # Plot each cluster
    unique_labels = set(labels)
    for label in unique_labels:
        cluster_points = X_graph_emb_draw[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label}')

    # Plot mark for subgraphs with malicious nodes
    malicious_nodes_num = subgraphs_stats_df["predicted_anomalies_malicious"] + subgraphs_stats_df[
        "predicted_normal_malicious"]
    cluster_points = X_graph_emb_draw[malicious_nodes_num > 0]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1],s=250, marker="D", facecolors="none",edgecolors="red",label='Has Malicious Nodes')

    # Plot mark subgraphs with actions within attack timerange
    cluster_points = X_graph_emb_draw[subgraphs_stats_df["percentage_of_actions_in_attack_time"] == 0]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=250, marker="s", facecolors="none", edgecolors="blue",label='No Edges in Attack Time')
    plt.title('Agglomerative Clustering')
    # plt.xlabel('Feature 1')
    # plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    out_path = args.root_path + "figures/" + args.exp_name + "/" + args.model.replace(".model","") + "/Agglomerative_clustering_ExpParam.pdf"
    out_path = rename_path_with_param(out_path)
    ensure_dir(out_path)
    plt.savefig(out_path)
    # plt.show()
    plt.close()
    print("Number of clusters is: ", len(unique_labels))
    unique, counts = np.unique(labels, return_counts=True)
    print("Number of subgraphs per cluster:\n",np.asarray((unique, counts)).T)
    return labels

# def ClusterWithKMeans(X_graph_emb, n_clusters=4):
#     # KNN_clustering = KNeighborsClassifier(n_neighbors=3)
#     kmeans = KMeans(n_clusters=n_clusters)
#     kmeans.fit(X_graph_emb)
#     labels = kmeans.labels_
#     centroids = kmeans.cluster_centers_
#     plt.figure(figsize=(8, 6))
#     for i in range(n_clusters):
#         # Data points for this cluster
#         cluster_points = X_graph_emb[labels == i]
#         # Plot the cluster points
#         plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i + 1}')
#
#     # Plot the centroids
#     plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='black', s=200, label='Centroids')
#     plt.title('K-Means Clustering')
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#     out_path = args.root_path + "figures/" + args.exp_name + "/" + args.model + "_KMeans_clustering_" +str(args.number_of_hops)+"_hop.pdf"
#     ensure_dir(out_path)
#     plt.savefig(out_path)
#     unique, counts = np.unique(labels, return_counts=True)
#     print("Number of subgraphs per cluster:\n", np.asarray((unique, counts)).T)
#     return labels


def graph_embedding(id, graph, nodes_embeddings):
    print("Graph embedding for subgraph {}, with {} nodes and {} edges".format(id,graph.number_of_nodes(),graph.number_of_edges()))
    graph_data = encode_subgraphs(id, graph, nodes_embeddings=nodes_embeddings)
    node_features = graph_data.__getitem__("nodes_emb")
    batch = (graph_data.batch if (graph_data.batch is not None)
        else torch.tensor((), dtype=torch.long).new_zeros(graph_data.num_nodes)
    )
    attention_layer = AttentionModule(args)
    graph_emb = attention_layer(node_features, batch)
    graph_data.graph_emb = graph_emb
    return graph_data

def graph_embedding_per_subgraph(id, graph):
    debug = time.time()
    global nodes_features
    print("Graph embedding for subgraph {}, with {} nodes and {} edges".format(id,graph.number_of_nodes(),graph.number_of_edges()))
    graph_data = encode_subgraphs_v2(id, graph, nodes_features=nodes_features)
    # graph_data = encode_subgraphs(id, graph,nodes_features=nodes_features)
    print("Encoding takes {} seconds.".format(time.time() - debug))
    graph_data = nodes_embeddings_per_subgraph(graph_data)
    debug = time.time()
    node_features = graph_data.__getitem__("nodes_emb")
    batch = (graph_data.batch if (graph_data.batch is not None)
        else torch.tensor((), dtype=torch.long).new_zeros(graph_data.num_nodes)
    )
    attention_layer = AttentionModule(args)
    graph_emb = attention_layer(node_features, batch)
    graph_data.graph_emb = graph_emb
    print("Embedding takes {} seconds.".format(time.time() - debug))
    return graph_data

def nodes_embeddings_per_subgraph(graph_data):
    global detectors, full_graph_key2int, subject_nodes
    device = 'cpu'
    x = graph_data.x.to(device)
    edge_index = graph_data.edge_index.to(device)
    edge_type = graph_data.edge_type.to(device)
    all_nodes_emb = {}
    nodes_embeddings_per_node_type = torch.zeros((graph_data.num_nodes,args.node_emb_size), dtype=torch.float)
    for subject_node in subject_nodes:
        all_nodes_emb[subject_node] = detectors[subject_node].model(x, edge_index, edge_type)
        node_type_mask = graph_data.node_type == full_graph_key2int[subject_node]
        nodes_embeddings_per_node_type[node_type_mask] = all_nodes_emb[subject_node][node_type_mask]
    graph_data.nodes_emb = nodes_embeddings_per_node_type
    return graph_data
def encode_subgraphs(id,g,nodes_embeddings=None,nodes_features=None):
    # Encode edges with the same mapping as the original graph
    global full_graph_key2int
    for s, o in g.edges():
        for k, edge_attr in g.get_edge_data(s, o).items():
            try:
                g.edges[s, o, k]["edge_type"] = full_graph_key2int[(g.nodes[s]['node_type'], edge_attr["action"], g.nodes[o]['node_type'])]
            except Exception as e:
                print("Remove undefined edge type. The error %s. The edge attributes %s ", e, edge_attr["action"])
                g.remove_edge(s, o, k)
    mapping = {uuid: j for j, uuid in enumerate(g.nodes())}
    inv_mapping = {j: uuid for j, uuid in enumerate(g.nodes())}
    g = nx.relabel_nodes(g, mapping)
    for node,node_type in g.nodes.data("node_type"):
        g.nodes[node]["node_type_label"] = full_graph_key2int[node_type]
        if nodes_embeddings is not None:
            g.nodes[node]["nodes_emb"] = torch.from_numpy(nodes_embeddings.loc[nodes_embeddings["node_uuid"]==inv_mapping[node],nodes_embeddings.columns != "node_uuid"].to_numpy()).view(-1).float()
        elif nodes_features is not None:
            g.nodes[node]["nodes_feat"] = torch.from_numpy(nodes_features.loc[nodes_features["node_uuid"] == inv_mapping[node], nodes_features.columns != "node_uuid"].to_numpy()).view(-1).float()
    if nodes_embeddings is not None:
        dgl_g = dgl.from_networkx(g, node_attrs=["nodes_emb", "node_type_label"], edge_attrs=["edge_type"])
    elif nodes_features is not None:
        dgl_g = dgl.from_networkx(g, node_attrs=["nodes_feat", "node_type_label"], edge_attrs=["edge_type"])
    g.clear()
    edge_index = torch.tensor([dgl_g.edges()[0].tolist(), dgl_g.edges()[1].tolist()])
    graph_data = Data(edge_index=edge_index, num_nodes=dgl_g.number_of_nodes(),node_type=dgl_g.ndata["node_type_label"],edge_type=dgl_g.edata['edge_type'], id=str(id))
    if nodes_embeddings is not None:
        graph_data.nodes_emb = dgl_g.ndata['nodes_emb']
    elif nodes_features is not None:
        graph_data.x = dgl_g.ndata['nodes_feat']
    del dgl_g
    return graph_data

def encode_subgraphs_v2(id,g,nodes_embeddings=None,nodes_features=None):
    # Encode edges with the same mapping as the original graph
    global full_graph_key2int
    graph_df = nx.to_pandas_edgelist(g, edge_key='ekey')
    node_types_dic = g.nodes.data("node_type")
    graph_df["edge_type"] = graph_df.apply(lambda x: full_graph_key2int[(node_types_dic[x.source], x.action, node_types_dic[x.target])], axis =1)
    edge_attr = graph_df.set_index(['source','target','ekey']).to_dict('index')
    nx.set_edge_attributes(g, edge_attr)
    del graph_df,edge_attr,node_types_dic
    allNodes_df = pd.DataFrame(g.nodes.data("node_type"), columns=["node", "node_type"])
    allNodes_df["node_type_label"] = allNodes_df["node_type"].apply(lambda x: full_graph_key2int[x])
    if nodes_embeddings is not None:
        allNodes_df["nodes_emb"] = allNodes_df["node"].apply(lambda x: torch.from_numpy(nodes_embeddings.loc[nodes_embeddings["node_uuid"] == x, nodes_embeddings.columns != "node_uuid"].to_numpy()).view(-1).float())
    if nodes_features is not None:
        allNodes_df["nodes_feat"] = allNodes_df["node"].apply(lambda x: torch.from_numpy(nodes_features.loc[nodes_features["node_uuid"] == x, nodes_features.columns != "node_uuid"].to_numpy()).view(-1).float())
    node_attr = allNodes_df.set_index('node').to_dict('index')
    nx.set_node_attributes(g, node_attr)
    del allNodes_df, node_attr
    mapping = {uuid: j for j, uuid in enumerate(g.nodes())}
    g = nx.relabel_nodes(g, mapping)
    del mapping
    if nodes_embeddings is not None:
        dgl_g = dgl.from_networkx(g, node_attrs=["nodes_emb", "node_type_label"], edge_attrs=["edge_type"])
    elif nodes_features is not None:
        dgl_g = dgl.from_networkx(g, node_attrs=["nodes_feat", "node_type_label"], edge_attrs=["edge_type"])
    g.clear()
    edge_index = torch.tensor([dgl_g.edges()[0].tolist(), dgl_g.edges()[1].tolist()])
    graph_data = Data(edge_index=edge_index, num_nodes=dgl_g.number_of_nodes(),node_type=dgl_g.ndata["node_type_label"],edge_type=dgl_g.edata['edge_type'], id=str(id))
    if nodes_embeddings is not None:
        graph_data.nodes_emb = dgl_g.ndata['nodes_emb']
    elif nodes_features is not None:
        graph_data.x = dgl_g.ndata['nodes_feat']
    del dgl_g
    return graph_data

def get_connected_sugraphs(subgraph_obj):
    subgraphs_lst = [subgraph_obj.subgraph(nodes).copy() for nodes in nx.weakly_connected_components(subgraph_obj) if len(nodes) > 1]
    # subgraphs_lst = sorted(subgraphs_lst, key=lambda g: g.number_of_nodes(),reverse=True)
    disconnected_nodes = [nodes for nodes in nx.weakly_connected_components(subgraph_obj) if len(nodes) == 1]
    disconnected_nodes = [n.pop() for n in disconnected_nodes]
    print("Number of disconnected nodes", len(disconnected_nodes))
    print("Number of correlated subgraphs:", len(subgraphs_lst))

    subgraphs_stats_df = explore_all_subgraphs(subgraphs_lst)

    return subgraphs_lst,disconnected_nodes, subgraphs_stats_df

def subgraph_anomaly_score(subgraph):
    global anomalies_nodes
    all_correlated_nodes = list({subgraph_nodes for subgraph_nodes in list(subgraph.nodes())})
    #### debug ####
    # Should anomaly score be based on all correlated nodes ? #
    correlated_anomalies_nodes = anomalies_nodes[anomalies_nodes["node_uuid"].isin(all_correlated_nodes)]
    sg_anomaly_score = correlated_anomalies_nodes["Prediction_probability"].sum()
    return sg_anomaly_score

def explore_subgraph(id,subgraph):
    global anomalies_nodes_uuid, malicious_nodes_uuid
    global all_predicted_anomalies_benign, all_predicted_anomalies_malicious, all_predicted_normal_benign, all_predicted_normal_malicious
    action_timestamps = [my_tz.localize(datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')) for _, _, timestamp in
                         subgraph.edges.data("timestamp")]
    first_action_dateime = min(action_timestamps)
    last_action_dateime = max(action_timestamps)
    print("--------------------------------------")
    print("subgraph {} actions are between:".format(id), first_action_dateime.strftime("%Y-%m-%d %H:%M:%S"), " and ",
          last_action_dateime.strftime("%Y-%m-%d %H:%M:%S"))
    n_actions = subgraph.number_of_edges()
    n_nodes = subgraph.number_of_nodes()
    print("Number of Nodes:",n_nodes, ", Number of edges:", n_actions)
    # x = list(subgraph.nodes.data("type"))
    # unique_nodes_types = list(set([y[1] for y in x]))
    # print("\nUnique nodes type:", unique_nodes_types)
    # for i in unique_nodes_types:
    #     print(i, ": ", len([node_id for node_id, node_type in subgraph.nodes.data("type") if node_type == i]))
    # x = list(subgraph.edges.data("type"))
    # unique_edges_types = list(set([y[2] for y in x]))
    # print("\nUnique edges type:", unique_edges_types)
    # for i in unique_edges_types:
    #     print(i, ": ", len([node_id for node_id, _, node_type in subgraph.edges.data("type") if node_type == i]))
    predicted_anomalies_benign = {node_id for node_id in subgraph.nodes() if
                                  (node_id not in malicious_nodes_uuid) & (node_id in anomalies_nodes_uuid)}
    predicted_anomalies_malicious = {node_id for node_id in subgraph.nodes() if
                                     (node_id in malicious_nodes_uuid) & (node_id in anomalies_nodes_uuid)}
    predicted_normal_benign = {node_id for node_id in subgraph.nodes() if
                                     (node_id not in malicious_nodes_uuid) & (node_id not in anomalies_nodes_uuid)}
    predicted_normal_malicious = {node_id for node_id in subgraph.nodes() if
                               (node_id in malicious_nodes_uuid) & (node_id not in anomalies_nodes_uuid)}

    attack_time_range = get_attack_time_range(args.host)
    n_actions_in_attack_time_range = {}
    total_n_actions_in_attack_time_range = 0
    for attack in attack_time_range.keys():
        start_attack = my_tz.localize(datetime.strptime(attack_time_range[attack][0], "%Y-%m-%d %H:%M")) - timedelta(minutes=1)
        end_attack = my_tz.localize(datetime.strptime(attack_time_range[attack][1], "%Y-%m-%d %H:%M")) + timedelta(minutes=1)
        n_actions_in_attack_time_range[attack] = len([ timestamp for timestamp in action_timestamps if start_attack <= timestamp <= end_attack])
        if n_actions_in_attack_time_range[attack] > 0:
            print("The subgraph has {} actions ({}%) within {} attack time range".format(
                n_actions_in_attack_time_range[attack], round(n_actions_in_attack_time_range[attack] / n_actions,3), attack))
            total_n_actions_in_attack_time_range += n_actions_in_attack_time_range[attack]
    perc_actions_in_attack_time_range = total_n_actions_in_attack_time_range / n_actions

    # Get anomaly score per subgraph
    sg_anomaly_score = subgraph_anomaly_score(subgraph)

    subgraph_statistics = [id,n_nodes, n_actions, len(predicted_anomalies_benign),
                           len(predicted_anomalies_malicious), len(predicted_normal_benign),
                           len(predicted_normal_malicious),len(predicted_normal_malicious) + len(predicted_anomalies_malicious), first_action_dateime, last_action_dateime,
                           total_n_actions_in_attack_time_range, perc_actions_in_attack_time_range,sg_anomaly_score]
    all_predicted_anomalies_benign.update(predicted_anomalies_benign)
    all_predicted_anomalies_malicious.update(predicted_anomalies_malicious)
    all_predicted_normal_benign.update(predicted_normal_benign)
    all_predicted_normal_malicious.update(predicted_normal_malicious)
    return subgraph_statistics


def draw_subgraph(i,subgraph):
    if args.get_node_attrs:
        # nodes_l des_labels[node] = n`ode_attr["node_type"]
        nodes_labels ={}
        for node, node_attr in list(subgraph.nodes.data()):
            if "node_attr" in node_attr.keys():
                if node_attr["node_attr"] is not None:
                    nodes_labels[node] = node_attr["node_attr"]
                else:
                    nodes_labels[node] = node_attr["node_type"]
            else:
                nodes_labels[node] = node_attr["node_type"]
    else:
        nodes_labels = dict(subgraph.nodes.data("node_type"))
    plt.figure(figsize=(15, 15))

    pos = nx.nx_agraph.graphviz_layout(subgraph)
    # pos = nx.multipartite_layout(subgraph,subset_key="node_type",align='horizontal')
    global anomalies_nodes_uuid, malicious_nodes_uuid
    predicted_anomalies_benign = {node_id for node_id in subgraph.nodes() if
                                  (node_id not in malicious_nodes_uuid) & (node_id in anomalies_nodes_uuid)}
    predicted_anomalies_malicious = {node_id for node_id in subgraph.nodes() if
                                     (node_id in malicious_nodes_uuid) & (node_id in anomalies_nodes_uuid)}
    predicted_normal_benign = {node_id for node_id in subgraph.nodes() if
                               (node_id not in malicious_nodes_uuid) & (node_id not in anomalies_nodes_uuid)}
    predicted_normal_malicious = {node_id for node_id in subgraph.nodes() if
                                  (node_id in malicious_nodes_uuid) & (node_id not in anomalies_nodes_uuid)}

    nx.draw_networkx_nodes(subgraph, pos, nodelist=predicted_anomalies_malicious, node_color="red", node_size=1000,
                           node_shape="o",alpha=0.6)
    nx.draw_networkx_nodes(subgraph, pos, nodelist=predicted_anomalies_benign, node_color="orange", node_size=1000,
                           node_shape="o", alpha=0.6)
    nx.draw_networkx_nodes(subgraph, pos, nodelist=predicted_normal_malicious, node_color="brown", node_size=1000,
                           node_shape="o",alpha=0.6)
    nx.draw_networkx_nodes(subgraph, pos, nodelist=predicted_normal_benign, node_color="skyblue", node_size=1000,
                           node_shape="o",alpha=0.6)
    nx.draw_networkx_labels(subgraph, pos, labels=nodes_labels, font_size=10, font_color="black", font_weight="bold",
                            alpha=0.6)
    edges_labels = {}
    for e1, e2, edge_attr in list(subgraph.edges(data=True)):
        if (e1, e2) in edges_labels.keys():
            edges_labels[(e1, e2)].append(edge_attr["action"])
        else:
            edges_labels[(e1, e2)] = [edge_attr["action"]]
    for (e1, e2) in edges_labels.keys():
        edges_labels[(e1, e2)] = list(set(edges_labels[(e1, e2)]))
    # edges_labels = {(e1, e2): edge_attr['action'] for e1, e2, edge_attr in list(subgraph.edges(data=True))}
    nx.draw_networkx_edges(subgraph, pos, alpha=0.8, width=1, arrows=True, edge_color="grey")
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edges_labels)
    plt.axis('off')
    out_path = args.root_path + "investigation/" + args.exp_name + "/" + args.model.replace(".model","") +"/run"+str(run)+"_ExpParam_correlated_subgraph_" + str(i) + ".pdf"
    out_path = rename_path_with_param(out_path)
    ensure_dir(out_path)
    plt.savefig(out_path)
    # plt.show()
    plt.close()
    return

# def construct_subgraphs(anomalies_nodes):
#     start_construction_time = time.time()
#     global anomalies_nodes_uuid, malicious_nodes_uuid
#     graph_df_path = args.root_path + args.source_graph + "_graph_df.csv"
#     graph_df = pd.read_csv(graph_df_path, sep="\t")
#
#     anomalies_direct_edges_df = graph_df.loc[(graph_df["source-id"].isin(anomalies_nodes_uuid)) & (graph_df["destination-id"].isin(anomalies_nodes_uuid))]
#     # Remove duplicate edges with the same timestamp
#     anomalies_direct_edges_df = anomalies_direct_edges_df.drop_duplicates()
#     anomalies_direct_edges_df.columns = ["subject_uuid", "subject_type", "object_uuid", "object_type", "action", "timestamp"]
#     print("Number of direct edges between anomalies nodes", len(anomalies_direct_edges_df))
#     subgraph_obj = nx.from_pandas_edgelist(
#         anomalies_direct_edges_df,
#         source="subject_uuid",
#         target="object_uuid",
#         edge_attr=["action", "timestamp"],
#         create_using=nx.MultiDiGraph()
#     )
#     # Add other anomalies nodes
#     subgraph_obj.add_nodes_from(anomalies_nodes["node_uuid"])
#     node_attr = anomalies_nodes[["node_uuid", "node_type", "Label", "Model_prediction"]].set_index('node_uuid').to_dict(
#         'index')
#     nx.set_node_attributes(subgraph_obj, node_attr)
#
#     subgraphs_lst, disconnected_nodes, subgraphs_stats_df = get_connected_sugraphs(subgraph_obj)
#
#     print("The construction time is: ", time.time() - start_construction_time, "seconds.")
#     return subgraphs_lst, disconnected_nodes, subgraphs_stats_df

def graph_embedding_subgraphs_lst(subgraphs_lst):
    start_embedding_time = time.time()
    if args.node_emb_per_subgraph:
        global detectors, nodes_features
        feat_path = args.root_path + "features/" + args.exp_name + "/all_features_node_uuid.csv"
        nodes_features = pd.read_csv(feat_path)
        nodes_features = nodes_features.drop_duplicates("node_uuid")
        detectors = {}
        for subject_node in subject_nodes:
            model_path = args.root_path + "models/" + args.exp_name + "/" + subject_node + "_" + args.model.replace(".model","") + ".model"
            detectors[subject_node] = torch.load(model_path)
        encoded_subgraphs_data_lst = [graph_embedding_per_subgraph(id, graph) for id, graph in
                                      enumerate(subgraphs_lst)]
    else:
        emb_path = args.root_path + "embedding/" + args.exp_name + "/" + args.model.replace(".model","") + ".csv"
        nodes_embeddings = pd.read_csv(emb_path)
        nodes_embeddings = nodes_embeddings.drop_duplicates("node_uuid")
        encoded_subgraphs_data_lst = [graph_embedding(id, graph, nodes_embeddings) for id, graph in
                                      enumerate(subgraphs_lst)]
    print("The graph embedding time time is: ", time.time() - start_embedding_time, "seconds.")
    return encoded_subgraphs_data_lst

def get_edges_mapping_of_full_graph():
    to_remove_pedicates = []
    to_remove_subject_object = []
    to_keep_edge_idx_map = []
    dataset = PygNodePropPredDataset_custom(name=args.exp_name, root=args.root_path, numofClasses=2)
    print(getrusage(RUSAGE_SELF))
    data = dataset[0]
    subject_nodes = list(data.y_dict.keys())
    split_idx = dataset.get_idx_split('node_type')

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

    edge_index_dict = data.edge_index_dict
    key_lst = list(edge_index_dict.keys())
    ##############add inverse edges ###################
    for key in key_lst:
        r, c = edge_index_dict[(key[0], key[1], key[2])]
        edge_index_dict[(key[2], 'inv_' + key[1], key[0])] = torch.stack([c, r])

    out = group_hetero_graph(data.edge_index_dict, data.num_nodes_dict)
    edge_index, edge_type, node_type, local_node_idx, local2global, key2int = out

    homo_data = Data(edge_index=edge_index, edge_attr=edge_type,
                     node_type=node_type, local_node_idx=local_node_idx,
                     num_nodes=node_type.size(0))
    # del edge_type, node_type, local_node_idx
    testing_idx = []
    mapping_nodes_df = pd.DataFrame()
    for subject_node in subject_nodes:
        testing_idx.extend(local2global[subject_node][split_idx['test'][subject_node]].tolist())
        file_path = args.root_path + args.exp_name + "/mapping/" + subject_node + "_entidx2name.csv"
        mapping_nodes_df_tmp = pd.read_csv(file_path, header=None, skiprows=1, names=["node_id", "node_uuid"])
        mapping_nodes_df_tmp["node_id"] = local2global[subject_node][mapping_nodes_df_tmp["node_id"]]
        mapping_nodes_df = pd.concat([mapping_nodes_df, mapping_nodes_df_tmp])

    mapping_dict = mapping_nodes_df.set_index("node_id")["node_uuid"].to_dict()
    testing_uuid = {mapping_dict[idx] for idx in testing_idx}
    print("DEBUG: Number of testing cases ", len(testing_uuid))
    print("DEBUG: ",key2int)
    print("DEBUG: edge_index stats", homo_data.edge_index.sum(),homo_data.edge_index.min(),homo_data.edge_index.max() )

    return key2int , subject_nodes, mapping_dict, testing_uuid, homo_data.edge_index

def investigate_subgraphs(subgraphs,cluster=None):
    global all_covered_ioc, covered_ioc_dic
    for subgraph_id,subgraph in subgraphs.items():
        print("investigating subgraph: ", subgraph_id)
        allNodes_dic, covered_ioc = label_query_graph_ioc(subgraph)
        all_covered_ioc.update(covered_ioc)
        covered_ioc_dic[subgraph_id] = covered_ioc
        attack_description_df = get_attack_description(subgraph,allNodes_dic,subgraph_id)
        out_path = args.root_path + "investigation/" + args.exp_name + "/" + args.model.replace(".model","") +"/run"+str(run)+"_ExpParam_attack_description_subgraph_" + str(subgraph_id)+".csv"
        checkpoint_save_csv(attack_description_df, out_path)
        del allNodes_dic, covered_ioc, attack_description_df
    return

def label_query_graph_ioc(subgraph):
    global query_graphs_IOCs, query_graphs_IOCs_set
    # attributes = get_subgraphs_attributes(args.host)
    allNodes_dic = dict(subgraph.nodes.data())
    covered_ioc = {}
    ## DEBUG: Need to adjust for Nodlink attributes ##
    if args.dataset == "nodlink":
        sg_Nodes_df = pd.DataFrame.from_dict(allNodes_dic, orient='index').reset_index()
        sg_Nodes_df.columns = ['node', 'node_type', 'node_attr']
        sg_Nodes_df = sg_Nodes_df.dropna(subset=['node_attr'])
        for attack, attack_IOC in query_graphs_IOCs.items():
            if len(query_graphs_IOCs_set)==0:break
            for ip_ioc in attack_IOC["ip"]:
                nodes_matched_iocs = sg_Nodes_df[sg_Nodes_df["node_attr"].str.contains(ip_ioc)]
                covered_ioc_tmp = set(nodes_matched_iocs["node_attr"])
                if len(covered_ioc_tmp) == 0: continue
                if attack not in covered_ioc:
                    covered_ioc[attack] = covered_ioc_tmp
                else:
                    covered_ioc[attack].update(covered_ioc_tmp)
                del covered_ioc_tmp
                for i, row in nodes_matched_iocs.iterrows():
                    allNodes_dic[row["node"]]["ioc"] = row["node_attr"]
                    if "match_attack" in allNodes_dic[row["node"]].keys():
                        allNodes_dic[row["node"]]["match_attack"].add(attack)
                    else:
                        allNodes_dic[row["node"]]["match_attack"] = {attack}
                del nodes_matched_iocs
            for file_ioc in attack_IOC["file"]:
                nodes_matched_iocs = sg_Nodes_df[sg_Nodes_df["node_attr"].str.lower().str.contains(file_ioc.lower())]
                covered_ioc_tmp = set(nodes_matched_iocs["node_attr"])
                if len(covered_ioc_tmp) == 0: continue
                if attack not in covered_ioc:
                    covered_ioc[attack] = covered_ioc_tmp
                else:
                    covered_ioc[attack].update(covered_ioc_tmp)
                del covered_ioc_tmp
                for i, row in nodes_matched_iocs.iterrows():
                    allNodes_dic[row["node"]]["ioc"] = row["node_attr"]
                    if "match_attack" in allNodes_dic[row["node"]].keys():
                        allNodes_dic[row["node"]]["match_attack"].add(attack)
                    else:
                        allNodes_dic[row["node"]]["match_attack"] = {attack}
                del nodes_matched_iocs
        del sg_Nodes_df
    else:
        for node_uuid, node_dic in allNodes_dic.items():
            if node_dic["node_attr"] is not None:
                for attack, attack_IOC in query_graphs_IOCs.items():
                    if len(query_graphs_IOCs_set) == 0: break
                    if node_dic["node_type"].lower() in ["flow","netflowobject","net"]:
                        if args.dataset == "optc":
                            token_attr = node_dic["node_attr"].split(" ")[0]
                        else:
                            token_attr = node_dic["node_attr"]
                        if token_attr in attack_IOC["ip"]:
                            allNodes_dic[node_uuid]["ioc"] = node_dic["node_attr"]
                            # allNodes_dic[node_uuid]["match_attack"] = attack
                            if "match_attack" in allNodes_dic[node_uuid].keys():
                                allNodes_dic[node_uuid]["match_attack"].add(attack)
                            else:
                                allNodes_dic[node_uuid]["match_attack"] = {attack}
                            if attack in covered_ioc.keys():
                                covered_ioc[attack].add(node_dic["node_attr"])
                            else:
                                covered_ioc[attack] = {node_dic["node_attr"]}
                    # if node_dic["node_type"].lower() == "file":
                    else:
                        if args.dataset == "optc":
                            token_attr = node_dic["node_attr"].split("\\")[-1].split(".")[0].lower()
                        else:
                            token_attr = node_dic["node_attr"].lower()
                        if token_attr in attack_IOC["file"]:
                            allNodes_dic[node_uuid]["ioc"] = node_dic["node_attr"]
                            if "match_attack" in allNodes_dic[node_uuid].keys():
                                allNodes_dic[node_uuid]["match_attack"].add(attack)
                            else:
                                allNodes_dic[node_uuid]["match_attack"] = {attack}
                            if attack in covered_ioc.keys():
                                covered_ioc[attack].add(node_dic["node_attr"])
                            else:
                                covered_ioc[attack] = {node_dic["node_attr"]}
    covered_ioc_all_attacks = set()
    if len(covered_ioc) > 0:
        for attack,iocs in covered_ioc.items():
            covered_ioc_all_attacks.update(iocs)
            print("The covered IOCS of attack", attack, "are:", iocs)
            covered_percentage = round(len(iocs) / (len(query_graphs_IOCs[attack]["ip"]) + len(query_graphs_IOCs[attack]["file"])), 3)
            print("The Subgraph has", covered_percentage, "percent of ",attack," Query Graph IOCs")
    else:
        print("The subgraph doesn't cover any attack IOCs")
    return allNodes_dic , covered_ioc_all_attacks

def parse_name_from_attr(node_attr,node_type):
    if node_attr is None or node_attr in ["","nan"]:
        return None
    if node_type in ["flow", "netflowobject", "net"]:
        if "->" in node_attr:
            node_attr = node_attr.split('->')[0].split(':')[0] + "->" + node_attr.split('->')[-1].split(':')[0]
        if ":" in node_attr:
            node_attr = node_attr.split(':')[0]
        if "," in node_attr:
            node_attr = node_attr.split(',')[1].split("/")[0]
        if " " in node_attr:
            node_attr = node_attr.split(' ')[0]
    else:
        if "/" in node_attr:
            node_attr = node_attr.split('/')[-1]
        if "\\" in node_attr:
            node_attr = node_attr.split('\\')[-1]
    return node_attr
def get_attack_description(subgraph,allNodes_dic,subgraph_id):
    # attributes = get_subgraphs_attributes(args.host)
    subgraphs_df = nx.to_pandas_edgelist(subgraph, edge_key='ekey')
    # out_path = args.root_path + "investigation/" + args.exp_name + "/" + args.model.replace(".model","") +"/run"+str(run)+"_ExpParam_subgraph_df_" + str(subgraph_id) + ".csv"
    # checkpoint_save_csv(subgraphs_df, out_path)
    subgraphs_df = subgraphs_df[["source", "target", "action", "timestamp"]]
    # subgraphs_df['timestamp'] = subgraphs_df['timestamp'].apply(lambda t: t.strftime('%Y-%m-%d %H:%M:%S'))
    # subgraphs_df = subgraphs_df.drop_duplicates()
    attack_description = []
    for i, row in subgraphs_df.iterrows():
        event = {}
        subject_attr = allNodes_dic[row["source"]]
        subject_attr["node_attr"] = parse_name_from_attr(subject_attr["node_attr"],subject_attr['node_type'])
        if subject_attr["node_attr"] is not None:
            event["description"] = "The " + subject_attr['node_type'].lower()+ ": " + subject_attr["node_attr"]
        else:
            event["description"] = "A " + subject_attr['node_type'].lower()
        # if 'ioc' in subject_attr.keys():
        #     event["description"] += "( "+ subject_attr["node_attr"] + " is an IOC for the attack " + str(subject_attr["match_attack"]) + " )"
        event["description"] += " " + row['action'].upper().replace("EVENT_","")
        object_attr = allNodes_dic[row["target"]]
        object_attr["node_attr"] = parse_name_from_attr(object_attr["node_attr"], object_attr['node_type'].lower())
        if object_attr["node_attr"] is not None:
            event["description"] += " the " + object_attr['node_type'].lower() + ": " + object_attr["node_attr"]
        else:
            event["description"] += " a " + object_attr['node_type'].lower()
        # if 'ioc' in object_attr.keys():
        #     event["description"] += "( "+ object_attr["node_attr"] + " is an IOC for the attack " + str(object_attr["match_attack"]) + " )"
        event["timestamp"] = row["timestamp"]
        attack_description.append(event)
    attack_description_df = pd.DataFrame(attack_description).drop_duplicates()
    attack_description_df = attack_description_df.sort_values(by=['timestamp'], ignore_index=True)
    # display(attack_description_df)
    return attack_description_df

def rename_path_with_param(out_path):
    global max_edges
    if args.inv_exp_name:
        exp_param = args.inv_exp_name
    else:
        if args.controlled_construction:
            exp_param = "_controlled_constructed"
        elif args.construct_from_anomaly_subgraph:
            exp_param = "_controlled_in_" + str(args.number_of_hops) + "_hop_anomaly_subgraph"
        elif args.backwards_only:
            exp_param = "_" + str(args.number_of_hops) + "_hop_backward"
        elif args.retraverse_disconnected_nodes:
            exp_param = "_" + str(args.number_of_hops) + "_hop_retraverse_disconnected"
        elif args.anomalies_or_process_only:
            exp_param = "_" + str(args.number_of_hops) + "_hop_anomaliesOrProcess"
        if args.correlate_anomalous_once:
            exp_param += "_correlate_once"
        if args.unvisited_seeds:
            exp_param += "_UnvisitedSeeds"
        if args.remove_duplicated_subgraph:
            exp_param += "_remove_duplication"
        if max_edges == 0:
            exp_param += "_NoSizeLimit"
        else:
            exp_param += "_max_edges"+str(max_edges)
    out_path = out_path.replace("ExpParam", exp_param)
    return out_path

def get_covered_ioc_nodes_df_IOCS(nodes_attrs_df):
    global query_graphs_IOCs, query_graphs_IOCs_set
    covered_ioc = {}
    if args.dataset == "nodlink":
        ## DEBUG: Needs to adjust for Nodlink attributes ##
        for attack, attack_IOC in query_graphs_IOCs.items():
            if len(query_graphs_IOCs_set) == 0: break
            for ip_ioc in attack_IOC["ip"]:
                covered_ioc_tmp = set(nodes_attrs_df[nodes_attrs_df["node_attr"].str.contains(ip_ioc)]["node_attr"])
                if len(covered_ioc_tmp) == 0: continue
                if attack not in covered_ioc:
                    covered_ioc[attack] = covered_ioc_tmp
                else:
                    covered_ioc[attack].update(covered_ioc_tmp)
                del covered_ioc_tmp
            for file_ioc in attack_IOC["file"]:
                covered_ioc_tmp = set(nodes_attrs_df[nodes_attrs_df["node_attr"].str.lower().str.contains(file_ioc.lower())]["node_attr"])
                if len(covered_ioc_tmp) == 0: continue
                if attack not in covered_ioc:
                    covered_ioc[attack] = covered_ioc_tmp
                else:
                    covered_ioc[attack].update(covered_ioc_tmp)
                del covered_ioc_tmp
    else:
        if args.dataset == "optc":
            nodes_attrs_df["node_attr"].apply(lambda x: x.split(" ")[0].split("\\")[-1])
        for attack, attack_IOC in query_graphs_IOCs.items():
            if args.dataset == "optc":
                covered_ioc[attack] = {node_attr for node_attr in
                                   nodes_attrs_df[nodes_attrs_df["node_type"].isin(["flow", "netflowobject","net"])]["node_attr"]
                                   if node_attr.split(" ")[0] in attack_IOC["ip"]}
            else:
                covered_ioc[attack] = {node_attr for node_attr in
                                       nodes_attrs_df[nodes_attrs_df["node_type"].isin(["flow", "netflowobject","net"])][
                                           "node_attr"]
                                       if node_attr in attack_IOC["ip"]}
            if args.dataset == "optc":
                covered_ioc[attack].update({node_attr for node_attr in
                                            nodes_attrs_df[~nodes_attrs_df["node_type"].isin(["flow", "netflowobject","net"])][
                                                "node_attr"]
                                            if node_attr.split("\\")[-1].split(".")[0].lower() in attack_IOC["file"]})
            else:
                covered_ioc[attack].update({node_attr for node_attr in
                                        nodes_attrs_df[~nodes_attrs_df["node_type"].isin(["flow", "netflowobject","net"])]["node_attr"]
                                        if node_attr.lower() in attack_IOC["file"]})
    all_covered_ioc_set = set()
    if len(covered_ioc) > 0:
        for attack, iocs in covered_ioc.items():
            all_covered_ioc_set.update(iocs)
            print("Number of covered IOCS of attack", attack, " is:", len(iocs))
            print("The covered IOCS are:", iocs)
        print("The total number of covered IOCs is", len(all_covered_ioc_set))
    else:
        print("doesn't cover any attack IOCs")
    print("________________________________________________")
    return
def check_covered_IOC_in_all_nodes():
    # print(sparql_queries['get_nodes_attributes'])
    read_sparql = SPARQLWrapper(repository_url)
    read_sparql.setReturnFormat(JSON)
    read_sparql.setQuery(sparql_queries['get_nodes_attributes'])
    nodes_attrs = read_sparql.queryAndConvert()
    nodes_attrs_df = pd.DataFrame(nodes_attrs['results']['bindings'])
    nodes_attrs_df = nodes_attrs_df.map(lambda x: x['value'] if type(x) is dict else x).drop_duplicates(subset=['node_uuid','node_type'])
    nodes_attrs_df.loc[:, 'node_type'] = nodes_attrs_df['node_type'].str.split('/').str[-1].str.lower()
    print("Check Covered IOCs in all dataset nodes")
    get_covered_ioc_nodes_df_IOCS(nodes_attrs_df)

    return nodes_attrs_df

# def trim_memory() -> int:
#     libc = ctypes.CDLL("libc.so.6")
#     return libc.malloc_trim(0)
# def release_memory(client):
#     client.restart()
#     client.run(gc.collect)
#     client.run(trim_memory)

def detect_anomalous_subgraphs(run, seed,nodes_attrs_df):
    # global run_start_time
    run_start_time = time.time()
    seed_everything(seed)

    global full_graph_key2int, subject_nodes, mapping_dict, testing_uuid, edge_index
    # full_graph_key2int, subject_nodes, mapping_dict, testing_uuid, edge_index = get_edges_mapping_of_full_graph()
    global anomalies_nodes
    anomalies_nodes = read_anomalies_nodes(testing_uuid,run)

    global anomalies_nodes_uuid, malicious_nodes_uuid

    # Node_anomaly_TP, Node_anomaly_TN, Node_anomaly_FP, Node_anomaly_FN = initial_evaluation(set(anomalies_nodes_uuid),
    #                                                                                         testing_uuid,
    #                                                                                         set(malicious_nodes_uuid))
    # TP, TN, FP, FN = len(Node_anomaly_TP), len(Node_anomaly_TN), len(Node_anomaly_FP), len(Node_anomaly_FN)
    # print_predict_evaluation_metrics(TP, TN, FP, FN, "Evaluate Node anomaly detection before subgraph construction")
    # if args.consider_related_nodes:
    #     alerts_2hop, anomaly_2hop_results_df = evaluation_with_2hop_neighbours(set(anomalies_nodes_uuid), testing_uuid,
    #                                                                            set(malicious_nodes_uuid),
    #                                                                            edge_index.tolist(), mapping_dict)

    print("Check Covered IOCs in all raised anomalies nodes")
    if args.get_node_attrs:
        get_covered_ioc_nodes_df_IOCS(nodes_attrs_df[nodes_attrs_df["node_uuid"].isin(anomalies_nodes_uuid)])

    # if args.controlled_construction :
    #     subgraphs_lst, disconnected_nodes, subgraphs_stats_df, all_correlated_nodes = construct_subgraphs_by_criteria(k=args.top_k)
    # elif args.construct_from_anomaly_subgraph:
    #     subgraphs_lst, disconnected_nodes, subgraphs_stats_df, all_correlated_nodes, anomaly_results_df_run = construct_subgraphs_by_criteria_from_anomaly_subgraph(number_of_hops=args.number_of_hops, k=args.top_k,k_percent=args.top_k_percent)
    # else:
    #     subgraphs_lst, disconnected_nodes, subgraphs_stats_df = construct_subgraphs_by_query(number_of_hops=args.number_of_hops)
    # subgraphs_lst, disconnected_nodes, subgraphs_stats_df = construct_subgraphs(anomalies_nodes, malicious_nodes_df)
    subgraphs_lst, disconnected_nodes, subgraphs_stats_df, all_correlated_nodes, anomaly_results_df_run = construct_subgraphs_by_criteria_from_anomaly_subgraph(
        run_start_time,number_of_hops=args.number_of_hops, k=args.top_k, k_percent=args.top_k_percent)
    if subgraphs_lst is None:
        return anomaly_results_df_run
    out_path = args.root_path + "results/" + args.exp_name + "/" + args.model.replace(".model","") + "/run"+str(run)+"_ExpParam_constructed_subgraphs_nx.pt"
    checkpoint(subgraphs_lst, out_path)


    if args.get_node_attrs:
        if len(disconnected_nodes) >0:
            print("Check IOCs in disconnected anomalies nodes")
            get_covered_ioc_nodes_df_IOCS(nodes_attrs_df[nodes_attrs_df["node_uuid"].isin(disconnected_nodes)])
        if len(all_correlated_nodes) > 0:
            print("Check IOCs in anomalies nodes correlated into subgraphs")
            get_covered_ioc_nodes_df_IOCS(nodes_attrs_df[nodes_attrs_df["node_uuid"].isin(all_correlated_nodes)])

    global all_covered_ioc, covered_ioc_dic
    all_covered_ioc = set()
    covered_ioc_dic = {}
    if len(subgraphs_lst) > 0:
        if (len(subgraphs_lst) == 1) or (not args.cluster_subgraphs):
            print("No clustering")
            subgraphs_sample = { i:subgraph for i,subgraph in enumerate(subgraphs_lst) }
            investigate_subgraphs(subgraphs_sample)
            subgraphs_stats_df = subgraphs_stats_df.sort_values(by=["subgraph_anomaly_score"], ascending=False)
        else:
            encoded_subgraphs_data_lst = graph_embedding_subgraphs_lst(subgraphs_lst)
            clustered_subgraphs_data_lst = clustering_graphs(encoded_subgraphs_data_lst,subgraphs_stats_df)
            subgraphs_stats_df["cluster_label"] = [graph_data.cluster_label for graph_data in clustered_subgraphs_data_lst]
            out_path = args.root_path + "results/" + args.exp_name + "/" + args.model.replace(".model","") + "/run"+str(run)+"_ExpParam_clustered_subgraphs_data_lst.pt"
            checkpoint(clustered_subgraphs_data_lst, out_path)
            for cluster_i in range(len(subgraphs_stats_df["cluster_label"].unique())):
                subgraphs_idx = subgraphs_stats_df[subgraphs_stats_df["cluster_label"]==cluster_i]["ID"].tolist()
                print("Cluster", cluster_i,"contains subgraphs", subgraphs_idx)
                if args.subgraphs_sample > 0:
                    print("investigating a sample of", args.subgraphs_sample, " subgraphs from cluster", cluster_i)
                    subgraphs_idx = sample(subgraphs_idx, args.subgraphs_sample)
                else:
                    print("investigating all subgraphs from cluster", cluster_i)
                subgraphs_sample ={}
                for subgraph_id in subgraphs_idx:
                    subgraphs_sample[subgraph_id] = subgraphs_lst[subgraph_id]
                investigate_subgraphs(subgraphs_sample,cluster_i)
            subgraphs_stats_df = subgraphs_stats_df.sort_values(by=["subgraph_anomaly_score"], ascending=False)
            clusters_stats_df = subgraphs_stats_df.groupby('cluster_label').agg(
                {'ID': 'count', 'number_of_nodes': ['min', 'mean', 'max'], 'number_of_edges': ['min', 'mean', 'max']}).reset_index()
            clusters_stats_df.columns = ["cluster_label", "number_of_subgraphs", "min_nodes", "mean_nodes", "max_nodes", "min_edges", "mean_edges", "max_edges"]
            clusters_stats_df["n_Benign_subgraphs_outside_attack_time"] = [sum(
                subgraphs_stats_df[subgraphs_stats_df["cluster_label"] == cluster_i][
                    "number_of_actions_in_attack_time"] == 0) for cluster_i in range(
                len(subgraphs_stats_df["cluster_label"].unique()))]
            clusters_stats_df["subgraphs_needs_investigation"] = [
                subgraphs_stats_df[(subgraphs_stats_df["cluster_label"] == cluster_i) & (subgraphs_stats_df["number_of_actions_in_attack_time"] != 0)]["ID"].tolist()
                for cluster_i in range(
                    len(subgraphs_stats_df["cluster_label"].unique()))]
            detected_subgraphs_IDs = []
            for cluster_i in range(len(subgraphs_stats_df["cluster_label"].unique())):
                subgraphs_stats_df_cluster = subgraphs_stats_df[subgraphs_stats_df["cluster_label"] == cluster_i]
                sorted_subgraphs_stats_df_cluster = subgraphs_stats_df_cluster[["ID", "subgraph_anomaly_score"]].sort_values(by="subgraph_anomaly_score", ascending=False)
                detected_subgraphs_IDs.extend(sorted_subgraphs_stats_df_cluster[sorted_subgraphs_stats_df_cluster["subgraph_anomaly_score"] >= sorted_subgraphs_stats_df_cluster[
                        "subgraph_anomaly_score"].quantile(0.95)]["ID"].tolist())
            detected_subgraphs_lsl = [subgraphs_lst[i] for i in detected_subgraphs_IDs]
            print("*******************************************************")
            print("evaluate with anomalous subgraphs that ranked on top 95 percentile of each cluster ")
            print("Number of subgraphs is ", len(detected_subgraphs_lsl))
            all_detected_nodes, detected_subgraphs_anomalies_nodes, disregarded_anomalies_nodes = get_coorelated_and_disregardes_nodes(detected_subgraphs_lsl)
            # re_evaluate_anomaly_detection(all_detected_nodes, detected_subgraphs_anomalies_nodes,disregarded_anomalies_nodes)
            re_evaluate_anomaly_detection_v2(detected_subgraphs_anomalies_nodes)
            out_path = args.root_path + "results/" + args.exp_name + "/" + args.model.replace(".model","") + "/run"+str(run)+"_ExpParam_all_clusters_statistics.csv"
            checkpoint_save_csv(clusters_stats_df, out_path)
        subgraphs_stats_df["covered_ioc"] = None
        for subgraph_id, covered_ioc in covered_ioc_dic.items():
            subgraphs_stats_df.loc[subgraphs_stats_df["ID"]==subgraph_id, "covered_ioc"] = ', '.join(str(x) for x in covered_ioc)
        print("Number of covered IOCs correlated into subgraphs :", len(all_covered_ioc))
        print("Covered IOCs in all constructed subgraphs", all_covered_ioc)
        out_path = args.root_path + "results/" + args.exp_name + "/" + args.model.replace(".model","") +"/run"+str(run)+"_ExpParam_correlated_subgraphs_statistics.csv"
        checkpoint_save_csv(subgraphs_stats_df, out_path)
    anomaly_results_df_run["run"] = str(run)
    return anomaly_results_df_run

if __name__ == '__main__':
    print(args)
    start_time = time.time()
    seed = 360
    anomaly_results_df = pd.DataFrame()
    global query_graphs_IOCs, query_graphs_IOCs_set
    ioc_f = args.root_path + "query_graphs_IOCs.json"
    with open(ioc_f) as f:
        query_graphs_IOCs = json.load(f)
    query_graphs_IOCs_set = set()
    for attack, iocs in query_graphs_IOCs.items():
        query_graphs_IOCs_set.update(iocs["file"])
        query_graphs_IOCs_set.update(iocs["ip"])

    global max_edges
    max_edges = args.max_edges

    if args.get_node_attrs:
        nodes_attrs_df = check_covered_IOC_in_all_nodes()
    global full_graph_key2int, subject_nodes, mapping_dict, testing_uuid, edge_index
    full_graph_key2int, subject_nodes, mapping_dict, testing_uuid, edge_index = get_edges_mapping_of_full_graph()
    for run in range(args.runs):
        print("******************************************")
        print("Run number:", run)
        print("Seed: ", seed)
        anomaly_results_df_run = detect_anomalous_subgraphs(run, seed,nodes_attrs_df)
        anomaly_results_df = pd.concat([anomaly_results_df, anomaly_results_df_run])
        seed = np.random.randint(0, 1000)
        # clear_globals()
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
    out_path = args.root_path + "results/" + args.exp_name + "/" + args.model.replace(".model","") +"/subgraph_anomaly_results_summary.csv"
    ensure_dir(out_path)
    anomaly_results_df.to_csv(out_path, index=None)

    print("Total time: ", time.time() - start_time, "seconds.")
