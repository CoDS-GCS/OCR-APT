import pandas as pd
import numpy as np
import argparse
from resource import *
import datetime
import os
import csv
import pytz
from networkx.readwrite import json_graph
import json
import networkx as nx
import glob
from database_config import get_subgraphs_attributes
import time
import pickle

parser = argparse.ArgumentParser(description='DARPA to RDF')
parser.add_argument('--host', type=str, default='cadets')
parser.add_argument('--source-graph', type=str, default="cadets")
parser.add_argument('--source-graph-nx', type=str, default="complete_cadets_pg")
parser.add_argument('--root-path', type=str, default="../dataset/darpa_tc3/cadets/pygod/")
parser.add_argument('--adjust-uuid', action="store_true", default=False)

args = parser.parse_args()
print(args)

prefix = "https://DARPA_TC3.graph/" + args.host +"/"
is_a = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
def read_json_graph(filename):
    with open(filename) as f:
        js_graph = json.load(f)
    return json_graph.node_link_graph(js_graph)

def map_node_type(attr_nodes_df):
    for node_type in attr_nodes_df["node_type"].unique():
        if node_type == "FLOW":
            attr_nodes_df["node_type"] = "NetFlowObject"
        elif node_type == "PROCESS":
            attr_nodes_df["node_type"] = "SUBJECT_PROCESS"
        elif node_type == "FILE":
            attr_nodes_df["node_type"] = "FILE_OBJECT_FILE"
        elif node_type == "PIPE":
            attr_nodes_df["node_type"] = "UnnamedPipeObject"
    return attr_nodes_df

def get_graph_attributes(file_path,allNodes_lst,out_path,json_graph=True):
    print("getting graph attributes from ",file_path)
    if json_graph:
        provenance_graph = read_json_graph(file_path)
    else:
        with open(file_path, 'rb') as f:
            provenance_graph = pickle.load(f)
    attributes = get_subgraphs_attributes(args.host)
    allNodes_df = pd.DataFrame()
    for node_attr_key in attributes.values():
        if node_attr_key != "NA":
            if args.adjust_uuid:
                attr_nodes_df = pd.DataFrame.from_dict({n: [d['type'], d[node_attr_key]] for n, d in provenance_graph.nodes.items() if node_attr_key in d}, orient='index')
                attr_nodes_df = attr_nodes_df.rename_axis('node').reset_index()
                attr_nodes_df.columns = ["node","node_type","node_attr"]
                attr_nodes_df = attr_nodes_df.dropna(subset=['node','node_type'])
                attr_nodes_df = map_node_type(attr_nodes_df)
                attr_nodes_df["node"] = attr_nodes_df["node"] + "-" + attr_nodes_df["node_type"].str.lower()
                del attr_nodes_df["node_type"]
            else:
                attr_nodes_df = pd.DataFrame(provenance_graph.nodes.data(node_attr_key), columns=["node", "node_attr"]).dropna(subset=['node'])
            attr_nodes_df = attr_nodes_df[attr_nodes_df['node'].isin(allNodes_lst)]
            allNodes_df = pd.concat([allNodes_df, attr_nodes_df])
            del attr_nodes_df
    n_found_attrs = len(allNodes_df)
    print("Number of node attributes",n_found_attrs)
    allNodes_df["node_attr"] = allNodes_df["node_attr"].apply(
        lambda x: x.split("=>")[-1].split("\\")[-1].split("/")[-1])
    turtle_df = allNodes_df.apply( lambda x: "node:" + str(x.node)+ " graph:node-attribute "+ '"' + str(x.node_attr) + '" .', axis =1)
    turtle_df.to_csv(out_path, mode='a', index=None, header=None, sep="\t", quoting=csv.QUOTE_NONE, quotechar="\\",
                     escapechar="\\")
    print("Done writing ", n_found_attrs," node attributes")
    del turtle_df, allNodes_df
    provenance_graph.clear()

    return n_found_attrs


def fix_node_uuid(graph_df):
    graph_df["source-id"] = graph_df["source-id"] + "-" + graph_df["source-type"].str.lower()
    graph_df["destination-id"] = graph_df["destination-id"] + "-" + graph_df["destination-type"].str.lower()
    return graph_df

def get_allNodes_graph(file_path):
    print("loading graph from", file_path)
    headers = ["source-id", "source-type", "destination-id", "destination-type", "edge-type", "timestamp"]
    graph_df = pd.read_csv(file_path, header=None, sep="\t")
    graph_df.columns = headers
    if args.adjust_uuid:
        graph_df = fix_node_uuid(graph_df)
    source_df = graph_df[["source-id", "source-type"]]
    source_df.columns = ["node", "type"]
    dest_df = graph_df[["destination-id", "destination-type"]]
    dest_df.columns = ["node", "type"]
    allNodes_df = pd.concat([source_df, dest_df]).drop_duplicates()
    print("Number of unique nodes is ",len(allNodes_df["node"].unique()))
    return allNodes_df

if __name__ == '__main__':
    start_time = datetime.datetime.now()
    train_path = args.root_path + args.source_graph + "_train.txt"
    allNodes_df_train = get_allNodes_graph(train_path)

    test_path = args.root_path + args.source_graph + "_test.txt"
    allNodes_df_test = get_allNodes_graph(test_path)
    allNodes_df = pd.concat([allNodes_df_train, allNodes_df_test]).drop_duplicates()
    allNodes_lst = allNodes_df["node"].unique().tolist()

    print("total number of source nodes",len(allNodes_lst))

    out_path = args.root_path + args.source_graph + "_attributes_rdfs.ttl"
    ensure_dir(out_path)
    turtle = []
    turtle.append(["@prefix graph: <" + prefix + "> ."])
    turtle.append(["@prefix", "node:", "<" + prefix + "node/> ."])
    turtle.append(["@prefix", "xsd:", "<http://www.w3.org/2001/XMLSchema#> ."])
    turtle.append(["@prefix", "a:", "<" + is_a + "> ."])
    turtle_df = pd.DataFrame(turtle)
    turtle_df.to_csv(out_path, index=None, header=None, sep="\t", quoting=csv.QUOTE_NONE, quotechar="\\",
                   escapechar="\\")
    del turtle,turtle_df

    total_n_found_attrs = 0
    train_path_nx = args.root_path + "provenance_graphs/" + args.source_graph_nx + "_benign_train.json"
    n_found_attrs = get_graph_attributes(train_path_nx,allNodes_lst,out_path)
    if n_found_attrs > 0:
        total_n_found_attrs += n_found_attrs
    test_paths = args.root_path + "provenance_graphs/" + args.source_graph_nx + "*_test.json"
    for test_path_nx in glob.glob(test_paths):
        n_found_attrs = get_graph_attributes(test_path_nx,allNodes_lst,out_path)
        if n_found_attrs > 0:
            total_n_found_attrs += n_found_attrs
    test_paths = args.root_path + "provenance_graphs/" + args.source_graph_nx + "*_test.pt"
    for test_path_nx in glob.glob(test_paths):
        n_found_attrs = get_graph_attributes(test_path_nx, allNodes_lst, out_path, json_graph=False)
        if n_found_attrs > 0:
            total_n_found_attrs += n_found_attrs

    print("Total number of found attributes : ", total_n_found_attrs)
    end_time = datetime.datetime.now()
    print("Total processing time :", end_time - start_time)
    print(getrusage(RUSAGE_SELF))


# __________________ DRAFT _____________________
# if len(allNodes_df) > 0:
#     for node, node_attrs in allNodes_df:
#         attr_type = attributes[node_attrs['type'].lower()]
#         if attr_type != "NA":
#             attr_value = node_attrs[attr_type].split("=>")[-1].split("\\")[-1].split("/")[-1]
#             turtle.append(["node:" + str(node), "graph:node-attribute", '"' + str(attr_value) + '"'])
#             n_found_attrs +=1

# print("Debug time is {}".format(time.time() - debug))

# debug = time.time()
# node_attrs_lst = list(provenance_graph.nodes.data())
# filtered_node_attrs_lst = [(node, node_attrs) for node, node_attrs in node_attrs_lst if node in allNodes_lst]
# print("Debug time is {}".format(time.time() - debug))
# del node_attrs_lst
# n_nodes = len(filtered_node_attrs_lst)
# print("Number of nodes in both dataset = ", n_nodes)
# n_found_attrs = 0
# if len(filtered_node_attrs_lst) > 0:
#     attributes = get_subgraphs_attributes(args.host)
#     for node, node_attrs in filtered_node_attrs_lst:
#         attr_type = attributes[node_attrs['type'].lower()]
#         if attr_type != "NA":
#             attr_value = node_attrs[attr_type].split("=>")[-1].split("\\")[-1].split("/")[-1]
#             turtle.append(["node:" + str(node), "graph:node-attribute", '"' + str(attr_value) + '"'])
#             n_found_attrs +=1
# del filtered_node_attrs_lst

# rdfs_df = pd.DataFrame(turtle, columns=['s', 'p', 'o', 't'])
# del turtle
# rdfs_df.loc[rdfs_df['t'].notna(), 's'] = rdfs_df.loc[rdfs_df['t'].notna(), 's'].apply(
#     lambda x: "<< " + str(x))
# rdfs_df.loc[rdfs_df['t'].notna(), 't'] = rdfs_df.loc[rdfs_df['t'].notna(), 't'].apply(
#     lambda x: " >> " + 'graph:timestamp "' + str(x) + '"')
# rdfs_df["end"] = "."

# print("Total number of nodes in both dataset : ", total_number_of_nodes)


# rdfs_df.to_csv(out_path,mode='a', index=None, header=None, sep="\t", quoting=csv.QUOTE_NONE, quotechar="\\",escapechar="\\")