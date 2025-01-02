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
from statistics import mean

parser = argparse.ArgumentParser(description='DARPA to RDF')
parser.add_argument('--dataset', type=str,required=True)
parser.add_argument('--host', type=str,required=True)
parser.add_argument('--source-graph', type=str,required=True)
parser.add_argument('--root-path', type=str,required=True)
parser.add_argument('--min-node-representation', type=int, default=5)
parser.add_argument('--filter-node-type', action="store_true", default=False)
parser.add_argument('--rdfs', action="store_true", default=False)
parser.add_argument('--graph-nx', action="store_true", default=False)
# parser.add_argument('--graph-optc', action="store_true", default=False)
# parser.add_argument('--graph-nodlink', action="store_true", default=False)
parser.add_argument('--adjust-uuid', action="store_true", default=False)
# parser.add_argument('--encode-attributes', action="store_true", default=False)
from torch_geometric.seed import seed_everything

args = parser.parse_args()
print(args)
assert args.dataset in ['tc3', 'optc', 'nodlink']
assert args.host in ['cadets', 'trace', 'theia', 'fivedirections','SysClient0051','SysClient0501','SysClient0201','SimulatedUbuntu','SimulatedW10','SimulatedWS12']
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return
def delete_file(file_path):
    ###################################Delete Folder if exist #############################
    try:
        os.remove(file_path)
        print("Folder Deleted")
    except OSError as e:
        print("Error Deleting : %s : %s" % (file_path, e.strerror))
    ####################
    return
if args.dataset == "optc":
    prefix = "https://DARPA_OPTC.graph/" + args.host + "/"
elif args.dataset == "nodlink":
    prefix = "https://NODLINK.graph/" + args.host + "/"
else:
    prefix = "https://DARPA_TC3.graph/" + args.host +"/"
is_a = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

def read_json_graph(filename):
    with open(filename) as f:
        js_graph = json.load(f)
    return json_graph.node_link_graph(js_graph)

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

def convert_to_RDF(file_path,isTrain,remove_node_types_lst,testing_allNodes_df=None):
    print(file_path)
    if args.graph_nx:
        headers = ["source-id", "destination-id", "edge-type", "timestamp"]
        provenance_graph = read_json_graph(file_path)
        graph_df = nx.to_pandas_edgelist(provenance_graph, edge_key='ekey')
        graph_df = graph_df[["source", "target", "type", "timestamp"]]
        graph_df.columns = headers
        allNodes_df = pd.DataFrame(provenance_graph.nodes.data("type"), columns=["node", "type"])
        graph_df = pd.merge(graph_df, allNodes_df, left_on="source-id", right_on="node")
        graph_df = graph_df.rename(columns={'type': 'source-type'})
        graph_df = pd.merge(graph_df, allNodes_df, left_on="destination-id", right_on="node")
        graph_df = graph_df.rename(columns={'type': 'destination-type'})
        node_attrs_lst = list(provenance_graph.nodes.data())
        provenance_graph.clear()
    elif args.dataset in ["optc","nodlink"]:
        file_path_csv = file_path.replace(".txt", "_edges.csv")
        graph_df = pd.read_csv(file_path_csv)
    else:
        headers = ["source-id", "source-type", "destination-id", "destination-type", "edge-type", "timestamp"]
        graph_df = pd.read_csv(file_path, header=None, sep="\t")
        graph_df.columns = headers
    if args.adjust_uuid:
        graph_df = fix_node_uuid(graph_df)
    if args.host in ["SimulatedW10", "SimulatedWS12"]:
        graph_df["edge-type"] = graph_df["edge-type"].apply(lambda x:x.split("/")[-1])
    graph_df[~graph_df["edge-type"].isnull()]
    if not args.graph_nx:
        allNodes_df = get_all_nodes_df(graph_df)

    if args.filter_node_type:
        print("Keeping only node types:", keep_node_types_lst)
        graph_df = graph_df[graph_df['source-type'].isin(keep_node_types_lst)]
        graph_df = graph_df[graph_df['destination-type'].isin(keep_node_types_lst)]
        allNodes_df = get_all_nodes_df(graph_df)
    elif args.min_node_representation:
        print("Removing node types:",remove_node_types_lst)
        graph_df = graph_df[~graph_df['source-type'].isin(remove_node_types_lst)]
        graph_df = graph_df[~graph_df['destination-type'].isin(remove_node_types_lst)]
        allNodes_df = get_all_nodes_df(graph_df)

    if isTrain == True:
        # Remove node appeared in testing from training set
        testing_training_nodeUUID_lst = allNodes_df.merge(testing_allNodes_df, on=['node', 'type'], how='left', indicator=True)
        allNodes_df = testing_training_nodeUUID_lst[testing_training_nodeUUID_lst['_merge'] == 'left_only'].drop(columns=['_merge'])

    print("Number of edges ", len(graph_df))
    print("Number of nodes ", len(allNodes_df))
    print("Number of unique nodes ", len(allNodes_df["node"].unique()))
    if args.dataset == "optc":
        date_format = "%Y-%m-%d %H:%M:%S"
        # graph_df["timestamp"] = graph_df["timestamp"].apply(lambda x:x[:19])
        timestamp_lst = graph_df["timestamp"].apply(lambda x: datetime.datetime.strptime(x[:19], date_format)).tolist()
        start_date = min(timestamp_lst)
        end_date = max(graph_df["timestamp"])
    else:
        start_date = min(graph_df["timestamp"])
        start_date = datetime.datetime.fromtimestamp(start_date // 1000000000, tz=pytz.timezone("America/Nipigon"))
        end_date = max(graph_df["timestamp"])
        end_date = datetime.datetime.fromtimestamp(end_date // 1000000000, tz=pytz.timezone("America/Nipigon"))
    print("The graph time range from ", start_date, "to", end_date)
    triples = []
    if args.rdfs:
        turtle = []
        turtle.append(["@prefix", "graph:", "<" + prefix + ">", None])
        turtle.append(["@prefix", "node:", "<" + prefix + "node/>", None])
        turtle.append(["@prefix", "xsd:", "<http://www.w3.org/2001/XMLSchema#>", None])
        turtle.append(["@prefix", "a:", "<" + is_a + ">", None])

    for row in allNodes_df.to_dict('records'):
        triples.append([prefix + "node/" + str(row["node"]), prefix + "node-type", prefix + row["type"]])
        triples.append([prefix + "node/" + str(row["node"]), prefix + "uuid", '"' + str(row["node"]) + '"'])
        if isTrain:
            triples.append([prefix + "node/" + str(row["node"]), prefix + "is_Train", "True"])
        else:
            triples.append([prefix + "node/" + str(row["node"]), prefix + "is_Train", "False"])
        if args.rdfs:
            turtle.append(["node:" + str(row["node"]), "graph:node-type", "graph:" + row["type"]])
            turtle.append(["node:" + str(row["node"]), "graph:uuid", '"' + str(row["node"]) + '"'])
            if isTrain:
                turtle.append(["node:" + str(row["node"]), "graph:is_Train", '"True"'])
            else:
                turtle.append(["node:" + str(row["node"]), "graph:is_Train", '"False"'])
    if args.graph_nx:
        attributes = get_subgraphs_attributes(args.host)
        for node,node_attrs in node_attrs_lst:
            attr_type = attributes[node_attrs['type'].lower()]
            if attr_type != "NA":
                attr_value = node_attrs[attr_type].split("=>")[-1].split("\\")[-1].split("/")[-1]
                triples.append([prefix + "node/" + str(node), prefix + "node-attribute", '"' + str(attr_value) + '"'])
                if args.rdfs:
                    turtle.append(["node:" + str(node), "graph:node-attribute", '"' + str(attr_value) + '"'])
    elif args.dataset in ["optc","nodlink"]:
        file_path_csv = file_path.replace(".txt", "_node_attrs.csv")
        node_attrs_df = pd.read_csv(file_path_csv)
        ############# fix_node_uuid ################
        if args.adjust_uuid:
            node_attrs_df["node"] = node_attrs_df["node"] + "-" + node_attrs_df["node_type"].str.lower()
        #############################################
        for id, row in node_attrs_df.iterrows():
            triples.append([prefix + "node/" + str(row['node']), prefix + "node-attribute", '"' + str(row['node_attr']) + '"'])
            if args.rdfs:
                turtle.append(["node:" + str(row['node']), "graph:node-attribute", '"' + str(row['node_attr']) + '"'])
        del node_attrs_df

    # types_df.loc[len(types_df.index)] = ["entity" , prefix + "is_Train", "Literal"]
    node_types = allNodes_df["type"].unique().tolist()
    print("Number of node types: ",len(node_types))
    print(node_types)
    for node_type in node_types:
        print("number of",node_type,"is:",len(allNodes_df[allNodes_df["type"] == node_type]))
    if args.dataset == "optc":
        file_path = args.root_path + "optc_ground_truth.txt"
        with open(file_path, 'r') as file:
            Is_malicious_lst = list(set(file.read().split()))
    else:
        file_path = args.root_path + args.host + "_ground_truth.txt"
        Is_malicious_df = pd.read_csv(file_path, header=None)
        Is_malicious_lst = Is_malicious_df[0].unique().tolist()
        del Is_malicious_df

    for uuid in allNodes_df["node"].unique().tolist():

        ############# fix_node_uuid ################
        if args.adjust_uuid:
            base_uuid = "-".join(uuid.split("-")[0:-1])
            if base_uuid in Is_malicious_lst:
                triples.append([prefix + "node/" + str(uuid), prefix + "is_malicious", "True"])
            else:
                triples.append([prefix + "node/" + str(uuid), prefix + "is_malicious", "False"])
            if args.rdfs:
                if base_uuid in Is_malicious_lst:
                    turtle.append(["node:" + str(uuid), "graph:is_malicious", '"True"'])
                else:
                    turtle.append(["node:" + str(uuid), "graph:is_malicious", '"False"'])
            #############################################
        else:
            if uuid in Is_malicious_lst:
                triples.append([prefix + "node/" + str(uuid), prefix + "is_malicious", "True"])
            else:
                triples.append([prefix + "node/" + str(uuid), prefix + "is_malicious", "False"])
            if args.rdfs:
                if uuid in Is_malicious_lst:
                    turtle.append(["node:" + str(uuid), "graph:is_malicious", '"True"'])
                else:
                    turtle.append(["node:" + str(uuid), "graph:is_malicious", '"False"'])


    del allNodes_df, Is_malicious_lst
    edge_types = graph_df["edge-type"].unique().tolist()
    print("Number of edge types", len(edge_types))
    print(edge_types)
    for node_type in node_types:
        triples.append([node_type, "a", prefix + 'node-type'])
        if args.rdfs:
            turtle.append(["graph:" + node_type, "a", 'graph:node-type'])
    for elem in edge_types:
        triples.append([elem, "a", prefix + 'edge-type'])
        if args.rdfs:
            turtle.append(["graph:" + elem, "a", 'graph:edge-type'])

    dict_columns = {elem: idx for idx, elem in enumerate(graph_df.columns.tolist())}
    for row in graph_df.itertuples(index=False):
        triples.append(
            [prefix + "node/" + str(row[dict_columns['source-id']]), prefix + row[dict_columns["edge-type"]],
             prefix + "node/" + str(row[dict_columns['destination-id']])])
        if args.rdfs:
            turtle.append(
                ["node:" + str(row[dict_columns['source-id']]), "graph:" + row[dict_columns["edge-type"]],
                 "node:" + str(row[dict_columns['destination-id']]), str(row[dict_columns["timestamp"]])])

    triples_df = pd.DataFrame(triples, columns=['s', 'p', 'o'])
    del triples

    print("Number of triples after converting", len(triples_df))
    print("Genrating RDF nt file")
    rdf_df = triples_df[["s", "p", "o"]]
    rdf_df["s"] = rdf_df["s"].apply(lambda x: "<" + str(x) + ">")
    rdf_df["p"] = rdf_df["p"].apply(lambda x: "<" + str(x) + ">")
    rdf_df["o"] = rdf_df["o"].apply(
        lambda x: "<" + str(x) + ">" if str(x).startswith("http") else ('"' + str(x) + '"'))
    rdf_df["end"] = "."
    rdfs_df =None
    if args.rdfs:
        rdfs_df = pd.DataFrame(turtle, columns=['s', 'p', 'o','t'])
        del turtle
        rdfs_df.loc[rdfs_df['t'].notna(), 's'] = rdfs_df.loc[rdfs_df['t'].notna(), 's'].apply(
            lambda x: "<< " + str(x))
        rdfs_df.loc[rdfs_df['t'].notna(), 't'] = rdfs_df.loc[rdfs_df['t'].notna(), 't'].apply(
            lambda x: " >> " + 'graph:timestamp "' + str(x) + '"')
        rdfs_df["end"] = "."


    return triples_df , rdf_df,rdfs_df , graph_df

def get_all_nodes_df(graph_df):
    source_df = graph_df[["source-id", "source-type"]].drop_duplicates()
    source_df.columns = ["node", "type"]
    dest_df = graph_df[["destination-id", "destination-type"]].drop_duplicates()
    dest_df.columns = ["node", "type"]
    allNodes_df = pd.concat([source_df, dest_df]).drop_duplicates()
    del source_df, dest_df
    return allNodes_df

def fix_node_uuid(graph_df):
    # graph_df["source-id"] = graph_df[['source-id', 'source-type']].agg('-'.join, axis=1)
    graph_df["source-id"] = graph_df["source-id"] + "-" + graph_df["source-type"].str.lower()
    graph_df["destination-id"] = graph_df["destination-id"] + "-" + graph_df["destination-type"].str.lower()
    return graph_df
def get_remove_nodeType_lst(train_path,test_path):
    remove_node_types_lst = ["USER_SESSION"]
    if args.dataset in ["optc","nodlink"]:
        file_path_csv = train_path.replace(".txt", "_edges.csv")
        train_graph_df = pd.read_csv(file_path_csv)
        file_path_csv = test_path.replace(".txt", "_edges.csv")
        test_graph_df = pd.read_csv(file_path_csv)
    else:
        headers = ["source-id", "source-type", "destination-id", "destination-type", "edge-type", "timestamp"]
        train_graph_df = pd.read_csv(train_path, header=None, sep="\t")
        train_graph_df.columns = headers
        test_graph_df = pd.read_csv(test_path, header=None, sep="\t")
        test_graph_df.columns = headers
    if args.adjust_uuid:
        train_graph_df = fix_node_uuid(train_graph_df)
    if args.adjust_uuid:
        test_graph_df = fix_node_uuid(test_graph_df)
    train_allNodes_df = get_all_nodes_df(train_graph_df)
    node_types_cnt = train_allNodes_df["type"].value_counts().to_frame().reset_index()
    remove_node_types_lst.extend(node_types_cnt[node_types_cnt["count"] <= args.min_node_representation]["type"].tolist())
    test_allNodes_df = get_all_nodes_df(test_graph_df)
    node_types_cnt = test_allNodes_df["type"].value_counts().to_frame().reset_index()
    remove_node_types_lst.extend(node_types_cnt[node_types_cnt["count"] <= args.min_node_representation]["type"].tolist())
    remove_node_types_lst = list(set(remove_node_types_lst))
    del test_graph_df, train_graph_df, node_types_cnt,train_allNodes_df,test_allNodes_df
    print("Removing node types:", remove_node_types_lst)

    return remove_node_types_lst


def explore_graph_stats(graph_df):
    print("----------------------------------------------")
    print("Analyzing the graph")
    print("Number of edges types ", len(graph_df["edge-type"].unique()))
    node_types = set(graph_df["source-type"].unique())
    node_types.update(set(graph_df["destination-type"].unique()))
    print("Number of nodes types ", len(node_types))
    allNodes_df = get_all_nodes_df(graph_df)
    graph_nx = nx.from_pandas_edgelist(
        graph_df,
        source="source-id",
        target="destination-id",
        edge_attr=["edge-type", "timestamp"],
        create_using=nx.MultiDiGraph()
    )
    allNodes_type = allNodes_df.set_index('node').to_dict('index')
    nx.set_node_attributes(graph_nx, allNodes_type)
    del allNodes_df, allNodes_type, graph_df
    print("Total number of nodes ", graph_nx.number_of_nodes())
    print("Total number of edges ", graph_nx.number_of_edges())
    graph_density = nx.density(graph_nx)
    print("The density of the graph is", graph_density)
    print("The density of the graph is {:.6f}".format(graph_density))
    print("Number of self loops edges", nx.number_of_selfloops(graph_nx))
    degree_sequence = [d for n, d in graph_nx.degree()]
    print(f"Minimum degree: {min(degree_sequence)}")
    print(f"Maximum degree: {max(degree_sequence)}")
    print(f"Average degree: {mean(degree_sequence):.6f}")
    print("----------------------------------------------")
    del graph_nx
    # degree_histogram = nx.degree_histogram(graph_nx)
    # print(degree_histogram)
    return

if __name__ == '__main__':
    seed = 360
    seed_everything(seed)
    start_time = datetime.datetime.now()
    print(getrusage(RUSAGE_SELF))
    remove_node_types_lst = []
    if args.graph_nx:
        train_path = args.root_path + "provenance_graphs/" + args.source_graph + "_benign_train.json"
        if args.min_node_representation:
            ############ To be implemented here
            # remove_node_types_lst = get_remove_nodeType_lst(train_path, test_path)
            print("To be implemented, filter node type with very few node representation")
        triples_df_train, rdf_df_train, rdfs_df_train, train_graph_df = convert_to_RDF(
            train_path, True, remove_node_types_lst)
        ##### Filter testing nodes from training set #####

        ########## To be fixed here ##########

        test_paths = args.root_path + "provenance_graphs/" + args.source_graph + "*_test.json"
        first_graph = True
        for test_path in glob.glob(test_paths):
            if first_graph:
                triples_df_test, rdf_df_test, rdfs_df_test, _, test_graph_df = convert_to_RDF(test_path, False,
                                                                                              remove_node_types_lst)
                first_graph = False
            else:
                triples_df_test_tmp, rdf_df_test_tmp, rdfs_df_test_tmp, _, test_graph_df_tmp = convert_to_RDF(test_path,
                                                                                                              False,
                                                                                                              remove_node_types_lst)
                triples_df_test = pd.concat([triples_df_test, triples_df_test_tmp])
                rdf_df_test = pd.concat([rdf_df_test, rdf_df_test_tmp])
                rdfs_df_test = pd.concat([rdfs_df_test, rdfs_df_test_tmp])
                test_graph_df = pd.concat([test_graph_df, test_graph_df_tmp])
                del triples_df_test_tmp, rdf_df_test_tmp, rdfs_df_test_tmp, test_graph_df_tmp
    else:
        train_path = args.root_path + args.source_graph + "_train.txt"
        test_path = args.root_path + args.source_graph + "_test.txt"
        if args.filter_node_type:
            keep_node_types_lst = ['FLOW', 'PROCESS', 'MODULE', 'FILE']
        elif args.min_node_representation:
            remove_node_types_lst = get_remove_nodeType_lst(train_path,test_path)


        triples_df_test, rdf_df_test, rdfs_df_test, test_graph_df = convert_to_RDF(test_path,False,
                                                                                                          remove_node_types_lst)
        testing_allNodes_df = get_all_nodes_df(test_graph_df)
        # all_testing_nodeUUID_lst = test_allNodes_df["node"].unique().tolist()
        # del test_allNodes_df

        triples_df_train, rdf_df_train, rdfs_df_train, train_graph_df = convert_to_RDF(train_path,True,
            remove_node_types_lst,testing_allNodes_df)


    print("Number of training triples",len(triples_df_train))
    print("Number of testing triples", len(triples_df_test))

    # concat train & test
    # types_df = pd.concat([types_df, types_df_test]).drop_duplicates()
    triples_df = pd.concat([triples_df_train, triples_df_test])
    rdf_df = pd.concat([rdf_df_train, rdf_df_test])
    if args.rdfs:
        rdfs_df = pd.concat([rdfs_df_train, rdfs_df_test])
    print("Total number of triples", len(triples_df))
    print("Number of unique nodes ", len(triples_df[triples_df["p"] == prefix + "node-type"]["s"].unique()))
    graph_df = pd.concat([train_graph_df, test_graph_df])
    explore_graph_stats(graph_df)
    out_path = args.root_path + args.source_graph + "_graph_df.csv"
    ensure_dir(out_path)
    delete_file(out_path)
    graph_df.to_csv(out_path, index=None, sep="\t")

    del triples_df_test, rdf_df_test, triples_df_train, rdf_df_train
    #save to csv
    print("Saving the graph to tsv")
    out_path = args.root_path + args.source_graph + ".tsv"
    ensure_dir(out_path)
    delete_file(out_path)
    triples_df.to_csv(out_path, index=None, sep="\t")

    out_path = args.root_path + args.source_graph + "_rdf.nt"
    ensure_dir(out_path)
    delete_file(out_path)
    rdf_df.to_csv(out_path, index=None, header=None, sep="\t", quoting=csv.QUOTE_NONE, quotechar="\\",
                         escapechar="\\")
    if args.rdfs:
        out_path = args.root_path + args.source_graph + "_rdfs.ttl"
        ensure_dir(out_path)
        delete_file(out_path)
        rdfs_df.to_csv(out_path, index=None, header=None, sep="\t", quoting=csv.QUOTE_NONE, quotechar="\\",
                      escapechar="\\")
    end_time = datetime.datetime.now()
    print("Total processing time :", end_time - start_time)
    print(getrusage(RUSAGE_SELF))
