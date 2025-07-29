import pandas as pd
import gzip
from datetime import datetime
import os
import shutil
import argparse
from torch_geometric.seed import seed_everything
from statistics import mean
import time
from sklearn.preprocessing import MinMaxScaler, normalize  # to standardize the features

from database_config import order_x_features

pd.set_option('display.max_columns', 100)

parser = argparse.ArgumentParser(description='RDF to PYG')
parser.add_argument('--dataset', type=str,required=True)
parser.add_argument('--host', type=str,required=True)
parser.add_argument('--root-path', type=str,required=True)
parser.add_argument('--exp-name', type=str,required=True)
parser.add_argument('--source-graph', type=str,required=True)
parser.add_argument('--get-timestamps-features', action="store_true", default=False)
parser.add_argument('--timestamps-in-minutes', action="store_true", default=False)
parser.add_argument('--get-idle-time', action="store_true", default=False)
parser.add_argument('--get-cumulative-active-time', action="store_true", default=False)
parser.add_argument('--get-lifespan', action="store_true", default=False)
parser.add_argument('--normalize-features', action="store_true", default=False)
parser.add_argument('--fill-with-mean', action="store_true", default=False)
parser.add_argument('--training-valid', action="store_true", default=False)
from sklearn.model_selection import train_test_split
import torch
torch.use_deterministic_algorithms(True)

# init_ru_maxrss = getrusage(RUSAGE_SELF).ru_maxrss
args = parser.parse_args()
print(args)
assert args.dataset in ['tc3', 'optc', 'nodlink']
assert args.host in ['cadets', 'trace', 'theia', 'SysClient0051','SysClient0501','SysClient0201','SimulatedUbuntu','SimulatedW10','SimulatedWS12']
def compress_gz(f_path):
    f_in = open(f_path, 'rb')
    f_out = gzip.open(f_path + ".gz", 'wb')
    f_out.writelines(f_in)
    f_out.close()
    f_in.close()

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
def delete_multiple_element(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)

def delete_folder(dir_path):
    ######### Delete Folder if exist
    try:
        shutil.rmtree(dir_path)
        print("Folder Deleted")
    except OSError as e:
        print("Deleting : %s : %s" % (dir_path, e.strerror))
    ####################
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

def splitbyStratifiedNodeTypes(g_tsv_df,this_type_node_idx,entites_dic,labels_rel_df,label_node):
    print("Get sample for", label_node, "node")
    print("Splitting train / valid / test.")
    split_df = g_tsv_df[g_tsv_df["p"] == split_rel]
    split_df = split_df[split_df["s"].isin(this_type_node_idx)].drop_duplicates()
    # print(split_df.head())

    split_df["s"] = split_df["s"].apply(lambda x: str(x).split("/")[-1]).astype(
        "str").apply(lambda x: entites_dic[label_node+"_dic"][str(x)] if x in entites_dic[
        label_node+"_dic"] else -1)


    if args.training_valid:
        print("Getting valid from training set")
        training_df = split_df[split_df["o"] == "True"]["s"]
        test_df = split_df[split_df["o"] == "False"]["s"]
        labels_rel_df.columns = ["s", "o_idx"]
        train_valid_df = pd.merge(training_df, labels_rel_df, on="s")
        train_df, valid_df = train_test_split(train_valid_df, test_size=0.15, random_state=1)
        train_df_malicious = train_df[train_df["o_idx"] == 1]
        valid_df = pd.concat([valid_df,train_df_malicious])
        train_df = train_df[train_df["o_idx"] == 0]
        del training_df, labels_rel_df
        del train_df['o_idx'], valid_df['o_idx']
    else:
        print("Getting valid from evaluation set")
        train_df = split_df[split_df["o"] == "True"]["s"]
        eval_df = split_df[split_df["o"] == "False"]["s"]
        print("Number of evaluation samples", len(eval_df))
        # #ensure only normal nodes in the training/validating set
        labels_rel_df.columns = ["s", "o_idx"]
        eval_df = pd.merge(eval_df, labels_rel_df, on="s")
        if len(eval_df["o_idx"].unique()) == 1:
            test_df, valid_df = train_test_split(eval_df, test_size=0.15, random_state=1)
        elif min(eval_df["o_idx"].value_counts()) <= 1:
            test_df, valid_df = train_test_split(eval_df, test_size=0.15, random_state=1)
        else:
            test_df, valid_df = train_test_split(eval_df, test_size=0.15, random_state=1,
                                                 stratify=eval_df["o_idx"])
        del test_df['o_idx'], valid_df['o_idx']

    print("Number of training samples", len(train_df))
    print("Number of validating samples",len(valid_df))
    print("Number of testing samples", len(test_df))
    return train_df,valid_df,test_df


def feature_engineering(graph_df,edge_types):
    all_nodes_uuid = set(graph_df["source-id"].unique().tolist() + graph_df["destination-id"].unique().tolist())
    sorted_edge_types = order_x_features(args.host,edge_types)
    edge_types_dic = {edge:0 for edge in sorted_edge_types}
    x_list = {node: edge_types_dic.copy() for node in all_nodes_uuid}
    if args.dataset == "optc":
        date_format = "%Y-%m-%d %H:%M:%S.%f"
    dict_columns = {elem: idx for idx, elem in enumerate(graph_df.columns.tolist())}
    for row in graph_df.itertuples(index=False):
        src_uuid = str(row[dict_columns['source-id']])
        dst_uuid = str(row[dict_columns['destination-id']])
        edge_type = str(row[dict_columns['edge-type']]).replace("EVENT_", "").lower()
        if args.dataset == "optc":
            edge_time = datetime.strptime(row[dict_columns['timestamp']], date_format)
        else:
            edge_time = row[dict_columns['timestamp']]
        x_list[src_uuid][str("out_" + edge_type)] += 1
        x_list[dst_uuid][str("in_" + edge_type)] += 1
        if args.get_timestamps_features:
            if "event_timestamp_lst" in x_list[src_uuid].keys():
                x_list[src_uuid]["event_timestamp_lst"].append(edge_time)
            else:
                x_list[src_uuid]["event_timestamp_lst"] = [edge_time]
            if "event_timestamp_lst" in x_list[dst_uuid].keys():
                x_list[dst_uuid]["event_timestamp_lst"].append(edge_time)
            else:
                x_list[dst_uuid]["event_timestamp_lst"] = [edge_time]
    if args.get_timestamps_features:
        second_threshold = 1.00
        for node_uuid in x_list.keys():
            if len(x_list[node_uuid]["event_timestamp_lst"]) > 1:
                x_list[node_uuid]["event_timestamp_lst"].sort()
                event_timestamps = pd.Series(x_list[node_uuid]["event_timestamp_lst"])
                gaps_durations = event_timestamps.diff().dropna()
                if args.dataset == "optc":
                    gaps_durations_sec = gaps_durations.dt.total_seconds()
                elif args.host in ["SimulatedW10", "SimulatedWS12"]:
                    gaps_durations_sec = gaps_durations / 1000
                else:
                    gaps_durations_sec = gaps_durations / 1000000000
                if args.get_cumulative_active_time:
                    active_durations = gaps_durations_sec[gaps_durations_sec < second_threshold]
                    idle_durations = gaps_durations_sec[gaps_durations_sec >= second_threshold]
                else:
                    idle_durations = gaps_durations_sec
                if args.get_idle_time:
                    if len(idle_durations) > 0:
                        x_list[node_uuid]["avg_idle_time"] = int(round(idle_durations.mean()))
                        x_list[node_uuid]["max_idle_time"] = int(round(idle_durations.max()))
                        x_list[node_uuid]["min_idle_time"] = int(round(idle_durations.min()))
                    else:
                        x_list[node_uuid]["avg_idle_time"] = 0
                        x_list[node_uuid]["max_idle_time"] = 0
                        x_list[node_uuid]["min_idle_time"] = 0
                if args.get_cumulative_active_time:
                    if len(active_durations) > 0:
                        x_list[node_uuid]["cumulative_active_time"] = int(round(active_durations.sum()))
                    else:
                        x_list[node_uuid]["cumulative_active_time"] = 0
                if args.get_lifespan:
                    x_list[node_uuid]["lifespan"] = int(round(gaps_durations_sec.sum()))
            else:
                if args.get_idle_time:
                    x_list[node_uuid]["avg_idle_time"] = 0
                    x_list[node_uuid]["max_idle_time"] = 0
                    x_list[node_uuid]["min_idle_time"] = 0
                if args.get_cumulative_active_time:
                    x_list[node_uuid]["cumulative_active_time"] = 0
                if args.get_lifespan:
                    x_list[node_uuid]["lifespan"] = 0
            del x_list[node_uuid]['event_timestamp_lst']
    x_list_df = pd.DataFrame.from_dict(x_list, orient='index')
    # # removes all columns in x_list_df that are entirely zeros. Not necessary to keep
    x_list_df = x_list_df.loc[:, (x_list_df != 0).any(axis=0)]
    del x_list

    if args.timestamps_in_minutes:
        timeFeatures_to_scale = []
        if args.get_idle_time:
            timeFeatures_to_scale.extend(["max_idle_time", "min_idle_time", "avg_idle_time"])
        if args.get_cumulative_active_time:
            timeFeatures_to_scale.append("cumulative_active_time")
        if args.get_lifespan:
            timeFeatures_to_scale.append("lifespan")
        if len(timeFeatures_to_scale) > 0:
            x_list_df[timeFeatures_to_scale] = x_list_df[timeFeatures_to_scale] / 60
    x_list_df = x_list_df.reset_index()
    x_list_df.rename(columns={'index': 'node_uuid'}, inplace=True)
    if args.fill_with_mean:
        print("Fill nan with mean")
        mean_features = x_list_df.loc[:, x_list_df.columns != 'node_uuid'].mean()
        x_list_df = x_list_df.fillna(mean_features)
    else:
        print("Fill nan with 0")
        x_list_df = x_list_df.fillna(0)
    if args.normalize_features:
        sorted_edge_types = [action for action in sorted_edge_types if action in x_list_df.columns]
        timestamp_features = [column for column in x_list_df.columns if (column not in sorted_edge_types) and (column !="node_uuid")]

        normalized_x_list_df = x_list_df[["node_uuid"]]

        # Normalize L2 Unit vector normalization (L2) for actions , and MinMax
        normalized_data = normalize(x_list_df[sorted_edge_types], norm='l2', axis=1)
        normalized_x_list_df[sorted_edge_types] = pd.DataFrame(normalized_data, columns=sorted_edge_types)
        if args.get_idle_time or args.get_cumulative_active_time or args.get_lifespan:
            scaler = MinMaxScaler()
            normalized_x_list_df[timestamp_features] = pd.DataFrame(scaler.fit_transform(x_list_df[timestamp_features]), columns=timestamp_features)
        del x_list_df
        x_list_df = normalized_x_list_df


    print("Total Number of features:", len(x_list_df.columns) - 1)
    print("Extracted features:", x_list_df.columns)

    return x_list_df



if __name__ == '__main__':
    seed = 360
    seed_everything(seed)
    start_convert = time.time()
    rdf_graph_name = args.source_graph
    dataset_name = args.exp_name

    dir_path = args.root_path + args.exp_name
    delete_folder(dir_path)
    zip_path = dir_path +".zip"
    delete_file(zip_path)

    split_rel = prefix + "is_Train"
    split_by = {"folder_name": "node_type", "split_data_type": "int", "train":4  ,
                "valid":5 , "test":6 }
    target_rel = prefix + "is_malicious"
    node_type_rel = prefix + "node-type"
    dic_results = {}
    Literals2Nodes = True
    output_root_path = args.root_path
    g_tsv_df = pd.read_csv(output_root_path + rdf_graph_name + ".tsv",encoding_errors='ignore',sep="\t")
    graph_df_path = output_root_path + args.source_graph + "_graph_df.csv"
    graph_df = pd.read_csv(graph_df_path, sep="\t")
    print("original_g_csv_df loaded , records length=", len(g_tsv_df))
    try:
        g_tsv_df = g_tsv_df.rename(columns={"Subject": "s", "Predicate": "p", "Object": "o"})
        g_tsv_df = g_tsv_df.rename(columns={0: "s", 1: "p", 2: "o"})
        label_nodes = g_tsv_df[g_tsv_df["p"]==prefix + "node-type"]["o"].unique()
        edge_types = g_tsv_df[g_tsv_df["o"]==prefix + "edge-type"]["s"].unique()
        g_tsv_df = g_tsv_df.dropna()
        print("len of g_tsv_df after dropna  ", len(g_tsv_df))
    except:
        print("g_tsv_df columns=", g_tsv_df.columns())
    dic_results[dataset_name] = {}
    dic_results[dataset_name]["usecase"] = dataset_name
    dic_results[dataset_name]["TriplesCount"] = len(g_tsv_df)
    relations_lst = edge_types.astype("str").tolist()
    print("relations_lst=", relations_lst)
    relations_df = pd.DataFrame(relations_lst, columns=["rel name"])
    relations_df["rel name"] = relations_df["rel name"].apply(lambda x: str(x).split("/")[-1])
    relations_df["rel idx"] = relations_df.index
    relations_df = relations_df[["rel idx", "rel name"]]
    map_folder = output_root_path + dataset_name + "/mapping"
    try:
        os.stat(map_folder)
    except:
        os.makedirs(map_folder)
    relations_df.to_csv(map_folder + "/relidx2relname.csv", index=None)
    compress_gz(map_folder + "/relidx2relname.csv")
    label_idx_df = pd.DataFrame({"label idx": [0, 1], "label name": [False, True]})
    dic_results[dataset_name]["ClassesCount"] = len(label_idx_df)
    label_idx_df.to_csv(map_folder + "/labelidx2labelname.csv", index=None)
    compress_gz(map_folder + "/labelidx2labelname.csv")
    ######## prepare relations mapping
    relations_entites_map = {}

    entites_dic = {}
    for node_type in label_nodes:
        entites_dic[node_type.split("/")[-1]] = set(g_tsv_df[g_tsv_df["o"]==node_type]["s"].apply(
                        lambda x: str(x).split("/")[-1]).unique())

    node_types_df = g_tsv_df[g_tsv_df["p"] == prefix + "node-type"][["s", "o"]].drop_duplicates()
    node_types_df = node_types_df.rename(columns={"s":"node","o": "type"})

    ######### Make sure all rec papers have target
    target_subjects_lst = g_tsv_df[g_tsv_df["p"] == target_rel]["s"].apply(
        lambda x: str(x).split("/")[-1]).unique().tolist()
    print("len of target_subjects_lst=", len(target_subjects_lst))
    # target_subjects_dic= {k: entites_dic['rec'][k] for k in target_subjects_lst}
    total_num_nodes = 0
    for label_node in label_nodes:
        label_node = label_node.split('/')[-1]
        entites_dic[label_node] = set.intersection(entites_dic[label_node], set(target_subjects_lst))
        print("len of entites_dic["+label_node+"]=", len(entites_dic[label_node]))
        total_num_nodes += len(entites_dic[label_node])
    print("Total encoded nodes is: ", total_num_nodes)
    ############# write entites index
    for key in list(entites_dic.keys()):
        entites_dic[key] = pd.DataFrame(list(entites_dic[key]), columns=['ent name']).astype(
            'str').sort_values(by="ent name").reset_index(drop=True)
        entites_dic[key] = entites_dic[key].drop_duplicates()
        entites_dic[key]["ent idx"] = entites_dic[key].index
        entites_dic[key] = entites_dic[key][["ent idx", "ent name"]]
        entites_dic[key + "_dic"] = pd.Series(entites_dic[key]["ent idx"].values,
                                              index=entites_dic[key]["ent name"]).to_dict()
        map_folder = output_root_path + dataset_name + "/mapping"
        try:
            os.stat(map_folder)
        except:
            os.makedirs(map_folder)
        entites_dic[key].to_csv(map_folder + "/" + key + "_entidx2name.csv", index=None)
        compress_gz(map_folder + "/" + key + "_entidx2name.csv")
    ########### write nodes statistics
    lst_node_has_feat = [
        list(
            filter(lambda entity: str(entity).endswith("_dic") == False, list(entites_dic.keys())))]
    lst_node_has_label = lst_node_has_feat.copy()
    lst_num_node_dict = lst_node_has_feat.copy()
    lst_has_feat = []
    lst_has_label = []
    lst_num_node = []
    label_nodes_names = [label_node.split('/')[-1] for label_node in label_nodes]
    for entity in lst_node_has_feat[0]:
        if str(entity) in label_nodes_names:
            lst_has_label.append("True")
            lst_has_feat.append("True")
        else:
            lst_has_label.append("False")
            lst_has_feat.append("False")

        lst_num_node.append(len(entites_dic[entity + "_dic"]))

    lst_node_has_feat.append(lst_has_feat)
    lst_node_has_label.append(lst_has_label)
    lst_num_node_dict.append(lst_num_node)

    map_folder = output_root_path + dataset_name + "/raw"
    print("map_folder=", map_folder)
    try:
        os.stat(map_folder)
    except:
        os.makedirs(map_folder)

    pd.DataFrame(lst_node_has_feat).to_csv(
        output_root_path + dataset_name + "/raw/nodetype-has-feat.csv", header=None,
        index=None)
    compress_gz(output_root_path + dataset_name + "/raw/nodetype-has-feat.csv")

    pd.DataFrame(lst_node_has_label).to_csv(
        output_root_path + dataset_name + "/raw/nodetype-has-label.csv",
        header=None, index=None)
    compress_gz(output_root_path + dataset_name + "/raw/nodetype-has-label.csv")

    pd.DataFrame(lst_num_node_dict).to_csv(
        output_root_path + dataset_name + "/raw/num-node-dict.csv", header=None,
        index=None)
    compress_gz(output_root_path + dataset_name + "/raw/num-node-dict.csv")

    ############# create label relation index
    label_idx_df["label idx"] = label_idx_df["label idx"].astype("int64")
    label_idx_df["label name"] = label_idx_df["label name"].apply(lambda x: str(x).split("/")[-1])
    label_idx_dic = pd.Series(label_idx_df["label idx"].values, index=label_idx_df["label name"]).to_dict()
    ######### drop multiple targets per subject keep first
    labels_rel_df = g_tsv_df[g_tsv_df["p"] == target_rel].reset_index(drop=True)
    labels_rel_df = labels_rel_df.sort_values(['s', 'o'], ascending=[True, True])
    labels_rel_df = labels_rel_df.drop_duplicates(subset=["s"], keep='first')
    cnt_train, cnt_valid, cnt_test = 0, 0, 0
    print("entites_dic=", list(entites_dic.keys()))
    lst_node_has_split = []
    for label_node in label_nodes:
        this_type_node_idx = node_types_df[node_types_df["type"] == label_node]["node"]
        label_node = label_node.split('/')[-1]
        labels_rel_df_temp = labels_rel_df[labels_rel_df["s"].isin(this_type_node_idx)]
        labels_rel_df_temp["s_idx"] = labels_rel_df_temp["s"].apply(
            lambda x: str(x).split("/")[-1])
        labels_rel_df_temp["s_idx"] = labels_rel_df_temp["s_idx"].astype("str")
        labels_rel_df_temp["s_idx"] = labels_rel_df_temp["s_idx"].apply(
            lambda x: entites_dic[label_node + "_dic"][x] if x in entites_dic[
                label_node + "_dic"].keys() else -1)
        labels_rel_df_notfound = labels_rel_df_temp[labels_rel_df_temp["s_idx"] == -1]
        labels_rel_df_temp = labels_rel_df_temp[labels_rel_df_temp["s_idx"] != -1]
        labels_rel_df_temp = labels_rel_df_temp.sort_values(by=["s_idx"]).reset_index(drop=True)

        labels_rel_df_temp["o_idx"] = labels_rel_df_temp["o"].apply(lambda x: str(x).split("/")[-1])
        labels_rel_df_temp["o_idx"] = labels_rel_df_temp["o_idx"].apply(
            lambda x: label_idx_dic[str(x)] if str(x) in label_idx_dic.keys() else -1)
        out_labels_df = labels_rel_df_temp[["o_idx"]]
        map_folder = output_root_path + dataset_name + "/raw/node-label/" + label_node
        try:
            os.stat(map_folder)
        except:
            os.makedirs(map_folder)
        out_labels_df.to_csv(map_folder + "/node-label.csv", header=None, index=None)
        compress_gz(map_folder + "/node-label.csv")


    ################# split parts (train/test/validate)
        labels_rel_df_temp = labels_rel_df_temp[["s_idx","o_idx"]]
        print("Split by", split_rel)
        train_df, valid_df, test_df = splitbyStratifiedNodeTypes(g_tsv_df,this_type_node_idx,entites_dic,labels_rel_df_temp,label_node)
        del labels_rel_df_temp
        if train_df is not None:
            cnt_train += len(train_df)
            cnt_valid += len(valid_df)
            cnt_test += len(test_df)
            map_folder = output_root_path + dataset_name + "/split/" + split_by[
            "folder_name"] + "/" + label_node
            try:
                os.stat(map_folder)
            except:
                os.makedirs(map_folder)
            train_df.to_csv(map_folder + "/train.csv", index=None, header=None)
            compress_gz(map_folder + "/train.csv")
            valid_df.to_csv(map_folder + "/valid.csv", index=None, header=None)
            compress_gz(map_folder + "/valid.csv")
            test_df.to_csv(map_folder + "/test.csv", index=None, header=None)
            compress_gz(map_folder + "/test.csv")
            lst_node_has_split.append(label_node)
    ###################### create nodetype-has-split.csv
    lst_has_split = []
    for node_type in label_nodes_names:
        if node_type in lst_node_has_split:
            lst_has_split.append("True")
        else:
            lst_has_split.append("False")
    node_has_split_df = [label_nodes_names,lst_has_split]
    pd.DataFrame(node_has_split_df).to_csv(
        output_root_path + dataset_name + "/split/" + split_by[
            "folder_name"] + "/nodetype-has-split.csv", header=None, index=None)
    compress_gz(output_root_path + dataset_name + "/split/" + split_by[
        "folder_name"] + "/nodetype-has-split.csv")
    print("***************************************************")
    print("Total Training Samples:",cnt_train)
    print("Total Validating Samples:", cnt_valid)
    print("Total Testing Samples:", cnt_test)

    ################# write entites relations for nodes only (non literals)
    # The updated way of mapping edges
    lst_relations_df = graph_df[['source-type', 'edge-type', 'destination-type']].drop_duplicates().reset_index(drop=True)

    total_num_edges = 0
    lst_relations = []
    for id, row in lst_relations_df.iterrows():
        subject, rel, object = row
        lst_relations.append([subject, rel, object])
        this_relations_dic = graph_df[(graph_df['source-type'] == subject) & (graph_df['edge-type'] == rel) & (graph_df['destination-type'] == object)]
        this_relations_dic["s_idx"] = this_relations_dic["source-id"].apply(lambda x: entites_dic[subject + "_dic"][x])
        this_relations_dic["o_idx"] = this_relations_dic["destination-id"].apply(lambda x: entites_dic[object + "_dic"][x])
        this_relations_dic = this_relations_dic.sort_values(by="s_idx").reset_index(drop=True)
        rel_out = this_relations_dic[["s_idx", "o_idx"]]
        if len(rel_out) > 0:
            map_folder = output_root_path + dataset_name + "/raw/relations/" + subject + "___" + \
                         rel.split("/")[-1] + "___" + object
            try:
                os.stat(map_folder)
            except:
                os.makedirs(map_folder)
            rel_out.to_csv(map_folder + "/edge.csv", index=None, header=None)
            compress_gz(map_folder + "/edge.csv")
            ##### write relations num
            f = open(map_folder + "/num-edge-list.csv", "w")
            f.write(str(len(this_relations_dic)))
            f.close()
            compress_gz(map_folder + "/num-edge-list.csv")
            ########## write relations idx
            rel_idx = relations_df[relations_df["rel name"] == rel.split("/")[-1]]["rel idx"].values[0]
            rel_out.insert(2, 'rel_idx', rel_idx)
            rel_idx_df = rel_out["rel_idx"]
            rel_idx_df.to_csv(map_folder + "/edge_reltype.csv", header=None, index=None)
            compress_gz(map_folder + "/edge_reltype.csv")
            total_num_edges += len(this_relations_dic)
            del this_relations_dic, rel_out, rel_idx_df, rel_idx
    pd.DataFrame(lst_relations).to_csv(
        output_root_path + dataset_name + "/raw/triplet-type-list.csv",
        header=None, index=None)
    compress_gz(output_root_path + dataset_name + "/raw/triplet-type-list.csv")
    print("Total encoded edges is: ", total_num_edges)

    ################ Prepare node features (edge type distribution)
    features_time = time.time()

    x_list_df = feature_engineering(graph_df,edge_types)
    feature_path = output_root_path + "/features/"+dataset_name
    delete_folder(feature_path)
    save_path = feature_path +"/all_features_node_uuid.csv"
    ensure_dir(save_path)
    x_list_df.to_csv(save_path, index=None)
    for label_node in label_nodes_names:
        file_path = output_root_path + dataset_name + "/mapping/" + label_node + "_entidx2name.csv"
        mapping_nodes_df = pd.read_csv(file_path, header=None, skiprows=1, names=["node_id", "node_uuid"])
        mapped_x_list_df = pd.merge(mapping_nodes_df, x_list_df, on="node_uuid")
        mapped_x_list_df = mapped_x_list_df.set_index("node_id")
        del mapped_x_list_df['node_uuid']
        x_list_tensor = torch.from_numpy(mapped_x_list_df.to_numpy()).float()
        save_path = output_root_path + dataset_name + "/features/" + label_node + "/node-features.pt"
        ensure_dir(save_path)
        torch.save(x_list_tensor, save_path)
    print("Feature engineering time:",time.time() - features_time," seconds")
    shutil.make_archive(output_root_path + dataset_name, 'zip',
                        root_dir=output_root_path, base_dir=dataset_name)
    print("Total Converting time:", time.time() - start_convert, " seconds")