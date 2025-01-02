#!/bin/sh
date=$(date +'%d_%m_%Y')
echo "Available datasets (cadets, theia, trace, ALL_TC3 , SysClient0051 , SysClient0501 , SysClient0201, ALL_OPTC )"
read -p "Enter the dataset: " dataset
read -p "Are you processing DARPA OpTC (y/N): ": Is_DARPA_OPTC
read -p "Enter the experiment name: " exp_name
read -p "Enter the model name: " model_path
echo "The dataset root path is: ${root_path}"
read -p "Enter the graph embedding size (32): " HiddenLayer
read -p "Enter the minimum number of nodes per subgraph (3): " min_nodes
if [[ "$Is_DARPA_OPTC" == "y" ]]
then
  SourceDataset=darpa_optc
else
  SourceDataset=darpa_tc3
fi

investigation_parameters=" --get-node-attrs --correlate-anomalous-once --remove-duplicated-subgraph --consider-related-nodes"
logs_name="controlled_in_1_hop_anomaly_subgraph_partition_then_sampling_MinNode${min_nodes}"
if [[ "$Is_DARPA_OPTC" == "y" ]]
then
  investigation_parameters+=" --graph-optc"
fi
read -p "Do you want to set maximum edges relative to query graphs ? (y/N)": qg_max_edges
if [[ "$qg_max_edges" == "y" ]]
then
  read -p "Enter the maximum edges per subgraph (25x Query Graphs) ( 0 for no edge limit): " max_edges
  investigation_parameters+=" --max-edges-mult-qg ${max_edges}"
  logs_name+="_MaxEdges${max_edges}"
else
  read -p "Enter the maximum edges per subgraph (default: 5000)( 0 for no edge limit): " max_edges
  investigation_parameters+=" --max-edges ${max_edges}"
  logs_name+="_MaxEdges${max_edges}"
fi
read -p "Do you want to consider top k percent of anomaly score instead of fixed number ? (y/N)": use_k_percent
if [[ "$use_k_percent" == "y" ]]
then
  read -p "Enter the top K percent subgraphs per node type to be investigated (0.03): " k_percent
  investigation_parameters+=" --top-k-percent ${k_percent}"
  logs_name+="_K${k_percent}"
else
  read -p "Enter the top K subgraphs per node type to be investigated (100): " k
  investigation_parameters+=" --top-k ${k}"
  logs_name+="_K${k}"
fi

#read -p "Do you want to consider top k processes only ? (y/N)": processCentric
#if [[ "$processCentric" == "y" ]]
#then
#  investigation_parameters+=" --process-centric"
#  logs_name+="_processCentric"
#fi
#read -p "Do you want to drop duplicated subgraphs ? (y/N)": drop_duplication
#if [[ "$drop_duplication" == "y" ]]
#then
#  investigation_parameters+=" --remove-duplicated-subgraph"
##  logs_name+="_NoDuplication"
#fi
#read -p "Do you want to correlate anomalous nodes only once? (y/N)": correlateOnce
#if [[ "$correlateOnce" == "y" ]]
#then
#  investigation_parameters+=" --correlate-anomalous-once "
##  logs_name+="_correlateOnce"
#fi

read -p "Do you want to draw constructed subgraphs? (y/N)": draw_subgraphs
if [[ "$draw_subgraphs" == "y" ]]
then
  investigation_parameters+=" --draw-subgraphs"
fi
#read -p "Do you want to cluster constructed subgraphs? (y/N)": cluster_subgraphs
#if [[ "$cluster_subgraphs" == "y" ]]
#then
#  investigation_parameters+=" --cluster-subgraphs"
#fi

#read -p "Enter the number of hops while investigation: " n_hops_investigate
#investigation_parameters=+=" --number-of-hops ${n_hops_investigate} "
#read -p "Do you want to traverse backwards only (y/N)": backwards
#if [[ "$backwards" == "y" ]]
#then
#  investigation_parameters+=" --backwards-only"
#fi
#python -B -u ../src/investigate_nodes.py --dataset ${dataset} --root-path ${root_path} --exp-name ${exp_name} --node-emb-per-subgraph --model ${model_path} --node-emb-size ${HiddenLayer} --tensor-neurons ${HiddenLayer} ${investigation_parameters} >> ../logs/${dataset}/${exp_name}/investigating_${n_hops_investigate}_hop_${model_path}_${date}.txt

logs_name+="_EarlyStop"
echo "investigating with controlled constructed subgraphs"
if [[ "$dataset" == "ALL_TC3" ]]
then
  dataset=cadets
  root_path="../dataset/darpa_tc3/${dataset}/pygod/"
  mkdir -p ../logs/${dataset}/${exp_name}
  python -B -u ../src/investigate_nodes.py --dataset ${dataset} --root-path ${root_path} --exp-name ${exp_name} --node-emb-per-subgraph --model ${model_path} --node-emb-size ${HiddenLayer} --tensor-neurons ${HiddenLayer} --construct-from-anomaly-subgraph --number-of-hops 1 ${investigation_parameters} --min-nodes ${min_nodes} --inv-exp-name ${logs_name} >> ../logs/${dataset}/${exp_name}/investigating_${logs_name}_FixedAttrs_sampledConnectedSubg_expandTo2hop_${date}.txt
  dataset=trace
  root_path="../dataset/darpa_tc3/${dataset}/pygod/"
  mkdir -p ../logs/${dataset}/${exp_name}
  python -B -u ../src/investigate_nodes.py --dataset ${dataset} --root-path ${root_path} --exp-name ${exp_name} --node-emb-per-subgraph --model ${model_path} --node-emb-size ${HiddenLayer} --tensor-neurons ${HiddenLayer} --construct-from-anomaly-subgraph --number-of-hops 1 ${investigation_parameters} --min-nodes ${min_nodes} --inv-exp-name ${logs_name} >> ../logs/${dataset}/${exp_name}/investigating_${logs_name}_FixedAttrs_sampledConnectedSubg_expandTo2hop_${date}.txt
  dataset=theia
  root_path="../dataset/darpa_tc3/${dataset}/pygod/"
  mkdir -p ../logs/${dataset}/${exp_name}
  python -B -u ../src/investigate_nodes.py --dataset ${dataset} --root-path ${root_path} --exp-name ${exp_name} --node-emb-per-subgraph --model ${model_path} --node-emb-size ${HiddenLayer} --tensor-neurons ${HiddenLayer} --construct-from-anomaly-subgraph --number-of-hops 1 ${investigation_parameters} --min-nodes ${min_nodes} --inv-exp-name ${logs_name} >> ../logs/${dataset}/${exp_name}/investigating_${logs_name}_FixedAttrs_sampledConnectedSubg_expandTo2hop_${date}.txt
else
  root_path="../dataset/${SourceDataset}/${dataset}/pygod/"
  mkdir -p ../logs/${dataset}/${exp_name}
  python -B -u ../src/investigate_nodes.py --dataset ${dataset} --root-path ${root_path} --exp-name ${exp_name} --node-emb-per-subgraph --model ${model_path} --node-emb-size ${HiddenLayer} --tensor-neurons ${HiddenLayer} --construct-from-anomaly-subgraph --number-of-hops 1 ${investigation_parameters} --min-nodes ${min_nodes} --inv-exp-name ${logs_name} >> ../logs/${dataset}/${exp_name}/investigating_${logs_name}_FixedAttrs_sampledConnectedSubg_expandTo2hop_${date}.txt
fi

#python -B -u ../src/investigate_nodes.py --dataset ${dataset} --root-path ${root_path} --exp-name ${exp_name} --node-emb-per-subgraph --model ${model_path} --node-emb-size ${HiddenLayer} --tensor-neurons ${HiddenLayer} --construct-from-anomaly-subgraph --number-of-hops 1 --top-k-percent 0.90 ${investigation_parameters} --min-nodes ${min_nodes}  >> ../logs/${dataset}/${exp_name}/investigating_controlled_in_1_hop_anomaly_subgraph${logs_name}_MinNode${min_nodes}_k95perc_${model_path}_${date}.txt
#
#python -B -u ../src/investigate_nodes.py --dataset ${dataset} --root-path ${root_path} --exp-name ${exp_name} --node-emb-per-subgraph --model ${model_path} --node-emb-size ${HiddenLayer} --tensor-neurons ${HiddenLayer} --construct-from-anomaly-subgraph --number-of-hops 1 --top-k 100 ${investigation_parameters} --min-nodes ${min_nodes}  >> ../logs/${dataset}/${exp_name}/investigating_controlled_in_1_hop_anomaly_subgraph${logs_name}_MinNode${min_nodes}_k100$_{model_path}_${date}.txt
#
#python -B -u ../src/investigate_nodes.py --dataset ${dataset} --root-path ${root_path} --exp-name ${exp_name} --node-emb-per-subgraph --model ${model_path} --node-emb-size ${HiddenLayer} --tensor-neurons ${HiddenLayer} --construct-from-anomaly-subgraph --number-of-hops 1 --top-k 100 --process-centric ${investigation_parameters} --min-nodes ${min_nodes}  >> ../logs/${dataset}/${exp_name}/investigating_controlled_in_1_hop_anomaly_subgraph${logs_name}_MinNode${min_nodes}_k100_processCentric${model_path}_${date}.txt


