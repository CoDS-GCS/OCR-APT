#!/bin/sh
date=$(date +'%d_%m_%Y')
echo "Available datasets ( tc3 , optc , nodlink )"
read -p "Enter the dataset name: " dataset

echo "Available hosts (ALL_TC3, cadets, theia, trace, fivedirections, ALL_OPTC, SysClient0051 , SysClient0501 , SysClient0201, ALL_NodLink, SimulatedUbuntu, SimulatedW10, SimulatedWS12)"
read -p "Enter the host name: " host
if [[ "$dataset" == "optc" ]]
then
  SourceDataset=darpa_optc
elif [[ "$dataset" == "nodlink" ]]
then
  SourceDataset=nodlink
else
  SourceDataset=darpa_tc3
fi
read -p "Enter the experiment name: " exp_name
read -p "Enter the model name: " model_path
#read -p "Enter the root working directory path: " root_path
read -p "Enter the main directory folder (pygod): " working_dir
read -p "Enter the size of hidden layer (32): " HiddenLayer

Construct_annomalies_subgraphs () {
  host=$1
  max_edges=$2
  top_k=$3
  HiddenLayer=$4
  model_path=$5
  n_hop=$6
  n_layers=$7
  beta=$8
  echo "investigating ${host} host"
  mkdir -p ../logs/${host}/${exp_name}
  investigation_parameters=" --min-nodes 3 --number-of-hops ${n_hop} --correlate-anomalous-once --remove-duplicated-subgraph --get-node-attrs --node-emb-per-subgraph --consider-related-nodes "
  logs_name="${n_hop}_hop_MaxEdges${max_edges}_K${top_k}_ly${n_layers}_Hly${HiddenLayer}_beta${beta}"
  investigation_parameters+=" --max-edges ${max_edges}"
  investigation_parameters+=" --top-k ${top_k}"
  python -B -u ../src/investigate_nodes.py --dataset ${dataset} --host ${host} --root-path ${root_path} --exp-name ${exp_name} --model ${model_path} --node-emb-size ${HiddenLayer} --tensor-neurons ${HiddenLayer} --construct-from-anomaly-subgraph  ${investigation_parameters}  --inv-exp-name ${logs_name} >> ../logs/${host}/${exp_name}/investigating_${logs_name}_expandUpTo2hop_${date}.txt
}

generate_llm_investigator_reports () {
  host=$1
}
# default training parameters
batch_size=0
HiddenLayer=32
n_layers=3
beta=0.5
learningRate=0.004
ep=100
dropout=0
minCon=0.001
maxCon=0.2
detector=OCRGCN

# default investigation parameters
top_k=15
max_edges=5000
n_hop=1


if [[ "$host" == "ALL" ]]
then
  echo "investigating all hosts"
  echo "investigating all DARPA OpTC hosts"
  dataset="optc"
  host="SysClient0051"
  root_path="../dataset/darpa_optc/${host}/1dayTrainingFilterStTerNoSlfLop_WithDirection/"
  exp_name="EdgeDist_IdleTimeSec_1day_filterStTer_NoSlfLop_withDirection_orderFeat_fixedTraining"
  model_path="EdgeDist_IdleTimeSec_1day_filterStTer_NoSlfLop_withDirection_orderFeat_fixedTraining_PyGoD_OCRGCN_ly3_Hly64_bs0_ep100_dynCon_1_maxCon0.2_flex0.001"
  Construct_annomalies_subgraphs ${host} ${max_edges} ${top_k} ${HiddenLayer} ${model_path} ${n_hop} ${n_layers} ${beta}
  host="SysClient0501"
  root_path="../dataset/darpa_optc/${host}/1dayTrainingFilterStTerNoSlfLop_WithDirection/"
  exp_name="EdgeDist_IdleTimeSec_FixTrainingFeat_withDirection"
  model_path="EdgeDist_IdleTimeSec_FixTrainingFeat_withDirection_PyGoD_OCRGCN_ly3_Hly64_bs0_ep100_dynCon_1_maxCon0.2_flex0.001"
  Construct_annomalies_subgraphs ${host} ${max_edges} ${top_k} ${HiddenLayer} ${model_path} ${n_hop} ${n_layers} ${beta}
  host="SysClient0201"
  root_path="../dataset/darpa_optc/${host}/1dayTrainingFilterStTerNoSlfLop_WithDirection/"
  exp_name="EdgeDist_IdleTimeSec_FixTrainingFeat_withDirection"
  model_path="EdgeDist_IdleTimeSec_FixTrainingFeat_withDirection_PyGoD_OCRGCN_ly3_Hly64_bs0_ep100_dynCon_1_maxCon0.2_flex0.001"
  Construct_annomalies_subgraphs ${host} ${max_edges} ${top_k} ${HiddenLayer} ${model_path} ${n_hop} ${n_layers} ${beta}
  SourceDataset=darpa_tc3
  dataset="tc3"
  model_path="EdgeDist_IdleTimeSec_attr_PyGoD_OCRGCN_ly3_Hly64_bs0_ep100_dynCon_1_maxCon0.2_flex0.001"
  exp_name="EdgeDist_IdleTimeSec_attr"
  host="cadets"
  root_path="../dataset/darpa_tc3/${host}/pygod/"
  Construct_annomalies_subgraphs ${host} ${max_edges} ${top_k} ${HiddenLayer} ${model_path} ${n_hop} ${n_layers} ${beta}
  host="trace"
  root_path="../dataset/darpa_tc3/${host}/pygod/"
  Construct_annomalies_subgraphs ${host} ${max_edges} ${top_k} ${HiddenLayer} ${model_path} ${n_hop} ${n_layers} ${beta}
  host="theia"
  root_path="../dataset/darpa_tc3/${host}/pygod/"
  Construct_annomalies_subgraphs ${host} ${max_edges} ${top_k} ${HiddenLayer} ${model_path} ${n_hop} ${n_layers} ${beta}
elif [[ "$host" == "ALL_TC3" ]]
then
  echo "investigating all DARPA TC3 hosts"
  dataset="tc3"
  host="theia"
  root_path="../dataset/${SourceDataset}/${host}/${working_dir}/"
  Construct_annomalies_subgraphs ${host} ${max_edges} ${top_k} ${HiddenLayer} ${model_path} ${n_hop} ${n_layers} ${beta}
  host="trace"
  root_path="../dataset/${SourceDataset}/${host}/${working_dir}/"
  Construct_annomalies_subgraphs ${host} ${max_edges} ${top_k} ${HiddenLayer} ${model_path} ${n_hop} ${n_layers} ${beta}
  host="cadets"
  root_path="../dataset/${SourceDataset}/${host}/${working_dir}/"
  Construct_annomalies_subgraphs ${host} ${max_edges} ${top_k} ${HiddenLayer} ${model_path} ${n_hop} ${n_layers} ${beta}
elif [[ "$host" == "ALL_OPTC" ]]
then
  echo"investigating all DARPA OpTC hosts"
  dataset="optc"
  host="SysClient0051"
  root_path="../dataset/${SourceDataset}/${host}/${working_dir}/"
  Construct_annomalies_subgraphs ${host} ${max_edges} ${top_k} ${HiddenLayer} ${model_path} ${n_hop} ${n_layers} ${beta}
  host="SysClient0501"
  root_path="../dataset/${SourceDataset}/${host}/${working_dir}/"
  Construct_annomalies_subgraphs ${host} ${max_edges} ${top_k} ${HiddenLayer} ${model_path} ${n_hop} ${n_layers} ${beta}
  host="SysClient0201"
  root_path="../dataset/${SourceDataset}/${host}/${working_dir}/"
  Construct_annomalies_subgraphs ${host} ${max_edges} ${top_k} ${HiddenLayer} ${model_path} ${n_hop} ${n_layers} ${beta}
elif [[ "$host" == "ALL_NodLink" ]]
then
  dataset="nodlink"
  host="SimulatedUbuntu"
  root_path="../dataset/${SourceDataset}/${host}/${working_dir}/"
  Construct_annomalies_subgraphs ${host} ${max_edges} ${top_k} ${HiddenLayer} ${model_path} ${n_hop} ${n_layers} ${beta}
  host="SimulatedWS12"
  root_path="../dataset/${SourceDataset}/${host}/${working_dir}/"
  Construct_annomalies_subgraphs ${host} ${max_edges} ${top_k} ${HiddenLayer} ${model_path} ${n_hop} ${n_layers} ${beta}
  host="SimulatedW10"
  root_path="../dataset/${SourceDataset}/${host}/${working_dir}/"
  Construct_annomalies_subgraphs ${host} ${max_edges} ${top_k} ${HiddenLayer} ${model_path} ${n_hop} ${n_layers} ${beta}
else
  root_path="../dataset/${SourceDataset}/${host}/${working_dir}/"
  Construct_annomalies_subgraphs ${host} ${max_edges} ${top_k} ${HiddenLayer} ${model_path} ${n_hop} ${n_layers} ${beta}
fi