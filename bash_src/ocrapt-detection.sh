#!/bin/sh
date=$(date +'%d_%m_%Y')
echo "Available datasets ( tc3 , optc , nodlink )"
read -p "Enter the dataset name: " dataset

echo "Available hosts (ALL_TC3: {cadets, theia, trace}, ALL_OPTC: {SysClient0051 , SysClient0501 , SysClient0201} ALL_NodLink: {SimulatedUbuntu, SimulatedW10, SimulatedWS12}"
read -p "Enter the host name: " host
read -p "Enter the experiment name: " exp_name
read -p "Enter the number of runs: " runs

read -p "Do you want to detect anomalous subgraphs uisng GNN? (y/N): " detectSubgraphs
if [[ "$detectSubgraphs" == "y" ]]
then
  read -p "Do you want to use the defaults trained GNN models? (Y/n): " useDefaultModels
  if [[ "$useDefaultModels" == "n" ]]
  then
    read -p "Enter the GNN model name: " load_model
  else
    load_model="OCRGCN_Dr0_ly3_bs0_ep100_beta0.5_LR0.005_Hly32_dynConVal0.001To0.05.model"
  fi
fi

read -p "Do you want to generate attack report using LLMs ?(y/N): " generateReports
if [[ "$generateReports" == "y" ]]
then
  if [[ "$detectSubgraphs" != "y" ]]
  then
    read -p "Do you want to use the defaults trained GNN models? (Y/n): " useDefaultModels
    if [[ "$useDefaultModels" == "n" ]]
    then
      read -p "Enter the GNN model name: " load_model
    else
      load_model="OCRGCN_Dr0_ly3_bs0_ep100_beta0.5_LR0.005_Hly32_dynConVal0.001To0.05.model"
    fi
    read -p "Enter the experiment name for anomalous subgraphs: " inv_logs_name
    read -p "Do you want to load previously indexed subgraphs (y/N): " load_index
  fi
  read -p "Enter the LLM investigation experiment name (output): " llm_exp_name
fi

if [[ "$dataset" == "optc" ]]
then
  SourceDataset=darpa_optc
elif [[ "$dataset" == "nodlink" ]]
then
  SourceDataset=nodlink
else
  SourceDataset=darpa_tc3
fi

Detect_Anomolous_Nodes () {
  detector=$1
  host=$2
  ep=$3
  beta=$4
  HiddenLayer=$5
  n_layers=$6
  batch_size=$7
  learningRate=$8
  minCon=$9
  maxCon=${10}
  load_model=${11}
  root_path=${12}
  parameters=" --dropout ${dropout} --batch-size ${batch_size} --epochs ${ep} --runs ${runs} --lr ${learningRate}  --num-layers ${n_layers} --beta ${beta} --multiple-models --dynamic-contamination --flexable-rate ${minCon} --max-contamination ${maxCon} "
  logs="DetectAnomalousNodes_${detector}_bs${batch_size}_ep${ep}_Dr${dropout}_ly${n_layers}_beta${beta}_LR${learningRate}_dynConVal${minCon}To${maxCon}_MultipleModels"
  if [[ "$HiddenLayer" == "D" ]]
  then
    parameters+=" --adjust-hidden-channels "
    logs+="_AutoHL"
  else
    parameters+=" --hidden-channels ${HiddenLayer} "
    logs+="_Hly${HiddenLayer}"
  fi
  logs+="_${date}.txt"
  echo "Parameters are: ${parameters}"
  echo "load from: ${load_model}"
  mkdir -p ../logs/${host}/${exp_name}
  python -B -u ../src/train_gnn_models.py --host ${host} --dataset ${dataset} --root-path ${root_path} --exp-name ${exp_name} --detector ${detector} ${parameters} --load-model ${load_model}  >> ../logs/${host}/${exp_name}/${logs}
}

Detect_Anomolous_Subgraph () {
  host=$1
  root_path=$2
  max_edges=$3
  top_k=$4
  model_path=$5
  n_hop=$6
  n_layers=$7
  abnormality=$8
  investigation_parameters=" --min-nodes 3 --max-edges ${max_edges} --number-of-hops ${n_hop} --runs ${runs} --remove-duplicated-subgraph --get-node-attrs --expand-2-hop no --correlate-anomalous-once --top-k ${top_k} --abnormality-level ${abnormality}"
  logs_name="expand_${n_hop}_hop_MaxEdges${max_edges}_K${top_k}_ly${n_layers}"
  mkdir -p ../logs/${host}/${exp_name}
  python -B -u ../src/detect_anomalous_subgraphs.py --host ${host} --dataset ${dataset} --root-path ${root_path} --exp-name ${exp_name} --model ${model_path}  --construct-from-anomaly-subgraph  ${investigation_parameters}  --inv-exp-name ${logs_name} ${more_param} >> ../logs/${host}/${exp_name}/DetectAnomalousSubgraphs_${logs_name}_${date}.txt
}

generate_llm_investigator_reports () {
  host=$1
  load_model=$2
  load_index=$3
  embed_model=$4
  anomalous=$5
  abnormality=$6
  root_path=$7
  parameters=" --llm-exp-name ${llm_exp_name}"
  inv_logs_name="expand_${n_hop}_hop_MaxEdges${max_edges}_K${top_k}"
  if [[ "$load_index" == "y" ]]
  then
    parameters+=" --load-index"
  fi
  mkdir -p ../logs/${host}/${exp_name}
  python -B -u ../src/ocrapt_llm_investigator.py --host ${host} --dataset ${dataset} --root-path ${root_path} --exp-name ${exp_name} --GNN-model-name ${load_model} --inv-exp-name ${inv_logs_name} --llm-embedding-model ${embed_model} --abnormality-level ${abnormality} --anomalous ${anomalous} ${parameters} >> ../logs/${host}/${exp_name}/llm_investigator_${llm_exp_name}_${date}.txt
}



# default training parameters
batch_size=0
HiddenLayer=32
n_layers=3
beta=0.5
learningRate=0.005
ep=100
dropout=0
minCon=0.001
maxCon=0.05
detector=OCRGCN


# default investigation parameters
max_edges=5000
n_hop=1
top_k=15
#
embed_model="text-embedding-3-large"
load_index="N"
anomalous="sub"
abnormality="Moderate"


execute_OCR_APT () {
  host=${1}
  root_path=../dataset/${SourceDataset}/${host}/experiments/
  echo "Run OCR-APT on host: ${host} "
  if [[ "$detectSubgraphs" == "y" ]]
  then
    echo "Detect anomolous nodes using the GNN model: ${load_model}"
    Detect_Anomolous_Nodes ${detector} ${host} ${ep} ${beta} ${HiddenLayer} ${n_layers} ${batch_size} ${learningRate} ${minCon} ${maxCon} ${load_model} ${root_path}
    echo "Detect anomolous subgraphs"
    Detect_Anomolous_Subgraph ${host} ${root_path} ${max_edges} ${top_k} ${load_model} ${n_hop} ${n_layers} ${abnormality}
  fi
  if [[ "$generateReports" == "y" ]]
  then
    echo "Generate attack reprots"
    generate_llm_investigator_reports ${host} ${load_model} ${load_index} ${embed_model} ${anomalous} ${abnormality} ${root_path}
  fi
  sleep 1m
}

if [[ "$host" == "ALL_TC3" ]]
then
  execute_OCR_APT cadets
  execute_OCR_APT theia
  execute_OCR_APT trace
elif [[ "$host" == "ALL_OPTC" ]]
then
  execute_OCR_APT SysClient0051
  execute_OCR_APT SysClient0201
  execute_OCR_APT SysClient0501
elif [[ "$host" == "ALL_NodLink" ]]
then
  execute_OCR_APT SimulatedUbuntu
  execute_OCR_APT SimulatedWS12
  execute_OCR_APT SimulatedW10
else
  execute_OCR_APT ${host}
fi

echo "Done"
