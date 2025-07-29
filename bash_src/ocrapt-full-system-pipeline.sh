#!/bin/sh
date=$(date +'%d_%m_%Y')
echo "Available datasets ( tc3 , optc , nodlink )"
read -p "Enter the dataset name: " dataset
echo "Available hosts (ALL_TC3: {cadets, theia, trace}, ALL_OPTC: {SysClient0051 , SysClient0501 , SysClient0201} ALL_NodLink: {SimulatedUbuntu, SimulatedW10, SimulatedWS12}"
read -p "Enter the host name: " host
read -p "Enter the experiment name: " exp_name
read -p "Enter the number of runs: " runs

if [[ "$dataset" == "optc" ]]
then
  SourceDataset=darpa_optc
elif [[ "$dataset" == "nodlink" ]]
then
  SourceDataset=nodlink
else
  SourceDataset=darpa_tc3
fi

# Transform to RDF parameters
read -p "Do you want to transform to RDF (y/N): " ToRDF
if [[ "$ToRDF" == "y" ]]
then
  rdf_parameter=" --rdfs --adjust-uuid --min-node-representation 5"
  if [[ "$dataset" == "tc3" ]]
  then
    read -p "Do you want to get node attributes from NetworkX graph(y/N): ": node_attrs_graph_nx
  fi
fi

# Encode to PYG parameters
read -p "Do you want to encode to PyG (y/N): " encodePYG
if [[ "$encodePYG" == "y" ]]
then
  pyg_parameters=" --get-timestamps-features --get-idle-time --normalize-features"
fi
# subgraph anomaly detection parameters
read -p "Do you want to train GNN model? (y/N): " trainGNN
if [[ "$trainGNN" != "y" ]]
then
  read -p "Enter the GNN model full name: " load_model
fi
read -p "Do you want to detect anomalous subgraphs with trained GNN model? (y/N): " detectSubgraphs
if [[ "$detectSubgraphs" != "y" ]]
then
  read -p "Enter the experiment name for anomalous subgraphs: " inv_logs_name
fi

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
# default subgraph construction parameters
max_edges=5000
n_hop=1
top_k=15


# LLM attack investigator parameters
read -p "Do you want to generate attack report using LLMs ?(y/N): " generateReports
if [[ "$generateReports" == "y" ]]
then
  read -p "Enter the LLM investigation experiment name (output): " llm_exp_name
  if [[ "$detectSubgraphs" != "y" ]]
  then
    read -p "Do you want to load previously indexed subgraphs (y/N): " load_index
  fi
fi
# default parameters
embed_model="text-embedding-3-large"
load_index="N"
anomalous="sub"
abnormality="Moderate"



transfor_to_RDF () {
  dataset=$1
  host=$2
  root_path=$3
  source_graph=${host}

  if [[ "$dataset" == "tc3" ]]
  then
    read -p "Do you want to get node attributes from NetworkX graph(y/N): ": node_attrs_graph_nx
  fi
  python -B -u ../src/transform_to_RDF.py --dataset ${dataset} --host ${host} --root-path ${root_path} --source-graph ${source_graph} ${rdf_parameter} >> ../logs/${host}/${exp_name}/Transform_to_RDF_${date}.txt
  if [[ "$node_attrs_graph_nx" == "y" ]]
  then
    python -B -u ../src/get_node_attributes.py --adjust-uuid --host ${host} --root-path ${root_path} --source-graph ${source_graph} --source-graph-nx ${nx_source_graph} >> ../logs/${host}/${exp_name}/Transform_to_RDF_${date}.txt
  fi
  echo "Done converting to RDF"
}

encode_to_PYG () {
  dataset=$1
  host=$2
  root_path=$3
  source_graph=${host}
  python -u -B ../src/encode_to_PyG.py --dataset ${dataset} --host ${host} --root-path ${root_path} --exp-name ${exp_name} --source-graph ${source_graph} ${pyg_parameters} >> ../logs/${host}/${exp_name}/Encode_RDF_to_PyG_${date}.txt
}

train_GNN_models () {
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
  root_path=${11}
  parameters=" --dropout ${dropout} --batch-size ${batch_size} --epochs ${ep} --runs ${runs} --lr ${learningRate}  --num-layers ${n_layers} --beta ${beta} --multiple-models --dynamic-contamination --flexable-rate ${minCon} --max-contamination ${maxCon}  "
  save_path="${detector}_Dr${dropout}_ly${n_layers}_bs${batch_size}_ep${ep}_beta${beta}_LR${learningRate}_Hly${HiddenLayer}"
  logs="Training_${detector}_bs${batch_size}_ep${ep}_Dr${dropout}_ly${n_layers}_beta${beta}_LR${learningRate}_dynCon${minCon}To${maxCon}_MultipleModels"
  if [[ "$HiddenLayer" == "D" ]]
  then
    parameters+=" --adjust-hidden-channels "
    logs+="_AutoHL"
  else
    parameters+=" --hidden-channels ${HiddenLayer} "
    logs+="_Hly${HiddenLayer}"
  fi
  save_path+="_dynConVal${minCon}To${maxCon}"
  if [ ! -f ${root_path}/results/${exp_name}/${save_path}/anomaly_results_summary.csv ]
  then
    save_path+=".model"
    logs+="_${date}.txt"
    echo "Parameters are: ${parameters}"
    echo "save to: ${save_path}"
    python -B -u ../src/train_gnn_models.py --host ${host} --dataset ${dataset} --root-path ${root_path} --exp-name ${exp_name} --detector ${detector} ${parameters} --save-model ${save_path} >> ../logs/${host}/${exp_name}/${logs}
  else
    echo "The model already trained before"
  fi
  load_model=${save_path}
}

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
  logs_name="expand_${n_hop}_hop_MaxEdges${max_edges}_K${top_k}"
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
  inv_logs_name="expand_${n_hop}_hop_MaxEdges${max_edges}_K${top_k}"
  parameters=" --llm-exp-name ${llm_exp_name}"
  if [[ "$load_index" == "y" ]]
  then
    parameters+=" --load-index"
  fi
  python -B -u ../src/ocrapt_llm_investigator.py --host ${host} --dataset ${dataset} --root-path ${root_path} --exp-name ${exp_name} --GNN-model-name ${load_model} --inv-exp-name ${inv_logs_name} --llm-embedding-model ${embed_model} --abnormality-level ${abnormality} --anomalous ${anomalous} ${parameters} >> ../logs/${host}/${exp_name}/llm_investigator_${llm_exp_name}_${date}.txt
}


execute_OCR_APT () {
  host=${1}
  mkdir -p ../logs/${host}/${exp_name}
  root_path=../dataset/${SourceDataset}/${host}/experiments/
  echo "Run OCR-APT on host: ${host} "
  if [[ "$ToRDF" == "y" ]]
  then
    echo "Converting to RDF"
    transfor_to_RDF ${dataset} ${host} ${root_path}
    echo "load RDFs provenance graphs into the RDF graph engine"
    read -p "Have you loaded provenance graphs into the database (y/N): " PG_loaded
  fi
  if [[ "$encodePYG" == "y" ]]
  then
    echo "Encoding to PyG"
    encode_to_PYG ${dataset} ${host} ${root_path}
  fi
  if [[ "$PG_loaded" == "y" ]]
  then
    if [[ "$trainGNN" == "y" ]]
    then
      echo "Train OCRGCN models, then utilize it to detect anomolous nodes"
      train_GNN_models ${detector} ${host} ${ep} ${beta} ${HiddenLayer} ${n_layers} ${batch_size} ${learningRate} ${minCon} ${maxCon} ${root_path}
    fi
    if [[ "$detectSubgraphs" == "y" ]]
    then
      echo "Detect anomolous subgraphs"
      Detect_Anomolous_Subgraph ${host} ${root_path} ${max_edges} ${top_k} ${load_model} ${n_hop} ${n_layers} ${abnormality}
    fi
    if [[ "$generateReports" == "y" ]]
    then
      echo "Generate attack reprots"
      generate_llm_investigator_reports ${host} ${load_model} ${load_index} ${embed_model} ${anomalous} ${abnormality} ${root_path}
    fi
  else
    echo "load provenance graphs into the database and run again"
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
