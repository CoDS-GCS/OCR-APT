#!/bin/sh
date=$(date +'%d_%m_%Y')
echo "Available datasets ( tc3 , optc , nodlink )"
read -p "Enter the dataset name: " dataset
#echo "Available datasets (cadets, theia, trace, fivedirections, SysClient0051 , SysClient0501 , SysClient0201)"
#read -p "Enter the dataset: " dataset
#read -p "Enter the experiment name: " exp_name
#read -p "Enter the root folder path (../dataset/darpa_tc3/cadets/pygod/): " root_path
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
#read -p "Enter the batch size: " batch_size
#read -p "Enter the number of epochs: " ep
#read -p "Enter the size of hidden layer (32): " HiddenLayer
#read -p "Enter the number of layers (2): " n_layers
#read -p "Enter the the contamination for one model training (0.1): " contamination

#train_model_inductive () {
#  detector=$1
#  dataset=$2
#  root_path="../dataset/darpa_tc3/${dataset}/pygod_v2/"
#  parameters=" --batch-size ${batch_size} --epochs ${ep} --runs ${runs}  --hidden-channels ${HiddenLayer} --num-layers ${n_layers}"
#  save_path="${detector}_ly${n_layers}_Hly${HiddenLayer}_bs${batch_size}_ep${ep}"
#  logs="training_${detector}_bs${batch_size}_ep${ep}_ly${n_layers}_Hly${HiddenLayer}"
#  parameters+=$3
#  save_path+=$4
#  logs+=$5
#  logs+="_${date}.txt"
#  save_path+=".model"
#  mkdir -p ../logs/${dataset}/${exp_name}
#  echo "Parameters are: ${parameters}"
#  echo "save to: ${save_path}"
#  python -B -u ../src/pygod_models_DARPA.py --dataset ${dataset} --root-path ${root_path} --exp-name ${exp_name} --detector ${detector} ${parameters} --save-model ${save_path} >> ../logs/${dataset}/${exp_name}/${logs}
#}
#test_model_transductive () {
#  detector=$1
#  dataset=$2
#  root_path="../dataset/darpa_tc3/${dataset}/pygod/"
#  parameters=" --batch-size ${batch_size} --epochs ${ep} --runs ${runs}  --hidden-channels ${HiddenLayer} --num-layers ${n_layers}"
#  load_path="${exp_name}_PyGoD_${detector}_ly${n_layers}_Hly${HiddenLayer}_bs${batch_size}_ep${ep}"
#  logs="testing_PyGoD_${detector}_bs${batch_size}_ep${ep}_ly${n_layers}_Hly${HiddenLayer}"
#  parameters+=$3
#  load_path+=$4
#  logs+=$5
#  logs+="_${date}.txt"
#  load_path+=".model"
#  echo "Parameters are: ${parameters}"
#  echo "load from: ${load_path}"
#  python -B -u ../src/pygod_models_DARPA_Transductive.py --dataset ${dataset} --root-path ${root_path} --exp-name ${exp_name} --detector ${detector} ${parameters} --load-model ${load_path} >> ../logs/${dataset}/${exp_name}/${logs}
#}

train_model_transductive () {
  detector=$1
  host=$2
  ep=$3
  beta=$4
  HiddenLayer=$5
  n_layers=$6
  batch_size=$7
  learningRate=$8
  Con=$9
#  root_path="../dataset/${SourceDataset}/${dataset}/pygod/"
  parameters=" --dropout ${dropout} --batch-size ${batch_size} --epochs ${ep} --runs ${runs} --lr ${learningRate} --hidden-channels ${HiddenLayer} --num-layers ${n_layers} --beta ${beta} --consider-related-nodes "
  if [[ "$Con" == "D" ]]
  then
    parameters+="--dynamic-contamination --dynamic-contamination-val "
  else
    parameters+="--contamination ${Con} "
  fi
  save_path="${detector}_Dr${dropout}_ly${n_layers}_Hly${HiddenLayer}_bs${batch_size}_ep${ep}_beta${beta}_LR${learningRate}_Con${Con}.model"
  logs="${detector}_bs${batch_size}_ep${ep}_Dr${dropout}_ly${n_layers}_Hly${HiddenLayer}_beta${beta}_LR${learningRate}_Con${Con}_${date}.txt"
  mkdir -p ../logs/${host}/${exp_name}
  echo "Parameters are: ${parameters}"
  echo "save to: ${save_path}"
  python -B -u ../src/pygod_models_DARPA_Transductive.py --dataset ${dataset} --host ${host} --root-path ${root_path} --exp-name ${exp_name} --detector ${detector} ${parameters} --save-model ${save_path} >> ../logs/${host}/${exp_name}/${logs}
}

load_model_transductive () {
  detector=$1
  host=$2
  ep=$3
  beta=$4
  HiddenLayer=$5
  n_layers=$6
  batch_size=$7
  learningRate=$8
  Con=$9
  load_model=${10}
#  root_path="../dataset/${SourceDataset}/${dataset}/pygod/"
  parameters=" --dropout ${dropout} --batch-size ${batch_size} --epochs ${ep} --runs ${runs} --lr ${learningRate} --hidden-channels ${HiddenLayer} --num-layers ${n_layers} --beta ${beta} --consider-related-nodes "
  if [[ "$Con" == "D" ]]
  then
    parameters+="--dynamic-contamination --dynamic-contamination-val "
  else
    parameters+="--contamination ${Con} "
  fi
  logs="${detector}_bs${batch_size}_ep${ep}_Dr${dropout}_ly${n_layers}_Hly${HiddenLayer}_beta${beta}_LR${learningRate}_Con${Con}_${date}.txt"
  echo "Parameters are: ${parameters}"
  echo "load from: ${load_path}"
  python -B -u ../src/pygod_models_DARPA_Transductive.py --dataset ${dataset} --host ${host} --root-path ${root_path} --exp-name ${exp_name} --detector ${detector} ${parameters} --load-model ${load_model} >> ../logs/${host}/${exp_name}/${logs}
}

train_multiple_model_transductive () {
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
#  runs=${11}
#  root_path="../dataset/${SourceDataset}/${dataset}/pygod/"
  parameters=" --dropout ${dropout} --batch-size ${batch_size} --epochs ${ep} --runs ${runs} --lr ${learningRate}  --num-layers ${n_layers} --beta ${beta} --multiple-models --dynamic-contamination --dynamic-contamination-val --flexable-rate ${minCon} --max-contamination ${maxCon} --consider-related-nodes  "

  save_path="${detector}_Dr${dropout}_ly${n_layers}_bs${batch_size}_ep${ep}_beta${beta}_LR${learningRate}_Hly${HiddenLayer}"
  logs="${detector}_bs${batch_size}_ep${ep}_Dr${dropout}_ly${n_layers}_beta${beta}_LR${learningRate}_dynConVal${minCon}To${maxCon}_MultipleModels"
  if [[ "$HiddenLayer" == "D" ]]
  then
    parameters+=" --adjust-hidden-channels "
    logs+="_AutoHL"
  else
    parameters+=" --hidden-channels ${HiddenLayer} "
    logs+="_Hly${HiddenLayer}"
  fi
  save_path+="_dynConVal${minCon}To${maxCon}"
  if [ ! -f ${root_path}/results/${exp_name}/${save_path}_anomaly_results_summary.csv ]
  then
    save_path+=".model"
    if [[ "$DEBUG" == "y" ]]
    then
      parameters+=" --debug-subjects 4"
      logs+="_DEBUG"
    fi
    logs+="_${date}.txt"
    mkdir -p ../logs/${host}/${exp_name}
    mkdir -p ../logs/${host}/${exp_name}
    echo "Parameters are: ${parameters}"
    echo "save to: ${save_path}"
    python -B -u ../src/pygod_models_DARPA_Transductive.py --host ${host} --dataset ${dataset} --root-path ${root_path} --exp-name ${exp_name} --detector ${detector} ${parameters} --save-model ${save_path} >> ../logs/${host}/${exp_name}/${logs}
  fi
}

load_multiple_model_transductive () {
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
#  runs=${12}
#  root_path="../dataset/${SourceDataset}/${dataset}/pygod/"
  parameters=" --dropout ${dropout} --batch-size ${batch_size} --epochs ${ep} --runs ${runs} --lr ${learningRate}  --num-layers ${n_layers} --beta ${beta} --multiple-models --dynamic-contamination --dynamic-contamination-val --flexable-rate ${minCon} --max-contamination ${maxCon} --consider-related-nodes "
  logs="Load_${detector}_bs${batch_size}_ep${ep}_Dr${dropout}_ly${n_layers}_beta${beta}_LR${learningRate}_dynConVal${minCon}To${maxCon}_MultipleModels"
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
  python -B -u ../src/pygod_models_DARPA_Transductive.py --host ${host} --dataset ${dataset} --root-path ${root_path} --exp-name ${exp_name} --detector ${detector} ${parameters} --load-model ${load_model}  >> ../logs/${host}/${exp_name}/${logs}
}

generate_llm_investigator_reports () {
  host=$1
  load_model=$2
  inv_logs_name=$3
  load_index=$4
  embed_model=$5
  anomalous=$6
  abnormality=$7
  parameters=" --llm-exp-name ${llm_exp_name}"
  if [[ "$load_index" == "y" ]]
  then
    parameters+=" --load-index"
  fi

  python -B -u ../src/ocrapt_llm_investigator.py --host ${host} --dataset ${dataset} --root-path ${root_path} --exp-name ${exp_name} --GNN-model-name ${load_model} --inv-exp-name ${inv_logs_name} --llm-embedding-model ${embed_model} --abnormality-level ${abnormality} --anomalous ${anomalous} ${parameters} >> ../logs/${host}/${exp_name}/llm_investigator_${llm_exp_name}_${date}.txt
}

Investigate_raised_alarms () {
  host=$1
  max_edges=$2
  k_percent=$3
  HiddenLayer=$4
  model_path=$5
  n_hop=$6
  n_layers=$7
  beta=$8
  expand=$9
  CorrelateOnce=${10}
#  runs=${11}
#  root_path="../dataset/${SourceDataset}/${dataset}/pygod/"
  investigation_parameters=" --min-nodes 3 --number-of-hops ${n_hop} --runs ${runs} --remove-duplicated-subgraph --get-node-attrs --node-emb-per-subgraph --consider-related-nodes "
  logs_name="expand_${n_hop}_hop_${expand}2hop_MaxEdges${max_edges}_ly${n_layers}_beta${beta}"
  if [[ "$CorrelateOnce" == "y" ]]
  then
    investigation_parameters+=" --correlate-anomalous-once "
    logs_name+="_CorrelateOnce"
  else
    logs_name+="_NoCorrelateOnce"
  fi
  if [[ "$HiddenLayer" == "D" ]]
  then
    logs_name+="_AutoHL"
  else
    investigation_parameters+=" --node-emb-size ${HiddenLayer} --tensor-neurons ${HiddenLayer} "
    logs_name+="_Hly${HiddenLayer}"
  fi
  investigation_parameters+=" --max-edges ${max_edges}"
  if [[ "$k_percent" == "N" ]]
  then
    investigation_parameters+=" --top-k ${top_k}"
    logs_name+="_K${top_k}"
  else
    investigation_parameters+=" --top-k-percent ${k_percent}"
    logs_name+="_K${k_percent}"
  fi
  investigation_parameters+=" --expand-2-hop ${expand}"
  python -B -u ../src/investigate_nodes.py --host ${host} --dataset ${dataset} --root-path ${root_path} --exp-name ${exp_name} --model ${model_path}  --construct-from-anomaly-subgraph  ${investigation_parameters}  --inv-exp-name ${logs_name} ${more_param} >> ../logs/${host}/${exp_name}/investigating_${logs_name}_${date}.txt
  echo ${logs_name}
}

# default training parameters
batch_size=0
decideHiddenLayer="N"
HiddenLayer=32
n_layers=3
beta=0.5
#learningRate=0.004
learningRate=0.005
ep=100
dropout=0
minCon=0.001
maxCon=0.05
detector=OCRGCN


# default investigation parameters
DEBUG="N"
max_edges=10000
n_hop=1
expand="no"
CorrelateOnce="y"
top_k=15
k_percent="N"
#host=SimulatedWS12
#root_path="../dataset/${SourceDataset}/${host}/OCR_APT/"
#HiddenLayer="D"
#learningRate=0.001
#echo "train Multiple models ${host} "
#train_multiple_model_transductive ${detector} ${host} ${ep} ${beta} ${HiddenLayer} ${n_layers} ${batch_size} ${learningRate} ${minCon} ${maxCon}
#load_model="${detector}_Dr${dropout}_ly${n_layers}_bs${batch_size}_ep${ep}_beta${beta}_LR${learningRate}_Hly${HiddenLayer}_dynConVal${minCon}To${maxCon}.model"
#Investigate_raised_alarms ${host} ${max_edges} ${k_percent} ${HiddenLayer} ${load_model} ${n_hop} ${n_layers} ${beta}
#
#host=SimulatedW10
#root_path="../dataset/${SourceDataset}/${host}/OCR_APT/"
#HiddenLayer="D"
#for learningRate in {0.0005,0.001,0.005};do
#  root_path="../dataset/${SourceDataset}/${host}/OCR_APT/"
#  echo "train Multiple models ${host} "
#  train_multiple_model_transductive ${detector} ${host} ${ep} ${beta} ${HiddenLayer} ${n_layers} ${batch_size} ${learningRate} ${minCon} ${maxCon}
#  load_model="${detector}_Dr${dropout}_ly${n_layers}_bs${batch_size}_ep${ep}_beta${beta}_LR${learningRate}_Hly${HiddenLayer}_dynConVal${minCon}To${maxCon}.model"
#  Investigate_raised_alarms ${host} ${max_edges} ${k_percent} ${HiddenLayer} ${load_model} ${n_hop} ${n_layers} ${beta}
#done
#
#host=SimulatedUbuntu
#root_path="../dataset/${SourceDataset}/${host}/OCR_APT/"
#HiddenLayer="D"
#for learningRate in {0.0005,0.001,0.005};do
#  root_path="../dataset/${SourceDataset}/${host}/OCR_APT/"
#  echo "train Multiple models ${host} "
#  train_multiple_model_transductive ${detector} ${host} ${ep} ${beta} ${HiddenLayer} ${n_layers} ${batch_size} ${learningRate} ${minCon} ${maxCon}
#  load_model="${detector}_Dr${dropout}_ly${n_layers}_bs${batch_size}_ep${ep}_beta${beta}_LR${learningRate}_Hly${HiddenLayer}_dynConVal${minCon}To${maxCon}.model"
#  Investigate_raised_alarms ${host} ${max_edges} ${k_percent} ${HiddenLayer} ${load_model} ${n_hop} ${n_layers} ${beta}
#done
#
#

#echo "train Multiple models ${host} "
#train_multiple_model_transductive ${detector} ${host} ${ep} ${beta} ${HiddenLayer} ${n_layers} ${batch_size} ${learningRate} ${minCon} ${maxCon}
#load_model="${detector}_Dr${dropout}_ly${n_layers}_bs${batch_size}_ep${ep}_beta${beta}_LR${learningRate}_Hly${HiddenLayer}_dynConVal${minCon}To${maxCon}.model"
#Investigate_raised_alarms ${host} ${max_edges} ${k_percent} ${HiddenLayer} ${load_model} ${n_hop} ${n_layers} ${beta}


## Slightly optimize subgraph construction ###




#dataset="optc"
#SourceDataset="darpa_optc"
#for host in {SysClient0051,SysClient0201};do
#  root_path="../dataset/${SourceDataset}/${host}/1dayTraining_withDirection_uuid/"
#  echo "train Multiple models ${host} "
##    train_multiple_model_transductive ${detector} ${host} ${ep} ${beta} ${HiddenLayer} ${n_layers} ${batch_size} ${learningRate} ${minCon} ${maxCon}
#  load_model="${detector}_Dr${dropout}_ly${n_layers}_bs${batch_size}_ep${ep}_beta${beta}_LR${learningRate}_Hly${HiddenLayer}_dynConVal${minCon}To${maxCon}.model"
#  Investigate_raised_alarms ${host} ${max_edges} ${k_percent} ${HiddenLayer} ${load_model} ${n_hop} ${n_layers} ${beta} ${expand}
#done
#
#dataset="tc3"
#SourceDataset="darpa_tc3"
#host="theia"
#root_path="../dataset/${SourceDataset}/${host}/Adjusted_UUID/"
#echo "train Multiple models ${host} "
##    train_multiple_model_transductive ${detector} ${host} ${ep} ${beta} ${HiddenLayer} ${n_layers} ${batch_size} ${learningRate} ${minCon} ${maxCon}
#load_model="${detector}_Dr${dropout}_ly${n_layers}_bs${batch_size}_ep${ep}_beta${beta}_LR${learningRate}_Hly${HiddenLayer}_dynConVal${minCon}To${maxCon}.model"
#Investigate_raised_alarms ${host} ${max_edges} ${k_percent} ${HiddenLayer} ${load_model} ${n_hop} ${n_layers} ${beta} ${expand}

#k_percent=0.03
#for max_edges in {10000,8000,7000,9000};do
#  dataset="tc3"
#  SourceDataset="darpa_tc3"
#  host="theia"
#  root_path="../dataset/${SourceDataset}/${host}/Adjusted_UUID/"
#  echo "train Multiple models ${host} "
#  #    train_multiple_model_transductive ${detector} ${host} ${ep} ${beta} ${HiddenLayer} ${n_layers} ${batch_size} ${learningRate} ${minCon} ${maxCon}
#  load_model="${detector}_Dr${dropout}_ly${n_layers}_bs${batch_size}_ep${ep}_beta${beta}_LR${learningRate}_Hly${HiddenLayer}_dynConVal${minCon}To${maxCon}.model"
#  Investigate_raised_alarms ${host} ${max_edges} ${k_percent} ${HiddenLayer} ${load_model} ${n_hop} ${n_layers} ${beta} ${expand}
#done
#
#for k_percent in {0.01,0.04,0.02};do
#  dataset="optc"
#  SourceDataset="darpa_optc"
#  for host in {SysClient0051,SysClient0201};do
#    root_path="../dataset/${SourceDataset}/${host}/1dayTraining_withDirection_uuid/"
#    echo "train Multiple models ${host} "
#  #    train_multiple_model_transductive ${detector} ${host} ${ep} ${beta} ${HiddenLayer} ${n_layers} ${batch_size} ${learningRate} ${minCon} ${maxCon}
#    load_model="${detector}_Dr${dropout}_ly${n_layers}_bs${batch_size}_ep${ep}_beta${beta}_LR${learningRate}_Hly${HiddenLayer}_dynConVal${minCon}To${maxCon}.model"
#    Investigate_raised_alarms ${host} ${max_edges} ${k_percent} ${HiddenLayer} ${load_model} ${n_hop} ${n_layers} ${beta} ${expand}
#  done
#
#  dataset="tc3"
#  SourceDataset="darpa_tc3"
#  host="theia"
#  root_path="../dataset/${SourceDataset}/${host}/Adjusted_UUID/"
#  echo "train Multiple models ${host} "
#  #    train_multiple_model_transductive ${detector} ${host} ${ep} ${beta} ${HiddenLayer} ${n_layers} ${batch_size} ${learningRate} ${minCon} ${maxCon}
#  load_model="${detector}_Dr${dropout}_ly${n_layers}_bs${batch_size}_ep${ep}_beta${beta}_LR${learningRate}_Hly${HiddenLayer}_dynConVal${minCon}To${maxCon}.model"
#  Investigate_raised_alarms ${host} ${max_edges} ${k_percent} ${HiddenLayer} ${load_model} ${n_hop} ${n_layers} ${beta} ${expand}
#done


##################### re_investigate to confirm last status -  #################################
#exp_name="Adjusted_UUID_normL2Actions_MinMax1IdleTimeSec_08_24"
#HiddenLayer="D"
#learningRate=0.005
#max_edges=6000
#n_layers=3
#k_percent=0.05
#
#dataset="optc"
#SourceDataset="darpa_optc"
#host=SysClient0501
#root_path="../dataset/${SourceDataset}/${host}/1dayTraining_withDirection_uuid/"
#echo "train Multiple models ${host} "
##    train_multiple_model_transductive ${detector} ${host} ${ep} ${beta} ${HiddenLayer} ${n_layers} ${batch_size} ${learningRate} ${minCon} ${maxCon}
#load_model="${detector}_Dr${dropout}_ly${n_layers}_bs${batch_size}_ep${ep}_beta${beta}_LR${learningRate}_Hly${HiddenLayer}_dynConVal${minCon}To${maxCon}.model"
#Investigate_raised_alarms ${host} ${max_edges} ${k_percent} ${HiddenLayer} ${load_model} ${n_hop} ${n_layers} ${beta} ${expand}
#
#
#dataset="tc3"
#SourceDataset="darpa_tc3"
#for host in {cadets,trace};do
#  root_path="../dataset/${SourceDataset}/${host}/Adjusted_UUID/"
#  echo "train Multiple models ${host} "
##    train_multiple_model_transductive ${detector} ${host} ${ep} ${beta} ${HiddenLayer} ${n_layers} ${batch_size} ${learningRate} ${minCon} ${maxCon}
#  load_model="${detector}_Dr${dropout}_ly${n_layers}_bs${batch_size}_ep${ep}_beta${beta}_LR${learningRate}_Hly${HiddenLayer}_dynConVal${minCon}To${maxCon}.model"
#  Investigate_raised_alarms ${host} ${max_edges} ${k_percent} ${HiddenLayer} ${load_model} ${n_hop} ${n_layers} ${beta} ${expand}
#done
#
#dataset="nodlink"
#SourceDataset="nodlink"
#HiddenLayer=32
#for host in {SimulatedWS12,SimulatedW10,SimulatedUbuntu};do
#  root_path="../dataset/${SourceDataset}/${host}/OCR_APT/"
#  echo "train Multiple models ${host} "
##  train_multiple_model_transductive ${detector} ${host} ${ep} ${beta} ${HiddenLayer} ${n_layers} ${batch_size} ${learningRate} ${minCon} ${maxCon}
#  load_model="${detector}_Dr${dropout}_ly${n_layers}_bs${batch_size}_ep${ep}_beta${beta}_LR${learningRate}_Hly${HiddenLayer}_dynConVal${minCon}To${maxCon}.model"
#  Investigate_raised_alarms ${host} ${max_edges} ${k_percent} ${HiddenLayer} ${load_model} ${n_hop} ${n_layers} ${beta} ${expand}
#done

#learningRate=0.005
#n_layers=3
#expand="upto"
#HiddenLayer="D"
#dataset="tc3"
#SourceDataset="darpa_tc3"
#host="cadets"
#root_path="../dataset/${SourceDataset}/${host}/Adjusted_UUID/"


#echo "train Multiple models ${host} "
#load_model="${detector}_Dr${dropout}_ly${n_layers}_bs${batch_size}_ep${ep}_beta${beta}_LR${learningRate}_Hly${HiddenLayer}_dynConVal${minCon}To${maxCon}.model"
#Investigate_raised_alarms ${host} ${max_edges} ${k_percent} ${HiddenLayer} ${load_model} ${n_hop} ${n_layers} ${beta} ${expand}
#
#max_edges=10000
#echo "train Multiple models ${host} "
#load_model="${detector}_Dr${dropout}_ly${n_layers}_bs${batch_size}_ep${ep}_beta${beta}_LR${learningRate}_Hly${HiddenLayer}_dynConVal${minCon}To${maxCon}.model"
#Investigate_raised_alarms ${host} ${max_edges} ${k_percent} ${HiddenLayer} ${load_model} ${n_hop} ${n_layers} ${beta} ${expand}

#for max_edges in {6000,10000,8000};do
#  for k_percent in {0.03,0.01,0.05};do
#    for expand in {upto,always,no};do
#      echo "train Multiple models ${host} "
#      load_model="${detector}_Dr${dropout}_ly${n_layers}_bs${batch_size}_ep${ep}_beta${beta}_LR${learningRate}_Hly${HiddenLayer}_dynConVal${minCon}To${maxCon}.model"
#      Investigate_raised_alarms ${host} ${max_edges} ${k_percent} ${HiddenLayer} ${load_model} ${n_hop} ${n_layers} ${beta} ${expand}
#    done
#  done
#done

#for expand in {upto,no,always};do
#  for host in {cadets,trace,theia};do
#    root_path="../dataset/${SourceDataset}/${host}/Adjusted_UUID/"
#    echo "train Multiple models ${host} "
#  #    train_multiple_model_transductive ${detector} ${host} ${ep} ${beta} ${HiddenLayer} ${n_layers} ${batch_size} ${learningRate} ${minCon} ${maxCon}
#    load_model="${detector}_Dr${dropout}_ly${n_layers}_bs${batch_size}_ep${ep}_beta${beta}_LR${learningRate}_Hly${HiddenLayer}_dynConVal${minCon}To${maxCon}.model"
#    Investigate_raised_alarms ${host} ${max_edges} ${k_percent} ${HiddenLayer} ${load_model} ${n_hop} ${n_layers} ${beta} ${expand}
#  done
#done

exp_name="Adjusted_UUID_normL2Actions_MinMax1IdleTimeSec_08_24"
top_k=15
max_edges=5000
expand="no"
k_percent="N"
#

### Optimize LLM
#llm_exp_name="OCRAPT_Optimize_Prompt"
#inv_logs_name="expand_1_hop_no2hop_MaxEdges5000_ly3_beta0.5_CorrelateOnce_AutoHL_K15"
#load_model="OCRGCN_Dr0_ly3_bs0_ep100_beta0.5_LR0.005_HlyD_dynConVal0.001To0.05.model"
#HiddenLayer="D"
#dataset="optc"
#SourceDataset="darpa_optc"
#host="SysClient0051"
#root_path="../dataset/${SourceDataset}/${host}/1dayTraining_withDirection_uuid/"
#load_index="y"
#generate_llm_investigator_reports ${host} ${load_model} ${inv_logs_name} ${load_index}

llm_exp_name="OCRAPT_fullAuto_MultiQ_ContextStages_AugCompExtHighest_Emb3lg_AnomSubj"
embed_model="text-embedding-3-large"
#embed_model="text-embedding-ada-002"
load_index="N"
anomalous="sub"
#anomalous="subobj"
abnormality="Moderate"

HiddenLayer="D"
dataset="optc"
SourceDataset="darpa_optc"
inv_logs_name="expand_1_hop_no2hop_MaxEdges5000_ly3_beta0.5_CorrelateOnce_AutoHL_K15"

#host="SysClient0501"
#root_path="../dataset/${SourceDataset}/${host}/1dayTraining_withDirection_uuid/"
#echo "Multiple models ${host} "
#load_model="${detector}_Dr${dropout}_ly${n_layers}_bs${batch_size}_ep${ep}_beta${beta}_LR${learningRate}_Hly${HiddenLayer}_dynConVal${minCon}To${maxCon}.model"
#generate_llm_investigator_reports ${host} ${load_model} ${inv_logs_name} ${load_index} ${embed_model} ${anomalous} ${abnormality}


#for host in {SysClient0201,SysClient0051,SysClient0501};do
#  root_path="../dataset/${SourceDataset}/${host}/1dayTraining_withDirection_uuid/"
#  echo "Multiple models ${host} "
#  #    train_multiple_model_transductive ${detector} ${host} ${ep} ${beta} ${HiddenLayer} ${n_layers} ${batch_size} ${learningRate} ${minCon} ${maxCon}
#  load_model="${detector}_Dr${dropout}_ly${n_layers}_bs${batch_size}_ep${ep}_beta${beta}_LR${learningRate}_Hly${HiddenLayer}_dynConVal${minCon}To${maxCon}.model"
##  load_multiple_model_transductive ${detector} ${host} ${ep} ${beta} ${HiddenLayer} ${n_layers} ${batch_size} ${learningRate} ${minCon} ${maxCon} ${load_model}
##  inv_logs_name=$(Investigate_raised_alarms ${host} ${max_edges} ${k_percent} ${HiddenLayer} ${load_model} ${n_hop} ${n_layers} ${beta} ${expand} ${CorrelateOnce})
#  generate_llm_investigator_reports ${host} ${load_model} ${inv_logs_name} ${load_index} ${embed_model} ${anomalous} ${abnormality}
#  sleep 1m
#done


#HiddenLayer="D"
#dataset="tc3"
#SourceDataset="darpa_tc3"
#for host in {trace,theia,cadets};do
#  root_path="../dataset/${SourceDataset}/${host}/Adjusted_UUID/"
#  echo "Multiple models ${host} "
##    train_multiple_model_transductive ${detector} ${host} ${ep} ${beta} ${HiddenLayer} ${n_layers} ${batch_size} ${learningRate} ${minCon} ${maxCon}
#  load_model="${detector}_Dr${dropout}_ly${n_layers}_bs${batch_size}_ep${ep}_beta${beta}_LR${learningRate}_Hly${HiddenLayer}_dynConVal${minCon}To${maxCon}.model"
##  load_multiple_model_transductive ${detector} ${host} ${ep} ${beta} ${HiddenLayer} ${n_layers} ${batch_size} ${learningRate} ${minCon} ${maxCon} ${load_model}
##  inv_logs_name=$(Investigate_raised_alarms ${host} ${max_edges} ${k_percent} ${HiddenLayer} ${load_model} ${n_hop} ${n_layers} ${beta} ${expand} ${CorrelateOnce})
#  generate_llm_investigator_reports ${host} ${load_model} ${inv_logs_name} ${load_index} ${embed_model} ${anomalous} ${abnormality}
#  sleep 1m
#done
#
#abnormality="Minor"
#host="cadets"
#root_path="../dataset/${SourceDataset}/${host}/Adjusted_UUID/"
#echo "Multiple models ${host} "
#load_model="${detector}_Dr${dropout}_ly${n_layers}_bs${batch_size}_ep${ep}_beta${beta}_LR${learningRate}_Hly${HiddenLayer}_dynConVal${minCon}To${maxCon}.model"
#generate_llm_investigator_reports ${host} ${load_model} ${inv_logs_name} ${load_index} ${embed_model} ${anomalous} ${abnormality}


HiddenLayer=32
inv_logs_name="expand_1_hop_no2hop_MaxEdges5000_ly3_beta0.5_CorrelateOnce_Hly32_K15"
dataset="nodlink"
SourceDataset="nodlink"
for host in {SimulatedWS12,SimulatedUbuntu,SimulatedW10};do
  root_path="../dataset/${SourceDataset}/${host}/OCR_APT/"
  echo "train Multiple models ${host} "
#  train_multiple_model_transductive ${detector} ${host} ${ep} ${beta} ${HiddenLayer} ${n_layers} ${batch_size} ${learningRate} ${minCon} ${maxCon}
  load_model="${detector}_Dr${dropout}_ly${n_layers}_bs${batch_size}_ep${ep}_beta${beta}_LR${learningRate}_Hly${HiddenLayer}_dynConVal${minCon}To${maxCon}.model"
#  load_multiple_model_transductive ${detector} ${host} ${ep} ${beta} ${HiddenLayer} ${n_layers} ${batch_size} ${learningRate} ${minCon} ${maxCon} ${load_model}
#  inv_logs_name=$(Investigate_raised_alarms ${host} ${max_edges} ${k_percent} ${HiddenLayer} ${load_model} ${n_hop} ${n_layers} ${beta} ${expand} ${CorrelateOnce})
  generate_llm_investigator_reports ${host} ${load_model} ${inv_logs_name} ${load_index} ${embed_model} ${anomalous} ${abnormality}
  sleep 1m
done


###########################################################################
## System Analysis ##
#exp_name="Adjusted_UUID_normL2Actions_MinMax1IdleTimeSec_08_24"
#top_k=15
#max_edges=5000
#expand="no"
#k_percent="N"
#
#HiddenLayer=32
#dataset="nodlink"
#SourceDataset="nodlink"
#detector=OCRGCN
#Con="D"
#for host in {SimulatedWS12,SimulatedUbuntu,SimulatedW10};do
#  echo "train one model for host ${host} "
#  root_path="../dataset/${SourceDataset}/${host}/OCR_APT/"
##  train_model_transductive ${detector} ${host} ${ep} ${beta} ${HiddenLayer} ${n_layers} ${batch_size} ${learningRate} ${Con}
#  load_model="${detector}_Dr${dropout}_ly${n_layers}_Hly${HiddenLayer}_bs${batch_size}_ep${ep}_beta${beta}_LR${learningRate}_Con${Con}.model"
##  load_model_transductive ${detector} ${host} ${ep} ${beta} ${HiddenLayer} ${n_layers} ${batch_size} ${learningRate} ${Con} ${load_model}
#  Investigate_raised_alarms ${host} ${max_edges} ${k_percent} ${HiddenLayer} ${load_model} ${n_hop} ${n_layers} ${beta} ${expand} ${CorrelateOnce}
#done
#
#for detector in {OCRGCN,OCGNN,GAE,CoLA,AnomalyDAE,CONAD};do
#  for host in {SimulatedWS12,SimulatedUbuntu,SimulatedW10};do
#    root_path="../dataset/${SourceDataset}/${host}/OCR_APT/"
#    echo "train Multiple models ${detector} for host ${host} "
##    train_multiple_model_transductive ${detector} ${host} ${ep} ${beta} ${HiddenLayer} ${n_layers} ${batch_size} ${learningRate} ${minCon} ${maxCon}
#    load_model="${detector}_Dr${dropout}_ly${n_layers}_bs${batch_size}_ep${ep}_beta${beta}_LR${learningRate}_Hly${HiddenLayer}_dynConVal${minCon}To${maxCon}.model"
##    load_multiple_model_transductive ${detector} ${host} ${ep} ${beta} ${HiddenLayer} ${n_layers} ${batch_size} ${learningRate} ${minCon} ${maxCon} ${load_model}
#    Investigate_raised_alarms ${host} ${max_edges} ${k_percent} ${HiddenLayer} ${load_model} ${n_hop} ${n_layers} ${beta} ${expand} ${CorrelateOnce}
#  done
#done

#exp_name="Adjusted_UUID_EdgeDistFeat"
#HiddenLayer=32
#dataset="nodlink"
#SourceDataset="nodlink"
#for host in {SimulatedWS12,SimulatedUbuntu,SimulatedW10};do
#  root_path="../dataset/${SourceDataset}/${host}/OCR_APT/"
#  echo "train Multiple models ${host} "
#  load_model="OCRGCN_Dr0_ly3_bs0_ep100_LR0.005_beta0.5_Hly32_dynConVal0.001To0.05.model"
#  load_multiple_model_transductive ${detector} ${host} ${ep} ${beta} ${HiddenLayer} ${n_layers} ${batch_size} ${learningRate} ${minCon} ${maxCon} ${load_model}
#  Investigate_raised_alarms ${host} ${max_edges} ${k_percent} ${HiddenLayer} ${load_model} ${n_hop} ${n_layers} ${beta} ${expand} ${CorrelateOnce}
#done



#HiddenLayer=32
#dataset="nodlink"
#SourceDataset="nodlink"
#for host in {SimulatedWS12,SimulatedUbuntu,SimulatedW10};do
#  root_path="../dataset/${SourceDataset}/${host}/OCR_APT/"
#  echo "train Multiple models ${host} "
#  train_multiple_model_transductive ${detector} ${host} ${ep} ${beta} ${HiddenLayer} ${n_layers} ${batch_size} ${learningRate} ${minCon} ${maxCon}
#  load_model="${detector}_Dr${dropout}_ly${n_layers}_bs${batch_size}_ep${ep}_beta${beta}_LR${learningRate}_Hly${HiddenLayer}_dynConVal${minCon}To${maxCon}.model"
#  Investigate_raised_alarms ${host} ${max_edges} ${k_percent} ${HiddenLayer} ${load_model} ${n_hop} ${n_layers} ${beta} ${expand} ${CorrelateOnce}
#done
#
#HiddenLayer=32
#dataset="optc"
#SourceDataset="darpa_optc"
#for host in {SysClient0051,SysClient0201,SysClient0501};do
#  root_path="../dataset/${SourceDataset}/${host}/1dayTraining_withDirection_uuid/"
#  echo "train Multiple models ${host} "
#  train_multiple_model_transductive ${detector} ${host} ${ep} ${beta} ${HiddenLayer} ${n_layers} ${batch_size} ${learningRate} ${minCon} ${maxCon}
#  load_model="${detector}_Dr${dropout}_ly${n_layers}_bs${batch_size}_ep${ep}_beta${beta}_LR${learningRate}_Hly${HiddenLayer}_dynConVal${minCon}To${maxCon}.model"
#  Investigate_raised_alarms ${host} ${max_edges} ${k_percent} ${HiddenLayer} ${load_model} ${n_hop} ${n_layers} ${beta} ${expand} ${CorrelateOnce}
#done




echo "Done Training"
