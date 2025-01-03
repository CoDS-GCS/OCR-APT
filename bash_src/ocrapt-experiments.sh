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

read -p "Enter the working directory folder (experiments): " working_dir

read -p "Enter the detector model (Main model: OCRGCN) (Other models: OCGNN, AnomalyDAE, GAE, CONAD, CoLA): " detector
read -p "Do you want to convert to RDF (y/N): " ToRDF
read -p "Do you want to preprocess to PyG (y/N): " preprocess
read -p "Do you want to investigate anomalies nodes (y/N): " investigate
read -p "Do you want to use the default parameters (Y/n): " default


if [[ "$default" == "n" ]]
then
  read -p "Enter the batch size: " batch_size
  read -p "Enter the number of epochs: " ep
  read -p "Enter the number of runs: " runs
  read -p "Do you want to decide hidden layer based on the input feature size (y/N): " decideHiddenLayer
  if [[ "$decideHiddenLayer" == "y" ]]
  then
    echo "Hidden layer will be decided automatically"
  else
    read -p "Enter the size of hidden layer (32): " HiddenLayer
  fi
  read -p "Enter the number of layers (2): " n_layers
  read -p "Enter the learning rate (0.004): " learningRate
  read -p "Enter Beta (0.5): " Beta
  read -p "Do you want to train with dynamic contamination (Y/n): " dynamicContamination
  if [[ "$dynamicContamination" == "n" ]]
  then
    read -p "Enter the proportion of outliers in the dataset (0.1): " contamination
  else
    read -p "Enter the Max contamination (0.5): " MaxContamination
    read -p "Enter the Min contamination (0.005): " MinContamination
  fi
  read -p "Do you want to train with multiple models per node type (y/N): " multipleModels
else
  echo "Using Default Parameters"
fi

if [[ "$ToRDF" == "y" ]]
then
  read -p "Do you want to get node attributes from NetworkX graph(y/N): ": node_attrs_graph_nx
  rdf_parameter=""
  read -p "Do you want to read from NetworkX graph(y/N): " graph_nx
  if [[ "$graph_nx" == "y" ]]
  then
    rdf_parameter+=" --graph-nx"
  fi
  read -p "Do you want to output graphs in RDFS format(y/N): " rdfs
  if [[ "$rdfs" == "y" ]]
  then
    rdf_parameter+=" --rdfs"
  fi
  read -p "Do you want to adjust UUID to avoid node duplication (y/N): " adjustUUID
  if [[ "$adjustUUID" == "y" ]]
  then
    rdf_parameter+=" --adjust-uuid"
  fi
  read -p "Enter the minimum nodes per node type (default: 10): ": min_node_type
  rdf_parameter+=" --min-node-representation ${min_node_type}"
  if [[ "$dataset" == "optc" ]]
  then
    read -p "Do you want to keep only ['FLOW', 'PROCESS', 'MODULE', 'FILE'] node types? (y/N): " FilterNodeType
    if [[ "$FilterNodeType" == "y" ]]
    then
      rdf_parameter+=" --filter-node-type"
    fi
  fi
else
  echo "Skip converting to RDF"
fi

if [[ "$preprocess" == "y" ]]
then
  read -p "Do you want to take validation from training set (y/N): ": TrainingValid
  if [[ "$TrainingValid" == "y" ]]
  then
    pyg_parameters+=" --training-valid"
  fi
  read -p "Do you want to extract features from timestamps (y/N): ": timestamps_features
  if [[ "$timestamps_features" == "y" ]]
  then
    pyg_parameters+=" --get-timestamps-features "
    read -p "Do you want to extract idle time (y/N): ": IdleTime
    if [[ "$IdleTime" == "y" ]]
    then
      pyg_parameters+=" --get-idle-time"
    fi
  fi
  read -p "Do you want to normalize features (y/N): ": Normalize
  if [[ "$Normalize" == "y" ]]
  then
    pyg_parameters+=" --normalize-features"
  fi
else
  echo "Skip converting to PyG"
fi

if [[ "$investigate" == "y" ]]
then
  read -p "Enter the number of hops while investigation: " n_hops_investigate
  investigation_parameters=" --number-of-hops ${n_hops_investigate} --correlate-anomalous-once --remove-duplicated-subgraph --get-node-attrs"
  logs_name="_controlled_in_${n_hops_investigate}_hop"
  read -p "Enter the maximum edges per subgraph (default: 5000)( 0 for no edge limit): " max_edges
  investigation_parameters+=" --max-edges ${max_edges}"
  logs_name+="_MaxEdges${max_edges}"
  read -p "Enter the top K seed nodes per type to be investigated (default: 15): " k_nodes
  investigation_parameters+=" --top-k ${k_nodes}"
  logs_name+="_K${k_nodes}"
  read -p "Do you want to draw constructed subgraphs (y/N):" draw_subgraphs
  if [[ "$draw_subgraphs" == "y" ]]
  then
    investigation_parameters+=" --draw-subgraphs"
  fi
  read -p "Enter the LLM investigation experiment name: " llm_exp_name
  read -p "Enter the subgraph abnormality level (default: Moderate): " abnormality
  read -p "Do you want to load previously indexed subgraphs (y/N): " load_index
  llm_parameters=" --llm-exp-name ${llm_exp_name} --abnormality-level ${abnormality}  "
  if [[ "$load_index" == "y" ]]
  then
    llm_parameters+=" --load-index"
  fi
fi


if [[ "$default" == "n" ]]
then
  dropout=0
  parameters=" --batch-size ${batch_size} --epochs ${ep} --runs ${runs}   --num-layers ${n_layers} --lr ${learningRate} --beta ${Beta} "
  save_path="${detector}_Dr${dropout}_ly${n_layers}_bs${batch_size}_ep${ep}_LR${learningRate}_beta${Beta}"
  logs="training_PyGoD_${detector}_bs${batch_size}_ep${ep}_ly${n_layers}_LR${learningRate}_beta${Beta}"
  if [[ "$decideHiddenLayer" == "y" ]]
  then
    save_path=+"_AutoHL"
    parameters+=" --adjust-hidden-channels "
    logs+="_AutoHL"
  else
    save_path+="_Hly${HiddenLayer}"
    parameters+=" --hidden-channels ${HiddenLayer} "
    logs+="_Hly${HiddenLayer}"
  fi
  if [[ "$dynamicContamination" == "n" ]]
  then
    parameters+=" --contamination ${contamination}"
    logs+="_con${contamination}"
    save_path+="_con${contamination}"
  else
    parameters+=" --dynamic-contamination --flexable-rate ${MinContamination} --max-contamination ${MaxContamination}"
    logs+="_dynConVal${MinContamination}To${MaxContamination}"
    save_path+="_dynConVal${MinContamination}To${MaxContamination}"
  fi
  if [[ "$multipleModels" == "y" ]]
  then
    parameters+=" --multiple-models"
    logs+="_MultipleModels"
  fi
  logs+="_${date}.txt"
  model_path=${save_path}
  save_path+=".model"
else
  save_path="${detector}_baseLine.model"
  logs="training_PyGoD_${detector}_baseLine_${date}.txt"
fi


execute_OCR_APT () {
  host=${1}
  root_path=../dataset/${SourceDataset}/${host}/${working_dir}/
  mkdir -p ../logs/${host}/${exp_name}
  source_graph="${host}"
  nx_source_graph="complete_${host}_pg"
  if [[ "$ToRDF" == "y" ]]
  then
    python -B -u ../src/transform_to_RDF.py --dataset ${dataset} --host ${host} --root-path ${root_path} --source-graph ${source_graph} ${rdf_parameter} >> ../logs/${host}/${exp_name}/DARPA_to_RDF_${date}.txt
    if [[ "$node_attrs_graph_nx" == "y" ]]
    then
      if [[ "$adjustUUID" == "y" ]]
      then
        node_attr_param=" --adjust-uuid "
      fi
      python -B -u ../src/get_node_attributes.py ${node_attr_param} --host ${host} --root-path ${root_path} --source-graph ${source_graph} --source-graph-nx ${nx_source_graph} >> ../logs/${host}/${exp_name}/DARPA_to_RDF_${date}.txt
    fi
    echo "Done converting to RDF"
  fi

  if [[ "$preprocess" == "y" ]]
  then
    python -u -B ../src/encode_to_PyG.py --dataset ${dataset} --host ${host} --root-path ${root_path} --exp-name ${exp_name} --source-graph ${source_graph} ${pyg_parameters} >> ../logs/${host}/${exp_name}/RDF_to_PyG_${date}.txt
    echo "Done converting to PyG"
  fi

  echo "start training"

  python -B -u ../src/train_gnn_models.py --dataset ${dataset} --host ${host} --root-path ${root_path} --exp-name ${exp_name} --detector ${detector} ${parameters} --save-model ${save_path} >> ../logs/${host}/${exp_name}/${logs}

  echo "Done training"

  if [[ "$investigate" == "y" ]]
  then
    python -B -u ../src/detect_anomalous_subgraphs.py --dataset ${dataset} --host ${host} --root-path ${root_path} --exp-name ${exp_name} --model ${model_path} --construct-from-anomaly-subgraph ${investigation_parameters} --min-nodes 3 --inv-exp-name ${logs_name} >> ../logs/${host}/${exp_name}/investigating${logs_name}_${date}.txt
    python -B -u ../src/ocrapt_llm_investigator.py --dataset ${dataset} --host ${host} --root-path ${root_path} --exp-name ${exp_name} --GNN-model-name ${model_path} --inv-exp-name ${logs_name} --anomalous "sub" --llm-embedding-model "text-embedding-3-large" ${llm_parameters} >> ../logs/${host}/${exp_name}/llm_investigator_${llm_exp_name}_${date}.txt
  fi
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
