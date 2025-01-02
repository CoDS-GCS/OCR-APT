#!/bin/sh
date=$(date +'%d_%m_%Y')
echo "Available datasets ( tc3 , optc , nodlink )"
read -p "Enter the dataset name: " dataset

echo "Available hosts (ALL_TC3, cadets, theia, trace, fivedirections, ALL_OPTC, SysClient0051 , SysClient0501 , SysClient0201, ALL_NodLink, SimulatedUbuntu, SimulatedW10, SimulatedWS12)"
read -p "Enter the host name: " host
#read -p "Are you processing DARPA OpTC (y/N): ": Is_DARPA_OPTC
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

#read -p "Enter the root folder path (../dataset/darpa_tc3/cadets/pygod/): " root_path
read -p "Enter the main directory folder (pygod): " working_dir

read -p "Enter the detector model (OCGNN, DOMINANT, AnomalyDAE, GAE, CONAD, CoLA, GUIDE, OCRGCN): " detector
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

#  read -p "Do you want to test with ensembled models (y/N): " ensembleModels
#  read -p "Do you want to initialize randomly (y/N): " Random
  read -p "Do you want to train with multiple models per node type (y/N): " multipleModels
#  if [[ "$multipleModels" == "y" ]]
#  then
#    read -p "Do you want to avoid nan/0 features per model (y/N): " AvoidNANFeatures
#    read -p "Do you want to perform features reduction by PCA (y/N): " FeaturePCA
#  fi
#  read -p "Do you want to visualize training (Y/n): " visualizeTraining
#  read -p "Do you want to save node embeddings (Y/n): " saveNodeEmbedding
  read -p "Do you want to consider 2 hop from detected anomalous (y/N): " consider2Hop
else
  echo "Using Default Parameters"
fi

if [[ "$ToRDF" == "y" ]]
then
  read -p "Enter the source graph postfix (ex. _1day): " source_graph_postfix
  read -p "Do you want to get node attributes from NetworkX graph(y/N): ": node_attrs_graph_nx
#  if [[ "$node_attrs_graph_nx" == "y" ]]
#  then
#    read -p "Enter the networkX source graph (e.g. complete_cadets_pg): ": nx_source_graph
#  fi
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
    rdf_parameter+=" --graph-optc"
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
  read -p "Enter the source graph postfix (ex. _1day): " source_graph_postfix
  read -p "Do you want to take validation from training set (y/N): ": TrainingValid
  if [[ "$TrainingValid" == "y" ]]
  then
    pyg_parameters+=" --training-valid"
  fi
  read -p "Do you want to extract features from timestamps (y/N): ": timestamps_features
  if [[ "$timestamps_features" == "y" ]]
  then
    pyg_parameters+=" --get-timestamps-features "
#    read -p "Do you want to extract total active time (y/N): ": activetime
#    if [[ "$activetime" == "y" ]]
#    then
#      pyg_parameters+=" --get-total-active-time"
#      read -p "Do you want to extract start and end active hours (y/N): ": activeHours
#      if [[ "$activeHours" == "y" ]]
#      then
#        pyg_parameters+=" --get-active-hour"
#      fi
#    fi
    read -p "Do you want to extract idle time (y/N): ": IdleTime
    if [[ "$IdleTime" == "y" ]]
    then
      pyg_parameters+=" --get-idle-time"
    fi
#    read -p "Do you want to round timestamps to minutes (y/N): ": InMinutes
#    if [[ "$InMinutes" == "y" ]]
#    then
#      pyg_parameters+=" --timestamps-in-minutes"
#    fi
#    read -p "Do you want to round timestamps to seconds (y/N): ": InSeconds
#    if [[ "$InSeconds" == "y" ]]
#    then
#      pyg_parameters+=" --timestamps-in-seconds"
#    fi
#    read -p "Do you want to parse timestamps to days, hours, minutes, seconds, [milliseconds] (y/N): ": parseMilliseconds
#    if [[ "$parseMilliseconds" == "y" ]]
#    then
#      pyg_parameters+=" --parse-timestamps"
#    fi
  fi
#  read -p "Do you want to extract time series for avg n_actions per hour in day  (y/N): ": timeseries
#  if [[ "$timeseries" == "y" ]]
#  then
#    pyg_parameters+=" --get-time-series"
#    read -p "Do you want to parse time series to mean ,min, max, most frequent  (y/N): ": parseTimeseries
#    if [[ "$parseTimeseries" == "y" ]]
#    then
#      pyg_parameters+=" --parse-time-series"
#    fi
#  fi
  read -p "Do you want to normalize features (y/N): ": Normalize
  if [[ "$Normalize" == "y" ]]
  then
    pyg_parameters+=" --normalize-features"
  fi
#  read -p "Do you want to fill nan features with mean (y/N): ": fillMean
#  if [[ "$fillMean" == "y" ]]
#  then
#    pyg_parameters+=" --fill-with-mean"
#  fi

else
  echo "Skip converting to PyG"
fi

if [[ "$investigate" == "y" ]]
then
  read -p "Enter the number of hops while investigation: " n_hops_investigate
  investigation_parameters=" --number-of-hops ${n_hops_investigate} --correlate-anomalous-once --remove-duplicated-subgraph --get-node-attrs"
  logs_name="_controlled_in_${n_hops_investigate}_hop_Fixed"
  read -p "Enter the maximum edges per subgraph (default: 5000)( 0 for no edge limit): " max_edges
  investigation_parameters+=" --max-edges ${max_edges}"
  logs_name+="_MaxEdges${max_edges}"
  read -p "Enter the top K seed nodes per type to be investigated (default: 15): " k_nodes
  investigation_parameters+=" --top-k ${k_nodes}"
  logs_name+="_K${k_nodes}"
  if [[ "$consider2Hop" == "y" ]]
  then
    investigation_parameters+=" --consider-related-nodes "
    logs_name+="_Consider2Hop"
  fi
#  read -p "Do you want to drop duplicated subgraphs ? (y/N)": drop_duplication
#  if [[ "$drop_duplication" == "y" ]]
#  then
#    investigation_parameters+=" --remove-duplicated-subgraph"
#  fi
#  read -p "Do you want to correlate anomalous nodes only once? (y/N)": correlateOnce
#  if [[ "$correlateOnce" == "y" ]]
#  then
#    investigation_parameters+=" --correlate-anomalous-once "
#  fi
  read -p "Do you want to draw constructed subgraphs (y/N)": draw_subgraphs
  if [[ "$draw_subgraphs" == "y" ]]
  then
    investigation_parameters+=" --draw-subgraphs"
  fi

#  read -p "Do you want to draw node attributes (y/N)": draw_attributes
#  if [[ "$draw_attributes" == "y" ]]
#  then
#    investigation_parameters+=" --get-node-attrs"
#  fi
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
#  if [[ "$visualizeTraining" == "n" ]]
#  then
#    echo "No training visualization"
#  else
#    parameters+=" --visualize-training"
#  fi
#  if [[ "$saveNodeEmbedding" == "n" ]]
#  then
#    echo "Will Not save Node Embeddings"
#  else
#    parameters+=" --save-emb"
#  fi
  if [[ "$dynamicContamination" == "n" ]]
  then
    parameters+=" --contamination ${contamination}"
    logs+="_con${contamination}"
    save_path+="_con${contamination}"
  else
    parameters+=" --dynamic-contamination --dynamic-contamination-val --flexable-rate ${MinContamination} --max-contamination ${MaxContamination}"
    logs+="_dynConVal${MinContamination}To${MaxContamination}"
    save_path+="_dynConVal${MinContamination}To${MaxContamination}"
  fi
  if [[ "$multipleModels" == "y" ]]
  then
    parameters+=" --multiple-models"
    logs+="_MultipleModels"
  fi
  if [[ "$consider2Hop" == "y" ]]
  then
    parameters+=" --consider-related-nodes "
    logs+="_Consider2Hop"
  fi


#    if [[ "$AvoidNANFeatures" == "y" ]]
#    then
#      parameters+=" --features-per-node-type"
#      logs+="_AvoidNANFeatures"
#      save_path+="_avoidNanFeatures"
#    fi
#  fi

#  if [[ "$ensembleModels" == "y" ]]
#  then
#    parameters+=" --ensemble-models"
#    logs+="_EnsembleModels"
#  fi
#  if [[ "$Random" == "y" ]]
#  then
#    parameters+=" --random-features"
#    logs+="_Random_Features"
#    save_path+="_Random_Features"
#  fi
#  if [[ "$FeaturePCA" == "y" ]]
#  then
#    parameters+=" --feature-reduction-PCA"
#    logs+="_PCA"
#    save_path+="_pca"
#  fi
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
  source_graph="${host}${source_graph_postfix}"
  nx_source_graph="complete_${host}_pg"
  if [[ "$ToRDF" == "y" ]]
  then
    python -B -u ../src/pygod_DARPA_to_RDF_Transductive.py --dataset ${dataset} --host ${host} --root-path ${root_path} --source-graph ${source_graph} ${rdf_parameter} >> ../logs/${host}/${exp_name}/DARPA_to_RDF_${date}.txt
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
    python -u -B ../src/pygod_DARPA_to_PyG_Transductive.py --dataset ${dataset} --host ${host} --root-path ${root_path} --exp-name ${exp_name} --source-graph ${source_graph} ${pyg_parameters} >> ../logs/${host}/${exp_name}/RDF_to_PyG_${date}.txt
    echo "Done converting to PyG"
  fi

  echo "start training"

  python -B -u ../src/pygod_models_DARPA_Transductive.py --dataset ${dataset} --host ${host} --root-path ${root_path} --exp-name ${exp_name} --detector ${detector} ${parameters} --save-model ${save_path} >> ../logs/${host}/${exp_name}/${logs}

  echo "Done training"

  if [[ "$investigate" == "y" ]]
  then
    python -B -u ../src/investigate_nodes.py --dataset ${dataset} --host ${host} --root-path ${root_path} --exp-name ${exp_name} --node-emb-per-subgraph --model ${model_path} --node-emb-size ${HiddenLayer} --tensor-neurons ${HiddenLayer} --construct-from-anomaly-subgraph ${investigation_parameters} --min-nodes 3 --inv-exp-name ${logs_name} >> ../logs/${host}/${exp_name}/investigating${logs_name}_${date}.txt
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
