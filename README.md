# OCR-APT

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17010220.svg)](https://doi.org/10.5281/zenodo.17010220)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16987705.svg)](https://doi.org/10.5281/zenodo.16987705)

**OCR-APT** is an APT detection system designed to identify anomalous nodes and subgraphs, prioritize alerts based on abnormality levels, and reconstruct attack stories to support comprehensive investigations.  

The system leverages **GNN-based subgraph anomaly detection** to uncover suspicious activities and **LLM-based reporting** to generate human-like attack narratives.  

This repository contains the code for the paper **OCR-APT: Reconstructing APT Stories through Subgraph Anomaly Detection and LLMs**, accepted at ACM CCS 2025.

---
## Repository Roadmap

The input to OCR-APT is audit logs in CSV format.  
The system is composed of multiple Python and Bash scripts that work together.  

- **`/src`** – Python scripts:
  - **`sparql_queries.py`** – Defines SPARQL queries for constructing subgraphs from the GraphDB database.  
  - **`llm_prompt.py`** – Contains prompts used by the LLM-based attack investigator.  
  - **`transform_to_RDF.py`** – Converts raw audit logs into RDF format for ingestion into GraphDB.  
  - **`encode_to_PyG.py`** – Encodes provenance subgraphs into PyTorch Geometric (PyG) data structures for model training and inference.  
  - **`train_gnn_models.py`** – Trains our one-class GNN model (`ocrgcn.py`) on benign data and applies it to identify anomalous nodes.  
  - **`detect_anomalous_subgraphs.py`** – Constructs subgraphs and detects anomalous ones using trained models.  
  - **`ocrapt_llm_investigator.py`** – Leverages LLMs to generate concise, human-readable attack investigation reports from anomalous subgraphs.  
- **`/bash_src`** – Bash scripts for managing the pipeline:  
  - **`ocrapt-full-system-pipeline.sh`** – Runs the complete OCR-APT workflow, from data preprocessing to report generation.  
  - **`ocrapt-detection.sh`** – Runs only the detection phase (GNN-based anomaly detection and report generation).  
- **`/recovered_reports`** – Contains reports generated in our experiments.  
- **`/logs`** – Default directory for system-generated logs.  
- **`/dataset`** – Provides training/testing audit logs, ground truth labels, experiment checkpoints, trained GNN models, and results (including anomalous nodes, subgraphs, and recovered reports). Our datasets are released in this [record](https://doi.org/10.5281/zenodo.16987705).  

---
## System Architecture

![System Architecture](OCR-APT-system.png)

---

## Setup OCR-APT

1. **Create the Conda environment**  
   The user should install Conda. From inside the `bash_src` directory, run the following script to create the environment using `requirements.txt`:
```bash
   conda create -n env-ocrapt python=3.9
   conda activate env-ocrapt
   bash setup_environment.sh
   ```

2. **Set up GraphDB with RDF-Star**  
   - [Download and install GraphDB](https://graphdb.ontotext.com/documentation/11.0/graphdb-desktop-installation.html).  
   - **Configure GraphDB instance**. For GraphDB Desktop, properties could be configured from the *Setting* window in the GraphDB Desktop application window ([More details](https://graphdb.ontotext.com/documentation/11.1/graphdb-desktop-installation.html)). Set a property with name: `graphdb.workbench.importDirectory` and value: `<PATH_TO_GraphDB_INSTANCE>/GraphDB/loading_files/`. For GraphDB standalone server (this step is not required if using GraphDB Desktop), properties could be configured using the `conf/graphdb.properties` file, located in the GraphDB home directory ([More details](https://graphdb.ontotext.com/documentation/11.1/graphdb-standalone-server.html)). User could edit the `graphdb.properties` file directly or use the following command to set the property:
    ```bash
      graphdb -s -Dgraphdb.workbench.importDirectory=<PATH_TO_GraphDB_INSTANCE>/GraphDB/loading_files/
     ```
   - Launch GraphDB Desktop, and access the GraphDB Workbench from `http://localhost:7200/` (The default port is 7200)
   - Use Create a repository for each dataset using the GraphDB Workbench. Go to Setup ‣ Repositories, click Create New repository ‣ GraphDB Repository, write the dataset name in Repositoy ID and select the **RDFS-Plus (Optimized)** ruleset ([More Details](https://graphdb.ontotext.com/documentation/11.0/creating-a-repository.html)).   
   - Download `loading_files.tar.xz` from our dataset [record](https://doi.org/10.5281/zenodo.16987705), which contains the RDF provenance graphs in turtle format. Extract them and move them to `<PATH_TO_GraphDB_INSTANCE>/GraphDB/.`.
   - Load datasets into their repositories using the GraphDB Workbench. Go to Import ‣ Server files, write the dataset Base IRI (e.g., `https://DARPA_OPTC.graph`), select Named graph and write the host IRI (e.g., `https://DARPA_OPTC.graph/SysClient0051` ([More Details](https://graphdb.ontotext.com/documentation/11.0/loading-data-using-the-workbench.html)). For DARPA TC3 datasets load both the main PG file (e.g., `cadets_rdfs.ttl`) and the attribute file (e.g., `cadets_attributes_rdfs.ttl`) to the same Named GraphFor DARPA TC3 datasets load both the main PG file (e.g., `cadets_rdfs.ttl`) and the attribute file (e.g., `cadets_attributes_rdfs.ttl`) to the same Named Graph. 
   Use the following IRI for our datasets/hosts:
   ```json
   https://DARPA_TC3.graph/cadets
   https://DARPA_TC3.graph/theia
   https://DARPA_TC3.graph/trace
   https://DARPA_OPTC.graph/SysClient0201
   https://DARPA_OPTC.graph/SysClient0501
   https://DARPA_OPTC.graph/SysClient0051
   https://NODLINK.graph/SimulatedUbuntu
   https://NODLINK.graph/SimulatedWS12
   https://NODLINK.graph/SimulatedW10
   ```
   

3. **Configure system settings**  
   Create a `config.json` file in the OCR-APT working directory as follows: (Replace the placeholders with your repository URLs and OpenAI API keys)
   ```json
   {
     "repository_url_tc3": "<TC3_URL>",
     "repository_url_optc": "<OPTC_URL>",
     "repository_url_nodlink": "<NODLINK_URL>",
     "openai_api_key": "<API_KEY>"
   }
   ```
   

4. **Prepare datasets and models**  
   Download `dataset.tar.xz` from our dataset [record](https://doi.org/10.5281/zenodo.16987705), which contains data snapshots, ground truth labels, and trained models.  
   Extract them and move the `dataset` directory into the OCR-APT working directory.  

5. **Run the detection pipeline**  
   - From inside the `bash_src` directory, Run detection with pre-trained models:  
     ```bash
     bash ocrapt-detection.sh
     ```
   - From inside the `bash_src` directory, Run the full system pipeline (preprocessing + retraining + detection):  
     ```bash
     bash ocrapt-full-system-pipeline.sh
     ```
   > **Note:** Preprocessed files are already available [here](https://doi.org/10.5281/zenodo.16987705), so preprocessing can be skipped if desired.  

---

## Experiments with Locally Deployed LLMs

### Experiment Summary
We evaluated OCR-APT using **locally deployed LLMs** and compared the generated reports with those produced by ChatGPT.  

- Deployment: **LLAMA3 (8B parameters)** on a machine with 4 CPU cores, 8 GB GPU, and 22 GB RAM.  
- Optimization: Tested multiple local embedding models and analyzed outputs to determine the most effective setup.  

**Key finding:** LLAMA3, combined with the best-performing embedding model, generated reports **comparable in quality to ChatGPT**.  

Detailed experimental results are available in this [spreadsheet](Experiments_with_locally_deployed_LLMs.xlsx).  
