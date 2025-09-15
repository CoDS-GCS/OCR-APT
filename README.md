# OCR-APT

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17107271.svg)](https://doi.org/10.5281/zenodo.17107271)
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
   Install Conda, then from inside the `bash_src` directory run the following commands to create and activate the environment using `requirements.txt`:
```bash
   conda create -n env-ocrapt python=3.9
   conda activate env-ocrapt
   bash create_env.sh
   ```

2. **Set up GraphDB with RDF-Star**  
   - [Download and install GraphDB](https://graphdb.ontotext.com/documentation/11.0/graphdb-desktop-installation.html).  
   - **Configure GraphDB instance**: 
     - For GraphDB Desktop, open the Settings window and set the property:
       - Name: `graphdb.workbench.importDirectory`
       - Value: `<PATH_TO_GraphDB_INSTANCE>/GraphDB/loading_files/` ([More details](https://graphdb.ontotext.com/documentation/11.1/graphdb-desktop-installation.html))
     - For the standalone server (not required for Desktop), edit `conf/graphdb.properties` in the GraphDB home directory or use the following command:
        ```
        graphdb -s -Dgraphdb.workbench.importDirectory=<PATH_TO_GraphDB_INSTANCE>/GraphDB/loading_files/
        ```
   - Launch GraphDB Desktop and open the Workbench at `http://localhost:7200/` (default port: 7200).  
   - Create a repository for each dataset: **Setup → Repositories → Create New Repository → GraphDB Repository**. Set the dataset name as the **Repository ID** and select the **RDFS-Plus (Optimized)** ruleset ([More details](https://graphdb.ontotext.com/documentation/11.0/creating-a-repository.html)).  
   - Download `loading_files.tar.xz` from our dataset [record](https://doi.org/10.5281/zenodo.16987705), which contains RDF provenance graphs in Turtle format. Extract and move them to `<PATH_TO_GraphDB_INSTANCE>/GraphDB/`.
   - Load datasets into their repositories using the Workbench: **Import → Server files**. Set the dataset **Base IRI** (e.g., `https://NODLINK.graph`), choose **Named Graph**, and provide the host IRI (e.g., `https://NODLINK.graph/SimulatedW10`). [For more details](https://graphdb.ontotext.com/documentation/11.0/loading-data-using-the-workbench.html).
     - For **DARPA TC3 datasets**, load both the main provenance graph file (e.g.,`cadets_rdfs.ttl`) and the attribute file (e.g.,`cadets_attributes_rdfs.ttl`) into the same Named Graph .
     - Files should be loaded to datasets as follows:
       - DARPA TC3: 
         - load `cadets_rdfs.ttl` then `cadets_attributes_rdfs.ttl` with the host IRI `https://DARPA_TC3.graph/cadets`
         - load `theia_rdfs.ttl` then `theia_attributes_rdfs.ttl` with the host IRI `https://DARPA_TC3.graph/theia`
         - load `trace_rdfs.ttl` then `trace_attributes_rdfs.ttl` with the host IRI `https://DARPA_TC3.graph/trace`
       - DARPA OpTC:
         - load `SysClient0201_1day_rdfs.ttl` with the host IRI `https://DARPA_OPTC.graph/SysClient0201`
         - load `SysClient0501_1day_rdfs.ttl` with the host IRI `https://DARPA_OPTC.graph/SysClient0501`
         - load `SysClient0051_1day_rdfs.ttl` with the host IRI `https://DARPA_OPTC.graph/SysClient0051`
       - NODLINK:
         - load `SimulatedUbuntu_rdfs.ttl` with the host IRI `https://NODLINK.graph/SimulatedUbuntu`
         - load `SimulatedWS12_rdfs.ttl` with the host IRI `https://NODLINK.graph/SimulatedWS12`
         - load `SimulatedW10_rdfs.ttl` with the host IRI `https://NODLINK.graph/SimulatedW10`



3. **Configure system settings**  
   Create a `config.json` file in the OCR-APT working directory as follows (replace the placeholders with your repository URLs and OpenAI API key):  
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
   Extract it and move the `dataset` directory into the OCR-APT working directory.  

5. **Run the detection pipeline**  
   - From inside the `bash_src` directory, run detection with pre-trained models:  
     ```bash
     bash ocrapt-detection.sh
     ```
   - To run the full system pipeline (preprocessing + retraining + detection):  
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
