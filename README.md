# OCR-APT

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17107271.svg)](https://doi.org/10.5281/zenodo.17107271)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17148221.svg)](https://doi.org/10.5281/zenodo.17148221)

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
- **`/dataset`** – Provides training/testing audit logs, ground truth labels, experiment checkpoints, trained GNN models, and results (including anomalous nodes, subgraphs, and recovered reports). Our datasets are released in this [record](https://doi.org/10.5281/zenodo.17148221).  

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
   - Download and install GraphDB Desktop from this [link](https://graphdb.ontotext.com/documentation/11.0/graphdb-desktop-installation.html).  
   - Download `GraphDB_repositories.tar.xz` from our dataset [record](https://doi.org/10.5281/zenodo.17148221). This archive contains a copy of the GraphDB repositories folder. 
     - To set up quickly, extract the archive and replace the entire `repositories` directory under `<PATH_TO_GraphDB_INSTANCE>/.graphdb/data/` with the one from `GraphDB_repositories/repositories/`.
     - If you prefer to keep your existing repositories, extract the archive and copy only the three provided repositories into the same location.
     - This setup is sufficient for running our evaluation datasets. For instructions on adding new repositories, see `Configure_GraphDB.md`, which provides a step-by-step guide.
   - Launch GraphDB Desktop and open the Workbench at `http://localhost:7200/` (default port: 7200). 
     - The expected result is to find three repositories under **Setup → Repositories** in the GraphDB Workbench. Their IDs should be:  
       - `darpa-tc3`  
       - `darpa-optc-1day`  
       - `simulated-nodlink` 

3. **Configure system settings**  
   Create a `config.json` file in the OCR-APT working directory as follows (Users should replace the placeholders with their OpenAI API key):  
   ```json
   {
     "repository_url_tc3": "http://localhost:7200//repositories/darpa-tc3",
     "repository_url_optc": "http://localhost:7200/repositories/darpa-optc-1day",
     "repository_url_nodlink": "http://localhost:7200/repositories/simulated-nodlink",
     "openai_api_key": "<API_KEY>"
   }
   ```
   
4. **Prepare datasets and models**  
   Download `dataset.tar.xz` from our dataset [record](https://doi.org/10.5281/zenodo.17148221), which contains data snapshots, ground truth labels, and trained models.  
   Extract it and move the `dataset` directory into the OCR-APT working directory.  


5. **Run the detection pipeline**  
   - From inside the `bash_src` directory, run OCR-APT detection using pre-trained models:  
     ```bash
     bash ocrapt-detection.sh
     ```
 
   - The expected output is three log files created in `/logs/<HOST>/Full_Script_Test`:
     - The file `DetectAnomalousNodes_*.txt`: node anomaly detection results using trained OCRGCN models.
     - The file `DetectAnomalousSubgraphs_*.txt`: anomalous subgraph detection results, including the detection performance of OCR-APT. 
     - The file `llm_investigator_output_*.txt`: human-readable attack investigation reports generated by the LLM-based module.
   - A sample run logs for the SimulatedW10 host is provided under `/logs/SimulatedW10/Full_Script_Test`, which you can use to verify successful execution.
   - To run the full system pipeline (preprocessing + retraining + detection):  
      ```bash
      bash ocrapt-full-system-pipeline.sh
      ```
   > **Note:** Preprocessed files are already available [here](https://doi.org/10.5281/zenodo.17148221), so preprocessing can be skipped if desired.
   
---

## Experiments with Locally Deployed LLMs

### Experiment Summary
We evaluated OCR-APT using **locally deployed LLMs** and compared the generated reports with those produced by ChatGPT.  

- Deployment: **LLAMA3 (8B parameters)** on a machine with 4 CPU cores, 8 GB GPU, and 22 GB RAM.  
- Optimization: Tested multiple local embedding models and analyzed outputs to determine the most effective setup.  

**Key finding:** LLAMA3, combined with the best-performing embedding model, generated reports **comparable in quality to ChatGPT**.  

Detailed experimental results are available in this [spreadsheet](Experiments_with_locally_deployed_LLMs.xlsx).  
