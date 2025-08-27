# OCR-APT

**OCR-APT** is an APT detection system designed to identify anomalous nodes and subgraphs, prioritize alerts based on abnormality levels, and reconstruct attack stories to support comprehensive investigations.  

The system leverages **GNN-based subgraph anomaly detection** to uncover suspicious activities and **LLM-based reporting** to generate human-like attack narratives.  

This is the code for the paper **OCR-APT: Reconstructing APT Stories through Subgraph Anomaly Detection and LLMs**, accepted to ACM CCS2025

---

## Repository Roadmap

The input to OCR-APT is audit logs in CSV format.  
The system consists of multiple Python and Bash scripts that work together interactively.  

- **`/src`** – Python scripts:
  - **`sparql_queries.py`** – SPARQL queries for constructing subgraphs from the GraphDB database.  
  - **`llm_prompt.py`** – Prompts used by the LLM-based attack investigator.  
- **`/bash_src`** – Bash scripts for managing the pipeline.  
- **`/recovered_reports`** – Recovered reports generated in our experiments.  
- **`/logs`** – Default directory for generated system logs.  
- **`/dataset`** – Contains training/testing audit logs, ground truth labels, experiment checkpoints, trained GNN models, and results (including anomalous nodes, subgraphs, and recovered reports).  

---

## System Architecture

![System Architecture](OCR-APT-system.png)

---

## Setup OCR-APT

1. **Create the Conda environment**  
   Run the following script to create the environment using `environment.yml` and `requirements.txt`:  
   ```bash
   bash /bash_src/create_env.sh
   ```

2. **Set up GraphDB with RDF-Star**  
   - [Download and install GraphDB](https://graphdb.ontotext.com/documentation/11.0/graphdb-desktop-installation.html).  
   - Add the following configuration:  
     ```
     graphdb.workbench.importDirectory = <PATH_TO_GraphDB_INSTANCE>/GraphDB/loading_files/
     graphdb.connector.maxHttpHeaderSize = 1000000
     ```
   - [Create a repository](https://graphdb.ontotext.com/documentation/11.0/creating-a-repository.html) for each dataset and select the **RDFS-Plus (Optimized)** ruleset.  
   - Download loading_files.tar.xz from our Zenodo [repository](https://doi.org/10.5281/zenodo.16969449), which contain the RDF provenance graphs in turtle format. Then extract them and move them to:  
     ```
     <PATH_TO_GraphDB_INSTANCE>/GraphDB/.
     ```
   - [Load datasets](https://graphdb.ontotext.com/documentation/11.0/loading-data-using-the-workbench.html) into their repositories using the *Importing server files* option.  

3. **Configure system settings**  
   Create a `config.json` file with your repository URLs and OpenAI API key:  
   ```json
   {
     "repository_url_tc3": "<TC3_URL>",
     "repository_url_optc": "<OPTC_URL>",
     "repository_url_nodlink": "<NODLINK_URL>",
     "openai_api_key": "<API_KEY>"
   }
   ```

4. **Prepare datasets and models**  
   Download "dataset.tar.xz" from our Zenodo [repository](https://doi.org/10.5281/zenodo.16969449), which contain data snapshots, ground truth labels, and trained models.  
   Extract them and move the `dataset` directory into the OCR-APT working directory.  

5. **Run the detection pipeline**  
   - Run detection with pre-trained models:  
     ```bash
     bash /bash_src/ocrapt-detection.sh
     ```
   - Run the full system pipeline (preprocessing + retraining + detection):  
     ```bash
     bash /bash_src/ocrapt-full-system-pipeline.sh
     ```
   > **Note:** Preprocessed files are already available [here]() (placeholder), so preprocessing can be skipped if desired.  

---

## Experiments with Locally Deployed LLMs

### Experiment Summary
We evaluated OCR-APT using **locally deployed LLMs** and compared the generated reports with those produced by ChatGPT.  

- Deployment: **LLAMA3 (8B parameters)** on a machine with:
  - 4 CPU cores  
  - 8 GB GPU  
  - 22 GB RAM  
- Optimization: Tested multiple local embedding models and analyzed outputs to determine the most effective setup.  

**Key finding:** LLAMA3, combined with the best-performing embedding model, generated reports **comparable in quality to ChatGPT**.  

Detailed experimental results are available in this [spreadsheet](Experiments_with_locally_deployed_LLMs.xlsx).  

[//]: # (## Citation )

[//]: # (### Bibtex)

[//]: # (```)

[//]: # (@article{aly2025ocr,)

[//]: # (  title={{OCR-APT}: Reconstructing APT Stories through Subgraph Anomaly Detection and LLMs},)

[//]: # (  author={Aly, Ahmed and Mansour, Essam and Youssef, Amr },)

[//]: # (  booktitle={Proceedings of the 2025 ACM SIGSAC Conference on Computer and Communications Security},)

[//]: # (  year={2025})

[//]: # (})

[//]: # (```)