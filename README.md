# OCR-APT
OCR-APT is an APT detection system capable of identifying anomalous nodes and subgraphs, performing alerts triage based on abnormality levels, and recovering attack stories to support comprehensive attack investigation.
OCR-APT employs GNN-based subgraph anomaly detection to identify suspicious activities and utilizes LLMs to generate human-like attack reports.  
This is the repository of the submitted paper: **OCR-APT: Reconstructing APT Stories through Subgraph Anomaly Detection and LLMs.**

## Repository Roadmap
The input to the system is audit logs in a CSV format.
The system consists of multiple Python scripts and bash scripts that command them in an interactive way.
- `/src` directory holds all Python scripts.
  - `/src/sparql_queries.py` all SPARQL queries used to construct subgraphs from the `GraphDB` database 
  - `/src/llm_prompt.py` all prompts used by the LLM-based attack investigator  
- `/bash_src` directory holds all bash scripts.
- `/recovered_reports` directory contains all recovered reports in our experiments.
- `/logs` directory is the default location for all generated system logs.
- `/dataset` directory is the default location for training and testing audit logs, ground truth labels, experiments checkpoints, trained GNN models and results. Results include detected anomalous nodes and subgraphs, and recovered attack reports.

## OCR-APT system Architecture 
![System Architecture](OCR-APT-system.png)


## Setup OCR-APT 
1. Run `/bash_src/create_env.sh` to create the Conda environment using `environment.yml` and `requirements.txt`. 
2. Set up GraphDB with RDF* support and create the necessary database repositories.
    - Follow this [link](https://graphdb.ontotext.com/documentation/11.0/graphdb-desktop-installation.html) to Download and install GraphDB 
    - Add the following configuration `graphdb.workbench.importDirectory = <PATH_TO_GraphDB_INSTANCE>/GraphDB/loading_files/` and `graphdb.connector.maxHttpHeaderSize = 1000000`.
    - Follow this [link](https://graphdb.ontotext.com/documentation/11.0/creating-a-repository.html) to create a repository for each dataset. Select the RDFS-Plus (Optimized) ruleset.  
    - get the RDF datasets from this [link](), Uncompress, and move them to `<PATH_TO_GraphDB_INSTANCE>/GraphDB/.`.
    - Follow this [link](https://graphdb.ontotext.com/documentation/11.0/loading-data-using-the-workbench.html) to load datasets into their corresponding repositories, use (Importing server files) option. 
3. Create a `config.json` file containing the URLs of your database repositories and your OpenAI API keys. 
4. Get the data snapshots, ground truth labels, and trained models from this [link](). Uncompress and move the `dataset` directory to OCR-APT working directory.
5. Run the full system pipeline using this bash script: [ocrapt-full-system-pipeline.sh](bash_src/ocrapt-full-system-pipeline.sh)
   - You can skip the transformation to RDF step, as RDF processed files are released.
   - To directly test the detection pipeline using our trained models, you can use this bash script: [ocrapt-detection.sh](bash_src/ocrapt-detection.sh).

## Experiments with locally deployed LLMs
### Experiment Summary: 
We evaluated OCR-APT using locally deployed LLMs and compared its generated reports with those produced by ChatGPT.  
Specifically, we deployed **LLAMA3 (8B parameters)** on a machine with 4 CPU cores, an 8GB GPU, and 22GB of RAM. To optimize performance, we experimented with different local embedding models, analyzing their outputs to determine the most effective one.  
Our findings show that **LLAMA3, when paired with the best-performing embedding model, produced reports of comparable quality to ChatGPT**.  

Experimental results are available in this [spreadsheet](Experiments_with_locally_deployed_LLMs.xlsx).  
