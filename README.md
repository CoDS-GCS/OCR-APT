# OCR-APT
OCR-APT is an APT detection system capable of identifying anomalous nodes and subgraphs, performing alerts triage based on abnormality levels, and recovering attack stories to support comprehensive attack investigation.
OCR-APT employs GNN-based subgraph anomaly detection to identify suspicious activities and utilizes LLMs to generate human-like attack reports.
This is the repository of the submitted paper: **OCR-APT: Reconstructing APT Stories through Subgraph Anomaly Detection and LLMs.**

## Repository Roadmap
The input to the system are kernel audit logs in a csv format.
The system consist of multiple python scripts and other bash script to command them in an interactive way.
- `/src` directory holds all python scripts.
- `/bash_src` directory holds all bash scripts.
- `/recovered_reports` directory contains all recovered reports in our experiments.
- `/groundtruth` directory contains the ground truth labels, each file contains malicious nodes UUID in a specific dataset/host. It should be moved to the dataset directory for reproducing results
- `/logs` directory is the default location for all generated system logs
- `/dataset` directory is the default location for training and testing audit logs, ground truth labels, experiments checkpoints, trained GNN models and results. Results include detected anomalous nodes and subgraphs, and recovered attack reports.

## OCR-APT system Architecture 
![System Architecture](OCR-APT-system.png)
