## Configure GraphDB
The `README.md` provides a quick-start setup for running OCR-APT on the evaluation datasets (DARPA TC3, DARPA OpTC, and NODLINK).

This document offers detailed configuration steps for setting up GraphDB repositories, which may be useful when working with additional datasets.

## Set up GraphDB with RDF-Star  
   - Download and install GraphDB Desktop from this [link](https://graphdb.ontotext.com/documentation/11.0/graphdb-desktop-installation.html).  
   - **Configure GraphDB instance**: 
     - For GraphDB Desktop, open the Settings window and set the property:
       - Name: `graphdb.workbench.importDirectory`
       - Value: `<PATH_TO_GraphDB_INSTANCE>/GraphDB/loading_files/` ([More details](https://graphdb.ontotext.com/documentation/11.1/graphdb-desktop-installation.html))
     - For the standalone server (not required for Desktop), edit `conf/graphdb.properties` in the GraphDB home directory or use the following command:
        ```
        graphdb -s -Dgraphdb.workbench.importDirectory=<PATH_TO_GraphDB_INSTANCE>/GraphDB/loading_files/
        ```
   - Launch GraphDB Desktop and open the Workbench at `http://localhost:7200/` (default port: 7200).
   - Create three repositories, one for each dataset: Setup → Repositories → Create New Repository → GraphDB Repository.
     - Set the dataset name as the Repository ID.
     - Select the RDFS-Plus (Optimized) ruleset. ([More details](https://graphdb.ontotext.com/documentation/11.0/creating-a-repository.html)).
     - Use the following Repository IDs for reproducibility:
       - `darpa-tc3`
       - `darpa-optc-1day`
       - `simulated-nodlink`
   - Download `loading_files.tar.xz` from our dataset [record](https://doi.org/10.5281/zenodo.16987705), which contains RDF provenance graphs in Turtle format. Extract and move them to `<PATH_TO_GraphDB_INSTANCE>/GraphDB/`.
   - Load datasets into their repositories using the Workbench: **Import → Server files**. Set the dataset **Base IRI** (e.g., `https://NODLINK.graph`), choose **Named Graph**, and provide the host IRI (e.g., `https://NODLINK.graph/SimulatedW10`). [For more details](https://graphdb.ontotext.com/documentation/11.0/loading-data-using-the-workbench.html).
     - For **DARPA TC3 datasets**, load both the main provenance graph file (e.g.,`cadets_rdfs.ttl`) and the attribute file (e.g.,`cadets_attributes_rdfs.ttl`) into the same Named Graph .
   - Files should be loaded to datasets repositories as follows:
     - DARPA TC3 (Repository ID: darpa-tc3): 
       - load `cadets_rdfs.ttl` then `cadets_attributes_rdfs.ttl` with the host IRI `https://DARPA_TC3.graph/cadets`
       - load `theia_rdfs.ttl` then `theia_attributes_rdfs.ttl` with the host IRI `https://DARPA_TC3.graph/theia`
       - load `trace_rdfs.ttl` then `trace_attributes_rdfs.ttl` with the host IRI `https://DARPA_TC3.graph/trace`
     - DARPA OpTC (Repository ID: darpa-optc-1day):
       - load `SysClient0201_1day_rdfs.ttl` with the host IRI `https://DARPA_OPTC.graph/SysClient0201`
       - load `SysClient0501_1day_rdfs.ttl` with the host IRI `https://DARPA_OPTC.graph/SysClient0501`
       - load `SysClient0051_1day_rdfs.ttl` with the host IRI `https://DARPA_OPTC.graph/SysClient0051`
     - NODLINK (Repository ID: simulated-nodlink):
       - load `SimulatedUbuntu_rdfs.ttl` with the host IRI `https://NODLINK.graph/SimulatedUbuntu`
       - load `SimulatedWS12_rdfs.ttl` with the host IRI `https://NODLINK.graph/SimulatedWS12`
       - load `SimulatedW10_rdfs.ttl` with the host IRI `https://NODLINK.graph/SimulatedW10`
   - The expected result is to see three repositories listed under Setup → Repositories in the GraphDB Workbench with IDs:
     - `darpa-tc3`
     - `darpa-optc-1day`
     - `simulated-nodlink`