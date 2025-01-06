# Attack Report: Anomalous Activity Detected in optc_anomalous_subgraph_12.csv

## Summary of Attack Behavior

The logs from `optc_anomalous_subgraph_12.csv` indicate a series of suspicious activities primarily executed by the process `svchost.exe` on September 24, 2019. The actions taken during this incident suggest a potential APT attack, characterized by file modifications, creations, deletions, and module loads. 

### Key Events:
- **File Modifications and Creations**: Multiple files with the `.rslc` extension were created and modified, indicating potential manipulation of system resources or configurations.
- **File Deletions**: Several files were deleted, which could signify an attempt to cover tracks or remove evidence of malicious activity.
- **Module Loads**: The loading of various DLL modules, such as `gpapi.dll`, `MrmDeploy.dll`, and `container.dll`, suggests the execution of additional functionalities that may be related to the attack.
- **Read and Write Operations**: The process engaged in reading and writing operations on various files, which could be indicative of data exfiltration or manipulation.

### APT Stages Identified:
- **Initial Compromise**: The creation of new files and modification of existing ones may indicate the initial compromise of the system.
- **Internal Reconnaissance**: The reading of various files suggests an attempt to gather information about the system.
- **Covering Tracks**: The deletion of files points to efforts to erase traces of the attack.

---

## Indicators of Compromise (IoCs)

| IoC                                                                 | Security Context                                                                                     |
|---------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| 8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_35.rslc | Potentially malicious file; could be used for unauthorized access or control.                      |
| 8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_33.rslc | Similar to above; indicates possible manipulation of system settings or configurations.             |
| 8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_31.rslc | Indicates potential data manipulation; could be part of a larger attack vector.                    |
| 8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_26.rslc | May contain sensitive information; could be targeted for exfiltration.                             |
| 8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_43.rslc | Potentially used for unauthorized access; requires further investigation.                           |
| 8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_61.rslc | Indicates possible exploitation; could be linked to malicious activities.                           |
| 8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_59.rslc | May be involved in data manipulation; further analysis needed.                                      |
| 8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_57.rslc | Could be part of a larger malicious framework; requires monitoring.                                 |
| 8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_55.rslc | Indicates potential for exploitation; should be flagged for further scrutiny.                       |
| 8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_53.rslc | May contain sensitive data; could be targeted for unauthorized access.                             |
| 8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_49.rslc | Potentially malicious; could be used for data exfiltration or system manipulation.                |
| 8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_45.rslc | Indicates possible unauthorized access; requires further investigation.                             |
| 8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_63.rslc | May be linked to malicious activities; should be monitored closely.                                 |
| 8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_61.rslc | Indicates potential for exploitation; further analysis needed.                                      |
| 8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_59.rslc | May be involved in data manipulation; further analysis needed.                                      |
| 8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_57.rslc | Could be part of a larger malicious framework; requires monitoring.                                 |
| 8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_55.rslc | Indicates potential for exploitation; should be flagged for further scrutiny.                       |
| 8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_53.rslc | May contain sensitive data; could be targeted for unauthorized access.                             |
| 8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_49.rslc | Potentially malicious; could be used for data exfiltration or system manipulation.                |
| 8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_45.rslc | Indicates possible unauthorized access; requires further investigation.                             |
| 8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_81.rslc | Potentially malicious file; could be used for unauthorized access or control.                      |
| 8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_77.rslc | Similar to above; indicates possible manipulation of system settings or configurations.             |
| 8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_75.rslc | Indicates potential data manipulation; could be part of a larger attack vector.                    |
| 8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_73.rslc | May contain sensitive information; could be targeted for exfiltration.                             |
| 8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_71.rslc | Potentially used for unauthorized access; requires further investigation.                           |
| 8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_69.rslc | Indicates possible exploitation; could be linked to malicious activities.                           |
| 8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_73.rslc | May be involved in data manipulation; further analysis needed.                                      |
| 8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_69.rslc | Could be part of a larger malicious framework; requires monitoring.                                 |
| 8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_83.rslc | Indicates potential for exploitation; further analysis needed.                                      |
| 8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_81.rslc | May be involved in data manipulation; further analysis needed.                                      |
| 8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_77.rslc | Could be part of a larger malicious framework; requires monitoring.                                 |
| 8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_73.rslc | Indicates potential for exploitation; should be flagged for further scrutiny.                       |
| 8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_4.rslc  | Potentially malicious; could be used for data exfiltration or system manipulation.                |
| 8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_3.rslc  | Indicates possible unauthorized access; requires further investigation.                             |
| 8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_26.rslc | May contain sensitive data; could be targeted for unauthorized access.                             |
| 8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_21.rslc | Potentially used for unauthorized access; requires further investigation.                           |
| 8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_20.rslc | Indicates possible exploitation; could be linked to malicious activities.                           |
| 8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_2.rslc  | May be involved in data manipulation; further analysis needed.                                      |
| 8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_19.rslc | Could be part of a larger malicious framework; requires monitoring.                                 |
| 8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_13.rslc | Indicates potential for exploitation; further analysis needed.                                      |
| 8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_12.rslc | May contain sensitive data; could be targeted for unauthorized access.                             |
| 8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_10.rslc | Potentially malicious; could be used for data exfiltration or system manipulation.                |
| 8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_9.rslc  | Indicates possible unauthorized access; requires further investigation.                             |
| 8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_6.rslc  | May be involved in data manipulation; further analysis needed.                                      |

---

## Chronological Log of Actions

### September 24, 2019

#### 13:38
- `svchost.exe` MODIFY the file: `8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_49.rslc`
- `svchost.exe` MODIFY the file: `8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_45.rslc`
- `svchost.exe` MODIFY the file: `8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_43.rslc`
- `svchost.exe` CREATE the file: `8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_67.rslc`
- `svchost.exe` MODIFY the file: `8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_37.rslc`
- `svchost.exe` MODIFY the file: `8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_33.rslc`
- `svchost.exe` MODIFY the file: `8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_31.rslc`
- `svchost.exe` LOAD the module: `gpapi.dll`
- `svchost.exe` LOAD the module: `MrmDeploy.dll`
- `svchost.exe` LOAD the module: `AppxApplicabilityEngine.dll`
- `svchost.exe` DELETE the file: `ActivationStore.dat`
- `svchost.exe` MODIFY the file: `8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_41.rslc`
- `svchost.exe` READ the file: `ActivationStore.dat`
- `svchost.exe` READ the file: `865901629.pri`
- `svchost.exe` READ the file: `3856886151.pri`
- `svchost.exe` READ the file: `3344272130.pri`
- `svchost.exe` READ the file: `3307166592.pri`
- `svchost.exe` READ the file: `1102797358.pri`
- `svchost.exe` OPEN the process: `svchost.exe`

#### 13:39
- `svchost.exe` WRITE the file: `8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_77.rslc`
- `svchost.exe` MODIFY the file: `8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_73.rslc`
- `svchost.exe` MODIFY the file: `8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_71.rslc`
- `svchost.exe` MODIFY the file: `8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_69.rslc`
- `svchost.exe` LOAD the module: `container.dll`
- `svchost.exe` LOAD the module: `IPHLPAPI.DLL`
- `svchost.exe` WRITE the file: `8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_71.rslc`
- `svchost.exe` DELETE the file: `8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_83.rslc`
- `svchost.exe` DELETE the file: `8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_81.rslc`
- `svchost.exe` DELETE the file: `8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_77.rslc`
- `svchost.exe` DELETE the file: `8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_75.rslc`
- `svchost.exe` DELETE the file: `8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_73.rslc`
- `svchost.exe` DELETE the file: `8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_71.rslc`

---

This report summarizes the suspicious activities detected in the logs, highlighting the potential risks and necessary actions for further investigation and mitigation.