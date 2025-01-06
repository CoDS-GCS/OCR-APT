# Attack Report: Anomalous Activity Detected in optc_anomalous_subgraph_0.csv

## Summary of Attack Behavior

The logs from `optc_anomalous_subgraph_0.csv` indicate a series of suspicious activities primarily executed by the `svchost.exe` process on September 25, 2019. The actions taken during this incident suggest a potential APT attack, characterized by file modifications, deletions, and creations, which are typical behaviors associated with data manipulation and persistence strategies.

### Key Events:
- **File Modifications and Deletions**: Multiple files were modified and deleted, including several `.rslc` files, which may indicate an attempt to cover tracks or manipulate system states.
- **File Creations**: New files were created, including several with GUIDs, which could be indicative of new malicious payloads or configurations being established.
- **Reading Sensitive Files**: The process read files such as `AppxSignature.p7x` and DLLs related to application deployment, suggesting an attempt to gather information or manipulate application states.
- **Potential Persistence Mechanisms**: The creation of files like `TempState` and modifications to `INetHistory` indicate efforts to maintain persistence and possibly exfiltrate data.

### APT Stages Identified:
- **Initial Compromise**: The reading of application-related files may suggest an initial compromise through exploitation of application vulnerabilities.
- **Command and Control**: The creation of new files and modification of existing ones could indicate establishing a command and control mechanism.
- **Covering Tracks**: The deletion of files and modification of `INetHistory` suggest efforts to erase traces of the attack.

## Table of IoCs Detected

| IoC                                                                 | Security Context                                                                                     |
|---------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| 5ca5016d-f3cc-43a6-8794-aa362af158cf_S-1-5-21-..._2.rslc          | Potentially malicious file, likely used for persistence or data manipulation.                       |
| 5ca5016d-f3cc-43a6-8794-aa362af158cf_S-1-5-21-..._18.rslc         | Similar to above, may indicate a new payload or configuration file.                                 |
| 5ca5016d-f3cc-43a6-8794-aa362af158cf_S-1-5-21-..._17.rslc         | Indicates potential malicious activity; further analysis required to determine legitimacy.           |
| 5ca5016d-f3cc-43a6-8794-aa362af158cf_S-1-5-21-..._16.rslc         | Likely part of a malicious operation; could be used for data exfiltration or command execution.    |
| 5ca5016d-f3cc-43a6-8794-aa362af158cf_S-1-5-21-..._15.rslc         | Indicates potential exploitation; further investigation needed.                                      |
| 5ca5016d-f3cc-43a6-8794-aa362af158cf_S-1-5-21-..._13.rslc         | May be associated with malicious activity; requires deeper analysis.                                 |
| 5ca5016d-f3cc-43a6-8794-aa362af158cf_S-1-5-21-..._12.rslc         | Potentially used for persistence; further investigation warranted.                                   |
| 5ca5016d-f3cc-43a6-8794-aa362af158cf_S-1-5-21-..._11.rslc         | Indicates possible malicious intent; further analysis required.                                      |
| 5ca5016d-f3cc-43a6-8794-aa362af158cf_S-1-5-21-..._10.rslc         | Likely part of a malicious operation; could be used for data exfiltration or command execution.    |
| 1527c705-839a-4832-9118-54d4Bd6a0c89_10.0.15063.1716...            | Application file; legitimate usage but could be exploited if compromised.                          |
| 1334242792.pri                                                      | Potentially sensitive file; requires further investigation to determine its role in the attack.    |
| 1698925018.pri                                                      | Similar to above; may contain sensitive information.                                                |
| Holoshell_10.0.15063.1746_neutral__cw5n1h2txyewy.xml              | Application configuration file; could be exploited for malicious purposes.                          |
| c5e2524a-ea46-4f67-841f-6a9465d9d515_cw5n1h2txyewy               | Potentially malicious file; further analysis needed.                                               |
| 0239d121-8ed3-4b1e-95eb-5b7f9a6314ea_S-1-5-21-..._4.rslc          | Indicates possible malicious activity; requires deeper analysis.                                     |
| 0239d121-8ed3-4b1e-95eb-5b7f9a6314ea_S-1-5-21-..._3.rslc          | Similar to above; may indicate a new payload or configuration file.                                 |
| 0239d121-8ed3-4b1e-95eb-5b7f9a6314ea_S-1-5-21-..._12.rslc         | Potentially used for persistence; further investigation warranted.                                   |
| 0239d121-8ed3-4b1e-95eb-5b7f9a6314ea_S-1-5-21-..._11.rslc         | Indicates possible malicious intent; further analysis required.                                      |
| 0239d121-8ed3-4b1e-95eb-5b7f9a6314ea_S-1-5-21-..._10.rslc         | Likely part of a malicious operation; could be used for data exfiltration or command execution.    |
| windows.immersivecontrolpanel_6.2.0.0_neutral_neutral_cw5n1h2txyewy | Legitimate Windows component; could be exploited if compromised.                                    |
| resources.e9aab164.pri                                             | Potentially sensitive file; requires further investigation to determine its role in the attack.    |
| resources.dba79ab6.pri                                             | Similar to above; may contain sensitive information.                                                |
| pagefile.sys                                                       | System file; legitimate usage but could be exploited if compromised.                                |
| resources.b93b0697.pri                                            | Potentially sensitive file; requires further investigation to determine its role in the attack.    |
| resources.8883896e.pri                                            | Similar to above; may contain sensitive information.                                                |
| resources.857e5af3.pri                                            | Potentially sensitive file; requires further investigation to determine its role in the attack.    |
| resources.5efe4060.pri                                            | Similar to above; may contain sensitive information.                                                |
| resources.44828b84.pri                                            | Potentially sensitive file; requires further investigation to determine its role in the attack.    |
| resources.38e1ccbd.pri                                            | Similar to above; may contain sensitive information.                                                |
| resources.387e40a3.pri                                            | Potentially sensitive file; requires further investigation to determine its role in the attack.    |
| resources.2bb76f1c.pri                                            | Similar to above; may contain sensitive information.                                                |
| AppxSignature.p7x                                                 | Application signature file; legitimate usage but could be exploited if compromised.                |
| APPXDEPLOYMENTEXTENSIONS.DESKTOP.DLL                             | Legitimate Windows component; could be exploited if compromised.                                    |
| INetHistory                                                       | System file; legitimate usage but could be exploited if compromised.                                |
| TempState                                                         | Temporary file; could be used for malicious purposes if manipulated.                                |

## Chronological Log of Actions

### September 25, 2019

- **13:43**: 
  - `svchost.exe` modified files related to `0239d121-8ed3-4b1e-95eb-5b7f9a6314ea`.
  - Loaded multiple modules including `shell32.dll`, `profext.dll`, and others.
  - Deleted several `.rslc` files and read `.pri` files.
  
- **13:44**: 
  - `svchost.exe` read `AppxSignature.p7x` and `APPXDEPLOYMENTEXTENSIONS.DESKTOP.DLL`.
  - Modified `INetHistory` and several `.rslc` files.
  - Created `TempState`.

- **13:49**: 
  - `svchost.exe` modified and deleted multiple `.rslc` files.
  
- **14:00**: 
  - `svchost.exe` created multiple new `.rslc` files and read `.pri` files.
  - Wrote to `Holoshell_10.0.15063.1746_neutral__cw5n1h2txyewy.xml` and created `c5e2524a-ea46-4f67-841f-6a9465d9d515_cw5n1h2txyewy`.

This report highlights the suspicious activities associated with the `svchost.exe` process, indicating potential malicious behavior consistent with APT tactics. Further investigation is warranted to assess the full impact and scope of the incident.