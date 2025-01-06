# Attack Report: Anomalous Activity Detected in optc_anomalous_subgraph_16.csv

## Summary of Attack Behavior

The logs from the document indicate a series of anomalous activities executed by the `python.exe` process on September 25, 2019. The behavior suggests a potential APT attack characterized by multiple outbound and inbound message flows, module loading, and file reading activities. 

### Key Events:
- **Initial Compromise**: The process initiated with the loading of various dynamic link libraries (DLLs) and Python modules, indicating a setup phase for potential exploitation.
- **Command and Control**: The process established outbound connections to external IP addresses (e.g., `142.20.61.132`, `10.20.2.66`), suggesting attempts to communicate with a command and control server.
- **Data Exfiltration**: The repeated outbound messages to the same external IPs indicate a potential data exfiltration attempt, as the process communicated with these addresses multiple times.
- **Internal Reconnaissance**: The reading of various files, including `.pyc` files, suggests that the attacker may have been gathering information about the environment or preparing for further actions.

## Table of IoCs Detected

| IoC                                   | Security Context                                                                                     |
|---------------------------------------|-----------------------------------------------------------------------------------------------------|
| 142.20.61.132                         | External IP address; potential command and control server.                                         |
| 142.20.56.52                          | External IP address; potential command and control server.                                         |
| 142.20.61.131                         | External IP address; potential command and control server.                                         |
| 10.20.2.66                            | Internal IP address; may indicate lateral movement or internal reconnaissance.                      |
| MSVCR90.dll                           | Legitimate Microsoft Visual C++ runtime library; could be exploited if used in a malicious context.|
| python27.dll                          | Legitimate Python runtime; could be exploited if used in a malicious context.                      |
| crypt32.dll                           | Legitimate Windows cryptographic library; could be exploited for cryptographic attacks.            |
| MSVCP_WIN.DLL                         | Legitimate Microsoft Visual C++ library; could be exploited if used in a malicious context.       |
| combase.dll                           | Legitimate COM base library; could be exploited for COM-related attacks.                           |
| sppc.dll                              | Legitimate Windows library; could be exploited if used in a malicious context.                     |
| shfolder.dll                          | Legitimate Windows shell folder library; could be exploited if used in a malicious context.       |
| rpchttp.dll                           | Legitimate RPC over HTTP library; could be exploited for remote procedure call attacks.           |
| CONTAB32.DLL                          | Legitimate Windows library; could be exploited if used in a malicious context.                     |
| msvcr100.dll                          | Legitimate Microsoft Visual C++ runtime library; could be exploited if used in a malicious context.|
| msvcp100.dll                          | Legitimate Microsoft Visual C++ library; could be exploited if used in a malicious context.       |
| ms1_0.dll                             | Legitimate Windows authentication library; could be exploited for authentication bypass.           |
| dpapi.dll                             | Legitimate Data Protection API; could be exploited for credential theft.                           |
| d2d1.dll                              | Legitimate Direct2D library; could be exploited if used in a malicious context.                   |
| Wldap32.dll                           | Legitimate LDAP library; could be exploited for directory service attacks.                         |
| ResourcePolicyClient.dll              | Legitimate Windows resource policy library; could be exploited if used in a malicious context.    |
| RICHED20.DLL                          | Legitimate Rich Edit control library; could be exploited if used in a malicious context.          |
| OLMAPI32.DLL                          | Legitimate Outlook MAPI library; could be exploited for email-related attacks.                    |
| NtlmShared.dll                        | Legitimate NTLM authentication library; could be exploited for credential theft.                  |
| MAPIR.DLL                             | Legitimate MAPI library; could be exploited for email-related attacks.                             |
| GdiPlus.dll                           | Legitimate GDI+ library; could be exploited for graphic-related attacks.                           |
| EMSMDB32.DLL                          | Legitimate EMSMDB library; could be exploited for email-related attacks.                           |
| 2D5E2D34-BED5-4B9F-9793-A31E26E6806Ex0x5x0 | Suspicious file; potential indicator of malicious activity.                                        |
| IRDOSession.pyc                       | Python compiled file; could be part of a malicious script.                                         |
| IRDOItems.pyc                         | Python compiled file; could be part of a malicious script.                                         |
| IRDOMail.pyc                          | Python compiled file; could be part of a malicious script.                                         |
| IRestrictionProperty.pyc              | Python compiled file; could be part of a malicious script.                                         |
| ITableFilter.pyc                      | Python compiled file; could be part of a malicious script.                                         |
| IRDOFolder.pyc                        | Python compiled file; could be part of a malicious script.                                         |
| _IRestriction.pyc                     | Python compiled file; could be part of a malicious script.                                         |
| _IMAPITable.pyc                       | Python compiled file; could be part of a malicious script.                                         |
| IRDOAttachments.pyc                   | Python compiled file; could be part of a malicious script.                                         |
| IRDOAddressEntry.pyc                  | Python compiled file; could be part of a malicious script.                                         |
| IRDOAttachment.pyc                    | Python compiled file; could be part of a malicious script.                                         |

## Chronological Log of Actions

### 14:21
- Loaded various modules including `MSVCR90.dll`, `python27.dll`, `crypt32.dll`, and others.
  
### 14:22
- Initiated outbound message flow to `10.20.2.66` (5 times).

### 14:23
- Established outbound connections to `142.20.61.132` and `142.20.61.131`.
- Inbound message flow from `142.20.56.52` (4 times).
- Read multiple files including `win.ini`, `emsmdb32.dll`, and others.
- Loaded additional modules including `sppc.dll`, `shfolder.dll`, and others.

### 14:24
- Outbound message flow to `142.20.61.132` (12 times).
- Inbound message flow from `142.20.56.52` (12 times).
- Read multiple `.pyc` files.

### 14:25
- Continued outbound and inbound message flows to `142.20.61.132` and `142.20.56.52`.

### 14:32
- Outbound message flow to `142.20.61.132` (12 times).
- Inbound message flow from `142.20.56.52` (12 times).
- Outbound message flow to `142.20.61.131` (3 times).

### 14:33
- Continued outbound and inbound message flows to `142.20.61.132` and `142.20.56.52`.

### 14:34
- Continued outbound and inbound message flows to `142.20.61.132` and `142.20.56.52`.

### 14:35
- Continued outbound and inbound message flows to `142.20.61.132` and `142.20.56.52`.

### 14:36
- Continued outbound and inbound message flows to `142.20.61.132` and `142.20.56.52`.

### 14:37
- Continued outbound and inbound message flows to `142.20.61.132` and `142.20.56.52`.

### 14:38
- Final outbound and inbound message flows to `142.20.61.132` and `142.20.56.52`.

This report highlights the suspicious activities and potential indicators of compromise that warrant further investigation to mitigate any potential threats.