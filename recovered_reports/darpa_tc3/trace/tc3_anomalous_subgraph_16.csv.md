# Attack Report: tc3_anomalous_subgraph_16.csv

## Summary of Attack Behavior

On April 13, 2018, the Thunderbird process exhibited a series of anomalous behaviors indicative of a potential Advanced Persistent Threat (APT) attack. The incident began at 14:38, with multiple instances of inter-process communication (RECVMSG) and memory protection (MPROTECT) actions, suggesting an attempt to manipulate or exfiltrate data. The process also engaged in file operations, including writing to pipes and linking files, which could indicate data staging or preparation for exfiltration.

As the incident progressed, the Thunderbird process executed a series of file operations, including reading and closing the file "maps," and renaming files such as "prefs.js" and "virtualFolders.dat." The renaming of these files may suggest an attempt to obscure or hide malicious activity. The process ultimately exited at 14:39, following a final set of write operations and unlinking of file links, which could indicate an effort to cover tracks.

The key events suggest the following APT stages:
- **Internal Reconnaissance**: RECVMSG and READ operations on "maps."
- **Command and Control**: MPROTECT and WRITE operations on pipes and srcsink.
- **Covering Tracks**: RENAME and UNLINK operations on files.

## Table of IoCs Detected

| IoC                     | Security Context                                                                                     |
|-------------------------|------------------------------------------------------------------------------------------------------|
| srcsink                 | Typically used for inter-process communication; could be exploited for data exfiltration.           |
| pipe                    | A legitimate IPC mechanism; may be used to transfer data between processes, potentially exploited.   |
| fileLink                | Used for linking files; could indicate manipulation of file paths for malicious purposes.            |
| maps                    | A file that may contain configuration or mapping data; could be targeted for reconnaissance.        |
| prefs.js                | A configuration file for Thunderbird; renaming may indicate an attempt to hide malicious changes.   |
| virtualFolders.dat      | A data file related to Thunderbird's virtual folders; renaming suggests potential data manipulation. |

## Chronological Log of Actions

### 14:38
- RECVMSG a srcsink (24 times)
- MPROTECT a memory (4 times)
- WRITE a pipe (3 times)
- LINK a fileLink (2 times)
- WRITE a srcsink (2 times)
- MMAP a memory (2 times)
- CLOSE the file: maps
- CLONE the process: thunderbird
- UNLINK a fileLink
- READ the file: maps
- OPEN the file: maps

### 14:39
- RECVMSG a srcsink (7 times)
- WRITE a pipe (4 times)
- WRITE a srcsink (2 times)
- EXIT the process: thunderbird
- RENAME the file: prefs.js
- MPROTECT a memory
- UNLINK a fileLink
- RENAME the file: virtualFolders.dat