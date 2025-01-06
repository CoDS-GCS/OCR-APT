# Attack Report: tc3_anomalous_subgraph_18.csv

## Summary of Attack Behavior

On April 13, 2018, the process "thunderbird" exhibited a series of anomalous behaviors indicative of a potential APT attack. The logs reveal multiple read and write operations on pipes and a srcsink, suggesting data manipulation or exfiltration attempts. The process also engaged in memory protection and file operations, including opening, closing, and renaming files, which may indicate an attempt to hide malicious activities or maintain persistence.

Key events include:
- **Initial Compromise**: The process initiated with multiple read and write operations on pipes and srcsink, indicating potential data interception or manipulation.
- **Internal Reconnaissance**: The opening and reading of the file "maps" suggest reconnaissance activities to gather information about the system.
- **Command and Control**: The linking of an external IP address (127.0.1.1:+24082) may indicate an attempt to establish a command and control channel.
- **Covering Tracks**: The unlinking and renaming of the file "aborted-session-ping" could be an effort to erase traces of the attack.

The process exited shortly after these activities, which may indicate an attempt to evade detection.

## Table of IoCs Detected

| IoC                        | Security Context                                                                                     |
|----------------------------|-----------------------------------------------------------------------------------------------------|
| 127.0.1.1:+24082           | This IP address is a loopback address, typically used for local communication. However, its usage in a linking context raises suspicion of potential command and control activity. |
| aborted-session-ping       | This file name suggests it may be related to session management. Its renaming and unlinking could indicate an attempt to cover tracks or remove evidence of malicious activity. |

## Chronological Log of Actions

### April 13, 2018

**13:41**
- READ a pipe (16 times)
- WRITE a pipe (13 times)
- RECVMSG a srcsink (8 times)
- WRITE a srcsink (7 times)
- MPROTECT a memory (4 times)
- READ the file: maps
- OPEN the file: maps
- OPEN a fileDir
- MMAP a memory
- CLOSE the file: maps
- LINK the file: 127.0.1.1:+24082
- CLONE the process: thunderbird
- CLOSE a fileDir
- RENAME a file
- SENDMSG a srcsink

**13:42**
- READ a pipe (34 times)
- WRITE a pipe (31 times)
- RECVMSG a srcsink (20 times)
- WRITE a srcsink (19 times)
- MPROTECT a memory (2 times)
- EXIT the process: thunderbird
- CLOSE a fileDir
- RENAME a file
- OPEN a fileDir
- UNLINK the file: aborted-session-ping
- RENAME the file: aborted-session-ping

This report highlights the suspicious activities of the "thunderbird" process, indicating potential APT behavior and the need for further investigation.