# Attack Report: tc3_anomalous_subgraph_23.csv

## Summary of Attack Behavior

The logs from the document indicate a series of suspicious activities that suggest a potential Advanced Persistent Threat (APT) operation. The timeline reveals multiple instances of file access, process execution, and network communication, which are characteristic of various APT stages.

1. **Initial Compromise**: The logs show the execution of the `lsof` command, which is often used for listing open files and can be exploited to gather information about the system. This could indicate an initial reconnaissance phase where the attacker is assessing the environment.

2. **Internal Reconnaissance**: The frequent access to system files such as `kmem`, `services`, and `.lsof_ta1-cadets` suggests that the attacker is gathering information about the system's memory and services, which is typical during the internal reconnaissance stage.

3. **Command and Control**: The presence of an external IP address (128.55.12.10) indicates potential command and control communication, which is a critical phase in APT operations.

4. **Data Exfiltration**: The repeated writing to `gather_stats_uma.txt` may suggest that the attacker is collecting and possibly exfiltrating data.

5. **Covering Tracks**: The use of various system files and processes, along with the execution of commands like `head`, indicates attempts to manipulate or hide the attacker's presence.

## Table of IoCs Detected

| IoC                     | Security Context                                                                                     |
|-------------------------|-----------------------------------------------------------------------------------------------------|
| 128.55.12.10           | External IP address potentially used for command and control. High likelihood of exploitation.     |
| lsof                    | A legitimate command for listing open files; however, its usage in this context raises suspicion.   |
| gather_stats_uma.txt   | A file that may be used for data collection; frequent writes suggest potential data exfiltration.   |
| .lsof_ta1-cadets       | A temporary file likely used for logging; its access pattern indicates potential malicious activity. |
| services                | A system file that lists active services; frequent access suggests reconnaissance.                  |
| kmem                    | A memory file that can provide sensitive information; its access indicates potential exploitation.   |
| termcap.db              | A database file for terminal capabilities; its access may indicate attempts to manipulate terminal settings. |
| top_procs.txt          | A file that may log active processes; its access could indicate monitoring of system activity.      |
| gpt                     | A file related to GUID partition tables; its access may indicate attempts to manipulate disk partitions. |
| fd                      | A file descriptor that can be used to access files; its usage in this context is suspicious.       |
| dtrace                  | A tracing tool that can be used for debugging; its access may indicate attempts to monitor system behavior. |
| dev                     | A directory for device files; access may indicate attempts to interact with hardware components.    |
| reroot                  | A file related to system reboot processes; its access may indicate attempts to manipulate system state. |
| gptid                   | A file related to GUID partition identifiers; its access may indicate attempts to manipulate disk structures. |
| usb                     | A directory for USB devices; access may indicate attempts to interact with removable media.         |
| mem                     | A file that provides access to system memory; its access indicates potential exploitation.          |

## Chronological Log of Actions

### 2018-04-11
- **16:41**
  - READ: kmem (2 times)
  - LSEEK: kmem (2 times)
  - WRITE: gather_stats_uma.txt (2 times)
  - CLOSE: .lsof_ta1-cadets
  - OPEN: gpt
  - OPEN: fd
  - OPEN: dtrace
  - OPEN: dev
  - OPEN: .lsof_ta1-cadets
  - LSEEK: services
  - FORK: process
  - CLOSE: dev
  - CLOSE: services
  - CLOSE: reroot
  - CLOSE: gptid
  - CLOSE: gpt
  - CLOSE: fd
  - CLOSE: dtrace
  - EXECUTE: lsof
  - CLOSE: usb
  - OPEN: gptid
  - WRITE: .lsof_ta1-cadets
  - READ: services
  - OPEN: usb
  - OPEN: services
  - OPEN: reroot
  - OPEN: mem
  - OPEN: kmem

- **22:55**
  - WRITE: gather_stats_uma.txt
  - READ: services
  - READ: kmem
  - READ: .lsof_ta1-cadets
  - OPEN: mem
  - OPEN: services
  - OPEN: kmem
  - OPEN: .lsof_ta1-cadets
  - LSEEK: services
  - LSEEK: kmem
  - FORK: process
  - EXECUTE: lsof
  - CLOSE: .lsof_ta1-cadets

### 2018-04-12
- **00:07**
  - CREATE_OBJECT: pipe
  - FORK: process
  - CLOSE: top_procs.txt
  - CLOSE: termcap.db
  - EXECUTE: head
  - OPEN: termcap.db
  - WRITE: top_procs.txt
  - READ: termcap.db
  - OPEN: top_procs.txt

- **18:08**
  - READ: services
  - READ: kmem
  - READ: .lsof_ta1-cadets
  - OPEN: services
  - OPEN: mem
  - OPEN: kmem
  - OPEN: .lsof_ta1-cadets
  - FORK: process
  - EXECUTE: lsof
  - EXECUTE: date
  - CLOSE: services
  - CLOSE: .lsof_ta1-cadets
  - LSEEK: kmem
  - WRITE: gather_stats_uma.txt
  - LSEEK: services