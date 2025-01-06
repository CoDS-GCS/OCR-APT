# Attack Report: tc3_anomalous_subgraph_21.csv

## Summary of Attack Behavior

On April 11, 2018, at 23:58, a series of log events were recorded that indicate potential anomalous behavior consistent with an advanced persistent threat (APT) attack. The logs show that a process closed multiple files, including `411.pkg-backup`, `libarchive.so.7`, and `libbz2.so.4`. Following this, the same process forked a new process, which may suggest an attempt to escalate privileges or establish persistence. The logs also indicate that the process memory-mapped (`MMAP`) the same libraries that were opened and closed, which is a common technique used by attackers to manipulate or execute code in memory without writing it to disk, thereby evading detection.

This sequence of events suggests that the attack may be in the **Privilege Escalation** or **Maintain Persistence** stage, as the attacker appears to be leveraging legitimate system libraries to execute potentially malicious actions.

## Table of Indicators of Compromise (IoCs)

| IoC                  | Security Context                                                                                     |
|----------------------|------------------------------------------------------------------------------------------------------|
| `411.pkg-backup`     | A legitimate backup file, but its closure may indicate an attempt to remove traces of malicious activity. High exploitation likelihood if used inappropriately. |
| `libarchive.so.7`    | A standard library for handling archive files. Its usage in `MMAP` and `OPEN` actions raises concerns about potential exploitation for code execution. Moderate to high exploitation likelihood. |
| `libbz2.so.4`        | A compression library commonly used in various applications. Similar to `libarchive.so.7`, its manipulation could indicate an attempt to execute arbitrary code. Moderate to high exploitation likelihood. |

## Chronological Log of Actions

| Timestamp           | Action Description                                      |
|---------------------|--------------------------------------------------------|
| 2018-04-11 23:58    | A process CLOSE the file: `411.pkg-backup`.           |
| 2018-04-11 23:58    | A process CLOSE the file: `libarchive.so.7`.          |
| 2018-04-11 23:58    | A process CLOSE the file: `libbz2.so.4`.              |
| 2018-04-11 23:58    | A process FORK a process.                              |
| 2018-04-11 23:58    | A process MMAP the file: `libarchive.so.7`.           |
| 2018-04-11 23:58    | A process MMAP the file: `libbz2.so.4`.               |
| 2018-04-11 23:58    | A process OPEN the file: `libarchive.so.7`.           |
| 2018-04-11 23:58    | A process OPEN the file: `libbz2.so.4`.               |

This report highlights the suspicious activities observed in the logs, indicating potential malicious intent and the need for further investigation into the processes involved.