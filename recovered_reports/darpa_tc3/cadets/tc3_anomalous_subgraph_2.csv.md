# Attack Report: tc3_anomalous_subgraph_2.csv

## Summary of Attack Behavior

On April 11, 2018, at 16:37, a series of actions were logged that indicate potential malicious activity. The logs show that multiple shared libraries were opened, memory-mapped, and subsequently closed by a process. Notably, the executable `ifconfig` was also executed during this timeframe. The sequence of events suggests that the attacker may have been conducting reconnaissance or attempting to manipulate network configurations, which aligns with the **Internal Reconnaissance** stage of an APT attack. The use of system libraries and network configuration tools raises concerns about the legitimacy of the actions taken, indicating a possible compromise.

## Table of Indicators of Compromise (IoCs)

| IoC                  | Security Context                                                                                     |
|----------------------|------------------------------------------------------------------------------------------------------|
| lib80211.so.1       | A legitimate library for wireless networking. However, its usage in this context may indicate exploitation. |
| libbsdxml.so.4      | A library used for XML parsing in BSD systems. Its presence in the logs could suggest potential misuse. |
| libjail.so.1        | A library that provides jail functionality for process isolation. Its manipulation may indicate an attempt to evade detection. |
| libsbuf.so.6        | A library for buffered I/O operations. While legitimate, its usage in conjunction with other IoCs raises suspicion. |
| ifconfig             | A standard network configuration tool. Its execution in this context may indicate an attempt to alter network settings maliciously. |

## Chronological Log of Actions

| Timestamp           | Action                                           |
|---------------------|--------------------------------------------------|
| 2018-04-11 16:37    | A process CLOSE a file                          |
| 2018-04-11 16:37    | A process CLOSE the file: lib80211.so.1       |
| 2018-04-11 16:37    | A process CLOSE the file: libbsdxml.so.4      |
| 2018-04-11 16:37    | A process CLOSE the file: libjail.so.1        |
| 2018-04-11 16:37    | A process CLOSE the file: libsbuf.so.6        |
| 2018-04-11 16:37    | A process EXECUTE the file: ifconfig           |
| 2018-04-11 16:37    | A process MMAP the file: lib80211.so.1        |
| 2018-04-11 16:37    | A process MMAP the file: libbsdxml.so.4       |
| 2018-04-11 16:37    | A process MMAP the file: libjail.so.1         |
| 2018-04-11 16:37    | A process MMAP the file: libsbuf.so.6         |
| 2018-04-11 16:37    | A process OPEN the file: lib80211.so.1        |
| 2018-04-11 16:37    | A process OPEN the file: libbsdxml.so.4       |
| 2018-04-11 16:37    | A process OPEN the file: libjail.so.1         |
| 2018-04-11 16:37    | A process OPEN the file: libsbuf.so.6         |

This report highlights the suspicious activities logged on the specified date and time, indicating potential malicious intent and the need for further investigation into the processes and files involved.