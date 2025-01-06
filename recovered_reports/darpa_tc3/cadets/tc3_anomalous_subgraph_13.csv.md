# Attack Report: tc3_anomalous_subgraph_13.csv

## Summary of Attack Behavior

The logs from the document indicate a series of actions that suggest a potential advanced persistent threat (APT) operation. The key events occurred on two separate dates, April 11 and April 12, 2018, involving the execution of processes, file manipulations, and memory mapping of critical system files. 

### Key Events:
- **April 11, 2018, 21:37**: The process executed `vmstat`, which is a legitimate system monitoring tool, but its usage in this context raises suspicion. The process also opened, wrote to, and closed the file `kernel_zones.txt`, indicating potential reconnaissance or data manipulation activities.
- **April 12, 2018, 09:40**: Similar actions were repeated with the file `kernel_mem.txt`, including execution of `vmstat`, which again suggests a focus on system memory and kernel information. The repeated use of the same files and processes over two days indicates a methodical approach typical of APTs, likely aiming for internal reconnaissance and data exfiltration.

These actions align with the **Internal Reconnaissance** stage of an APT attack, where the attacker gathers information about the system and its environment to plan further actions.

## Table of Indicators of Compromise (IoCs)

| IoC                  | Security Context                                                                                     |
|----------------------|-----------------------------------------------------------------------------------------------------|
| kernel_zones.txt     | A system file that may contain sensitive kernel memory information. Legitimate usage for monitoring, but could be exploited for reconnaissance. |
| libkvm.so.7         | A shared library used for kernel memory access. Legitimate for system operations, but its exploitation can lead to unauthorized access to kernel data. |
| libmemstat.so.3     | A library for memory statistics. Legitimate for performance monitoring, but can be misused to gather sensitive memory information. |
| vmstat               | A legitimate system monitoring tool. While useful for performance analysis, its execution in a suspicious context raises red flags. |
| kernel_mem.txt      | A file that may store kernel memory data. Legitimate for system diagnostics, but could be targeted for data exfiltration. |

## Chronological Log of Actions

### April 11, 2018
- **21:37**: A process CLOSE a file.
- **21:37**: A process CLOSE the file: `kernel_zones.txt`.
- **21:37**: A process CLOSE the file: `libkvm.so.7`.
- **21:37**: A process CLOSE the file: `libmemstat.so.3`.
- **21:37**: A process EXECUTE the file: `vmstat`.
- **21:37**: A process FORK a process.
- **21:37**: A process MMAP the file: `libkvm.so.7`.
- **21:37**: A process MMAP the file: `libmemstat.so.3`.
- **21:37**: A process OPEN the file: `kernel_zones.txt`.
- **21:37**: A process OPEN the file: `libkvm.so.7`.
- **21:37**: A process OPEN the file: `libmemstat.so.3`.
- **21:37**: A process WRITE the file: `kernel_zones.txt`.

### April 12, 2018
- **09:40**: A process CLOSE a file.
- **09:40**: A process CLOSE the file: `kernel_mem.txt`.
- **09:40**: A process CLOSE the file: `libkvm.so.7`.
- **09:40**: A process CLOSE the file: `libmemstat.so.3`.
- **09:40**: A process EXECUTE the file: `vmstat`.
- **09:40**: A process FORK a process.
- **09:40**: A process MMAP the file: `libkvm.so.7`.
- **09:40**: A process MMAP the file: `libmemstat.so.3`.
- **09:40**: A process OPEN the file: `kernel_mem.txt`.
- **09:40**: A process OPEN the file: `libkvm.so.7`.
- **09:40**: A process OPEN the file: `libmemstat.so.3`.
- **09:40**: A process WRITE the file: `kernel_mem.txt`.

This report highlights the suspicious activities observed in the logs, indicating a potential APT operation focused on internal reconnaissance and data manipulation. Further investigation is recommended to assess the impact and scope of the activities.