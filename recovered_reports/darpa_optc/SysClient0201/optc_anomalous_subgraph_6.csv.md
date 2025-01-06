# Attack Report: Anomalous Activity Detected in optc_anomalous_subgraph_6.csv

## Summary of Attack Behavior

The logs from `optc_anomalous_subgraph_6.csv` indicate a series of anomalous activities primarily involving the processes `python.exe`, `svchost.exe`, `conhost.exe`, and `csrss.exe`. The timeline reveals multiple instances of process creation, opening, and termination, suggesting potential malicious behavior consistent with an Advanced Persistent Threat (APT) attack.

Key events include:
- **Initial Compromise**: The repeated opening and termination of `python.exe` at various timestamps (e.g., 09:54, 14:24) may indicate an attempt to execute a malicious script or payload.
- **Internal Reconnaissance**: The frequent interactions between `svchost.exe` and `csrss.exe` suggest reconnaissance activities, as these processes are often involved in system management and user session handling.
- **Command and Control**: The consistent opening of `svchost.exe` and `conhost.exe` processes indicates potential command and control (C2) communications, as these processes can be leveraged by attackers to maintain control over compromised systems.
- **Covering Tracks**: The termination of processes like `conhost.exe` and `csrss.exe` at various points may indicate efforts to cover tracks and remove evidence of the attack.

Overall, the logs reflect a sophisticated approach to maintaining persistence and executing commands, characteristic of APT behavior.

## Table of IoCs Detected

| IoC            | Security Context                                                                                     |
|----------------|------------------------------------------------------------------------------------------------------|
| python.exe     | A legitimate scripting language often used for automation and data analysis. High likelihood of exploitation if used to execute malicious scripts. |
| svchost.exe    | A critical system process that hosts multiple Windows services. Can be exploited to run malicious services or processes. |
| conhost.exe    | A legitimate console host for command-line applications. Can be exploited to execute commands stealthily. |
| csrss.exe      | A critical Windows process responsible for handling user sessions. High risk if exploited, as it can lead to privilege escalation. |

## Chronological Log of Actions (Organized by Minute)

### 09:54
- `python.exe` OPEN
- `python.exe` TERMINATE
- `python.exe` CREATE

### 09:58
- `svchost.exe` OPEN `csrss.exe`

### 10:02
- `svchost.exe` OPEN `python.exe`

### 10:09
- `svchost.exe` OPEN `conhost.exe`
- `conhost.exe` TERMINATE `conhost.exe`
- `conhost.exe` OPEN `conhost.exe`

### 10:11
- `svchost.exe` OPEN `conhost.exe` (2 times)
- `csrss.exe` OPEN `csrss.exe`
- `svchost.exe` TERMINATE `svchost.exe`
- `svchost.exe` OPEN `svchost.exe`
- `svchost.exe` OPEN `csrss.exe`

### 10:12
- `svchost.exe` OPEN `conhost.exe`

### 10:19
- `svchost.exe` OPEN `conhost.exe`

### 10:20
- `conhost.exe` TERMINATE `conhost.exe`
- `conhost.exe` OPEN `conhost.exe`

### 10:23
- `svchost.exe` OPEN `csrss.exe`

### 10:30
- `svchost.exe` OPEN `conhost.exe`

### 10:31
- `conhost.exe` OPEN `conhost.exe`

### 10:32
- `svchost.exe` OPEN `conhost.exe`

### 10:33
- `svchost.exe` OPEN `svchost.exe` (2 times)

### 10:36
- `svchost.exe` OPEN `conhost.exe`
- `conhost.exe` OPEN `conhost.exe`

### 10:41
- `svchost.exe` TERMINATE `svchost.exe`
- `svchost.exe` OPEN `svchost.exe`
- `svchost.exe` OPEN `csrss.exe`
- `csrss.exe` TERMINATE `csrss.exe`
- `csrss.exe` OPEN `csrss.exe`

### 10:52
- `conhost.exe` TERMINATE `conhost.exe`
- `svchost.exe` OPEN `csrss.exe`

### 10:53
- `svchost.exe` OPEN `conhost.exe`

### 10:54
- `svchost.exe` OPEN `python.exe`
- `svchost.exe` OPEN `conhost.exe`

### 14:11
- `conhost.exe` TERMINATE `conhost.exe`
- `svchost.exe` OPEN `csrss.exe`
- `csrss.exe` OPEN `csrss.exe`

### 14:17
- `conhost.exe` OPEN `conhost.exe`

### 14:18
- `svchost.exe` OPEN `conhost.exe`

### 14:24
- `svchost.exe` OPEN `python.exe`
- `python.exe` OPEN `python.exe`

### 14:25
- `svchost.exe` OPEN `python.exe`
- `csrss.exe` TERMINATE `csrss.exe`

### 14:26
- `svchost.exe` OPEN `csrss.exe`

### 14:33
- `svchost.exe` OPEN `conhost.exe`
- `conhost.exe` OPEN `conhost.exe`

### 14:38
- `svchost.exe` OPEN `conhost.exe` (2 times)
- `conhost.exe` TERMINATE `conhost.exe`

### 14:41
- `conhost.exe` OPEN `csrss.exe` (3 times)
- `csrss.exe` TERMINATE `csrss.exe` (2 times)
- `svchost.exe` OPEN `csrss.exe` (2 times)
- `conhost.exe` CREATE `conhost.exe`
- `csrss.exe` OPEN `csrss.exe`
- `csrss.exe` CREATE `csrss.exe`
- `csrss.exe` CREATE `conhost.exe`
- `conhost.exe` TERMINATE `conhost.exe`
- `conhost.exe` OPEN `conhost.exe`
- `svchost.exe` OPEN `conhost.exe`

### 14:42
- `conhost.exe` TERMINATE `conhost.exe`
- `conhost.exe` OPEN `conhost.exe`

### 14:48
- `svchost.exe` OPEN `conhost.exe`

### 14:49
- `python.exe` OPEN `python.exe`

### 14:50
- `svchost.exe` OPEN `python.exe`

### 14:58
- `svchost.exe` OPEN `conhost.exe`
- `conhost.exe` TERMINATE `conhost.exe`
- `conhost.exe` OPEN `conhost.exe`

### 14:59
- `svchost.exe` OPEN `conhost.exe`
- `conhost.exe` TERMINATE `conhost.exe`

### 15:00
- `svchost.exe` OPEN `conhost.exe`

### 15:01
- `svchost.exe` OPEN `csrss.exe`
- `svchost.exe` OPEN `conhost.exe`

### 15:02
- `svchost.exe` OPEN `conhost.exe`
- `conhost.exe` OPEN `conhost.exe`

### 15:03
- `svchost.exe` OPEN `conhost.exe`
- `conhost.exe` TERMINATE `conhost.exe`

### 15:04
- `conhost.exe` TERMINATE `conhost.exe`
- `svchost.exe` TERMINATE `svchost.exe`

### 15:06
- `svchost.exe` OPEN `svchost.exe`

### 15:07
- `svchost.exe` OPEN `svchost.exe`

### 15:08
- `conhost.exe` TERMINATE `conhost.exe`

### 15:09
- `svchost.exe` OPEN `conhost.exe`

### 15:11
- `conhost.exe` TERMINATE `conhost.exe`
- `csrss.exe` TERMINATE `csrss.exe`
- `csrss.exe` OPEN `csrss.exe`
- `conhost.exe` OPEN `csrss.exe`
- `conhost.exe` OPEN `conhost.exe`
- `conhost.exe` CREATE `csrss.exe`
- `svchost.exe` OPEN `csrss.exe`
- `svchost.exe` OPEN `conhost.exe`

### 15:29
- `svchost.exe` OPEN `conhost.exe`

### 15:30
- `svchost.exe` OPEN `csrss.exe` 

This report highlights the suspicious activities observed in the logs, indicating a potential APT attack that requires further investigation and remediation.