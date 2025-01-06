# Attack Report: Anomalous Activity Detected in optc_anomalous_subgraph_11.csv

## Summary of Attack Behavior

The logs from the document indicate a series of anomalous activities primarily involving the processes `svchost.exe`, `conhost.exe`, and `python.exe`. The events span from 12:32 to 15:08 on September 24, 2019, showcasing a pattern of process creation, opening, and termination that suggests potential malicious behavior.

Key events include:
- **Process Creation and Termination**: Multiple instances of `svchost.exe` creating and terminating processes, including `conhost.exe` and `python.exe`, indicate a possible attempt to manipulate system processes for unauthorized actions.
- **Frequent Open Actions**: The `svchost.exe` process frequently opened other instances of itself and other processes, which is characteristic of lateral movement or privilege escalation attempts.
- **Repeated Termination of `conhost.exe` and `python.exe`**: The termination of these processes could suggest an effort to cover tracks or disrupt legitimate operations.

These behaviors align with the **Internal Reconnaissance** and **Lateral Movement** stages of an APT attack, where the attacker seeks to explore the environment and escalate privileges.

## Indicators of Compromise (IoCs)

| IoC                        | Security Context                                                                                     |
|----------------------------|------------------------------------------------------------------------------------------------------|
| `conhost.exe`             | A legitimate Windows process used to host command-line applications. However, it can be exploited to execute malicious commands. |
| `svchost.exe`             | A critical system process that can be exploited by malware to run malicious services. Its frequent creation and termination are suspicious. |
| `python.exe`              | A legitimate interpreter for Python scripts, but can be used to execute malicious scripts if compromised. |
| `cmd.exe`                 | The Windows command prompt, often used for legitimate administrative tasks, but can be exploited for command execution by attackers. |
| `UserNotPresentSession.etl` | A log file that may contain session data. Its access by `svchost.exe` could indicate an attempt to gather information about user sessions. |

## Chronological Log of Actions

### 12:32 - 12:59
- 12:32: `svchost.exe` OPEN the process: `conhost.exe`
- 12:32: `svchost.exe` OPEN the process: `svchost.exe`
- 12:34: `svchost.exe` OPEN the process: `svchost.exe` (2 times)
- 12:34: `svchost.exe` OPEN the process: `conhost.exe`
- 12:36: `svchost.exe` OPEN the process: `conhost.exe` (2 times)
- 12:36: `svchost.exe` OPEN the process: `svchost.exe` (2 times)
- 12:39: `svchost.exe` OPEN the process: `conhost.exe`
- 12:39: `svchost.exe` OPEN the process: `svchost.exe`
- 12:41: `svchost.exe` OPEN the process: `svchost.exe` (2 times)
- 12:41: `svchost.exe` OPEN the process: `conhost.exe`
- 12:42: `conhost.exe` TERMINATE the process: `conhost.exe`
- 12:43: `conhost.exe` TERMINATE the process: `conhost.exe`
- 12:43: `svchost.exe` OPEN the process: `conhost.exe`
- 12:43: `svchost.exe` OPEN the process: `svchost.exe`
- 12:46: `svchost.exe` OPEN the process: `conhost.exe` (2 times)
- 12:46: `svchost.exe` OPEN the process: `svchost.exe` (2 times)
- 12:46: `conhost.exe` TERMINATE the process: `conhost.exe`
- 12:48: `svchost.exe` OPEN the process: `conhost.exe`
- 12:48: `svchost.exe` OPEN the process: `svchost.exe`
- 12:51: `svchost.exe` OPEN the process: `svchost.exe` (3 times)
- 12:51: `svchost.exe` OPEN the process: `conhost.exe` (2 times)
- 12:53: `svchost.exe` OPEN the process: `svchost.exe`
- 12:53: `svchost.exe` TERMINATE the process: `svchost.exe`
- 12:54: `svchost.exe` OPEN the process: `conhost.exe`
- 12:54: `svchost.exe` OPEN the process: `svchost.exe`
- 12:55: `svchost.exe` READ the file: `UserNotPresentSession.etl` (2 times)
- 12:55: `conhost.exe` TERMINATE the process: `conhost.exe`
- 12:55: `svchost.exe` OPEN the process: `conhost.exe`
- 12:56: `svchost.exe` OPEN the process: `python.exe` (4 times)
- 12:56: `svchost.exe` OPEN the process: `conhost.exe` (3 times)
- 12:56: `python.exe` TERMINATE the process: `python.exe` (2 times)
- 12:56: `svchost.exe` OPEN the process: `svchost.exe` (2 times)

### 14:44 - 15:08
- 14:44: `svchost.exe` OPEN the process: `svchost.exe`
- 14:45: `svchost.exe` OPEN the process: `svchost.exe` (5 times)
- 14:45: `svchost.exe` TERMINATE the process: `svchost.exe`
- 14:46: `svchost.exe` OPEN the process: `conhost.exe` (2 times)
- 14:46: `svchost.exe` OPEN the process: `svchost.exe` (2 times)
- 14:46: `conhost.exe` OPEN the process: `conhost.exe`
- 14:46: `conhost.exe` TERMINATE the process: `conhost.exe`
- 14:47: `svchost.exe` OPEN the process: `svchost.exe` (5 times)
- 14:47: `svchost.exe` TERMINATE the process: `svchost.exe` (2 times)
- 14:48: `svchost.exe` OPEN the process: `svchost.exe`
- 14:49: `svchost.exe` OPEN the process: `svchost.exe` (5 times)
- 14:49: `conhost.exe` OPEN the process: `cmd.exe`
- 14:49: `svchost.exe` CREATE the process: `cmd.exe`
- 14:49: `svchost.exe` TERMINATE the process: `svchost.exe`
- 14:50: `svchost.exe` OPEN the process: `svchost.exe`
- 14:51: `svchost.exe` OPEN the process: `svchost.exe` (6 times)
- 14:51: `cmd.exe` TERMINATE the process: `cmd.exe`
- 14:51: `conhost.exe` OPEN the process: `cmd.exe`
- 14:51: `svchost.exe` TERMINATE the process: `svchost.exe`
- 14:52: `svchost.exe` OPEN the process: `svchost.exe` (2 times)
- 14:53: `svchost.exe` OPEN the process: `svchost.exe`
- 14:54: `svchost.exe` OPEN the process: `svchost.exe` (9 times)
- 14:54: `svchost.exe` TERMINATE the process: `svchost.exe`
- 14:55: `svchost.exe` OPEN the process: `svchost.exe` (3 times)
- 14:55: `svchost.exe` TERMINATE the process: `svchost.exe` (2 times)
- 14:55: `conhost.exe` OPEN the process: `conhost.exe`
- 14:55: `conhost.exe` TERMINATE the process: `conhost.exe`
- 14:55: `svchost.exe` CREATE the process: `svchost.exe`
- 14:55: `svchost.exe` OPEN the process: `conhost.exe`
- 14:55: `svchost.exe` OPEN the process: `python.exe` 

This report highlights the suspicious activities observed in the logs, indicating potential malicious intent and the need for further investigation into the processes involved.