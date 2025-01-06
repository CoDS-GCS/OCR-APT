# Attack Report: Anomalous Activity Detected in optc_anomalous_subgraph_8.csv

## Summary of Attack Behavior

The logs from `optc_anomalous_subgraph_8.csv` indicate a series of suspicious activities primarily involving the processes `svchost.exe`, `conhost.exe`, and `python.exe`. The timeline of events suggests a potential Advanced Persistent Threat (APT) attack characterized by the following stages:

1. **Initial Compromise**: The attack appears to have initiated with the `svchost.exe` process modifying system files and creating tasks, which is a common tactic for establishing a foothold in the system.
   
2. **Internal Reconnaissance**: The frequent creation and modification of tasks by `svchost.exe` indicate an attempt to gather information about the system and its processes.

3. **Command and Control**: The outbound flow initiated by `svchost.exe` suggests that the attacker may have established a command and control channel to communicate with external servers.

4. **Privilege Escalation**: The repeated creation and modification of tasks, along with the spawning of `conhost.exe`, may indicate attempts to escalate privileges within the system.

5. **Maintain Persistence**: The continuous creation and modification of tasks over an extended period (from 10:54 to 15:47) suggest efforts to maintain persistence within the compromised environment.

6. **Covering Tracks**: The termination of processes and modification of tasks may indicate attempts to erase traces of the attack.

## Table of IoCs Detected

| IoC            | Security Context                                                                                     |
|----------------|------------------------------------------------------------------------------------------------------|
| python.exe     | A legitimate process used for running Python scripts. However, its presence in unusual contexts may indicate exploitation or malicious use. |
| svchost.exe    | A critical Windows process that hosts multiple services. Its misuse can indicate malware activity, especially when it performs unexpected actions like creating tasks or modifying files. |
| conhost.exe    | A legitimate console host process. However, frequent spawning and termination can suggest malicious activity, particularly in conjunction with other suspicious processes. |

## Chronological Log of Actions

| Time (UTC)         | Action                                                                                     |
|--------------------|--------------------------------------------------------------------------------------------|
| 2019-09-23 10:54   | svchost.exe MODIFY the file: ntuser.ini                                                  |
| 2019-09-23 10:54   | svchost.exe READ the file: cmd.exe                                                       |
| 2019-09-23 10:54   | svchost.exe MODIFY the file: mantra_content.tar                                          |
| 2019-09-23 10:55   | svchost.exe START a task (3 times)                                                       |
| 2019-09-23 10:55   | svchost.exe OPEN the process: conhost.exe                                                |
| 2019-09-23 10:55   | svchost.exe START_OUTBOUND the flow                                                      |
| 2019-09-23 10:56   | conhost.exe OPEN the process: conhost.exe                                                |
| 2019-09-23 10:56   | svchost.exe CREATE the process: conhost.exe                                              |
| 2019-09-23 10:56   | svchost.exe MODIFY a task                                                                  |
| 2019-09-23 10:56   | svchost.exe START a task                                                                   |
| 2019-09-23 10:57   | svchost.exe START a task                                                                   |
| 2019-09-23 10:58   | svchost.exe START a task                                                                   |
| 2019-09-23 10:59   | svchost.exe START a task                                                                   |
| 2019-09-23 11:00   | svchost.exe START a task (2 times)                                                        |
| 2019-09-23 11:01   | svchost.exe START a task                                                                   |
| 2019-09-23 11:02   | svchost.exe START a task                                                                   |
| 2019-09-23 11:03   | svchost.exe START a task                                                                   |
| 2019-09-23 11:04   | svchost.exe START a task                                                                   |
| 2019-09-23 11:05   | svchost.exe START a task                                                                   |
| 2019-09-23 11:06   | svchost.exe START a task                                                                   |
| 2019-09-23 11:07   | svchost.exe START a task                                                                   |
| 2019-09-23 11:08   | svchost.exe START a task                                                                   |
| 2019-09-23 11:08   | svchost.exe CREATE the process: conhost.exe                                              |
| 2019-09-23 11:08   | conhost.exe TERMINATE the process: conhost.exe                                            |
| 2019-09-23 11:08   | conhost.exe OPEN the process: conhost.exe                                                |
| 2019-09-23 11:09   | svchost.exe START a task (3 times)                                                       |
| 2019-09-23 11:10   | svchost.exe START a task                                                                   |
| 2019-09-23 11:11   | conhost.exe OPEN the process: conhost.exe (3 times)                                      |
| 2019-09-23 11:11   | svchost.exe START a task (2 times)                                                        |
| 2019-09-23 11:11   | conhost.exe TERMINATE the process: conhost.exe                                            |
| 2019-09-23 11:11   | svchost.exe CREATE a task                                                                  |
| 2019-09-23 11:11   | svchost.exe CREATE the process: conhost.exe                                              |
| 2019-09-23 11:11   | svchost.exe DELETE a task                                                                  |
| 2019-09-23 11:12   | svchost.exe START a task                                                                   |
| 2019-09-23 11:13   | svchost.exe START a task                                                                   |
| 2019-09-23 11:14   | svchost.exe START a task                                                                   |
| 2019-09-23 11:15   | python.exe OPEN the process: python.exe                                                  |
| 2019-09-23 12:27   | python.exe TERMINATE the process: python.exe                                             |
| 2019-09-23 12:28   | svchost.exe START a task                                                                   |
| 2019-09-23 12:29   | svchost.exe START a task                                                                   |
| 2019-09-23 12:30   | svchost.exe START a task                                                                   |
| 2019-09-23 12:30   | svchost.exe MODIFY a task                                                                  |
| 2019-09-23 12:31   | svchost.exe START a task                                                                   |
| 2019-09-23 12:32   | svchost.exe START a task                                                                   |
| 2019-09-23 12:33   | svchost.exe MODIFY a task (2 times)                                                       |
| 2019-09-23 12:33   | svchost.exe START a task                                                                   |
| 2019-09-23 12:33   | svchost.exe MODIFY the file: 0                                                            |
| 2019-09-23 12:34   | svchost.exe START a task                                                                   |
| 2019-09-23 12:35   | svchost.exe START a task (3 times)                                                        |
| 2019-09-23 12:36   | svchost.exe START a task                                                                   |
| 2019-09-23 12:37   | svchost.exe START a task                                                                   |
| 2019-09-23 12:38   | svchost.exe START a task                                                                   |
| 2019-09-23 12:39   | svchost.exe START a task (2 times)                                                        |
| 2019-09-23 12:40   | svchost.exe START a task                                                                   |
| 2019-09-23 12:41   | svchost.exe START a task (2 times)                                                        |
| 2019-09-23 12:41   | svchost.exe DELETE a task                                                                  |
| 2019-09-23 12:41   | svchost.exe CREATE a task                                                                  |
| 2019-09-23 12:42   | svchost.exe START a task                                                                   |
| 2019-09-23 12:43   | svchost.exe START a task                                                                   |
| 2019-09-23 12:44   | svchost.exe START a task                                                                   |
| 2019-09-23 12:45   | svchost.exe START a task                                                                   |
| 2019-09-23 12:46   | svchost.exe START a task                                                                   |
| 2019-09-23 12:47   | svchost.exe START a task                                                                   |
| 2019-09-23 12:48   | svchost.exe START a task                                                                   |
| 2019-09-23 12:49   | svchost.exe MODIFY a task                                                                  |
| 2019-09-23 12:49   | svchost.exe START a task                                                                   |
| 2019-09-23 12:50   | svchost.exe START a task                                                                   |
| 2019-09-23 12:51   | svchost.exe START a task                                                                   |
| 2019-09-23 12:53   | conhost.exe TERMINATE the process: conhost.exe                                            |
| 2019-09-23 12:53   | svchost.exe CREATE the process: conhost.exe                                              |
| 2019-09-23 12:53   | svchost.exe START a task                                                                   |
| 2019-09-23 12:54   | svchost.exe START a task                                                                   |
| 2019-09-23 12:55   | svchost.exe START a task                                                                   |
| 2019-09-23 12:56   | svchost.exe START a task                                                                   |
| 2019-09-23 15:24   | svchost.exe START a task (2 times)                                                        |
| 2019-09-23 15:24   | svchost.exe MODIFY the file: mantra_content.tar                                          |
| 2019-09-23 15:24   | svchost.exe MODIFY a task                                                                  |
| 2019-09-23 15:25   | svchost.exe START a task                                                                   |
| 2019-09-23 15:26   | svchost.exe START a task                                                                   |
| 2019-09-23 15:27   | conhost.exe OPEN the process: conhost.exe                                                |
| 2019-09-23 15:27   | svchost.exe CREATE the process: conhost.exe                                              |
| 2019-09-23 15:27   | svchost.exe START a task                                                                   |
| 2019-09-23 15:28   | svchost.exe START a task                                                                   |
| 2019-09-23 15:29   | svchost.exe START a task                                                                   |
| 2019-09-23 15:30   | svchost.exe START a task                                                                   |
| 2019-09-23 15:30   | conhost.exe OPEN the process: conhost.exe                                                |
| 2019-09-23 15:30   | svchost.exe CREATE the process: conhost.exe                                              |
| 2019-09-23 15:31   | svchost.exe START a task                                                                   |
| 2019-09-23 15:32   | svchost.exe START a task                                                                   |
| 2019-09-23 15:33   | svchost.exe START a task                                                                   |
| 2019-09-23 15:34   | svchost.exe START a task                                                                   |
| 2019-09-23 15:35   | svchost.exe START a task (2 times)                                                        |
| 2019-09-23 15:36   | svchost.exe START a task                                                                   |
| 2019-09-23 15:37   | svchost.exe START a task (3 times)                                                        |
| 2019-09-23 15:37   | svchost.exe CREATE the process: sc.exe                                                   |
| 2019-09-23 15:38   | svchost.exe START a task                                                                   |
| 2019-09-23 15:39   | svchost.exe START a task                                                                   |
| 2019-09-23 15:40   | svchost.exe START a task                                                                   |
| 2019-09-23 15:41   | svchost.exe START a task (3 times)                                                        |
| 2019-09-23 15:41   | svchost.exe CREATE a task                                                                  |
| 2019-09-23 15:41   | svchost.exe DELETE a task                                                                  |
| 2019-09-23 15:42   | svchost.exe START a task                                                                   |
| 2019-09-23 15:43   | svchost.exe START a task                                                                   |
| 2019-09-23 15:44   | svchost.exe START a task                                                                   |
| 2019-09-23 15:45   | svchost.exe START a task                                                                   |
| 2019-09-23 15:46   | svchost.exe START a task                                                                   |
| 2019-09-23 15:47   | svchost.exe START a task                                                                   |