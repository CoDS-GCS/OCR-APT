# Attack Report: tc3_anomalous_subgraph_7.csv

## Summary of Attack Behavior

The logs from the document indicate a sequence of actions that suggest potential anomalous behavior consistent with an APT attack. The key events are as follows:

1. **Process Execution**: At **2018-04-12 14:35**, a process named `uname` was executed. This command is typically used to display system information, which could indicate an attempt at internal reconnaissance by the attacker to gather information about the system.

2. **Process Forking**: At the same timestamp, a process was forked. This action may suggest that the attacker is attempting to create a new process to carry out further actions, potentially indicating the **Command and Control** stage of the APT lifecycle.

3. **Process Exit**: At **2018-04-12 14:38**, a process exited. This could indicate the completion of the attacker's actions or a cleanup phase, which is often associated with the **Covering Tracks** stage.

Overall, the sequence of events suggests an initial reconnaissance followed by potential command execution, with the possibility of the attacker attempting to cover their tracks.

## Table of Indicators of Compromise (IoCs)

| IoC    | Security Context                                                                                     |
|--------|------------------------------------------------------------------------------------------------------|
| uname  | A legitimate command used to display system information. However, its execution in this context may indicate reconnaissance by an attacker. |
| FORK   | A system call used to create a new process. While common in legitimate applications, its usage in conjunction with other suspicious activities raises the likelihood of exploitation. |
| EXIT   | Indicates the termination of a process. This can be a normal operation, but in the context of suspicious activity, it may suggest an attempt to hide traces of malicious actions. |

## Chronological Log of Actions

| Time                | Action                          |
|---------------------|---------------------------------|
| 2018-04-12 14:35    | A process EXECUTE the file: uname |
| 2018-04-12 14:35    | A process FORK a process        |
| 2018-04-12 14:38    | A process EXIT a process        | 

This report highlights the potential malicious behavior observed in the logs, emphasizing the need for further investigation and monitoring of the affected systems.