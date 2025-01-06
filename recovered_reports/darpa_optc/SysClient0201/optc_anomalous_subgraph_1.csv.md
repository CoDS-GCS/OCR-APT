# Attack Report: Anomalous Activity Detected in optc_anomalous_subgraph_1.csv

## Summary of Attack Behavior

The logs from the document indicate a series of suspicious activities primarily involving the process `svchost.exe`. The attack appears to have been executed in multiple stages, including **Initial Compromise**, **Command and Control**, and **Maintain Persistence**. 

Key events include:

- **Initial Compromise**: The process `svchost.exe` initiated multiple commands to create and delete files associated with `r2te1030`, suggesting an attempt to establish a foothold on the system. This was observed on **2019-09-23 12:58**, where several files were deleted and then recreated, indicating potential malware behavior.

- **Command and Control**: The external IP address `142.20.56.202` was repeatedly contacted, with multiple inbound messages logged. This activity was noted throughout the day, particularly at **12:54**, **12:55**, and **14:00**, indicating a possible command and control server communicating with the compromised host.

- **Maintain Persistence**: The continuous execution of shell commands and the creation of files related to `r2te1030` suggest that the attacker aimed to maintain persistence on the system. This was evident from the logs showing multiple command executions at various timestamps.

## Table of IoCs Detected

| IoC                     | Security Context                                                                                     |
|-------------------------|-----------------------------------------------------------------------------------------------------|
| 142.20.56.202           | An external IP address frequently contacted, indicating potential command and control activity.     |
| r2te1030.tmp            | A temporary file created by `svchost.exe`, likely associated with malicious activity.              |
| r2te1030.out            | An output file created by `svchost.exe`, potentially used for logging malicious actions.           |
| r2te1030.err            | An error file created by `svchost.exe`, possibly used to capture errors during malicious execution.|
| r2te1030.dll            | A dynamic link library file created by `svchost.exe`, likely containing malicious code.            |
| r2te1030.cmdline        | A command line file created by `svchost.exe`, potentially used to store execution parameters.      |
| r2te1030                | The main executable or script associated with the malicious activity, indicating exploitation risk. |

## Chronological Log of Actions

### 2019-09-23

- **11:39**: `svchost.exe MESSAGE_INBOUND the flow: 142.20.56.202 (10 times)`
- **11:39**: `svchost.exe ADD a registry`
- **11:40**: `svchost.exe COMMAND a shell (19 times)`
- **11:40**: `svchost.exe MESSAGE_INBOUND the flow: 142.20.56.202 (9 times)`
- **11:40**: `svchost.exe OPEN the process: svchost.exe`
- **11:41**: `svchost.exe COMMAND a shell (15 times)`
- **11:41**: `svchost.exe MESSAGE_INBOUND the flow: 142.20.56.202 (11 times)`
- **11:41**: `svchost.exe READ the file: System.ni.dll.aux`
- **11:41**: `svchost.exe READ the file: System.Management.Automation.ni.dll.aux`
- **11:41**: `svchost.exe READ the file: System.Core.ni.dll.aux`
- **11:41**: `svchost.exe READ the file: Microsoft.Management.Infrastructure.ni.dll.aux`
- **11:41**: `svchost.exe READ the file: Microsoft.Management.Infrastructure.Native.ni.dll.aux`
- **11:42**: `svchost.exe MESSAGE_INBOUND the flow: 142.20.56.202 (11 times)`
- **11:42**: `svchost.exe COMMAND a shell (11 times)`
- **11:43**: `svchost.exe MESSAGE_INBOUND the flow: 142.20.56.202 (12 times)`
- **11:43**: `svchost.exe COMMAND a shell (12 times)`
- **11:44**: `svchost.exe COMMAND a shell (12 times)`
- **11:44**: `svchost.exe MESSAGE_INBOUND the flow: 142.20.56.202 (10 times)`
- **11:45**: `svchost.exe COMMAND a shell (12 times)`
- **11:45**: `svchost.exe MESSAGE_INBOUND the flow: 142.20.56.202 (10 times)`
- **11:46**: `svchost.exe COMMAND a shell (11 times)`
- **11:46**: `svchost.exe MESSAGE_INBOUND the flow: 142.20.56.202 (9 times)`
- **11:47**: `svchost.exe COMMAND a shell (11 times)`
- **11:47**: `svchost.exe MESSAGE_INBOUND the flow: 142.20.56.202 (8 times)`
- **11:48**: `svchost.exe COMMAND a shell (11 times)`
- **11:48**: `svchost.exe MESSAGE_INBOUND the flow: 142.20.56.202 (11 times)`
- **11:49**: `svchost.exe COMMAND a shell (13 times)`
- **11:49**: `svchost.exe MESSAGE_INBOUND the flow: 142.20.56.202 (10 times)`
- **11:50**: `svchost.exe COMMAND a shell (11 times)`
- **12:00**: `svchost.exe COMMAND a shell (12 times)`
- **12:00**: `svchost.exe MESSAGE_INBOUND the flow: 142.20.56.202 (11 times)`
- **12:01**: `svchost.exe COMMAND a shell (12 times)`
- **12:01**: `svchost.exe MESSAGE_INBOUND the flow: 142.20.56.202 (7 times)`
- **12:02**: `svchost.exe COMMAND a shell (12 times)`
- **12:02**: `svchost.exe MESSAGE_INBOUND the flow: 142.20.56.202 (9 times)`
- **12:03**: `svchost.exe COMMAND a shell (12 times)`
- **12:03**: `svchost.exe MESSAGE_INBOUND the flow: 142.20.56.202 (10 times)`
- **12:04**: `svchost.exe COMMAND a shell (11 times)`
- **12:04**: `svchost.exe MESSAGE_INBOUND the flow: 142.20.56.202 (7 times)`
- **12:05**: `svchost.exe COMMAND a shell (12 times)`
- **12:05**: `svchost.exe MESSAGE_INBOUND the flow: 142.20.56.202 (8 times)`
- **12:06**: `svchost.exe COMMAND a shell (11 times)`
- **12:06**: `svchost.exe MESSAGE_INBOUND the flow: 142.20.56.202 (10 times)`
- **12:07**: `svchost.exe COMMAND a shell (12 times)`
- **12:07**: `svchost.exe MESSAGE_INBOUND the flow: 142.20.56.202 (10 times)`
- **12:08**: `svchost.exe COMMAND a shell (12 times)`
- **12:08**: `svchost.exe MESSAGE_INBOUND the flow: 142.20.56.202 (11 times)`
- **12:09**: `svchost.exe COMMAND a shell (13 times)`
- **12:09**: `svchost.exe MESSAGE_INBOUND the flow: 142.20.56.202 (8 times)`
- **12:10**: `svchost.exe COMMAND a shell (11 times)`
- **12:10**: `svchost.exe MESSAGE_INBOUND the flow: 142.20.56.202 (8 times)`
- **12:11**: `svchost.exe COMMAND a shell (11 times)`
- **12:11**: `svchost.exe MESSAGE_INBOUND the flow: 142.20.56.202 (11 times)`
- **12:12**: `svchost.exe COMMAND a shell (13 times)`
- **12:12**: `svchost.exe MESSAGE_INBOUND the flow: 142.20.56.202 (11 times)`
- **12:13**: `svchost.exe COMMAND a shell (12 times)`
- **12:13**: `svchost.exe MESSAGE_INBOUND the flow: 142.20.56.202 (10 times)`
- **12:14**: `svchost.exe COMMAND a shell (14 times)`
- **12:54**: `svchost.exe MESSAGE_INBOUND the flow: 142.20.56.202 (10 times)`
- **12:55**: `svchost.exe COMMAND a shell (12 times)`
- **12:55**: `svchost.exe MESSAGE_INBOUND the flow: 142.20.56.202 (11 times)`
- **12:56**: `svchost.exe COMMAND a shell (13 times)`
- **12:56**: `svchost.exe MESSAGE_INBOUND the flow: 142.20.56.202 (11 times)`
- **12:57**: `svchost.exe COMMAND a shell (13 times)`
- **12:57**: `svchost.exe MESSAGE_INBOUND the flow: 142.20.56.202 (10 times)`
- **12:58**: `svchost.exe DELETE the file: r2te1030.tmp`
- **12:58**: `svchost.exe DELETE the file: r2te1030.out`
- **12:58**: `svchost.exe DELETE the file: r2te1030.err`
- **12:58**: `svchost.exe DELETE the file: r2te1030.dll`
- **12:58**: `svchost.exe DELETE the file: r2te1030.cmdline`
- **12:58**: `svchost.exe CREATE the file: r2te1030.out`
- **12:58**: `svchost.exe CREATE the file: r2te1030.tmp`
- **12:58**: `svchost.exe CREATE the file: r2te1030.err`
- **12:58**: `svchost.exe CREATE the file: r2te1030.dll`
- **12:58**: `svchost.exe CREATE the file: r2te1030.cmdline`
- **12:58**: `svchost.exe CREATE the file: r2te1030`
- **12:58**: `svchost.exe READ the file: Microsoft.Management.Infrastructure.ni.dll.aux`
- **12:58**: `svchost.exe READ the file: Microsoft.Management.Infrastructure.Native.ni.dll.aux`
- **12:58**: `svchost.exe READ the file: $SECURE:$SDH:$INDEX_ALLOCATION`
- **12:58**: `svchost.exe OPEN the process: csrss.exe`
- **12:58**: `svchost.exe WRITE the file: r2te1030.out`
- **12:58**: `svchost.exe WRITE the file: r2te1030.cmdline`
- **12:58**: `svchost.exe READ the file: System.Core.ni.dll.aux`
- **12:58**: `svchost.exe READ the file: r2te1030.dll`
- **12:58**: `svchost.exe READ the file: System.ni.dll.aux`
- **12:58**: `svchost.exe READ the file: System.Management.Automation.ni.dll.aux`
- **14:00**: `svchost.exe COMMAND a shell (11 times)`
- **14:00**: `svchost.exe MESSAGE_INBOUND the flow: 142.20.56.202 (10 times)`
- **14:01**: `svchost.exe COMMAND a shell (12 times)`
- **14:01**: `svchost.exe MESSAGE_INBOUND the flow: 142.20.56.202 (11 times)`
- **14:02**: `svchost.exe COMMAND a shell (13 times)`
- **14:02**: `svchost.exe MESSAGE_INBOUND the flow: 142.20.56.202 (10 times)`
- **14:03**: `svchost.exe COMMAND a shell (11 times)`
- **14:03**: `svchost.exe MESSAGE_INBOUND the flow: 142.20.56.202 (9 times)`
- **14:04**: `svchost.exe COMMAND a shell (12 times)`
- **14:04**: `svchost.exe MESSAGE_INBOUND the flow: 142.20.56.202 (11 times)`
- **14:05**: `svchost.exe COMMAND a shell (3 times)`
- **14:05**: `svchost.exe MESSAGE_INBOUND the flow: 142.20.56.202`
- **14:05**: `svchost.exe TERMINATE the process: svchost.exe` 

This report highlights the suspicious behavior of the `svchost.exe` process and the potential compromise of the system, warranting further investigation and remediation actions.