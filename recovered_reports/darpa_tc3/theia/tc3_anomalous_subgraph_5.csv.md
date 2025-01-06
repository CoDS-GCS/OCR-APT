# Attack Report: tc3_anomalous_subgraph_5.csv

## Summary of Attack Behavior

The logs indicate a series of suspicious activities primarily associated with the `mail` process on April 12, 2018. The timeline reveals a significant increase in the number of connections and data transfer actions, suggesting potential malicious behavior. 

- **Initial Compromise**: The attack appears to have begun with the `BOOT` process on April 3, 2018, which may have established a foothold in the system.
- **Internal Reconnaissance**: On April 12, 2018, multiple `CONNECT`, `SENDTO`, and `RECVFROM` actions were recorded, indicating the attacker was gathering information and possibly preparing for data exfiltration.
- **Command and Control**: The connection to the external IP address `128.55.12.110` on April 12, 2018, at 13:20 suggests a command and control (C2) communication, which is a critical indicator of an ongoing APT attack.

The logs show a pattern of repeated actions, particularly with the `CONNECT` and `SENDTO` processes, which escalated in frequency, indicating a potential attempt to exfiltrate data or maintain persistence within the network.

## Table of IoCs Detected

| IoC                | Security Context                                                                                     |
|--------------------|-----------------------------------------------------------------------------------------------------|
| 128.55.12.110      | This external IP address is associated with suspicious activity. It may be used for command and control communications. Legitimate usage is possible, but the high frequency of connections suggests exploitation likelihood is moderate to high. |

## Chronological Log of Actions

### April 12, 2018

- **13:17**
  - `mail SENDTO` a process
  - `mail RECVFROM` a process
  - `mail MMAP` a process
  - `mail MPROTECT` a process
  - `mail OPEN` a process
  - `mail READ` a process
  - `mail CLONE` a process
  - `mail CONNECT` a process
  - `mail EXECUTE` a process

- **13:18**
  - `mail RECVFROM` a process (2 times)
  - `mail CONNECT` a flow (2 times)
  - `mail CONNECT` a process (2 times)
  - `mail SENDTO` a process (2 times)

- **13:19**
  - `mail CONNECT` a process (6 times)
  - `mail CONNECT` a flow (6 times)
  - `mail SENDTO` a process (5 times)
  - `mail RECVFROM` a process (5 times)

- **13:20**
  - `mail CONNECT` a process (11 times)
  - `mail RECVFROM` a process (10 times)
  - `mail CONNECT` a flow (10 times)
  - `mail SENDTO` a process (10 times)
  - `mail CONNECT` the flow: 128.55.12.110

- **13:21**
  - `mail CONNECT` a process (4 times)
  - `mail CONNECT` a flow (4 times)
  - `mail RECVFROM` a process (4 times)
  - `mail SENDTO` a process (4 times)

- **13:24**
  - `mail SENDTO` a process (3 times)
  - `mail RECVFROM` a process (2 times)
  - `mail CLONE` a process
  - `mail MMAP` a process

- **13:26**
  - `mail RECVFROM` a process

This report highlights the suspicious activities associated with the `mail` process and the potential APT attack, emphasizing the need for further investigation and mitigation strategies.