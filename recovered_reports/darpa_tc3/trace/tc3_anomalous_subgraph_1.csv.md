# Attack Report: tc3_anomalous_subgraph_1.csv

## Summary of Attack Behavior

The logs indicate a series of network connection and disconnection events involving multiple external IP addresses on April 13, 2018. The activity appears to be part of a coordinated effort to establish and terminate connections, which is characteristic of the **Command and Control** stage of an APT attack. 

Key events include:
- Multiple connections and closures of flows to various IP addresses at the same timestamps, suggesting potential lateral movement or reconnaissance activities.
- The process `tcexec` was actively connecting and closing flows, indicating a possible attempt to maintain persistence or establish a command channel.
- The exit of the `tcexec` process at 14:28 could signify the end of the session or a potential cleanup phase.

## Table of Indicators of Compromise (IoCs)

| IoC                | Security Context                                                                                     |
|--------------------|-----------------------------------------------------------------------------------------------------|
| 128.55.12.73       | External IP address; could be a legitimate server or a compromised host used for C2 communication. |
| 128.55.12.55       | External IP address; frequent connections suggest potential exploitation or lateral movement.        |
| 128.55.12.67       | External IP address; repeated connections and closures indicate suspicious activity.                |
| 128.55.12.166      | External IP address; may be involved in data exfiltration or command and control operations.       |
| 128.55.12.1        | External IP address; could be a legitimate service or a target for exploitation.                   |
| 128.55.12.103      | External IP address; potential for exploitation given its involvement in multiple connection events.|
| 128.55.12.110      | External IP address; similar to others, may indicate a compromised host or C2 server.              |
| 128.55.12.141      | External IP address; frequent connections raise suspicion of malicious intent.                      |

## Chronological Log of Actions

### April 13, 2018

- **14:21**
  - CLOSE the flow: 128.55.12.73
  - CONNECT the flow: 128.55.12.55
  - CONNECT the flow: 128.55.12.73
  - CLOSE the flow: 128.55.12.55

- **14:22**
  - CLOSE the flow: 128.55.12.67 (2 times)
  - CONNECT the flow: 128.55.12.67 (2 times)
  - CLOSE the flow: 128.55.12.166
  - CONNECT the flow: 128.55.12.166

- **14:23**
  - CLOSE the flow: 128.55.12.1
  - CLOSE the flow: 128.55.12.103
  - CLOSE the flow: 128.55.12.110
  - CLOSE the flow: 128.55.12.141
  - CLOSE the flow: 128.55.12.55
  - CLOSE the flow: 128.55.12.67
  - CONNECT the flow: 128.55.12.1
  - CONNECT the flow: 128.55.12.103
  - CONNECT the flow: 128.55.12.110
  - CONNECT the flow: 128.55.12.141
  - CONNECT the flow: 128.55.12.55
  - CONNECT the flow: 128.55.12.67

- **14:26**
  - CLOSE the flow: 128.55.12.1
  - CONNECT the flow: 128.55.12.1

- **14:28**
  - EXIT the process: tcexec

This report highlights the suspicious activities captured in the logs, indicating potential APT behavior and the need for further investigation into the identified IoCs.