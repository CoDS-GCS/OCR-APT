# Attack Report: tc3_anomalous_subgraph_7.csv

## Summary of Attack Behavior

The logs from the document indicate a series of network connection and disconnection events involving multiple IP addresses on April 13, 2018. The behavior observed suggests a potential APT attack characterized by rapid connection and disconnection of flows, which may indicate reconnaissance or lateral movement activities. 

Key events include:
- Multiple connections and closures of flows at the same timestamp, particularly around 14:21 and 14:23, indicating a high level of activity.
- The same IP addresses were connected and closed multiple times, which may suggest attempts to establish persistence or probe the network for vulnerabilities.
- The process `tcexec` was exited at 14:28, indicating the end of the session, which could signify the completion of the attack phase or an attempt to cover tracks.

This behavior aligns with the **Internal Reconnaissance** and **Lateral Movement** stages of an APT attack, where the attacker seeks to gather information about the network and move laterally to gain further access.

## Table of IoCs Detected

| IoC               | Security Context                                                                                     |
|-------------------|-----------------------------------------------------------------------------------------------------|
| 128.55.12.73      | Potentially a compromised host; frequent connections may indicate exploitation attempts.            |
| 128.55.12.55      | Similar to above; repeated connections and closures suggest probing or lateral movement.            |
| 128.55.12.67      | Active involvement in multiple connection attempts; could be a target or a compromised system.     |
| 128.55.12.166     | Engaged in connection attempts; further investigation needed to determine legitimacy.               |
| 128.55.12.1       | Involved in both connection and closure events; may indicate a critical system or target.          |
| 128.55.12.103     | Frequent connection activity; potential for exploitation or lateral movement.                       |
| 128.55.12.110     | Similar to above; requires further analysis to assess risk and legitimacy.                          |
| 128.55.12.141     | Engaged in multiple connection events; could be a compromised host or a target for exploitation.   |

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

This report highlights the suspicious activities observed in the logs, indicating potential APT behavior that warrants further investigation.