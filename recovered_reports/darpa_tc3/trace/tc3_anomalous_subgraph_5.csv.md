# Attack Report: tc3_anomalous_subgraph_5.csv

## Summary of Attack Behavior

The logs from the document indicate a series of network connection and disconnection events involving multiple IP addresses on April 13, 2018. The activity appears to be part of a coordinated effort to establish and terminate connections, which is characteristic of the **Command and Control** stage of an APT attack. 

Key events include:
- Multiple connections and closures of flows at the same timestamp, suggesting potential lateral movement or reconnaissance activities.
- The process `tcexec` was observed connecting and closing flows for various IP addresses, indicating possible attempts to establish persistence or exfiltrate data.
- The final log entry shows the `tcexec` process exiting, which may indicate the conclusion of the attack or a shift to a different phase.

## Table of Indicators of Compromise (IoCs)

| IoC                | Security Context                                                                                     |
|--------------------|-----------------------------------------------------------------------------------------------------|
| 128.55.12.73       | An IP address involved in both connection and closure events. Potentially exploited for lateral movement. |
| 128.55.12.55       | Frequently connected and closed, indicating possible command and control activity.                   |
| 128.55.12.67       | Engaged in multiple connection and closure events, suggesting it may be a target for exploitation.   |
| 128.55.12.166      | Connected and closed shortly after, indicating potential reconnaissance or data exfiltration attempts. |
| 128.55.12.1        | Involved in multiple connection events, indicating potential persistence or lateral movement.         |
| 128.55.12.103      | Engaged in connection events, possibly indicating a target for data exfiltration or lateral movement. |
| 128.55.12.110      | Similar to other IPs, involved in connection events, suggesting potential exploitation.               |
| 128.55.12.141      | Connected and closed, indicating possible command and control or data exfiltration activity.         |

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

This report outlines the suspicious activities observed in the logs, highlighting potential indicators of compromise and the nature of the attack behavior. Further investigation is recommended to assess the impact and scope of the incident.