# Attack Report: tc3_anomalous_subgraph_8.csv

## Summary of Attack Behavior

The logs from the document indicate a series of network connection and disconnection events involving multiple external IP addresses on April 13, 2018. The activity appears to be orchestrated by a process named `tcexec`, which executed a sequence of `CONNECT` and `CLOSE` commands. 

### Key Events:
- **14:21**: The process initiated connections to two IP addresses (128.55.12.55 and 128.55.12.73) while simultaneously closing connections to the same addresses.
- **14:22**: The process exhibited repetitive behavior by connecting and closing connections to the IP address 128.55.12.67, as well as connecting to 128.55.12.166.
- **14:23**: A significant number of connections were closed, including those to 128.55.12.1, 128.55.12.103, 128.55.12.110, and 128.55.12.141, followed by new connections to the same IPs.
- **14:26**: The process closed and immediately reconnected to 128.55.12.1.
- **14:28**: The process exited, indicating the end of the session.

This behavior suggests potential lateral movement and command and control activities, as the process frequently connected and disconnected from various IPs, which could indicate attempts to establish persistence or exfiltrate data.

## Table of IoCs Detected

| IoC                | Security Context                                                                                     |
|--------------------|-----------------------------------------------------------------------------------------------------|
| 128.55.12.73       | External IP address involved in multiple connection events. Potentially used for command and control. |
| 128.55.12.55       | External IP address with frequent connection and disconnection, indicating possible exploitation.      |
| 128.55.12.67       | Repeated connection and closure events suggest it may be a target for lateral movement.               |
| 128.55.12.166      | Connected during the incident, could be part of a larger network of compromised systems.            |
| 128.55.12.1        | Involved in multiple connection events, indicating potential data exfiltration or command control.   |
| 128.55.12.103      | Frequent connections suggest it may be a target for exploitation or lateral movement.                |
| 128.55.12.110      | Similar to above, involved in multiple connection events, indicating potential risk.                 |
| 128.55.12.141      | Connected during the incident, could be part of a compromised network.                              |

## Chronological Log of Actions

| Time (UTC)         | Action                                                                                     |
|---------------------|--------------------------------------------------------------------------------------------|
| 14:21               | tcexec CLOSE the flow: 128.55.12.73                                                      |
| 14:21               | tcexec CONNECT the flow: 128.55.12.55                                                    |
| 14:21               | tcexec CONNECT the flow: 128.55.12.73                                                    |
| 14:21               | tcexec CLOSE the flow: 128.55.12.55                                                     |
| 14:22               | tcexec CLOSE the flow: 128.55.12.67 (2 times)                                           |
| 14:22               | tcexec CONNECT the flow: 128.55.12.67 (2 times)                                         |
| 14:22               | tcexec CLOSE the flow: 128.55.12.166                                                    |
| 14:22               | tcexec CONNECT the flow: 128.55.12.166                                                  |
| 14:23               | tcexec CLOSE the flow: 128.55.12.1                                                      |
| 14:23               | tcexec CLOSE the flow: 128.55.12.103                                                    |
| 14:23               | tcexec CLOSE the flow: 128.55.12.110                                                    |
| 14:23               | tcexec CLOSE the flow: 128.55.12.141                                                    |
| 14:23               | tcexec CLOSE the flow: 128.55.12.55                                                     |
| 14:23               | tcexec CLOSE the flow: 128.55.12.67                                                     |
| 14:23               | tcexec CONNECT the flow: 128.55.12.1                                                    |
| 14:23               | tcexec CONNECT the flow: 128.55.12.103                                                  |
| 14:23               | tcexec CONNECT the flow: 128.55.12.110                                                  |
| 14:23               | tcexec CONNECT the flow: 128.55.12.141                                                  |
| 14:23               | tcexec CONNECT the flow: 128.55.12.55                                                   |
| 14:23               | tcexec CONNECT the flow: 128.55.12.67                                                   |
| 14:26               | tcexec CLOSE the flow: 128.55.12.1                                                      |
| 14:26               | tcexec CONNECT the flow: 128.55.12.1                                                    |
| 14:28               | tcexec EXIT the process: tcexec                                                          |

This report outlines the suspicious activities observed in the logs, highlighting potential indicators of compromise and the sequence of actions taken during the incident. Further investigation is recommended to assess the impact and scope of the activities associated with the identified IoCs.