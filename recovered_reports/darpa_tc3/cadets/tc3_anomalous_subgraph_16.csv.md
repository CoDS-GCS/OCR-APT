# Attack Report: tc3_anomalous_subgraph_16.csv

## Summary of Attack Behavior

The logs indicate a series of suspicious activities centered around the IP address **192.113.144.28** on **April 12, 2018**, at **14:35**. The sequence of events suggests a potential APT attack, primarily during the **Command and Control** stage. The attacker appears to have established a connection with the external IP, executed file operations, and communicated with the external server multiple times.

Key events include:
- **14:35**: The process initiated communication with the external IP **192.113.144.28** through `RECVFROM` and `SENDTO` actions, indicating the establishment of a command and control channel.
- **14:35**: The process opened and memory-mapped the file **libpcap.so.8**, which is commonly used for packet capturing and could be exploited for malicious purposes.
- **14:36 - 14:38**: Multiple `CLOSE`, `RECVFROM`, and `SENDTO` actions were logged, suggesting ongoing data exchange with the external IP, potentially for exfiltration or command execution.
- **14:38**: The process exited and created a pipe, indicating a possible attempt to maintain persistence or prepare for further actions.

Overall, the logs reflect a structured approach to establishing a foothold and communicating with an external entity, characteristic of APT behavior.

## Table of IoCs Detected

| IoC                  | Security Context                                                                                     |
|----------------------|-----------------------------------------------------------------------------------------------------|
| 192.113.144.28      | An external IP address associated with suspicious activity. Likely used for command and control.   |
| libpcap.so.8        | A legitimate library for packet capturing. Its usage in this context raises suspicion of exploitation. |
| 128.55.12.67        | An external IP address that was closed during the session. Requires further investigation for potential links to malicious activity. |
| 128.55.12.110       | Another external IP address involved in the session closure. Potentially linked to the attacker's infrastructure. |
| 128.55.12.10        | Similar to the above, this IP address was also closed. Further analysis needed to determine its role in the incident. |

## Chronological Log of Actions

### April 12, 2018

- **14:35**: 
  - RECVFROM the flow: **192.113.144.28**
  - CREATE_OBJECT a pipe
  - CONNECT the flow: **192.113.144.28**
  - OPEN the file: **libpcap.so.8**
  - MMAP the file: **libpcap.so.8**
  - SENDTO the flow: **192.113.144.28**

- **14:36**: 
  - CLOSE a file (6 times)
  - RECVFROM the flow: **192.113.144.28** (3 times)
  - SENDTO the flow: **192.113.144.28**
  - CLOSE the flow: **128.55.12.67**

- **14:37**: 
  - CLOSE a file (10 times)
  - RECVFROM the flow: **192.113.144.28** (5 times)
  - CLOSE the flow: **128.55.12.110**
  - CLOSE the flow: **128.55.12.10**
  - SENDTO the flow: **192.113.144.28**

- **14:38**: 
  - CLOSE a file (5 times)
  - RECVFROM the flow: **192.113.144.28** (3 times)
  - SENDTO the flow: **192.113.144.28** (2 times)
  - CLOSE a pipe
  - EXIT a process
  - CREATE_OBJECT a pipe

This report highlights the suspicious activities and potential indicators of compromise that warrant further investigation to mitigate any ongoing threats.