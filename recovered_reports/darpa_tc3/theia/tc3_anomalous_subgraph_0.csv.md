# Attack Report: tc3_anomalous_subgraph_0.csv

## Summary of Attack Behavior

The logs from the document indicate a series of suspicious activities primarily involving the Firefox process, which suggest potential malicious behavior consistent with an Advanced Persistent Threat (APT) attack. The timeline reveals multiple connections, data transmissions, and receptions to and from various external IP addresses, particularly focusing on the IPs 67.28.122.168 and 208.44.49.52. 

Key events include:

- **Initial Compromise**: The logs show multiple connections initiated by Firefox, indicating potential reconnaissance or exploitation attempts.
- **Command and Control**: The repeated connections and data exchanges with external IPs suggest that the attacker may have established a command and control channel.
- **Data Exfiltration**: The logs indicate multiple SENDTO actions, particularly to the IPs 67.28.122.168 and 208.44.49.52, which could imply data being sent to an external entity.

The activity is concentrated around the timestamps of April 12, 2018, with significant actions occurring between 14:27 and 14:57.

## Table of IoCs Detected

| IoC                  | Security Context                                                                                     |
|----------------------|-----------------------------------------------------------------------------------------------------|
| 67.28.122.168        | Known IP associated with suspicious activities; often used in data exfiltration attempts.          |
| 216.73.86.152       | External IP that may be linked to malicious command and control servers.                           |
| 208.17.90.10        | Potentially malicious IP; requires further investigation to determine legitimacy.                   |
| 208.44.49.52        | Frequently appears in logs related to data exfiltration; high likelihood of exploitation.          |
| 216.163.248.17      | External IP that may be involved in suspicious activities; further analysis needed.                |
| 143.166.224.244     | Less known IP; requires investigation to assess potential threat level.                            |
| 210.50.7.243        | External IP with limited reputation; further scrutiny is warranted.                                 |
| 83.222.15.109       | Potentially malicious IP; should be monitored for any further suspicious activities.                |

## Chronological Log of Actions Organized by Minute

### April 12, 2018

- **14:27**
  - firefox CONNECT a flow (2 times)
  - firefox SENDTO the flow: 67.28.122.168
  - firefox RECVFROM the flow: 216.73.86.152

- **14:28**
  - firefox CONNECT a flow (3 times)

- **14:29**
  - firefox CONNECT a flow (4 times)
  - firefox SENDTO the flow: 67.28.122.168

- **14:30**
  - firefox CONNECT a flow

- **14:31**
  - firefox CONNECT a flow (3 times)
  - firefox SENDTO the flow: 67.28.122.168
  - firefox RECVFROM the flow: 208.17.90.10
  - firefox RECVFROM a flow

- **14:32**
  - firefox CONNECT a flow

- **14:34**
  - firefox CONNECT a flow (4 times)

- **14:35**
  - firefox CONNECT a flow (4 times)
  - firefox RECVFROM a flow

- **14:36**
  - firefox CONNECT a flow (4 times)

- **14:37**
  - firefox CONNECT a flow (5 times)
  - firefox RECVFROM a flow
  - firefox RECVFROM the flow: 208.44.49.52
  - firefox SENDTO the flow: 208.44.49.52

- **14:38**
  - firefox CONNECT a flow (4 times)

- **14:39**
  - firefox CONNECT a flow (3 times)
  - firefox SENDTO the flow: 67.28.122.168
  - firefox RECVFROM a flow

- **14:40**
  - firefox CONNECT a flow (4 times)
  - firefox RECVFROM the flow: 208.44.49.52
  - firefox RECVFROM a flow
  - firefox SENDTO the flow: 208.44.49.52

- **14:41**
  - firefox CONNECT a flow (5 times)

- **14:42**
  - firefox CONNECT a flow (4 times)
  - firefox RECVFROM a flow (2 times)

- **14:43**
  - firefox CONNECT a flow (2 times)
  - firefox SENDTO the flow: 216.163.248.17
  - firefox RECVFROM the flow: 216.163.248.17

- **14:44**
  - firefox CONNECT a flow (5 times)
  - firefox SENDTO the flow: 208.44.49.52

- **14:45**
  - firefox CONNECT a flow (4 times)

- **14:46**
  - firefox CONNECT a flow (3 times)
  - firefox SENDTO the flow: 67.28.122.168

- **14:47**
  - firefox CONNECT a flow (3 times)
  - firefox RECVFROM the flow: 83.222.15.109

- **14:48**
  - firefox CONNECT a flow (6 times)
  - firefox RECVFROM a flow (2 times)
  - firefox SENDTO the flow: 67.28.122.168

- **14:49**
  - firefox CONNECT a flow (4 times)
  - firefox RECVFROM a flow

- **14:50**
  - firefox CONNECT a flow (4 times)
  - firefox RECVFROM a flow
  - firefox SENDTO the flow: 67.28.122.168

- **14:51**
  - firefox CONNECT a flow (5 times)
  - firefox RECVFROM a flow (2 times)

- **14:52**
  - firefox CONNECT a flow (3 times)

- **14:53**
  - firefox CONNECT a flow (3 times)
  - firefox SENDTO the flow: 67.28.122.168

- **14:54**
  - firefox CONNECT a flow (3 times)

- **14:55**
  - firefox CONNECT a flow (2 times)
  - firefox SENDTO the flow: 67.28.122.168

- **14:56**
  - firefox CONNECT a flow (4 times)

- **14:57**
  - firefox CONNECT a flow

This report highlights the suspicious activities and potential threats identified in the logs, emphasizing the need for further investigation and monitoring of the identified IoCs.