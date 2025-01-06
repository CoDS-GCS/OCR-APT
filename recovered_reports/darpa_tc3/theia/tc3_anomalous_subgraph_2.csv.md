# Attack Report: Anomalous Activity Detected

## Summary of Attack Behavior

The logs from the document `tc3_anomalous_subgraph_2.csv` indicate a series of anomalous activities related to mail connections on April 12, 2018. The events suggest a potential Command and Control (C2) stage of an Advanced Persistent Threat (APT) attack, characterized by multiple connections to a specific flow and an external IP address. 

Key events include:
- **13:18**: The process initiated a mail connection flow 2 times.
- **13:19**: The connection attempts increased to 6 times.
- **13:20**: The flow was connected 10 times, and a specific external IP address (128.55.12.110) was also connected 2 times.
- **13:21**: The connection attempts continued with 4 additional flows.

The increasing frequency of connection attempts, particularly to the external IP address, raises concerns about potential data exfiltration or unauthorized access.

## Table of Indicators of Compromise (IoCs)

| IoC               | Security Context                                                                                     |
|-------------------|------------------------------------------------------------------------------------------------------|
| 128.55.12.110     | This external IP address is associated with the mail connection flow. While it may be legitimate, the frequency of connections raises suspicion of exploitation or unauthorized access. |

## Chronological Log of Actions

| Timestamp          | Action Description                                      |
|--------------------|--------------------------------------------------------|
| 2018-04-12 13:18   | Mail CONNECT a flow (2 times)                         |
| 2018-04-12 13:19   | Mail CONNECT a flow (6 times)                         |
| 2018-04-12 13:20   | Mail CONNECT a flow (10 times)                        |
| 2018-04-12 13:20   | Mail CONNECT the flow: 128.55.12.110 (2 times)       |
| 2018-04-12 13:21   | Mail CONNECT a flow (4 times)                         |

This report highlights the need for further investigation into the identified external IP address and the unusual connection patterns to mitigate potential threats.