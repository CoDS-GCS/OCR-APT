# Attack Report: Anomalous Activity Detected

## Summary of Attack Behavior

The logs from the document `tc3_anomalous_subgraph_3.csv` indicate a series of anomalous connection attempts that suggest potential malicious activity. The events occurred on April 12, 2018, and involved multiple connections to a specific flow, culminating in a direct connection to an external IP address. 

### Key Events:
- **Initial Compromise**: The process began with multiple connection attempts to a flow, indicating potential reconnaissance or probing for vulnerabilities.
- **Command and Control**: The connection to the external IP address (128.55.12.110) suggests a possible command and control (C2) communication, which is a critical stage in APT attacks where the attacker maintains control over the compromised system.

The sequence of events shows a pattern of increasing connection attempts, which may indicate an escalation in the attacker's activity or an attempt to establish a persistent connection.

## Table of Indicators of Compromise (IoCs)

| IoC                | Security Context                                                                                     |
|--------------------|-----------------------------------------------------------------------------------------------------|
| 128.55.12.110      | This external IP address may be associated with malicious activity. It could be used for C2 communications, indicating a high likelihood of exploitation. Legitimate usage is possible, but the context of multiple connection attempts raises suspicion. |

## Chronological Log of Actions

| Timestamp               | Action Description                                      |
|-------------------------|--------------------------------------------------------|
| 2018-04-12 13:18       | Mail CONNECT a flow (2 times)                          |
| 2018-04-12 13:19       | Mail CONNECT a flow (6 times)                          |
| 2018-04-12 13:20       | Mail CONNECT a flow (10 times)                         |
| 2018-04-12 13:20       | Mail CONNECT the flow: 128.55.12.110                   |
| 2018-04-12 13:21       | Mail CONNECT a flow (4 times)                          |

This report highlights the potential threat posed by the detected anomalous behavior and the associated IoC, emphasizing the need for further investigation and monitoring of the identified external IP address.