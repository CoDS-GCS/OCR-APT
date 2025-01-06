# Attack Report: tc3_anomalous_subgraph_15.csv

## Summary of Attack Behavior

The logs from the document indicate a series of suspicious activities primarily involving the IP address **128.55.12.55**. The events suggest a potential APT attack characterized by multiple instances of file read and write operations, along with repeated acceptance of network flows from the same IP address. 

### Key Events:
- **Initial Compromise**: The process began with multiple ACCEPT events from **128.55.12.55**, indicating an initial connection to the network at various timestamps, notably at **18:38** and **15:30**.
- **Internal Reconnaissance**: Following the initial connections, there were numerous READ and WRITE operations on files, suggesting reconnaissance activities to gather information about the system.
- **Command and Control**: The consistent acceptance of flows from **128.55.12.55** at different times indicates a potential command and control mechanism being established.
- **Data Exfiltration**: The high frequency of file read and write operations, particularly at **18:55** and **19:15**, raises concerns about data being exfiltrated from the system.

The attack appears to be methodical, with the attacker leveraging the same IP address for multiple actions, indicating a focused effort to maintain persistence and control over the compromised environment.

## Table of IoCs Detected

| IoC                | Security Context                                                                                     |
|--------------------|-----------------------------------------------------------------------------------------------------|
| 128.55.12.55       | A frequently occurring IP address in the logs, indicating potential malicious activity. Legitimate usage may include internal network communications, but the high frequency and context suggest exploitation likelihood is moderate to high. |
| 128.55.12.166      | An external IP address that was accepted in the flow, indicating possible lateral movement or external command and control. Legitimate usage is low, and exploitation likelihood is high due to its anomalous appearance. |
| 128.55.12.110      | Another external IP address noted in the logs, suggesting potential unauthorized access. Legitimate usage is low, and exploitation likelihood is high, given the context of the other activities. |

## Chronological Log of Actions

| Timestamp           | Action                                                                                     |
|---------------------|--------------------------------------------------------------------------------------------|
| 2018-04-12 15:30    | ACCEPT the flow: 128.55.12.55 (2 times)                                                  |
| 2018-04-12 15:30    | READ a file (2 times)                                                                     |
| 2018-04-12 15:30    | WRITE a file (2 times)                                                                    |
| 2018-04-12 15:51    | ACCEPT the flow: 128.55.12.55 (2 times)                                                  |
| 2018-04-12 15:52    | ACCEPT the flow: 128.55.12.55 (3 times)                                                  |
| 2018-04-12 15:52    | READ a file (2 times)                                                                     |
| 2018-04-12 15:52    | WRITE a file (2 times)                                                                    |
| 2018-04-12 15:53    | READ a file (2 times)                                                                     |
| 2018-04-12 15:53    | WRITE a file (2 times)                                                                    |
| 2018-04-12 16:02    | READ a file (3 times)                                                                     |
| 2018-04-12 16:02    | WRITE a file (3 times)                                                                    |
| 2018-04-12 16:02    | ACCEPT the flow: 128.55.12.55 (3 times)                                                  |
| 2018-04-12 16:15    | ACCEPT the flow: 128.55.12.55                                                             |
| 2018-04-12 16:15    | READ a file                                                                                 |
| 2018-04-12 16:15    | WRITE a file                                                                                |
| 2018-04-12 16:23    | READ a file                                                                                 |
| 2018-04-12 16:23    | WRITE a file                                                                                |
| 2018-04-12 16:37    | WRITE a file (5 times)                                                                     |
| 2018-04-12 16:37    | ACCEPT the flow: 128.55.12.55 (5 times)                                                  |
| 2018-04-12 16:37    | READ a file (5 times)                                                                      |
| 2018-04-12 16:39    | ACCEPT the flow: 128.55.12.55                                                              |
| 2018-04-12 16:39    | READ a file                                                                                 |
| 2018-04-12 16:39    | WRITE a file                                                                                |
| 2018-04-12 16:50    | ACCEPT the flow: 128.55.12.55 (2 times)                                                  |
| 2018-04-12 16:51    | WRITE a file (2 times)                                                                     |
| 2018-04-12 16:51    | READ a file (2 times)                                                                      |
| 2018-04-12 16:53    | ACCEPT the flow: 128.55.12.55 (2 times)                                                  |
| 2018-04-12 16:54    | READ a file (2 times)                                                                      |
| 2018-04-12 16:54    | WRITE a file (2 times)                                                                     |
| 2018-04-12 16:54    | ACCEPT the flow: 128.55.12.55                                                              |
| 2018-04-12 16:55    | ACCEPT the flow: 128.55.12.55                                                              |
| 2018-04-12 18:38    | ACCEPT the flow: 128.55.12.55 (3 times)                                                  |
| 2018-04-12 18:38    | READ a file (3 times)                                                                     |
| 2018-04-12 18:38    | WRITE a file (3 times)                                                                    |
| 2018-04-12 18:39    | READ a file (2 times)                                                                     |
| 2018-04-12 18:39    | WRITE a file (2 times)                                                                    |
| 2018-04-12 18:50    | ACCEPT the flow: 128.55.12.55 (2 times)                                                  |
| 2018-04-12 18:50    | WRITE a file                                                                                |
| 2018-04-12 18:50    | READ a file                                                                                 |
| 2018-04-12 18:55    | READ a file (6 times)                                                                     |
| 2018-04-12 18:55    | WRITE a file (6 times)                                                                    |
| 2018-04-12 18:55    | ACCEPT the flow: 128.55.12.55 (4 times)                                                  |
| 2018-04-12 18:56    | ACCEPT the flow: 128.55.12.55 (4 times)                                                  |
| 2018-04-12 18:56    | READ a file (2 times)                                                                     |
| 2018-04-12 18:56    | WRITE a file (2 times)                                                                    |
| 2018-04-12 18:57    | ACCEPT the flow: 128.55.12.55 (5 times)                                                  |
| 2018-04-12 18:57    | READ a file (2 times)                                                                     |
| 2018-04-12 18:57    | WRITE a file (2 times)                                                                    |
| 2018-04-12 18:58    | WRITE a file (2 times)                                                                    |
| 2018-04-12 18:58    | READ a file (2 times)                                                                     |
| 2018-04-12 18:59    | ACCEPT the flow: 128.55.12.55 (2 times)                                                  |
| 2018-04-12 19:00    | READ a file                                                                                 |
| 2018-04-12 19:00    | WRITE a file                                                                                |
| 2018-04-12 19:01    | ACCEPT the flow: 128.55.12.55 (2 times)                                                  |
| 2018-04-12 19:01    | READ a file                                                                                 |
| 2018-04-12 19:01    | WRITE a file                                                                                |
| 2018-04-12 19:02    | READ a file                                                                                 |
| 2018-04-12 19:02    | WRITE a file                                                                                |
| 2018-04-12 19:04    | ACCEPT the flow: 128.55.12.55 (3 times)                                                  |
| 2018-04-12 19:05    | READ a file (2 times)                                                                     |
| 2018-04-12 19:05    | WRITE a file (2 times)                                                                    |
| 2018-04-12 19:09    | ACCEPT the flow: 128.55.12.55 (3 times)                                                  |
| 2018-04-12 19:09    | ACCEPT the flow: 128.55.12.67                                                              |
| 2018-04-12 19:15    | ACCEPT the flow: 128.55.12.55 (2 times)                                                  |
| 2018-04-12 19:15    | READ a file (2 times)                                                                     |
| 2018-04-12 19:15    | WRITE a file (2 times)                                                                    |
| 2018-04-12 19:19    | READ a file (2 times)                                                                     |
| 2018-04-12 19:19    | WRITE a file (2 times)                                                                    |
| 2018-04-12 19:28    | READ a file (2 times)                                                                     | 

This report highlights the suspicious activities associated with the identified IoCs and provides a detailed account of the actions taken during the incident. Further investigation is recommended to assess the full impact and potential remediation steps.