# Attack Report: Context of svchost.exe Activity

## Summary of Attack Behavior

On September 24, 2019, a series of suspicious activities were logged involving the `svchost.exe` process, indicating potential malicious behavior consistent with an Advanced Persistent Threat (APT) attack. The activities can be categorized into several stages of the APT lifecycle:

1. **Initial Compromise**: The process initiated outbound connections to multiple external IP addresses, including `10.20.0.2`, `10.50.2.101`, and `202.6.172.98`, suggesting an attempt to establish communication with a command and control (C2) server.

2. **Internal Reconnaissance**: The process read various files, including `setuptools-0.7.2-py2.7.egg` and `kickoff.log`, which may indicate an attempt to gather information about the system and its configuration.

3. **Command and Control**: Multiple outbound messages were sent to external IP addresses, including `10.50.5.11` and `142.20.61.132`, indicating ongoing communication with potential C2 infrastructure.

4. **Covering Tracks**: The process deleted the `usernames.csv` file and renamed configuration files, which is a common tactic used to obscure malicious activity and maintain persistence.

Overall, the logs indicate a coordinated effort to compromise the system, gather intelligence, and maintain communication with external entities, while also attempting to erase traces of the attack.

## Table of IoCs Detected

| IoC                                   | Security Context                                                                                     |
|---------------------------------------|-----------------------------------------------------------------------------------------------------|
| 10.20.0.2                             | Internal IP address; potential C2 server.                                                          |
| 10.50.2.101                           | Internal IP address; potential C2 server.                                                          |
| 127.0.0.1                             | Localhost; often used for internal communications.                                                  |
| 10.50.5.11                            | Internal IP address; potential C2 server.                                                          |
| 202.6.172.98                          | External IP address; potential malicious actor.                                                    |
| 142.20.57.246                         | External IP address; potential malicious actor.                                                    |
| ping.exe                              | Legitimate utility; could be exploited for network reconnaissance.                                  |
| csc.sys                               | System file; could be targeted for exploitation or manipulation.                                    |
| mantra-2.0-py2.7.egg                 | Python egg file; could indicate the use of a malicious Python package.                             |
| setuptools-0.7.2-py2.7.egg           | Python package; could be used for dependency management but may also be exploited.                 |
| kickoff.log                           | Log file; could contain sensitive information or logs of malicious activity.                        |
| usernames.csv                         | CSV file; could contain sensitive user information, targeted for exfiltration.                     |
| Various Python bytecode files         | Potentially malicious scripts; could be used for executing payloads or maintaining persistence.     |
| .winlogbeat.yml                       | Configuration file; could be used for logging but may also be manipulated for malicious purposes.   |
| .winlogbeat.yml.new                   | Temporary configuration file; could indicate attempts to modify logging behavior.                   |

## Chronological Log of Actions

### September 24, 2019

- **11:57**
  - `svchost.exe` renamed `.winlogbeat.yml.new` (12 times).
  - `svchost.exe` wrote `agent.log` (12 times).
  - `svchost.exe` wrote `.winlogbeat.yml` (9 times).
  - `svchost.exe` wrote `mantra.log` (7 times).
  - `svchost.exe` received inbound messages from `142.20.57.246` (6 times).
  - `svchost.exe` sent outbound messages to `10.20.2.66` (4 times).
  - `svchost.exe` sent outbound messages to `142.20.61.132` (3 times).
  - `svchost.exe` read `ping.exe` (3 times).
  - `svchost.exe` started outbound flow to `202.6.172.98` (2 times).
  - `svchost.exe` sent outbound messages to `202.6.172.98` (2 times).
  - `svchost.exe` read `lwabeat.exe`, `python27.dll`, and various JSON files.

- **14:10**
  - `svchost.exe` wrote `mantra.log` (8 times).
  - `svchost.exe` sent outbound messages to `202.6.172.98` (7 times).
  - `svchost.exe` received inbound messages from `142.20.57.246` (6 times).
  - `svchost.exe` read `ping.exe` (4 times).
  - `svchost.exe` created and modified `MistressMainframeDepression.pptx` and `EVN ELE.pdf`.
  - `svchost.exe` read various system files including `csc.sys` and `$UsnJrnl:$J`.

- **15:16**
  - `svchost.exe` initiated outbound flow to `10.20.0.2` (2 times).
  - `svchost.exe` wrote `kickoff.log` (2 times).
  - `svchost.exe` read multiple files, including `setuptools-0.7.2-py2.7.egg` and `usernames.csv`.
  - `svchost.exe` deleted `usernames.csv`.
  - `svchost.exe` received inbound messages from `10.50.2.101` and `127.0.0.1`.
  - `svchost.exe` sent outbound messages to `10.50.5.11` and `202.6.172.98`.

- **15:29**
  - `svchost.exe` sent outbound messages to `142.20.61.132` (11 times).
  - `svchost.exe` renamed `.winlogbeat.yml.new` (11 times).
  - `svchost.exe` wrote `mantra.log` (4 times).
  - `svchost.exe` read various files, including `Cortana.Sync.dll` and `DWM.EXE`.

This report highlights the suspicious activities associated with `svchost.exe`, indicating a potential APT attack characterized by initial compromise, internal reconnaissance, command and control, and efforts to cover tracks. The identified IoCs should be monitored for further investigation and mitigation.