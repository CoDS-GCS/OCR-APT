# Attack Report: tc3_anomalous_subgraph_17.csv

## Summary of Attack Behavior

The logs indicate a series of suspicious activities associated with the external IP address **53.158.101.118** on **April 12, 2018**. The attack appears to follow several stages of an Advanced Persistent Threat (APT) lifecycle, primarily focusing on **Command and Control** and **Data Exfiltration**.

1. **Initial Compromise**: The logs show multiple **SENDTO** and **RECVFROM** actions to and from the IP address **53.158.101.118**, indicating potential command and control communication.
2. **File Manipulation**: The processes involved opening, modifying, and unlinking files such as **urandom**, **netlog**, and **sendmail**, suggesting an attempt to manipulate system files for malicious purposes.
3. **Data Exfiltration**: The repeated communication with the external IP, especially the **SENDTO** actions, indicates a potential data exfiltration attempt, particularly with the **netlog** file, which may contain sensitive information.
4. **Covering Tracks**: The unlinking of files, including **sendmail** and **tmp.8w2uoV**, suggests an effort to erase traces of the attack.

## Table of IoCs Detected

| IoC                | Security Context                                                                                     |
|--------------------|-----------------------------------------------------------------------------------------------------|
| 53.158.101.118     | An external IP address associated with suspicious activity. High likelihood of exploitation.        |
| urandom            | A file typically used for generating random numbers. Legitimate usage, but can be exploited for entropy. |
| netlog             | A log file that may contain sensitive information. High likelihood of exploitation if accessed by an attacker. |
| sendmail           | A legitimate mail transfer agent. Can be exploited for sending unauthorized emails or data exfiltration. |
| tmp.8w2uoV        | A temporary file that may be used for various purposes. Its unlinking suggests potential malicious activity. |

## Chronological Log of Actions

### April 12, 2018

- **14:21**
  - A process SENDTO the flow: 53.158.101.118 (5 times).
  - A process RECVFROM the flow: 53.158.101.118 (5 times).
  - A process OPEN the file: .. 
  - A process MODIFY_PROCESS the file: .. 
  - A process MODIFY_PROCESS a process.
  - A process CONNECT the flow: 53.158.101.118.
  - A process OPEN the file: urandom.
  - A process READ the file: urandom.
  - A process CLOSE the file: .. 
  - A process CLOSE the file: urandom.

- **14:22**
  - A process SENDTO the flow: 53.158.101.118 (4 times).
  - A process RECVFROM the flow: 53.158.101.118 (4 times).
  - A process CLOSE the file: .. 
  - A process OPEN the file: .. 
  - A process UNLINK the file: tmp.8w2uoV.

- **14:23**
  - A process RECVFROM the flow: 53.158.101.118 (5 times).
  - A process SENDTO the flow: 53.158.101.118 (3 times).
  - A process WRITE the file: netlog.
  - A process OPEN the file: .. 
  - A process OPEN the file: netlog.
  - A process MODIFY_FILE_ATTRIBUTES the file: netlog.
  - A process CLOSE the file: netlog.
  - A process MODIFY_PROCESS the file: .. 
  - A process MODIFY_PROCESS a process.
  - A process CLOSE the file: .. 

- **14:24**
  - A process UNLINK the file: netlog.
  - A process SENDTO the flow: 53.158.101.118.
  - A process RECVFROM the flow: 53.158.101.118.

- **14:26**
  - A process RECVFROM the flow: 53.158.101.118 (4 times).
  - A process SENDTO the flow: 53.158.101.118 (3 times).
  - A process CLOSE the file: sendmail.
  - A process CLOSE the file: .. 
  - A process OPEN the file: .. 
  - A process MODIFY_FILE_ATTRIBUTES the file: sendmail.
  - A process OPEN the file: sendmail.
  - A process WRITE the file: sendmail.

- **14:27**
  - A process SENDTO the flow: 53.158.101.118.
  - A process RECVFROM the flow: 53.158.101.118.

- **14:28**
  - A process SENDTO the flow: 53.158.101.118 (9 times).
  - A process RECVFROM the flow: 53.158.101.118 (9 times).
  - A process MODIFY_PROCESS a process (3 times).
  - A process OPEN the file: .. (3 times).
  - A process CLOSE the file: .. (3 times).
  - A process MODIFY_PROCESS the file: .. (2 times).
  - A process UNLINK the file: sendmail.

- **14:29**
  - A process RECVFROM the flow: 53.158.101.118 (3 times).
  - A process SENDTO the flow: 53.158.101.118 (3 times).
  - A process CLOSE the file: .. 
  - A process OPEN the file: .. 

- **14:32**
  - A process RECVFROM the flow: 53.158.101.118 (4 times).
  - A process SENDTO the flow: 53.158.101.118 (3 times).

- **14:33**
  - A process RECVFROM the flow: 53.158.101.118.
  - A process SENDTO the flow: 53.158.101.118.

- **14:34**
  - A process RECVFROM the flow: 53.158.101.118.
  - A process SENDTO the flow: 53.158.101.118.

- **14:35**
  - A process RECVFROM the flow: 53.158.101.118.

- **14:38**
  - A process SENDTO the flow: 53.158.101.118 (3 times).
  - A process RECVFROM the flow: 53.158.101.118 (2 times).
  - A process OPEN the file: .. 
  - A process CLOSE the file: .. 

- **14:39**
  - A process RECVFROM the flow: 53.158.101.118.
  - A process SENDTO the flow: 53.158.101.118.