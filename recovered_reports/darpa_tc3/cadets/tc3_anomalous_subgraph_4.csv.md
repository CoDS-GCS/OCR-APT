# Attack Report: tc3_anomalous_subgraph_4.csv

## Summary of Attack Behavior

On April 11, 2018, at 23:01, a series of actions were logged that indicate potential anomalous behavior associated with an advanced persistent threat (APT) attack. The logs show multiple interactions with three files: `bounds`, `libncursesw.so.8`, and `msgs`. 

The sequence of events suggests that the attacker may have been attempting to manipulate the `bounds` file, as it was opened, read, written to, and had its attributes modified. The `libncursesw.so.8` file was also opened and memory-mapped, which could indicate an attempt to exploit this shared library for malicious purposes. The `msgs` file was executed, which raises concerns about its legitimacy and potential use as a vector for further exploitation.

These actions align with the **Internal Reconnaissance** and **Privilege Escalation** stages of the APT lifecycle, as the attacker appears to be gathering information and attempting to manipulate system files for further access.

## Table of Indicators of Compromise (IoCs)

| IoC                     | Security Context                                                                                     |
|-------------------------|------------------------------------------------------------------------------------------------------|
| bounds                  | A file that appears to be manipulated. Legitimate usage may include configuration or data storage, but its modification raises suspicion of exploitation. |
| libncursesw.so.8       | A shared library commonly used for text-based user interfaces. While legitimate, its memory mapping could indicate an attempt to exploit vulnerabilities. |
| msgs                    | A file that was executed. This could be a legitimate messaging file, but its execution in this context suggests potential malicious intent. |

## Chronological Log of Actions

| Timestamp            | Action                                      | File                     |
|----------------------|---------------------------------------------|--------------------------|
| 2018-04-11 23:01     | CLOSE                                       | bounds                   |
| 2018-04-11 23:01     | CLOSE                                       | libncursesw.so.8        |
| 2018-04-11 23:01     | CLOSE                                       | msgs                     |
| 2018-04-11 23:01     | EXECUTE                                     | msgs                     |
| 2018-04-11 23:01     | MMAP                                        | libncursesw.so.8        |
| 2018-04-11 23:01     | MODIFY_FILE_ATTRIBUTES                      | bounds                   |
| 2018-04-11 23:01     | OPEN                                        | bounds                   |
| 2018-04-11 23:01     | OPEN                                        | libncursesw.so.8        |
| 2018-04-11 23:01     | OPEN                                        | msgs                     |
| 2018-04-11 23:01     | READ                                        | bounds                   |
| 2018-04-11 23:01     | WRITE                                       | bounds                   |

This report highlights the suspicious activities surrounding the files involved, indicating a potential APT attack that warrants further investigation.