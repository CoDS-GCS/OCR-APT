# Attack Report: tc3_anomalous_subgraph_8.csv

## Summary of Attack Behavior

On April 11, 2018, at 23:58, a series of actions were logged that indicate potential anomalous behavior involving the file `libmd.so.6`. The sequence of events includes the opening, memory mapping (MMAP), execution, and closing of this file. The execution of the `sort` command coincides with these actions, suggesting that the process may have been used to manipulate or analyze data in a potentially malicious manner. 

Given the nature of the file `libmd.so.6`, which is a shared library, its usage in this context raises concerns about the possibility of exploitation. The actions taken during this incident align with the **Internal Reconnaissance** stage of an APT attack, where the attacker seeks to gather information about the system and its files.

## Table of Indicators of Compromise (IoCs)

| IoC            | Security Context                                                                                     |
|----------------|------------------------------------------------------------------------------------------------------|
| libmd.so.6     | A shared library file that is typically legitimate; however, its usage in this context raises suspicion of exploitation or manipulation. |
| sort           | A standard command-line utility for sorting data; while legitimate, its execution alongside suspicious file activity warrants further investigation. |

## Chronological Log of Actions

| Timestamp               | Action                                      |
|-------------------------|---------------------------------------------|
| 2018-04-11 23:58       | OPEN the file: libmd.so.6                  |
| 2018-04-11 23:58       | MMAP the file: libmd.so.6                  |
| 2018-04-11 23:58       | EXECUTE the file: sort                      |
| 2018-04-11 23:58       | CLOSE the file: libmd.so.6                  | 

This report highlights the need for further investigation into the actions surrounding `libmd.so.6` and the `sort` command to determine if they are part of a larger malicious activity.