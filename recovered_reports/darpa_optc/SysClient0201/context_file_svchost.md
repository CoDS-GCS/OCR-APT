# Attack Report: Context File SVCHOST

## Summary of Attack Behavior

The logs from the context_file_svchost indicate a series of actions that suggest a coordinated attack involving multiple processes. The attack appears to have begun with the creation and modification of various SVCHOST-related files, particularly `SVCHOST.EXE-2562231B.pf`, which was created and written to multiple times by various processes, including `GoogleUpdate.exe`, `SearchProtocolHost.exe`, and `powershell.exe`. 

The initial compromise stage is indicated by the creation of the `SVCHOST.EXE-2562231B.pf` file, which was subsequently modified by several processes, suggesting an attempt to maintain persistence. The presence of multiple processes reading and modifying the same files indicates internal reconnaissance and potential lateral movement within the system. 

The logs show that processes such as `powershell.exe`, `cmd.exe`, and `GoogleUpdate.exe` were heavily involved in both reading and writing to these files, which could indicate an attempt to manipulate system configurations or gather information about the environment. However, there are no clear indicators of data exfiltration or covering tracks in the logs provided.

## Table of IoCs Detected

| IoC                             | Security Context                                                                                     |
|---------------------------------|-----------------------------------------------------------------------------------------------------|
| SVCHOST.EXE-2562231B.pf        | A potentially malicious file created during the attack, likely used for persistence.               |
| svchost.exe                     | A legitimate Windows process, but often exploited by attackers for malicious purposes.              |
| tasklist.exe                    | A legitimate command-line utility, but can be used by attackers to gather information about running processes. |
| powershell.exe                  | A legitimate scripting environment that can be exploited for executing malicious scripts.           |
| GoogleUpdate.exe                | Typically a legitimate updater, but can be misused by attackers to maintain persistence.            |
| SearchProtocolHost.exe          | A legitimate process that can be exploited for reconnaissance or data manipulation.                 |
| conhost.exe                     | A legitimate console host process, but can be leveraged by attackers for command execution.        |
| wmiprvse.exe                    | A legitimate WMI process that can be exploited for system information gathering.                     |
| cmd.exe                         | A legitimate command-line interpreter that can be used for executing commands and scripts.          |
| SVCHOST.EXE-00ABB06A.pf        | A potentially malicious file, indicating further exploitation attempts.                             |
| SVCHOST.EXE-04C8CE0B.pf        | A potentially malicious file, indicating further exploitation attempts.                             |
| SVCHOST.EXE-135A30D8.pf        | A potentially malicious file, indicating further exploitation attempts.                             |
| SVCHOST.EXE-74432B26.pf        | A potentially malicious file, indicating further exploitation attempts.                             |
| SVCHOST.EXE-8DA0BAAD.pf        | A potentially malicious file, indicating further exploitation attempts.                             |
| SVCHOST.EXE-8FD92526.pf        | A potentially malicious file, indicating further exploitation attempts.                             |
| SVCHOST.EXE-A88B0E13.pf        | A potentially malicious file, indicating further exploitation attempts.                             |
| SVCHOST.EXE-CA1952BB.pf        | A potentially malicious file, indicating further exploitation attempts.                             |

## Chronological Log of Actions

### 2019-09-23 09:36
- `powershell.exe` READ the file: SVCHOST.EXE-A88B0E13.PF
- `GoogleUpdate.exe` MODIFY the file: SVCHOST.EXE-A88B0E13.PF
- `GoogleUpdate.exe` READ the file: SVCHOST.EXE-A88B0E13.PF
- `SearchProtocolHost.exe` MODIFY the file: SVCHOST.EXE-A88B0E13.PF
- `SearchProtocolHost.exe` READ the file: SVCHOST.EXE-A88B0E13.PF
- `cmd.exe` MODIFY the file: SVCHOST.EXE-A88B0E13.PF
- `cmd.exe` READ the file: SVCHOST.EXE-A88B0E13.PF
- `conhost.exe` MODIFY the file: SVCHOST.EXE-A88B0E13.PF
- `conhost.exe` READ the file: SVCHOST.EXE-A88B0E13.PF
- `powershell.exe` MODIFY the file: SVCHOST.EXE-A88B0E13.PF
- `svchost.exe` MODIFY the file: SVCHOST.EXE-A88B0E13.PF
- `tasklist.exe` MODIFY the file: SVCHOST.EXE-A88B0E13.PF
- `tasklist.exe` READ the file: SVCHOST.EXE-A88B0E13.PF
- `wmiprvse.exe` MODIFY the file: SVCHOST.EXE-A88B0E13.PF
- `wmiprvse.exe` READ the file: SVCHOST.EXE-A88B0E13.PF
- `svchost.exe` READ the file: SVCHOST.EXE-A88B0E13.PF

### 2019-09-23 09:41
- `powershell.exe` WRITE the file: SVCHOST.EXE-2562231B.pf
- `svchost.exe` CREATE the file: SVCHOST.EXE-2562231B.pf
- `svchost.exe` WRITE the file: SVCHOST.EXE-2562231B.pf
- `tasklist.exe` CREATE the file: SVCHOST.EXE-2562231B.pf
- `tasklist.exe` WRITE the file: SVCHOST.EXE-2562231B.pf
- `wmiprvse.exe` CREATE the file: SVCHOST.EXE-2562231B.pf
- `wmiprvse.exe` WRITE the file: SVCHOST.EXE-2562231B.pf
- `powershell.exe` CREATE the file: SVCHOST.EXE-2562231B.pf
- `GoogleUpdate.exe` CREATE the file: SVCHOST.EXE-2562231B.pf
- `GoogleUpdate.exe` WRITE the file: SVCHOST.EXE-2562231B.pf
- `SearchProtocolHost.exe` CREATE the file: SVCHOST.EXE-2562231B.pf
- `SearchProtocolHost.exe` WRITE the file: SVCHOST.EXE-2562231B.pf
- `cmd.exe` CREATE the file: SVCHOST.EXE-2562231B.pf

### 2019-09-23 12:12
- `powershell.exe` READ the file: SVCHOST.EXE-383BACA3.PF

### 2019-09-23 12:13
- `GoogleUpdate.exe` READ the file: SVCHOST.EXE-A88B0E13.PF
- `SearchProtocolHost.exe` READ the file: SVCHOST.EXE-A88B0E13.PF
- `cmd.exe` READ the file: SVCHOST.EXE-A88B0E13.PF
- `conhost.exe` READ the file: SVCHOST.EXE-A88B0E13.PF
- `powershell.exe` READ the file: SVCHOST.EXE-A88B0E13.PF
- `svchost.exe` READ the file: SVCHOST.EXE-A88B0E13.PF
- `wmiprvse.exe` READ the file: SVCHOST.EXE-A88B0E13.PF
- `tasklist.exe` READ the file: SVCHOST.EXE-A88B0E13.PF

### 2019-09-23 12:24
- `tasklist.exe` CREATE the file: SVCHOST.EXE-2562231B.pf
- `tasklist.exe` WRITE the file: SVCHOST.EXE-2562231B.pf
- `wmiprvse.exe` CREATE the file: SVCHOST.EXE-2562231B.pf
- `wmiprvse.exe` WRITE the file: SVCHOST.EXE-2562231B.pf
- `svchost.exe` WRITE the file: SVCHOST.EXE-2562231B.pf
- `GoogleUpdate.exe` CREATE the file: SVCHOST.EXE-2562231B.pf
- `GoogleUpdate.exe` WRITE the file: SVCHOST.EXE-2562231B.pf
- `SearchProtocolHost.exe` CREATE the file: SVCHOST.EXE-2562231B.pf
- `SearchProtocolHost.exe` WRITE the file: SVCHOST.EXE-2562231B.pf
- `cmd.exe` CREATE the file: SVCHOST.EXE-2562231B.pf
- `cmd.exe` WRITE the file: SVCHOST.EXE-2562231B.pf
- `conhost.exe` CREATE the file: SVCHOST.EXE-2562231B.pf
- `conhost.exe` WRITE the file: SVCHOST.EXE-2562231B.pf
- `powershell.exe` CREATE the file: SVCHOST.EXE-2562231B.pf
- `powershell.exe` WRITE the file: SVCHOST.EXE-2562231B.pf
- `svchost.exe` CREATE the file: SVCHOST.EXE-2562231B.pf

### 2019-09-23 12:35
- `GoogleUpdate.exe` CREATE the file: SVCHOST.EXE-2562231B.pf
- `GoogleUpdate.exe` WRITE the file: SVCHOST.EXE-2562231B.pf
- `SearchProtocolHost.exe` CREATE the file: SVCHOST.EXE-2562231B.pf
- `SearchProtocolHost.exe` WRITE the file: SVCHOST.EXE-2562231B.pf

### 2019-09-23 11:00
- `svchost.exe` MODIFY the file: SVCHOST.EXE-383BACA3.pf (3 times)
- `tasklist.exe` MODIFY the file: SVCHOST.EXE-383BACA3.pf (3 times)
- `GoogleUpdate.exe` MODIFY the file: SVCHOST.EXE-383BACA3.pf (3 times)
- `wmiprvse.exe` MODIFY the file: SVCHOST.EXE-383BACA3.pf (3 times)
- `SearchProtocolHost.exe` MODIFY the file: SVCHOST.EXE-383BACA3.pf (3 times)
- `cmd.exe` MODIFY the file: SVCHOST.EXE-383BACA3.pf (3 times)
- `conhost.exe` MODIFY the file: SVCHOST.EXE-383BACA3.pf (3 times)
- `powershell.exe` MODIFY the file: SVCHOST.EXE-383BACA3.pf (3 times)
- `tasklist.exe` READ the file: SVCHOST.EXE-383BACA3.pf (2 times)
- `wmiprvse.exe` READ the file: SVCHOST.EXE-383BACA3.pf (2 times)
- `powershell.exe` READ the file: SVCHOST.EXE-383BACA3.pf (2 times)
- `conhost.exe` READ the file: SVCHOST.EXE-383BACA3.pf (2 times)
- `cmd.exe` READ the file: SVCHOST.EXE-383BACA3.pf (2 times)
- `SearchProtocolHost.exe` READ the file: SVCHOST.EXE-383BACA3.pf (2 times)
- `GoogleUpdate.exe` READ the file: SVCHOST.EXE-383BACA3.pf (2 times)
- `svchost.exe` READ the file: SVCHOST.EXE-383BACA3.pf (2 times)

### 2019-09-23 11:03
- `GoogleUpdate.exe` MODIFY the file: svchost.exe
- `SearchProtocolHost.exe` MODIFY the file: svchost.exe

This report summarizes the key events and actions taken during the incident, highlighting the potential for exploitation and the involvement of various processes in the attack. Further investigation is recommended to assess the full impact and to implement necessary security measures.