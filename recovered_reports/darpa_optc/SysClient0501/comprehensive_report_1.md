# Comprehensive Attack Report: Anomalous Activity Detected

## Summary of Attack Behavior

The analysis of the provided reports indicates a coordinated and sophisticated attack, likely an advanced persistent threat (APT), characterized by multiple stages of exploitation and manipulation. The logs span various timestamps, primarily focusing on the activities of the `svchost.exe` process, which is commonly exploited by attackers to execute malicious actions while masquerading as a legitimate system process.

### Key Events and Stages Identified:

1. **Initial Compromise**:
   - The attack appears to have initiated with the exploitation of the `svchost.exe` process, which is known for its legitimate use in Windows but can be manipulated for malicious purposes. The presence of external IP addresses `202.6.172.98` and `142.20.57.246` suggests initial communication with command and control (C2) servers. Additionally, processes such as `powershell.exe`, `lsass.exe`, and `cmd.exe` indicate that the attacker may have gained initial access through a script or command execution.

2. **Internal Reconnaissance**:
   - The logs indicate multiple read and write operations on various files, including suspicious files such as `8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_35.rslc`, which may have been used to gather information about the system and its configurations. The use of `NETSTAT.EXE` and `PING.EXE` further suggests that the attacker was gathering information about the network and available hosts.

3. **Command and Control**:
   - Repeated outbound messages to `202.6.172.98` and inbound messages from `142.20.57.246` highlight the establishment of a C2 channel, allowing the attacker to maintain control over the compromised system. The consistent outbound messages from various processes to the IP address `202.6.172.98` suggest that this IP may be a command and control server, facilitating further instructions to the compromised system.

4. **Privilege Escalation**:
   - The use of legitimate modules such as `Microsoft.Powershell.Commands.Diagnostics.ni.dll` and `Microsoft.WSMan.Management.ni.dll` indicates attempts to escalate privileges and gain unauthorized access to sensitive system functionalities.

5. **Lateral Movement**:
   - The presence of multiple instances of `svchost.exe`, `conhost.exe`, and `python.exe` suggests potential lateral movement within the network, allowing the attacker to explore and exploit other systems.

6. **Maintain Persistence**:
   - Although no specific IoCs were identified for this stage, the continuous use of `svchost.exe` and the creation of suspicious files indicate efforts to maintain persistence within the environment. Processes like `GoogleUpdate.exe`, `schtasks.exe`, and `compattelrunner.exe` are often used to maintain persistence on a system, indicating that the attacker may have set up mechanisms to ensure continued access.

7. **Data Exfiltration**:
   - No direct evidence of data exfiltration was found in the logs, but the established C2 communication could facilitate such actions if the attacker chose to do so.

8. **Covering Tracks**:
   - The deletion of files and manipulation of system resources suggest attempts to cover tracks and erase evidence of the attack.

## Table of Indicators of Compromise (IoCs)

| IoC                                                                 | Security Context                                                                                     |
|---------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| `202.6.172.98`                                                      | An external IP address frequently used for C2 communications. High likelihood of exploitation.      |
| `142.20.57.246`                                                     | Another external IP address involved in inbound communications. Potentially malicious.               |
| `svchost.exe`                                                       | A legitimate Windows process that can be exploited for malicious purposes. Moderate to high risk.   |
| `8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_35.rslc` | Potentially malicious file; could be used for unauthorized access or control.                      |
| `8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_33.rslc` | Similar to above; indicates possible manipulation of system settings or configurations.             |
| `8d273d55-059f-4c89-9fd2-587b4bad1ce4_S-1-5-21-4190936083-3304963419-1584388968-1105_31.rslc` | Indicates potential data manipulation; could be part of a larger attack vector.                    |
| `Microsoft.Powershell.Commands.Diagnostics.ni.dll`                 | A legitimate PowerShell module that can be exploited for malicious scripting.                       |
| `Microsoft.WSMan.Management.ni.dll`                                 | A legitimate module for managing Windows systems remotely, which can be abused for unauthorized access. |
| `powershell.exe`                                                   | Legitimate Windows process often used for scripting; can be exploited for malicious commands.       |
| `lsass.exe`                                                        | Windows process for managing security policies; can be targeted for credential harvesting.          |
| `cmd.exe`                                                          | Command-line interpreter; can be used for executing commands and scripts maliciously.               |
| `wmiprvse.exe`                                                     | Windows Management Instrumentation process; can be exploited for remote management tasks.           |
| `GoogleUpdate.exe`                                                 | Legitimate updater for Google applications; can be misused for persistence.                         |
| `taskhostw.exe`                                                   | Windows process for running tasks; can be exploited for scheduled tasks.                           |
| `conhost.exe`                                                     | Console host process; can be used to execute commands in a console window.                         |
| `compattelrunner.exe`                                             | Windows process for compatibility telemetry; can be exploited for persistence.                      |
| `NETSTAT.EXE`                                                     | Network utility to display active connections; can be used for reconnaissance.                     |
| `PING.EXE`                                                        | Utility for testing network connectivity; can be used for reconnaissance.                          |
| `schtasks.exe`                                                   | Utility for managing scheduled tasks; can be exploited for persistence.                            |
| `winlogon.exe`                                                   | Windows logon process; can be targeted for credential harvesting.                                   |

## Chronological Log of Actions

### 2019-09-24 11:31
- `powershell.exe` MESSAGE_OUTBOUND the flow : `202.6.172.98` (2 times)
- `lsass.exe` START_OUTBOUND the flow : `202.6.172.98` (2 times)

### 2019-09-24 11:32
- `powershell.exe` MESSAGE_OUTBOUND the flow : `202.6.172.98` (5 times)
- `lsass.exe` START_OUTBOUND the flow : `202.6.172.98` (5 times)
- `lsass.exe` MESSAGE_OUTBOUND the flow : `202.6.172.98` (5 times)
- `cmd.exe` START_OUTBOUND the flow : `202.6.172.98` (5 times)
- `powershell.exe` START_OUTBOUND the flow : `202.6.172.98` (5 times)
- `wmiprvse.exe` START_OUTBOUND the flow : `202.6.172.98` (5 times)
- `cmd.exe` MESSAGE_OUTBOUND the flow : `202.6.172.98` (5 times)
- `GoogleUpdate.exe` START_OUTBOUND the flow : `202.6.172.98` (5 times)
- `GoogleUpdate.exe` MESSAGE_OUTBOUND the flow : `202.6.172.98` (5 times)
- `wmiprvse.exe` MESSAGE_OUTBOUND the flow : `202.6.172.98` (5 times)
- `svchost.exe` START_OUTBOUND the flow : `202.6.172.98` (5 times)
- `svchost.exe` MESSAGE_OUTBOUND the flow : `202.6.172.98` (5 times)
- `conhost.exe` MESSAGE_OUTBOUND the flow : `202.6.172.98` (3 times)
- `compattelrunner.exe` START_OUTBOUND the flow : `202.6.172.98` (2 times)
- `compattelrunner.exe` START_OUTBOUND the flow : `202.6.172.98` (2 times)
- `conhost.exe` MESSAGE_OUTBOUND the flow : `202.6.172.98` (2 times)
- `cmd.exe` MESSAGE_OUTBOUND the flow : `202.6.172.98` (2 times)
- `svchost.exe` START_OUTBOUND the flow : `202.6.172.98` (2 times)

### 2019-09-24 12:46
- `winlogon.exe` MESSAGE_OUTBOUND the flow : `202.6.172.98` (2 times)
- `winlogon.exe` START_OUTBOUND the flow : `202.6.172.98` (2 times)

### 2019-09-24 12:51
- `conhost.exe` MESSAGE_OUTBOUND the flow : `202.6.172.98` (2 times)
- `conhost.exe` START_OUTBOUND the flow : `202.6.172.98` (2 times)
- `lsass.exe` MESSAGE_OUTBOUND the flow : `202.6.172.98` (2 times)
- `lsass.exe` START_OUTBOUND the flow : `202.6.172.98` (2 times)
- `powershell.exe` MESSAGE_OUTBOUND the flow : `202.6.172.98` (2 times)
- `powershell.exe` START_OUTBOUND the flow : `202.6.172.98` (2 times)
- `schtasks.exe` MESSAGE_OUTBOUND the flow : `202.6.172.98` (2 times)
- `schtasks.exe` START_OUTBOUND the flow : `202.6.172.98` (2 times)
- `svchost.exe` MESSAGE_OUTBOUND the flow : `202.6.172.98` (2 times)
- `svchost.exe` START_OUTBOUND the flow : `202.6.172.98` (2 times)
- `compattelrunner.exe` START_OUTBOUND the flow : `202.6.172.98` (2 times)
- `GoogleUpdate.exe` START_OUTBOUND the flow : `202.6.172.98` (2 times)
- `NETSTAT.EXE` MESSAGE_OUTBOUND the flow : `202.6.172.98` (2 times)
- `GoogleUpdate.exe` MESSAGE_OUTBOUND the flow : `202.6.172.98` (2 times)
- `GoogleUpdate.exe` MESSAGE_OUTBOUND the flow : `202.6.172.98` (2 times)
- `GoogleUpdate.exe` START_OUTBOUND the flow : `202.6.172.98` (2 times)
- `NETSTAT.EXE` MESSAGE_OUTBOUND the flow : `202.6.172.98` (2 times)
- `conhost.exe` START_OUTBOUND the flow : `202.6.172.98` (2 times)
- `lsass.exe` MESSAGE_OUTBOUND the flow : `202.6.172.98` (2 times)
- `lsass.exe` START_OUTBOUND the flow : `202.6.172.98` (2 times)
- `powershell.exe` MESSAGE_OUTBOUND the flow : `202.6.172.98` (2 times)
- `powershell.exe` START_OUTBOUND the flow : `202.6.172.98` (2 times)
- `schtasks.exe` MESSAGE_OUTBOUND the flow : `202.6.172.98` (2 times)
- `schtasks.exe` START_OUTBOUND the flow : `202.6.172.98` (2 times)
- `svchost.exe` MESSAGE_OUTBOUND the flow : `202.6.172.98` (2 times)
- `svchost.exe` START_OUTBOUND the flow : `202.6.172.98` (2 times)
- `winlogon.exe` MESSAGE_OUTBOUND the flow : `202.6.172.98` (2 times)

## Conclusion

The logs from the provided reports reveal a complex and multi-faceted attack involving the exploitation of legitimate processes and the establishment of C2 communications. The identified IoCs warrant immediate investigation and remediation actions to mitigate the risks associated with this APT attack. Continuous monitoring and analysis of network traffic and system behavior are essential to prevent further exploitation and ensure the integrity of the affected systems.