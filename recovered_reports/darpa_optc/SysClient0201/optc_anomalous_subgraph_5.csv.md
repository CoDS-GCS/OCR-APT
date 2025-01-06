# Attack Report: Anomalous Activity Detected in optc_anomalous_subgraph_5.csv

## Summary of Attack Behavior

The logs from the document indicate a series of suspicious activities primarily involving the `svchost.exe` and `cmd.exe` processes. The timeline reveals that on September 23, 2019, at 10:03, `cmd.exe` initiated multiple module loads, which is a common behavior for command-line operations. However, the subsequent actions involving `svchost.exe` suggest a potential escalation of privileges or lateral movement within the system.

Notably, `svchost.exe` was observed reading a batch file named `DeleteArchiveSecurity.bat`, which raises concerns about potential malicious intent, as this file could be used to execute harmful commands. The repeated loading of critical system modules such as `KernelBase.dll`, `advapi32.dll`, and `bcryptprimitives.dll` indicates an attempt to leverage system functionalities, possibly for privilege escalation or persistence.

The overall behavior aligns with the **Internal Reconnaissance** and **Privilege Escalation** stages of an APT attack, where the attacker seeks to gather information and gain higher access levels within the system.

## Table of Indicators of Compromise (IoCs)

| IoC                          | Security Context                                                                                     |
|------------------------------|------------------------------------------------------------------------------------------------------|
| DeleteArchiveSecurity.bat    | A batch file that may contain commands to delete or manipulate files; potential for malicious use.  |
| cmd.exe                      | A legitimate command-line interpreter; often exploited for executing commands without user knowledge.|
| conhost.exe                  | Console host process for command-line applications; can be used to hide malicious activities.        |
| svchost.exe                  | A legitimate Windows service host; often targeted for exploitation due to its elevated privileges.   |
| KernelBase.dll               | A core Windows library; exploitation can lead to system-level access.                               |
| advapi32.dll                 | Provides advanced Windows API functions; can be exploited for privilege escalation.                  |
| bcryptprimitives.dll         | Used for cryptographic operations; potential for misuse in securing malicious payloads.              |
| appidapi.dll                 | Manages application identity; could be exploited for unauthorized access.                           |
| msvcp_win.dll                | Microsoft C++ runtime library; exploitation can lead to arbitrary code execution.                    |
| msvcrt.dll                   | C runtime library; often targeted for buffer overflow attacks.                                      |
| rpcrt4.dll                   | Remote procedure call runtime library; can be exploited for remote code execution.                   |
| sechost.dll                  | Security host library; critical for security operations; exploitation can lead to privilege escalation.|
| srpapi.dll                   | Security resource provider API; potential for misuse in security context manipulation.              |
| ucrtbase.dll                 | Universal C runtime library; can be exploited for various attacks, including memory corruption.     |
| win32u.dll                   | Provides Windows API functions; exploitation can lead to system-level access.                        |
| cmdext.dll                   | Command extensions for cmd.exe; can be used to execute additional commands.                         |
| combase.dll                  | Component Object Model base library; exploitation can lead to unauthorized access to COM objects.   |
| crypt32.dll                  | Provides cryptographic services; can be exploited for data exfiltration.                            |
| gdi32.dll                    | Graphics Device Interface library; exploitation can lead to arbitrary code execution.                |
| gdi32full.dll                | Full GDI library; similar risks as gdi32.dll.                                                      |
| kernel.appcore.dll           | Core Windows application library; exploitation can lead to system-level access.                     |
| msasn1.dll                   | ASN.1 encoding/decoding library; potential for exploitation in data handling.                       |

## Chronological Log of Actions

### September 23, 2019

- **10:03**: `cmd.exe` loads multiple modules including `ConhostV2.dll`, `kernel.appcore.dll`, `msvcp_win.dll`, `msvcrt.dll`, `oleaut32.dll`, `powrprof.dll`, `profapi.dll`, `rpcrt4.dll`, `sechost.dll`, `shell32.dll`, `ucrtbase.dll`, `win32u.dll`, and `windows.storage.dll`.
- **10:03**: `cmd.exe` opens the process `conhost.exe`.
- **10:03**: `conhost.exe` is terminated.
- **10:03**: `svchost.exe` creates the process `conhost.exe`.
- **10:03**: `svchost.exe` loads several modules including `KernelBase.dll`, `advapi32.dll`, `appidapi.dll`, `bcryptprimitives.dll`, `cmd.exe`, `cmdext.dll`, `combase.dll`, `crypt32.dll`, `gdi32.dll`, `gdi32full.dll`, `kernel.appcore.dll`, `msasn1.dll`, `msvcp_win.dll`, `msvcrt.dll`, `rpcrt4.dll`, `sechost.dll`, and `srpapi.dll`.
- **10:21**: `svchost.exe` loads additional modules including `appidapi.dll`, `bcryptprimitives.dll`, `msvcp_win.dll`, `msvcrt.dll`, `rpcrt4.dll`, `sechost.dll`, `srpapi.dll`, `ucrtbase.dll`, and `win32u.dll`.
- **10:22**: `svchost.exe` reads the file `DeleteArchiveSecurity.bat`.

This report highlights the suspicious activities observed in the logs, indicating potential malicious behavior that warrants further investigation.