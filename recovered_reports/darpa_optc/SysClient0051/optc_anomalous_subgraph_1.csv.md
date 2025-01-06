# Attack Report: Anomalous Activity Detected in optc_anomalous_subgraph_1.csv

## Summary of Attack Behavior

The logs from the document indicate a series of suspicious activities primarily involving the `svchost.exe` process on September 25, 2019. The timeline reveals multiple instances of module loading, outbound connections, and file reads, suggesting potential malicious behavior. 

Key events include:
- **Module Loading**: The `svchost.exe` process loaded several modules, including `cKfGW.exe`, which is not a standard Windows module and raises suspicion of being a potential threat.
- **Outbound Connections**: The process initiated an outbound connection to the IP address `53.192.68.50`, which could indicate a command and control (C2) communication.
- **File Reads**: The process read the file `SRVCLI.DLL`, which may be part of the attacker's toolkit or a method to gather information about the system.
- **Thread Termination**: The termination of a thread at 10:39 could indicate an attempt to cover tracks or terminate malicious activities.

These actions suggest that the attack is in the **Command and Control** stage, where the attacker is likely trying to establish a foothold and communicate with external servers.

## Indicators of Compromise (IoCs)

| IoC                | Security Context                                                                                     |
|--------------------|-----------------------------------------------------------------------------------------------------|
| 53.192.68.50      | An external IP address that may be associated with malicious activity. Requires further investigation to determine its legitimacy. |
| 142.20.56.52      | Another external IP address that was involved in inbound messaging. Potentially linked to C2 operations. |
| cKfGW.exe          | An executable that is not recognized as a standard Windows process. High likelihood of being a malicious payload. |

## Chronological Log of Actions

### September 25, 2019

- **10:15**
  - `svchost.exe` LOAD the module: wininet.dll
  - `svchost.exe` LOAD the module: winhttp.dll
  - `svchost.exe` LOAD the module: webio.dll
  - `svchost.exe` LOAD the module: userenv.dll
  - `svchost.exe` LOAD the module: mswsock.dll
  - `svchost.exe` LOAD the module: sechost.dll
  - `svchost.exe` LOAD the module: rasapi32.dll
  - `svchost.exe` LOAD the module: psmachine.dll
  - `svchost.exe` LOAD the module: ntmarta.dll
  - `svchost.exe` LOAD the module: ntdll.dll
  - `svchost.exe` LOAD the module: nsi.dll
  - `svchost.exe` LOAD the module: netutils.dll
  - `svchost.exe` LOAD the module: netapi32.dll
  - `svchost.exe` LOAD the module: msxml3.dll
  - `svchost.exe` LOAD the module: sspicli.dll
  - `svchost.exe` MESSAGE_OUTBOUND the flow: 53.192.68.50 (2 times)

- **10:28**
  - `svchost.exe` LOAD the module: wininet.dll
  - `svchost.exe` LOAD the module: winhttp.dll
  - `svchost.exe` LOAD the module: cKfGW.exe
  - `svchost.exe` START_OUTBOUND the flow: 53.192.68.50
  - `svchost.exe` MESSAGE_INBOUND the flow: 142.20.56.52 (2 times)
  - `svchost.exe` LOAD the module: wkscli.dll (2 times)
  - `svchost.exe` LOAD the module: netapi32.dll (2 times)
  - `svchost.exe` LOAD the module: mswsock.dll
  - `svchost.exe` LOAD the module: msvcrt.dll
  - `svchost.exe` LOAD the module: kernel32.dll
  - `svchost.exe` LOAD the module: imm32.dll
  - `svchost.exe` LOAD the module: dhcpcsvc6.dll
  - `svchost.exe` LOAD the module: dhcpcsvc.dll
  - `svchost.exe` LOAD the module: cscapi.dll
  - `svchost.exe` LOAD the module: cryptsp.dll
  - `svchost.exe` LOAD the module: cryptbase.dll
  - `svchost.exe` LOAD the module: bcryptprimitives.dll
  - `svchost.exe` LOAD the module: apphelp.dll
  - `svchost.exe` LOAD the module: advapi32.dll
  - `svchost.exe` LOAD the module: KernelBase.dll

- **10:39**
  - `svchost.exe` TERMINATE a thread

This report highlights the suspicious activities associated with the `svchost.exe` process, indicating a potential APT attack that requires further investigation and mitigation measures.