# Attack Report: Anomalous Activity Detected in optc_anomalous_subgraph_6.csv

## Summary of Attack Behavior

On September 25, 2019, at 14:21, a series of module loads were executed by the process `svchost.exe`, indicating potential malicious activity. The loading of multiple DLLs, including `PERFCOUNTER.DLL`, suggests that the attacker may be attempting to manipulate performance counters or gather system information, which aligns with the **Internal Reconnaissance** stage of an APT attack. The subsequent read action on `PERFCOUNTER.DLL` at 14:27 further indicates that the attacker is likely gathering data or preparing for further exploitation.

The use of `svchost.exe` is particularly concerning, as this process is commonly exploited by attackers to run malicious code under the guise of legitimate system processes. The presence of various DLLs associated with system performance and management raises the likelihood of exploitation, as these components can be leveraged to maintain persistence or escalate privileges.

## Table of Indicators of Compromise (IoCs)

| IoC                          | Security Context                                                                                     |
|------------------------------|-----------------------------------------------------------------------------------------------------|
| PERFCOUNTER.DLL             | A legitimate Windows DLL used for performance monitoring. High likelihood of exploitation for data gathering. |
| CLBCATQ.DLL                  | A COM library used for DCOM services. Can be exploited for remote code execution.                  |
| CORPERFMONEXT.DLL           | Related to performance monitoring extensions. Potentially exploited for unauthorized data access.   |
| ESSCLI.DLL                   | Part of the Windows Event Log service. Can be used to manipulate event logging.                    |
| MSCOREEI.DLL                | .NET runtime library. Can be exploited to run malicious .NET applications.                          |
| MSVCR120_CLR0400.DLL        | Microsoft Visual C++ runtime library. Can be used to execute malicious code.                        |
| NCOBJAPI.DLL                | Used for network object management. Potential for exploitation in network-based attacks.            |
| PDH.DLL                      | Performance Data Helper library. Can be exploited to gather system performance data.               |
| RASMAN.DLL                  | Remote Access Service Manager. Can be exploited for unauthorized remote access.                    |
| VERSION.dll                  | Provides version information for Windows components. Low likelihood of exploitation.                |
| WEVTAPI.DLL                  | Windows Event Log API. Can be exploited to manipulate event logs.                                  |
| WMICLNT.DLL                 | Windows Management Instrumentation client. Can be exploited for system management tasks.            |
| WMIPERFCLASS.DLL            | WMI performance class. Can be exploited for performance data manipulation.                          |
| WMIPROV.DLL                 | WMI provider. Can be exploited to execute arbitrary code.                                          |
| WMIPRVSE.EXE                | WMI provider host process. Can be exploited for privilege escalation.                               |
| fastprox.dll                 | Fast proxy service for DCOM. Can be exploited for remote code execution.                           |
| wbemcomn.dll                 | Common WMI library. Can be exploited for unauthorized access to system information.                 |
| wbemprox.dll                 | WMI proxy service. Can be exploited for remote management tasks.                                   |
| wbemsvc.dll                  | WMI service. Can be exploited to execute malicious scripts.                                        |
| wtsapi32.dll                 | Windows Terminal Services API. Can be exploited for remote access.                                 |

## Chronological Log of Actions

| Timestamp          | Action Description                                      |
|--------------------|--------------------------------------------------------|
| 2019-09-25 14:21   | svchost.exe LOAD the module: CLBCATQ.DLL              |
| 2019-09-25 14:21   | svchost.exe LOAD the module: CORPERFMONEXT.DLL       |
| 2019-09-25 14:21   | svchost.exe LOAD the module: ESSCLI.DLL               |
| 2019-09-25 14:21   | svchost.exe LOAD the module: MSCOREEI.DLL             |
| 2019-09-25 14:21   | svchost.exe LOAD the module: MSVCR120_CLR0400.DLL     |
| 2019-09-25 14:21   | svchost.exe LOAD the module: NCOBJAPI.DLL             |
| 2019-09-25 14:21   | svchost.exe LOAD the module: PDH.DLL                   |
| 2019-09-25 14:21   | svchost.exe LOAD the module: PERFCOUNTER.DLL          |
| 2019-09-25 14:21   | svchost.exe LOAD the module: RASMAN.DLL               |
| 2019-09-25 14:21   | svchost.exe LOAD the module: VERSION.dll               |
| 2019-09-25 14:21   | svchost.exe LOAD the module: WEVTAPI.DLL              |
| 2019-09-25 14:21   | svchost.exe LOAD the module: WMICLNT.DLL              |
| 2019-09-25 14:21   | svchost.exe LOAD the module: WMIPERFCLASS.DLL         |
| 2019-09-25 14:21   | svchost.exe LOAD the module: WMIPROV.DLL              |
| 2019-09-25 14:21   | svchost.exe LOAD the module: WMIPRVSE.EXE             |
| 2019-09-25 14:21   | svchost.exe LOAD the module: fastprox.dll              |
| 2019-09-25 14:21   | svchost.exe LOAD the module: wbemcomn.dll             |
| 2019-09-25 14:21   | svchost.exe LOAD the module: wbemprox.dll             |
| 2019-09-25 14:21   | svchost.exe LOAD the module: wbemsvc.dll              |
| 2019-09-25 14:21   | svchost.exe LOAD the module: wtsapi32.dll             |
| 2019-09-25 14:27   | svchost.exe READ the file: PERFCOUNTER.DLL           |

This report highlights the suspicious behavior associated with the `svchost.exe` process and the modules loaded during the incident, indicating potential malicious activity consistent with APT tactics. Further investigation is recommended to assess the impact and scope of the incident.