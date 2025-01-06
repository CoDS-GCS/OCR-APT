# Attack Report: Anomalous Activity Detected in optc_anomalous_subgraph_2.csv

## Summary of Attack Behavior

On September 25, 2019, at 14:21, a series of actions were logged involving the process `svchost.exe`, which is a legitimate Windows process responsible for hosting multiple Windows services. However, the extensive loading of various modules and reading of executable files, particularly `SEARCHUI.EXE`, indicates potential malicious behavior. The logs show that `svchost.exe` was involved in loading numerous DLLs and executing files that are typically associated with system operations and user interface functionalities.

The behavior observed can be categorized into the following stages of the APT attack lifecycle:

- **Initial Compromise**: The loading of `SEARCHUI.EXE` and other modules suggests that the attacker may have exploited a vulnerability in the Windows environment to gain initial access.
- **Internal Reconnaissance**: The extensive reading and loading of various system modules indicate that the attacker is gathering information about the system and its capabilities.
- **Command and Control**: The presence of modules related to Cortana and Windows UI suggests that the attacker may be establishing a command and control mechanism to interact with the compromised system.

## Indicators of Compromise (IoCs)

| IoC                                           | Security Context                                                                                     |
|-----------------------------------------------|------------------------------------------------------------------------------------------------------|
| `SEARCHUI.EXE`                               | A legitimate Windows process for search functionality; could be exploited for unauthorized access.   |
| `tzres.dll.mui`                              | A resource file for time zone settings; typically benign but could be manipulated for persistence.  |
| `Windows.Storage.ApplicationData.dll`        | Used for accessing application data; legitimate but could be exploited for data exfiltration.       |
| `Windows.Graphics.dll`                        | Handles graphics rendering; legitimate but could be used to hide malicious activities.               |
| `Windows.Cortana.ProxyStub.dll`              | Related to Cortana's functionality; could be exploited for command and control.                     |
| `Windows.ApplicationModel.Background.TimeBroker.dll` | Manages background tasks; legitimate but could be used for persistence.                             |
| `WinTypes.dll`                               | Contains Windows type definitions; typically benign but could be exploited.                         |
| `WS2_32.DLL`                                 | Windows Sockets API; legitimate but could be used for network communication by malware.             |
| `WININET.DLL`                                | Used for internet-related functions; could be exploited for data exfiltration.                      |
| `WINHTTP.DLL`                                | Handles HTTP requests; legitimate but could be used for command and control.                        |
| `WINDOWSCODECS.DLL`                          | Used for media codecs; typically benign but could be exploited.                                     |
| `WINDOWS.WEB.WINMD`                          | Related to web functionalities; could be exploited for unauthorized access.                         |
| `WINDOWS.UI.DLL`                             | Handles user interface elements; could be exploited for malicious UI manipulation.                   |
| `WINDOWS.MEDIA.SPEECH.DLL`                  | Used for speech recognition; could be exploited for unauthorized audio capture.                      |
| `CortanaApi.dll`                             | Related to Cortana's API; could be exploited for command and control.                               |
| `CORTANA.CORE.DLL`                           | Core functionalities of Cortana; could be exploited for malicious purposes.                         |
| `SMARTSCREENPS.DLL`                          | Part of Windows SmartScreen; could be exploited to bypass security checks.                          |
| `SHCORE.DLL`                                 | Provides core Windows functionalities; could be exploited for privilege escalation.                  |
| `SECHOST.DLL`                                | Handles security-related functions; could be exploited for malicious activities.                     |

## Chronological Log of Actions

### September 25, 2019

- **14:21**
  - `svchost.exe` LOAD the module: `WINDOWS.UI.WINMD`
  - `svchost.exe` READ the file: `tzres.dll.mui`
  - `svchost.exe` OPEN the process: `svchost.exe`
  - `svchost.exe` LOAD the module: `winnsi.dll`
  - `svchost.exe` LOAD the module: `windows.storage.dll`
  - `svchost.exe` LOAD the module: `windows.cortana.pal.desktop.dll`
  - `svchost.exe` LOAD the module: `win32u.dll`
  - `svchost.exe` LOAD the module: `webplatstorageserver.dll`
  - `svchost.exe` LOAD the module: `usermgrcli.dll`
  - `svchost.exe` LOAD the module: `srpapi.dll`
  - `svchost.exe` LOAD the module: `shlwapi.dll`
  - `svchost.exe` LOAD the module: `XmlLite.dll`
  - `svchost.exe` LOAD the module: `shell32.DLL`
  - `svchost.exe` LOAD the module: `profapi.dll`
  - `svchost.exe` LOAD the module: `powrprof.dll`
  - `svchost.exe` LOAD the module: `oleaut32.dll`
  - `svchost.exe` LOAD the module: `ole32.dll`
  - `svchost.exe` LOAD the module: `nsi.dll`
  - `svchost.exe` LOAD the module: `mswsock.dll`
  - `svchost.exe` LOAD the module: `logoncli.dll`
  - `svchost.exe` LOAD the module: `gdi32full.dll`
  - `svchost.exe` LOAD the module: `dwmapi.dll`
  - `svchost.exe` LOAD the module: `combase.dll`
  - `svchost.exe` LOAD the module: `apphelp.dll`
  - `svchost.exe` LOAD the module: `shellCommonCommonProxyStub.dll`
  - `svchost.exe` LOAD the module: `IERTUTIL.DLL`
  - `svchost.exe` LOAD the module: `GDI32.DLL`
  - `svchost.exe` LOAD the module: `EDGEMANAGER.DLL`
  - `svchost.exe` LOAD the module: `EDGEHTML.DLL`
  - `svchost.exe` LOAD the module: `DXGI.DLL`
  - `svchost.exe` LOAD the module: `DEVOBJ.dll`
  - `svchost.exe` LOAD the module: `DCOMP.DLL`
  - `svchost.exe` LOAD the module: `D3D11.DLL`
  
- **14:22**
  - `svchost.exe` READ the file: `SEARCHUI.EXE`
  - `svchost.exe` LOAD the module: `VEEventDispatcher.dll`
  - `svchost.exe` LOAD the module: `D3DCompiler_47.dll`

This report highlights the suspicious activities associated with the `svchost.exe` process, indicating potential malicious intent and the need for further investigation and remediation.