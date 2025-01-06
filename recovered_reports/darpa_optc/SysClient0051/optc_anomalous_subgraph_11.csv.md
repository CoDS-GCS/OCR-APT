# Attack Report: Anomalous Activity Detected in optc_anomalous_subgraph_11.csv

## Summary of Attack Behavior

The logs from the document indicate a series of suspicious activities primarily involving the process `svchost.exe` on September 25, 2019. The behavior suggests potential malicious activity, including outbound connections to known public DNS servers and the creation and deletion of various files associated with safe browsing features. 

### Key Events:
- **Initial Compromise**: The process `svchost.exe` initiated outbound connections to external IP addresses (8.8.4.4, 8.8.8.8, and 142.20.61.130) at 13:46, which are commonly used for DNS resolution and could indicate an attempt to communicate with a command and control (C2) server.
- **Internal Reconnaissance**: Multiple file reads and writes were observed, including access to `5f7b5f1e01b83767.automaticDestinations-ms`, which may contain user activity data.
- **Command and Control**: The outbound messages to external IPs suggest potential C2 communication, particularly with the IP 142.20.61.130, which is not a standard DNS server.
- **Data Manipulation**: The creation and deletion of files related to safe browsing features indicate attempts to manipulate browser security settings, potentially to facilitate further exploitation or data exfiltration.

## Table of IoCs Detected

| IoC                                           | Security Context                                                                                     |
|-----------------------------------------------|-----------------------------------------------------------------------------------------------------|
| 142.20.61.130                                 | External IP address; potential C2 server.                                                          |
| 8.8.4.4                                      | Google Public DNS server; legitimate but could be used for DNS tunneling.                         |
| 8.8.8.8                                      | Google Public DNS server; legitimate but could be used for DNS tunneling.                         |
| 5f7b5f1e01b83767.automaticDestinations-ms    | File related to user activity; could indicate reconnaissance or data collection.                   |
| JumpListIconsOld                              | Temporary file; frequent deletion may indicate attempts to cover tracks.                           |
| Safe Browsing Download Whitelist_new          | File related to safe browsing; creation and deletion suggest manipulation of security settings.     |
| Safe Browsing Download_new                     | File related to safe browsing; creation and deletion suggest manipulation of security settings.     |
| Safe Browsing Bloom_new                        | File related to safe browsing; creation and deletion suggest manipulation of security settings.     |
| 5d696d521de238c3.customDestinations-ms~RFf71df6.TMP | Temporary file; frequent creation and deletion may indicate malicious activity.                     |
| 5d696d521de238c3.customDestinations-ms~RFf6a8a7.TMP | Temporary file; frequent creation and deletion may indicate malicious activity.                     |
| Safe Browsing Extension Blacklist_new         | File related to safe browsing; creation and deletion suggest manipulation of security settings.     |
| Safe Browsing IP Blacklist_new                | File related to safe browsing; creation and deletion suggest manipulation of security settings.     |
| Safe Browsing Inclusion Whitelist_new          | File related to safe browsing; creation and deletion suggest manipulation of security settings.     |
| Safe Browsing UwS List_new                    | File related to safe browsing; creation and deletion suggest manipulation of security settings.     |
| scoped_dir_3940_32530                        | Temporary directory; could be used for malicious file storage.                                     |
| AcGenral.dll                                  | Legitimate Windows module; could be exploited for privilege escalation.                             |
| AudioSes.dll                                  | Legitimate Windows module; could be exploited for privilege escalation.                             |
| BluetoothApis.dll                             | Legitimate Windows module; could be exploited for privilege escalation.                             |
| chrome.dll                                    | Legitimate browser module; could be exploited for malicious activities.                             |
| Local State~RFf35872.TMP                      | Temporary file; frequent creation and deletion may indicate malicious activity.                     |
| .winlogbeat.yml                               | Configuration file for Winlogbeat; could be used for logging malicious activities.                 |
| 5d696d521de238c3.customDestinations-ms~RFf3e958.TMP | Temporary file; frequent creation and deletion may indicate malicious activity.                     |

## Chronological Log of Actions

### 13:42
- `svchost.exe` READ the file: HOSTS
- `svchost.exe` READ the file: Microsoft-Windows-Client-Features-WOW64-Package-AutoMerged-minkernel~31bf3856ad364e35~amd64~~10.0.15063.0.cat
- `svchost.exe` READ the file: Package_1138_for_KB4493474~31bf3856ad364e35~amd64~~10.0.1.5.cat
- `svchost.exe` READ the file: chrome.dll::$EA
- `svchost.exe` READ the file: color
- `svchost.exe` LOAD the module: msctf.dll
- `svchost.exe` LOAD the module: DWrite.dll
- `svchost.exe` LOAD the module: CHROME.EXE
- `svchost.exe` READ the file: docs.crx
- `svchost.exe` READ the file: drive.crx
- `svchost.exe` READ the file: external_extensions.json
- `svchost.exe` READ the file: rdpendp.dll
- `svchost.exe` READ the file: sRGB Color Space Profile.icm
- `svchost.exe` READ the file: search.crx
- `svchost.exe` READ the file: youtube.crx
- `svchost.exe` WRITE the file: Cookies-journal
- `svchost.exe` WRITE the file: First Run
- `svchost.exe` WRITE the file: etilqs_qKF3kepZy3jnc24
- `svchost.exe` CREATE the file: Cookies-journal
- `svchost.exe` CREATE the file: First Run
- `svchost.exe` CREATE the file: Top Sites-journal
- `svchost.exe` CREATE the file: etilqs_oBJllzdOdPguQJm
- `svchost.exe` CREATE the file: etilqs_qKF3kepZy3jnc24
- `svchost.exe` CREATE the file: scoped_dir_3940_13185
- `svchost.exe` CREATE the file: scoped_dir_3940_32530
- `svchost.exe` LOAD the module: AcGenral.dll
- `svchost.exe` LOAD the module: AudioSes.dll
- `svchost.exe` LOAD the module: BluetoothApis.dll
- `svchost.exe` LOAD the module: chrome.dll
- `svchost.exe` OPEN_INBOUND a flow (6 times)

### 13:43
- `svchost.exe` OPEN_INBOUND a flow (2 times)

### 13:45
- `svchost.exe` DELETE the file: 5d696d521de238c3.customDestinations-ms~RFf4d406.TMP
- `svchost.exe` LOAD the module: CoreMessaging.dll
- `svchost.exe` LOAD the module: CoreUIComponents.dll
- `svchost.exe` LOAD the module: TextInputFramework.dll
- `svchost.exe` LOAD the module: usermgrcli.dll
- `svchost.exe` OPEN_INBOUND a flow
- `svchost.exe` READ the file: 5f7b5f1e01b83767.automaticDestinations-ms
- `svchost.exe` READ the file: D413.tmp
- `svchost.exe` DELETE the file: 5d696d521de238c3.customDestinations-ms~RFf54917.TMP

### 13:46
- `svchost.exe` DELETE the file: JumpListIconsOld (2 times)
- `svchost.exe` OPEN_INBOUND a flow (2 times)
- `svchost.exe` WRITE the file: .winlogbeat.yml
- `svchost.exe` START_OUTBOUND the flow: 8.8.8.8
- `svchost.exe` READ the file: D413.tmp
- `svchost.exe` START_OUTBOUND the flow: 142.20.61.130
- `svchost.exe` START_OUTBOUND the flow: 8.8.4.4
- `svchost.exe` READ the file: 5f7b5f1e01b83767.automaticDestinations-ms
- `svchost.exe` CREATE the file: 5d696d521de238c3.customDestinations-ms~RFf5be37.TMP
- `svchost.exe` CREATE the file: 5d696d521de238c3.customDestinations-ms~RFf63367.TMP
- `svchost.exe` DELETE the file: 5d696d521de238c3.customDestinations-ms~RFf5be37.TMP
- `svchost.exe` DELETE the file: 5d696d521de238c3.customDestinations-ms~RFf63367.TMP
- `svchost.exe` MESSAGE_INBOUND the flow: 142.20.56.52
- `svchost.exe` MESSAGE_OUTBOUND the flow: 142.20.61.130
- `svchost.exe` MESSAGE_OUTBOUND the flow: 8.8.4.4
- `svchost.exe` MESSAGE_OUTBOUND the flow: 8.8.8.8

### 13:47
- `svchost.exe` OPEN_INBOUND a flow (2 times)
- `svchost.exe` READ the file: 5f7b5f1e01b83767.automaticDestinations-ms (2 times)

### 13:48
- `svchost.exe` DELETE the file: JumpListIconsOld (2 times)
- `svchost.exe` OPEN_INBOUND a flow (2 times)
- `svchost.exe` READ the file: 5f7b5f1e01b83767.automaticDestinations-ms (2 times)
- `svchost.exe` DELETE the file: Safe Browsing Download Whitelist_new
- `svchost.exe` DELETE the file: Safe Browsing Download_new
- `svchost.exe` DELETE the file: Safe Browsing Extension Blacklist_new
- `svchost.exe` DELETE the file: Safe Browsing IP Blacklist_new
- `svchost.exe` DELETE the file: Safe Browsing Inclusion Whitelist_new
- `svchost.exe` DELETE the file: Safe Browsing UwS List_new
- `svchost.exe` CREATE the file: 5d696d521de238c3.customDestinations-ms~RFf79316.TMP
- `svchost.exe` CREATE the file: Safe Browsing Download Whitelist_new
- `svchost.exe` CREATE the file: Safe Browsing Download_new
- `svchost.exe` CREATE the file: Safe Browsing Extension Blacklist_new
- `svchost.exe` CREATE the file: Safe Browsing IP Blacklist_new
- `svchost.exe` CREATE the file: Safe Browsing Inclusion Whitelist_new
- `svchost.exe` CREATE the file: Safe Browsing UwS List_new
- `svchost.exe` DELETE the file: 5d696d521de238c3.customDestinations-ms~RFf79316.TMP
- `svchost.exe` DELETE the file: Safe Browsing Bloom_new

### 13:49
- `svchost.exe` READ the file: 5f7b5f1e01b83767.automaticDestinations-ms (2 times)
- `svchost.exe` DELETE the file: JumpListIconsOld (2 times)
- `svchost.exe` DELETE the file: 5d696d521de238c3.customDestinations-ms~RFf8f2c6.TMP
- `svchost.exe` DELETE the file: 5d696d521de238c3.customDestinations-ms~RFf87d67.TMP

This report highlights the suspicious activities associated with the `svchost.exe` process, indicating potential malicious behavior that warrants further investigation.