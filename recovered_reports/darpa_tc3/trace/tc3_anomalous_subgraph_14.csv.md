# Attack Report: tc3_anomalous_subgraph_14.csv

## Summary of Attack Behavior

The logs from the document indicate a series of actions performed by the Thunderbird application on April 13, 2018, which suggest potential anomalous behavior consistent with an APT attack. The key events include multiple file operations, socket communications, and memory manipulations, which may indicate an attempt to exfiltrate data or maintain persistence within the system.

### Key Events:
- **Initial Compromise**: The logs show multiple file openings and reads, particularly of email-related files (e.g., `INBOX.msf`, `Sent-1.msf`, `Unsent Messages.msf`), which may indicate the attacker’s initial access to sensitive information.
- **Internal Reconnaissance**: The frequent reading of files such as `blist.sqlite` and `impab.mab` suggests the attacker was gathering information about the email accounts and contacts.
- **Command and Control**: The logs indicate multiple socket connections and communications, which could be indicative of establishing a command and control channel.
- **Data Exfiltration**: The repeated reads and writes to email files and the use of sockets suggest potential data exfiltration activities.
- **Covering Tracks**: The closing of various files and processes may indicate attempts to erase traces of the attack.

## Table of IoCs Detected

| IoC                             | Security Context                                                                                     |
|---------------------------------|------------------------------------------------------------------------------------------------------|
| INBOX.msf                       | A legitimate file used by Thunderbird to store incoming emails; exploitation could lead to data theft. |
| Sent-1.msf                      | A legitimate file for sent emails; access may reveal sensitive communications.                       |
| impab.mab                       | A legitimate address book file; exploitation could expose contact information.                      |
| startupCache.8.little           | A cache file used by Thunderbird; may contain remnants of previous sessions that could be exploited. |
| thunderbird.desktop              | A configuration file for the Thunderbird application; could be modified for persistence.            |
| 08e8e1c95fe2fc01f976f1e063a24ccd| A file with a hash indicating potential malicious content; requires further investigation.           |
| Drafts-1.msf                    | A legitimate file for draft emails; access may reveal unsubmitted communications.                   |
| Unsent Messages.msf             | A legitimate file for unsent emails; could contain sensitive information.                           |
| blist.sqlite                    | A legitimate SQLite database for managing contacts; exploitation could lead to data leakage.        |
| libXss.so.1                    | A legitimate library file; could be exploited for privilege escalation or code injection.            |
| libdconfsettings.so            | A legitimate library; potential exploitation could allow configuration changes.                      |
| mounts                          | A legitimate file related to mounted filesystems; could be used for reconnaissance.                 |

## Chronological Log of Actions

### April 13, 2018

- **13:40**
  - READ the file: Unsent Messages.msf
  - READ the file: blist.sqlite
  - READ the file: impab.mab
  - READ the file: libXss.so.1
  - READ the file: libdconfsettings.so
  - READ the file: mounts
  - READ the file: thunderbird.desktop
  - SENDMSG a srcsink
  - WRITE a socket
  - READ the file: maps
  - CONNECT a socket
  - LINK a fileLink
  - LINK the file: 127.0.1.1:+23977
  - MMAP a memory
  - MMAP the file: libXss.so.1
  - OPEN the file: maps
  - MMAP the file: startupCache.8.little
  - MODIFY_FILE_ATTRIBUTES a fileDir
  - MPROTECT a memory
  - OPEN the file: 08e8e1c95fe2fc01f976f1e063a24ccd
  - OPEN the file: INBOX.msf
  - OPEN the file: Sent-1.msf
  - OPEN the file: Unsent Messages.msf
  - OPEN the file: blist.sqlite
  - OPEN the file: impab.mab
  - OPEN the file: libXss.so.1
  - OPEN the file: libdconfsettings.so
  - MMAP the file: libdconfsettings.so

- **13:41**
  - READ a pipe (37 times)
  - WRITE a pipe (34 times)
  - RECVMSG a srcsink (22 times)
  - WRITE a srcsink (20 times)
  - READ the file: Sent-1.msf (7 times)
  - WRITE the file: Sent-1.msf (7 times)
  - MPROTECT a memory (5 times)

- **13:59**
  - READ the file: INBOX.msf (4 times)
  - WRITE the file: INBOX.msf (4 times)
  - READ a fileChar (2 times)
  - OPEN a fileDir (2 times)
  - OPEN a fileChar (2 times)
  - CLOSE a fileChar (2 times)
  - CLOSE a fileDir (2 times)
  - OPEN the file: impab.mab
  - OPEN the file: libXss.so.1
  - OPEN the file: libdconfsettings.so
  - OPEN the file: mounts
  - OPEN the file: startupCache.8.little
  - OPEN the file: thunderbird.desktop
  - READ a socket
  - READ the file: 08e8e1c95fe2fc01f976f1e063a24ccd
  - READ the file: Sent-1.msf
  - READ the file: Unsent Messages.msf
  - READ the file: blist.sqlite
  - READ the file: libXss.so.1
  - READ the file: libdconfsettings.so
  - READ the file: mounts
  - READ the file: thunderbird.desktop
  - WRITE a socket
  - READ the file: impab.mab
  - CLONE the process: thunderbird
  - CLOSE the file: 08e8e1c95fe2fc01f976f1e063a24ccd
  - CLOSE the file: Sent-1.msf
  - CLOSE the file: impab.mab
  - CLOSE the file: libXss.so.1
  - OPEN the file: blist.sqlite
  - CLOSE the file: mounts
  - CLOSE the file: thunderbird.desktop
  - CONNECT a socket
  - MMAP the file: libXss.so.1
  - MMAP the file: libdconfsettings.so
  - MMAP the file: startupCache.8.little
  - MODIFY_FILE_ATTRIBUTES a fileDir

- **14:01**
  - CLOSE the file: INBOX.msf
  - CLOSE the file: Sent-1.msf
  - CLOSE the file: impab.mab
  - EXIT the process: thunderbird
  - OPEN the file: Sent-1.msf
  - WRITE a socket
  - WRITE the file: Sent-1.msf
  - CLOSE the file: startupCache.8.little

- **14:02**
  - CLOSE a fileChar (2 times)
  - EXIT the process: python3 (2 times)
  - OPEN a fileChar (2 times)
  - READ a fileChar (2 times)

- **14:13**
  - READ the file: INBOX.msf (4 times)
  - WRITE the file: INBOX.msf (4 times)
  - OPEN a fileDir (3 times)
  - CLOSE a fileDir (3 times)
  - CLOSE a fileChar (3 times)
  - READ the file: Sent-1.msf (2 times)
  - READ a socket (2 times)
  - CONNECT a socket (2 times)
  - OPEN a fileChar (2 times)
  - READ a fileChar (2 times)
  - WRITE a socket (2 times)
  - CLONE the process: thunderbird
  - CLOSE the file: 08e8e1c95fe2fc01f976f1e063a24ccd
  - CLOSE the file: Drafts-1.msf
  - CLOSE the file: Sent-1.msf
  - CLOSE the file: impab.mab
  - CLOSE the file: libXss.so.1
  - CLOSE the file: libdconfsettings.so
  - CLOSE the file: mounts
  - CLOSE the file: thunderbird.desktop
  - READ the file: mounts
  - READ the file: thunderbird.desktop
  - READ the file: libXss.so.1

This report outlines the suspicious activities associated with the Thunderbird application, highlighting potential indicators of compromise and the stages of the APT attack. Further investigation is recommended to assess the extent of the compromise and to implement necessary security measures.