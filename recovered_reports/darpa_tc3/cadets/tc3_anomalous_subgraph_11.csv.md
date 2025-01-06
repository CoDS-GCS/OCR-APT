# Attack Report: tc3_anomalous_subgraph_11.csv

## Summary of Attack Behavior

The logs from the document indicate a series of suspicious activities primarily involving the `sshd` process and various PAM (Pluggable Authentication Module) libraries. The events span multiple timestamps on April 11, 2018, suggesting a coordinated effort to manipulate authentication mechanisms and access sensitive user data.

Key events include:
- Multiple openings and closings of PAM-related files (`pam_nologin.so.6`, `pam_login_access.so.6`, etc.) and libraries (`libypclnt.so.4`, `libopie.so.8`), indicating potential attempts to modify authentication processes.
- Frequent access to user login files (`utx.lastlogin`, `utx.active`) and audit files (`audit_user`, `audit_control`, `audit_class`), which may suggest internal reconnaissance to gather user information and system configurations.
- The presence of the `sshd` process indicates that remote access was likely being exploited, which aligns with the **Initial Compromise** and **Internal Reconnaissance** stages of an APT attack.

Overall, the behavior observed in the logs raises significant concerns regarding unauthorized access and potential exploitation of the system's authentication mechanisms.

## Table of IoCs Detected

| IoC                     | Security Context                                                                                     |
|-------------------------|-----------------------------------------------------------------------------------------------------|
| `sshd`                  | Legitimate SSH daemon for remote access; high likelihood of exploitation if misconfigured.         |
| `pam_unix.so.6`        | Standard PAM module for Unix authentication; potential for exploitation if modified.              |
| `pam_permit.so.6`      | PAM module that allows all authentication; can be exploited to bypass security checks.             |
| `pam_opieaccess.so.6`  | PAM module for OPIE authentication; legitimate but can be misused for unauthorized access.        |
| `pam_opie.so.6`        | PAM module for OPIE; legitimate usage but can be exploited if not properly secured.                |
| `pam_nologin.so.6`     | Prevents non-root users from logging in; can be exploited to restrict access.                      |
| `pam_login_access.so.6`| Controls access based on user login; potential for exploitation if misconfigured.                  |
| `libypclnt.so.4`       | Library for YP (Yellow Pages) client; legitimate but can be exploited for unauthorized access.     |
| `libopie.so.8`         | Library for OPIE; legitimate usage but can be exploited if vulnerabilities exist.                  |
| `hosts`                 | Standard hosts file; legitimate but can be manipulated for DNS spoofing.                          |
| `utx.lastlogin`         | Tracks last login times; legitimate but can be targeted for user information gathering.            |
| `audit_user`            | Contains user audit information; legitimate but critical for monitoring unauthorized access.       |
| `audit_control`         | Controls audit settings; legitimate but can be exploited to disable auditing.                      |
| `audit_class`          | Defines audit classes; legitimate but can be manipulated to evade detection.                       |
| `utx.active`           | Tracks active user sessions; legitimate but can be exploited for session hijacking.                |

## Chronological Log of Actions

### April 11, 2018

- **16:38**
  - OPEN: `pam_nologin.so.6`
  - OPEN: `libypclnt.so.4`
  
- **16:43**
  - CLOSE: `hosts`
  - CLOSE: `audit_control`
  - CLOSE: `pam_login_access.so.6`
  - OPEN: `libypclnt.so.4`
  - OPEN: `libopie.so.8`
  - OPEN: `hosts`
  - OPEN: `audit_user`
  - OPEN: `audit_control`
  - OPEN: `audit_class`
  - LSEEK: `utx.lastlogin`
  - LSEEK: `utx.active`
  - LSEEK: `audit_user`
  - CLOSE: `libopie.so.8`
  - CLOSE: `utx.lastlogin`
  - CLOSE: `utx.active`
  - CLOSE: `sshd`
  - CLOSE: `pam_unix.so.6`
  - CLOSE: `pam_permit.so.6`
  - CLOSE: `pam_opieaccess.so.6`
  - CLOSE: `pam_opie.so.6`
  - CLOSE: `pam_nologin.so.6`
  - CLOSE: `pam_login_access.so.6`
  - CLOSE: `libypclnt.so.4`
  - LSEEK: `audit_class`
  
- **16:58**
  - OPEN: `pam_opie.so.6`
  - OPEN: `pam_nologin.so.6`
  - OPEN: `pam_login_access.so.6`
  - OPEN: `libypclnt.so.4`
  - OPEN: `libopie.so.8`
  - OPEN: `audit_user`
  - OPEN: `audit_control`
  - OPEN: `audit_class`
  - LSEEK: `utx.lastlogin`
  - LSEEK: `utx.active`
  - LSEEK: `audit_user`
  - LSEEK: `audit_class`
  - CLOSE: `utx.lastlogin`
  - CLOSE: `utx.active`
  - CLOSE: `sshd`
  - OPEN: `hosts`
  - OPEN: `utx.lastlogin`
  
- **17:09**
  - CLOSE: `utx.active` (2 times)
  - OPEN: `utx.active` (2 times)
  - LSEEK: `utx.active` (2 times)
  - OPEN: `hosts`
  - OPEN: `audit_user`
  - OPEN: `audit_control`
  - OPEN: `audit_class`
  - LSEEK: `utx.lastlogin`
  - LSEEK: `audit_user`
  - LSEEK: `audit_class`
  - CLOSE: `utx.lastlogin`
  - CLOSE: `sshd`
  - CLOSE: `pam_permit.so.6`
  - CLOSE: `pam_opieaccess.so.6`
  - CLOSE: `pam_opie.so.6`
  - CLOSE: `pam_nologin.so.6`
  - CLOSE: `pam_login_access.so.6`
  - CLOSE: `libypclnt.so.4`
  - CLOSE: `libopie.so.8`
  - CLOSE: `hosts`
  - CLOSE: `audit_control`
  
- **19:47**
  - OPEN: `pam_login_access.so.6`
  - OPEN: `libypclnt.so.4`
  - OPEN: `libopie.so.8`
  - OPEN: `utx.lastlogin`
  - OPEN: `hosts`
  - OPEN: `audit_user`
  - OPEN: `audit_control`
  - OPEN: `audit_class`
  - LSEEK: `utx.lastlogin`
  - LSEEK: `utx.active`
  - LSEEK: `audit_user`
  - LSEEK: `audit_class`
  - CLOSE: `utx.lastlogin`
  - CLOSE: `utx.active`
  - CLOSE: `sshd`
  - CLOSE: `pam_permit.so.6`
  - CLOSE: `pam_opieaccess.so.6`
  - CLOSE: `pam_opie.so.6`
  - CLOSE: `pam_nologin.so.6`
  - CLOSE: `pam_login_access.so.6`
  - CLOSE: `libypclnt.so.4`
  - CLOSE: `libopie.so.8`
  - CLOSE: `hosts`
  - CLOSE: `audit_control`
  
- **19:49**
  - OPEN: `utx.active`
  - LSEEK: `utx.active`
  - CLOSE: `utx.active`
  
- **20:03**
  - OPEN: `utx.active` (2 times)
  - LSEEK: `utx.active` (2 times)
  - CLOSE: `utx.active` (2 times)
  - OPEN: `utx.lastlogin`
  - OPEN: `sshd`
  - LSEEK: `audit_class`
  - CLOSE: `pam_login_access.so.6`
  - CLOSE: `libypclnt.so.4`
  - CLOSE: `libopie.so.8`
  - CLOSE: `hosts`
  
This report highlights the suspicious activities and potential exploitation of the system's authentication mechanisms, warranting further investigation and remediation actions.