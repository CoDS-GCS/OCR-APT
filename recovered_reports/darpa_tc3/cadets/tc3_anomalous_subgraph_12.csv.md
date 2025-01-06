# Attack Report: tc3_anomalous_subgraph_12.csv

## Summary of Attack Behavior

The logs from the document indicate a series of suspicious activities primarily involving the SSH daemon (`sshd`) and various libraries associated with authentication and encryption. The events span multiple timestamps on April 12, 2018, suggesting a potential ongoing attack or exploitation attempt. 

Key events include:

- **Execution of `sshd`**: The SSH daemon was executed multiple times, which is a common method for establishing remote access during an attack. This indicates a potential Command and Control (C2) stage of the APT lifecycle.
- **Frequent opening and closing of sensitive files**: The logs show numerous instances of opening and closing files related to SSH keys and various libraries, which could indicate reconnaissance or preparation for further exploitation.
- **Modification of process directories**: The logs indicate modifications to process directories, which may suggest attempts to manipulate or hide malicious activities.

Overall, the behavior observed aligns with the **Command and Control** and **Internal Reconnaissance** stages of an APT attack, where the attacker seeks to establish a foothold and gather information about the system.

## Table of IoCs Detected

| IoC                          | Security Context                                                                                     |
|------------------------------|-----------------------------------------------------------------------------------------------------|
| UTC                          | A standard time format; legitimate usage, low exploitation likelihood.                             |
| hosts.allow                  | Configuration file for controlling access; legitimate usage, but can be exploited for access control. |
| libasn1.so.11                | Library for ASN.1 encoding; legitimate usage, moderate exploitation likelihood if manipulated.      |
| libblacklist.so.0            | Library for managing blacklists; legitimate usage, moderate exploitation likelihood.                |
| libbsm.so.3                  | Library for Basic Security Module; legitimate usage, moderate exploitation likelihood.              |
| libhx509.so.11               | Library for X.509 certificates; legitimate usage, moderate exploitation likelihood.                 |
| libkrb5.so.11                | Library for Kerberos authentication; legitimate usage, moderate exploitation likelihood.            |
| libgssapi.so.10              | Library for GSSAPI authentication; legitimate usage, moderate exploitation likelihood.              |
| libgssapi_krb5.so.10         | Kerberos GSSAPI library; legitimate usage, moderate exploitation likelihood.                        |
| libgssapi_spnego.so.10       | Library for SPNEGO authentication; legitimate usage, moderate exploitation likelihood.             |
| libheimbase.so.11            | Library for Heimdal Kerberos; legitimate usage, moderate exploitation likelihood.                  |
| libpam.so.6                  | Pluggable Authentication Module; legitimate usage, high exploitation likelihood if compromised.    |
| libprivateldns.so.5          | Library for private DNS; legitimate usage, moderate exploitation likelihood.                       |
| libprivatessh.so.5           | Library for private SSH; legitimate usage, moderate exploitation likelihood.                       |
| libthr.so.3                  | Thread library; legitimate usage, low exploitation likelihood.                                     |
| libwind.so.11                | Library for Wind Kerberos; legitimate usage, moderate exploitation likelihood.                     |
| libwrap.so.6                 | Library for TCP wrappers; legitimate usage, moderate exploitation likelihood.                      |
| login.conf                   | Configuration file for login settings; legitimate usage, moderate exploitation likelihood.         |
| mech                          | Mechanism configuration; legitimate usage, moderate exploitation likelihood.                       |
| openssl.cnf                  | Configuration file for OpenSSL; legitimate usage, moderate exploitation likelihood.                |
| posixrules                   | Configuration for POSIX compliance; legitimate usage, low exploitation likelihood.                 |
| resolv.conf                  | DNS resolver configuration; legitimate usage, low exploitation likelihood.                         |
| ssh_host_ed25519_key         | SSH key for authentication; legitimate usage, high exploitation likelihood if compromised.         |
| ssh_host_ed25519_key.pub     | Public SSH key; legitimate usage, high exploitation likelihood if compromised.                     |
| ssh_host_rsa_key             | RSA SSH key for authentication; legitimate usage, high exploitation likelihood if compromised.     |
| ssh_host_rsa_key.pub         | Public RSA SSH key; legitimate usage, high exploitation likelihood if compromised.                 |
| ssh_host_ecdsa_key.pub       | Public ECDSA SSH key; legitimate usage, high exploitation likelihood if compromised.               |

## Chronological Log of Actions

### April 12, 2018

- **10:07**
  - CLOSE the file: ssh_host_rsa_key
  - CLOSE the file: ssh_host_rsa_key.pub
  - EXECUTE the file: sshd
  - MODIFY_PROCESS a fileDir
  - CLOSE the file: posixrules
  - OPEN the file: libwind.so.11
  - OPEN the file: libwrap.so.6
  - OPEN the file: login.conf
  - OPEN the file: mech
  - OPEN the file: nsswitch.conf
  - OPEN the file: openssl.cnf
  - OPEN the file: posixrules
  - OPEN the file: resolv.conf
  - OPEN the file: ssh_host_ed25519_key
  - OPEN the file: ssh_host_ed25519_key.pub
  - OPEN the file: ssh_host_rsa_key
  - OPEN the file: ssh_host_rsa_key.pub
  - OPEN the file: ssh_host_ecdsa_key.pub
  - CLOSE the file: libprivatessh.so.5
  - CLOSE a file
  - CLOSE the file: UTC
  - CLOSE the file: hosts.allow
  - CLOSE the file: libasn1.so.11
  - CLOSE the file: libblacklist.so.0
  - CLOSE the file: libbsm.so.3
  - CLOSE the file: libcom_err.so.5
  - CLOSE the file: libgssapi.so.10
  - CLOSE the file: libgssapi_krb5.so.10
  - CLOSE the file: libgssapi_spnego.so.10
  - CLOSE the file: libheimbase.so.11
  - CLOSE the file: libhx509.so.11
  - CLOSE the file: libkrb5.so.11
  - CLOSE the file: libpam.so.6
  - CLOSE the file: libprivateldns.so.5
  - CLOSE the file: libcrypt.so.5

- **13:06**
  - EXECUTE the file: sshd
  - MODIFY_PROCESS a fileDir
  - OPEN the file: UTC
  - OPEN the file: hosts.allow
  - OPEN the file: libasn1.so.11
  - OPEN the file: libblacklist.so.0
  - OPEN the file: libbsm.so.3
  - OPEN the file: libcom_err.so.5
  - OPEN the file: libgssapi.so.10
  - OPEN the file: libgssapi_krb5.so.10
  - OPEN the file: libgssapi_spnego.so.10
  - OPEN the file: libheimbase.so.11
  - OPEN the file: libhx509.so.11
  - OPEN the file: libkrb5.so.11
  - OPEN the file: libpam.so.6
  - OPEN the file: libprivateldns.so.5
  - OPEN the file: libprivatessh.so.5
  - OPEN the file: libthr.so.3
  - OPEN the file: libwind.so.11
  - OPEN the file: libwrap.so.6
  - OPEN the file: login.conf
  - OPEN the file: mech
  - OPEN the file: nsswitch.conf
  - OPEN the file: openssl.cnf
  - OPEN the file: posixrules
  - OPEN the file: resolv.conf
  - OPEN the file: ssh_host_ecdsa_key.pub
  - OPEN the file: ssh_host_ed25519_key
  - OPEN the file: ssh_host_ed25519_key.pub
  - CLOSE the file: libcrypt.so.5
  - CLOSE the file: ssh_host_rsa_key
  - CLOSE the file: ssh_host_rsa_key.pub

- **21:46**
  - CLOSE the file: ssh_host_rsa_key
  - CLOSE the file: ssh_host_rsa_key.pub
  - OPEN the file: libgssapi.so.10
  - MODIFY_PROCESS a fileDir
  - OPEN the file: UTC
  - OPEN the file: hosts.allow
  - OPEN the file: libasn1.so.11
  - OPEN the file: libblacklist.so.0
  - OPEN the file: libbsm.so.3
  - OPEN the file: libcom_err.so.5
  - OPEN the file: libcrypt.so.5
  - EXECUTE the file: sshd
  - OPEN the file: ssh_host_rsa_key.pub

- **22:10**
  - OPEN the file: libhx509.so.11
  - OPEN the file: libkrb5.so.11
  - OPEN the file: libpam.so.6
  - OPEN the file: libprivateldns.so.5
  - OPEN the file: libprivatessh.so.5
  - OPEN the file: libthr.so.3
  - OPEN the file: libwind.so.11
  - OPEN the file: login.conf
  - OPEN the file: mech
  - OPEN the file: nsswitch.conf
  - OPEN the file: openssl.cnf
  - OPEN the file: posixrules
  - OPEN the file: resolv.conf
  - OPEN the file: ssh_host_ecdsa_key.pub
  - OPEN the file: ssh_host_ed25519_key
  - OPEN the file: libwrap.so.6
  - EXECUTE the file: sshd
  - MODIFY_PROCESS a fileDir
  - OPEN the file: UTC
  - OPEN the file: libheimbase.so.11
  - OPEN the file: libasn1.so.11
  - OPEN the file: libblacklist.so.0
  - OPEN the file: libbsm.so.3
  - CLOSE the file: libcom_err.so.5
  - OPEN the file: libcrypt.so.5
  - OPEN the file: libgssapi.so.10

This report highlights the suspicious activities observed in the logs, indicating potential malicious behavior that warrants further investigation.