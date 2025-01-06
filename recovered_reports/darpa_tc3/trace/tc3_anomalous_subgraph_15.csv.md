# Attack Report: tc3_anomalous_subgraph_15.csv

## Summary of Attack Behavior

The logs from the document indicate a series of actions performed by the `thunderbird` process on April 13, 2018, which suggest potential malicious activity. The sequence of events includes multiple reads and writes to pipes, memory mapping, and the use of message passing, which are indicative of a process attempting to communicate and manipulate data in a potentially unauthorized manner.

Key events include:
- **Initial Compromise**: The process begins with reading and writing to pipes and files, suggesting an initial setup phase.
- **Internal Reconnaissance**: The process opens and reads files such as `maps` and `libnotify.so.4`, indicating an attempt to gather information about the system.
- **Command and Control**: The frequent use of `RECVMSG` and `SENDMSG` indicates potential communication with a command and control server.
- **Privilege Escalation**: The use of `MMAP` and `MPROTECT` suggests attempts to manipulate memory, which could be indicative of privilege escalation tactics.
- **Maintain Persistence**: The cloning of the `thunderbird` process and the unlinking of file links suggest efforts to maintain persistence on the system.

The logs culminate in the exit of the `thunderbird` process, which may indicate the completion of the attack or a stealthy exit to avoid detection.

## Table of IoCs Detected

| IoC                     | Security Context                                                                                     |
|-------------------------|-----------------------------------------------------------------------------------------------------|
| 127.0.1.1:+26878       | This IP address appears to be a local address with a port, potentially used for internal communication. While it may be legitimate, its usage in this context raises suspicion of internal exploitation. |
| prefs.js                | A configuration file commonly used by applications. Its renaming suggests potential tampering or evasion tactics, indicating a moderate likelihood of exploitation. |
| libnotify.so.4         | A shared library used for notifications in Linux systems. Its access and manipulation could indicate an attempt to exploit system resources, suggesting a high likelihood of exploitation. |

## Chronological Log of Actions

### 14:13
- READ a pipe (15 times)
- WRITE a pipe (11 times)
- RECVMSG a srcsink (3 times)
- WRITE a srcsink (3 times)
- MMAP a memory (2 times)
- OPEN the file: maps
- MPROTECT a memory
- LINK the file: 127.0.1.1:+26878
- SENDMSG a srcsink
- READ the file: maps
- UNLINK a fileLink
- LINK a fileLink
- CLOSE the file: maps

### 14:14
- READ a pipe (25 times)
- WRITE a pipe (8 times)
- RENAME the file: prefs.js
- EXIT the process: thunderbird
- CLONE the process: thunderbird
- MPROTECT a memory
- MMAP a memory
- RECVMSG a srcsink
- READ a pipe (39 times)
- WRITE a pipe (19 times)
- RECVMSG a srcsink (14 times)
- WRITE a srcsink (5 times)
- MMAP a memory (3 times)

### 14:15
- READ the file: libnotify.so.4
- OPEN the file: libnotify.so.4
- MPROTECT a memory
- MMAP the file: libnotify.so.4
- CLOSE the file: libnotify.so.4

### 14:16
- READ a pipe (36 times)
- WRITE a pipe (4 times)
- RECVMSG a srcsink (4 times)
- WRITE a srcsink

### 14:17
- READ a pipe (30 times)
- RECVMSG a srcsink

### 14:18
- READ a pipe (29 times)
- RECVMSG a srcsink (2 times)
- WRITE a pipe

### 14:19
- READ a pipe (19 times)
- WRITE a pipe

### 14:20
- READ a pipe (18 times)

### 14:21
- READ a pipe (20 times)
- WRITE a pipe (2 times)

### 14:22
- READ a pipe (13 times)

### 14:23
- READ a pipe (15 times)

### 14:24
- READ a pipe (22 times)
- WRITE a pipe (6 times)

### 14:25
- READ a pipe (18 times)
- RECVMSG a srcsink (2 times)
- WRITE a pipe (2 times)
- WRITE a srcsink

### 14:26
- READ a pipe (20 times)
- RECVMSG a srcsink (2 times)
- WRITE a srcsink
- WRITE a pipe

### 14:27
- READ a pipe (15 times)
- MMAP a memory
- MPROTECT a memory
- WRITE a pipe

### 14:28
- READ a pipe (20 times)
- WRITE a pipe (2 times)

### 14:29
- READ a pipe (28 times)
- RECVMSG a srcsink (7 times)
- WRITE a pipe (5 times)
- WRITE a srcsink (3 times)

### 14:30
- READ a pipe (16 times)
- RECVMSG a srcsink (3 times)
- WRITE a srcsink (2 times)
- WRITE a pipe

### 14:31
- READ a pipe (22 times)
- WRITE a pipe

### 14:32
- READ a pipe (40 times)
- WRITE a pipe (36 times)
- RECVMSG a srcsink (22 times)
- WRITE a srcsink (20 times)
- MPROTECT a memory (7 times)
- MMAP a memory (2 times)

### 14:33
- CLONE the process: thunderbird
- EXIT the process: thunderbird
- UNLINK a fileLink