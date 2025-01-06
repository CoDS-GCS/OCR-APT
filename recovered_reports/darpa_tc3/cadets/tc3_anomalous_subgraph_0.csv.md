# Attack Report: tc3_anomalous_subgraph_0.csv

## Summary of Attack Behavior

The logs from the document indicate a series of anomalous activities primarily involving the repeated reading of the file `devctl` and sending data to a socket. The events span from April 11, 2018, to April 12, 2018, with a high frequency of interactions with the `devctl` file, suggesting potential reconnaissance or exploitation activities. 

Key events include:
- Multiple instances of reading the `devctl` file, which occurred frequently, indicating a possible attempt to gather sensitive information or manipulate system controls.
- A consistent pattern of sending data to a socket, which may suggest attempts to establish a command and control (C2) channel or exfiltrate data.

These behaviors align with the **Internal Reconnaissance** and **Command and Control** stages of an APT attack, where the attacker seeks to understand the environment and maintain communication with compromised systems.

## Table of IoCs Detected

| IoC      | Security Context                                                                                     |
|----------|------------------------------------------------------------------------------------------------------|
| devctl   | `devctl` is a legitimate system file used for device control in operating systems. However, its frequent access and manipulation can indicate exploitation attempts, especially if it is being used to gather sensitive information or control system behavior. |

## Chronological Log of Actions

### April 11, 2018
- **19:47** - A process SENDTO a socket.
- **19:49** - A process READ the file: devctl.
- **19:49** - A process SENDTO a socket.
- **20:03** - A process READ the file: devctl (2 times).
- **20:03** - A process SENDTO a socket (2 times).
- **20:14** - A process READ the file: devctl.
- **20:14** - A process SENDTO a socket.
- **20:15** - A process READ the file: devctl.
- **20:15** - A process SENDTO a socket.
- **20:19** - A process READ the file: devctl (2 times).
- **20:19** - A process SENDTO a socket (2 times).
- **20:32** - A process READ the file: devctl (2 times).
- **20:32** - A process SENDTO a socket (2 times).
- **20:38** - A process READ the file: devctl.
- **20:38** - A process SENDTO a socket.
- **20:52** - A process READ the file: devctl.
- **20:52** - A process SENDTO a socket.
- **20:53** - A process READ the file: devctl.
- **20:53** - A process SENDTO a socket.
- **21:03** - A process READ the file: devctl (2 times).
- **21:03** - A process SENDTO a socket (2 times).
- **21:12** - A process READ the file: devctl (2 times).
- **21:12** - A process SENDTO a socket (2 times).
- **21:27** - A process READ the file: devctl.
- **21:27** - A process SENDTO a socket.
- **21:29** - A process READ the file: devctl.
- **21:29** - A process SENDTO a socket.
- **21:38** - A process READ the file: devctl (2 times).
- **21:38** - A process SENDTO a socket (2 times).
- **21:49** - A process READ the file: devctl.
- **21:49** - A process SENDTO a socket.
- **21:51** - A process READ the file: devctl.
- **21:51** - A process SENDTO a socket.
- **22:03** - A process READ the file: devctl.
- **22:03** - A process SENDTO a socket.
- **22:05** - A process READ the file: devctl.
- **22:05** - A process SENDTO a socket.
- **22:15** - A process READ the file: devctl.
- **22:15** - A process SENDTO a socket.
- **22:29** - A process READ the file: devctl.
- **22:29** - A process SENDTO a socket.
- **22:39** - A process READ the file: devctl (2 times).
- **22:39** - A process SENDTO a socket (2 times).

### April 12, 2018
- **10:41** - A process SENDTO a socket.
- **10:43** - A process READ the file: devctl.
- **10:43** - A process SENDTO a socket.
- **10:45** - A process READ the file: devctl.
- **10:45** - A process SENDTO a socket.
- **10:48** - A process READ the file: devctl.
- **10:48** - A process SENDTO a socket.
- **10:55** - A process READ the file: devctl.
- **10:55** - A process SENDTO a socket.
- **11:07** - A process READ the file: devctl.
- **11:07** - A process SENDTO a socket.
- **11:09** - A process READ the file: devctl.
- **11:09** - A process SENDTO a socket.
- **11:11** - A process READ the file: devctl.
- **11:11** - A process SENDTO a socket.
- **11:17** - A process READ the file: devctl.
- **11:17** - A process SENDTO a socket.
- **11:18** - A process READ the file: devctl.
- **11:18** - A process SENDTO a socket.
- **11:19** - A process READ the file: devctl.
- **11:19** - A process SENDTO a socket.
- **11:24** - A process READ the file: devctl (2 times).
- **11:24** - A process SENDTO a socket (2 times).
- **11:32** - A process READ the file: devctl (2 times).
- **11:32** - A process SENDTO a socket (2 times).
- **11:39** - A process READ the file: devctl (2 times).
- **11:39** - A process SENDTO a socket (2 times).
- **11:40** - A process READ the file: devctl (2 times).
- **11:40** - A process SENDTO a socket (2 times).
- **11:45** - A process READ the file: devctl.
- **11:45** - A process SENDTO a socket.
- **11:46** - A process READ the file: devctl (3 times).
- **11:46** - A process SENDTO a socket (3 times).
- **11:47** - A process READ the file: devctl.
- **11:47** - A process SENDTO a socket.
- **11:52** - A process READ the file: devctl.
- **11:52** - A process SENDTO a socket.
- **11:54** - A process READ the file: devctl.
- **11:54** - A process SENDTO a socket.
- **11:57** - A process READ the file: devctl (2 times).
- **11:57** - A process SENDTO a socket (2 times).
- **12:01** - A process READ the file: devctl (2 times).
- **12:01** - A process SENDTO a socket (2 times).