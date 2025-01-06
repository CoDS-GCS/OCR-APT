# Attack Report: tc3_anomalous_subgraph_12.csv

## Summary of Attack Behavior

The logs indicate a series of network connection and disconnection events involving multiple IP addresses on April 13, 2018. The activity appears to be part of a coordinated operation, likely indicative of an APT attack. 

- **Initial Compromise**: The logs show multiple connections initiated at 14:21, suggesting the establishment of a foothold within the network.
- **Lateral Movement**: The repeated connections and closures of various flows, particularly at 14:23, indicate potential lateral movement within the network as the attacker navigates between different systems.
- **Maintain Persistence**: The connections made at 14:26, shortly after several closures, suggest attempts to maintain access to the compromised systems.
- **Covering Tracks**: The exit of the process at 14:28 may indicate an effort to cover tracks after the operations were completed.

## Table of IoCs Detected

| IoC               | Security Context                                                                                     |
|-------------------|-----------------------------------------------------------------------------------------------------|
| 128.55.12.73      | Potentially a compromised host; legitimate usage may include internal services, but exploitation likelihood is moderate. |
| 128.55.12.55      | Another internal IP; could be a legitimate service or a target for lateral movement, with moderate exploitation risk. |
| 128.55.12.67      | Similar to above, this IP shows multiple connection attempts, indicating potential exploitation.    |
| 128.55.12.166     | This IP was involved in both connection and closure events, suggesting it may be a target or a compromised host. |
| 128.55.12.1       | Active connections and closures indicate it may be a critical system; exploitation likelihood is high. |
| 128.55.12.103     | Involved in multiple connection events, suggesting it may be a target for lateral movement.         |
| 128.55.12.110     | Similar to 128.55.12.103, this IP shows signs of being targeted; exploitation likelihood is moderate. |
| 128.55.12.141     | Active in the logs, indicating potential compromise; legitimate usage may include internal applications. |

## Chronological Log of Actions

### 14:21
- **CONNECT**: 128.55.12.55
- **CONNECT**: 128.55.12.73
- **CLOSE**: 128.55.12.73
- **CLOSE**: 128.55.12.55

### 14:22
- **CLOSE**: 128.55.12.67 (2 times)
- **CONNECT**: 128.55.12.67 (2 times)
- **CLOSE**: 128.55.12.166
- **CONNECT**: 128.55.12.166

### 14:23
- **CLOSE**: 128.55.12.1
- **CLOSE**: 128.55.12.103
- **CLOSE**: 128.55.12.110
- **CLOSE**: 128.55.12.141
- **CLOSE**: 128.55.12.55
- **CLOSE**: 128.55.12.67
- **CONNECT**: 128.55.12.1
- **CONNECT**: 128.55.12.103
- **CONNECT**: 128.55.12.110
- **CONNECT**: 128.55.12.141
- **CONNECT**: 128.55.12.55
- **CONNECT**: 128.55.12.67

### 14:26
- **CLOSE**: 128.55.12.1
- **CONNECT**: 128.55.12.1

### 14:28
- **EXIT**: tcexec