# Attack Report: Context of svchost.exe Activity

## Summary of Attack Behavior

The logs from the `context_svchost` document indicate a series of suspicious activities that suggest a potential Advanced Persistent Threat (APT) operation. The timeline of events reveals multiple stages of the attack, including:

1. **Initial Compromise**: The presence of external IP addresses such as `142.20.56.202`, `10.50.5.10`, `132.197.158.98`, `10.20.0.2`, and `10.20.4.133` indicates possible command and control (C2) communications. The frequent inbound and outbound messages suggest that the compromised system is communicating with external entities.

2. **Internal Reconnaissance**: The logs show extensive reading of various Python bytecode files and egg packages, which may indicate an attempt to gather information about the environment and installed software.

3. **Maintain Persistence**: The creation and subsequent deletion of the file `usernames.csv` suggest an attempt to maintain persistence on the system, potentially for credential harvesting.

4. **Command and Control**: The outbound messages to the identified IP addresses indicate that the attacker is likely maintaining control over the compromised system.

The logs are timestamped primarily on September 23, 2019, with significant activity noted at 12:41, 15:11, 13:11, and 10:53.

## Table of IoCs Detected

| IoC                             | Security Context                                                                                     |
|---------------------------------|-----------------------------------------------------------------------------------------------------|
| 142.20.56.202                   | External IP address; potential C2 server.                                                          |
| 10.50.5.10                      | Internal IP address; frequent outbound communication suggests possible data exfiltration.          |
| 132.197.158.98                  | External IP address; potential C2 server.                                                          |
| 10.20.0.2                       | Internal IP address; involved in multiple outbound communications.                                  |
| 10.20.4.133                     | Internal IP address; involved in multiple inbound communications.                                   |
| __init__.pyc                    | Python bytecode file; indicates potential reconnaissance of Python applications.                   |
| setuptools-0.7.2-py2.7.egg     | Python package; may indicate the presence of a Python environment used for malicious purposes.      |
| idna-2.5-py2.7.egg             | Python package; may indicate the presence of a Python environment used for malicious purposes.      |
| enum34-1.1.6-py2.7.egg         | Python package; may indicate the presence of a Python environment used for malicious purposes.      |
| beautifulsoup4-4.6.0-py2.7.egg | Python package; may indicate the presence of a Python environment used for malicious purposes.      |
| chardet-3.0.4-py2.7.egg       | Python package; may indicate the presence of a Python environment used for malicious purposes.      |
| base64.pyc                      | Python bytecode file; indicates potential reconnaissance of Python applications.                   |
| base_module.pyc                 | Python bytecode file; indicates potential reconnaissance of Python applications.                   |
| blockfeeder.pyc                 | Python bytecode file; indicates potential reconnaissance of Python applications.                   |
| codecs.pyc                      | Python bytecode file; indicates potential reconnaissance of Python applications.                   |
| collections.pyc                 | Python bytecode file; indicates potential reconnaissance of Python applications.                   |
| contextlib.pyc                  | Python bytecode file; indicates potential reconnaissance of Python applications.                   |
| copy.pyc                        | Python bytecode file; indicates potential reconnaissance of Python applications.                   |
| copy_reg.pyc                    | Python bytecode file; indicates potential reconnaissance of Python applications.                   |
| _endian.pyc                     | Python bytecode file; indicates potential reconnaissance of Python applications.                   |
| _weakrefset.pyc                 | Python bytecode file; indicates potential reconnaissance of Python applications.                   |
| abc.pyc                         | Python bytecode file; indicates potential reconnaissance of Python applications.                   |
| aes.pyc                         | Python bytecode file; indicates potential reconnaissance of Python applications.                   |
| aliases.pyc                     | Python bytecode file; indicates potential reconnaissance of Python applications.                   |
| atexit.pyc                      | Python bytecode file; indicates potential reconnaissance of Python applications.                   |
| autologin.pyc                   | Python bytecode file; indicates potential reconnaissance of Python applications.                   |
| autologin_windows_worker.pyc    | Python bytecode file; indicates potential reconnaissance of Python applications.                   |
| ConfigParser.pyc                | Python bytecode file; indicates potential reconnaissance of Python applications.                   |
| SysClient0201_kickoff.json      | JSON file; may contain configuration or data related to the attack.                               |
| UserDict.pyc                    | Python bytecode file; indicates potential reconnaissance of Python applications.                   |
| __future__.pyc                  | Python bytecode file; indicates potential reconnaissance of Python applications.                   |
| _abcoll.pyc                     | Python bytecode file; indicates potential reconnaissance of Python applications.                   |
| usernames.csv                   | File created and deleted; indicates potential credential harvesting.                                 |

## Chronological Log of Actions

### 10:53
- START_OUTBOUND the flow: 142.20.61.131 (2 times)
- START_OUTBOUND the flow: 142.20.61.132 (2 times)
- WRITE the file: SEARCHPROTOCOLHOST.EXE-AFAD3EF9.pf
- READ the file: python27.dll
- READ the file: CONTAB32.DLL
- READ the file: Redemption.dll
- READ the file: Wldap32.dll
- READ the file: activeds.dll.mui
- READ the file: activeds.tlb
- READ the file: adsldp.dll
- READ the file: adsmsext.dll
- READ the file: desktop.ini
- CREATE the file: SEARCHFILTERHOST.EXE-AA7A1FDD.pf
- CREATE the file: SEARCHPROTOCOLHOST.EXE-AFAD3EF9.pf
- READ the file: $UsnJrnl:$J
- READ the file: mantra_corpus.txt
- READ the file: ntdsapi.dll

### 10:54
- WRITE the file: agent.log (12 times)
- MESSAGE_OUTBOUND the flow: 10.50.5.10 (12 times)
- CREATE the file: .winlogbeat.yml.new (10 times)
- RENAME the file: .winlogbeat.yml.new (10 times)
- MODIFY the file: HarddiskVolume1 (4 times)
- WRITE the file: WebCacheV01.dat (4 times)
- WRITE the file: .winlogbeat.yml (4 times)
- READ the file: APPXDEPLOYMENTSERVER.DLL (3 times)
- MODIFY the file: LWABEAT.EXE (3 times)
- READ the file: $UsnJrnl:$J (3 times)
- MODIFY the file: WIN32KFULL.SYS (2 times)
- MODIFY the file: UCRTBASE.DLL (2 times)
- MODIFY the file: URLMON.DLL (2 times)
- MODIFY the file: USER32.DLL (2 times)
- MODIFY the file: SYSMAIN.DLL (2 times)

### 12:41
- WRITE the file: agent.log (12 times)
- MESSAGE_INBOUND the flow: 142.20.56.202 (11 times)
- MESSAGE_OUTBOUND the flow: 10.50.5.10 (11 times)
- MESSAGE_OUTBOUND the flow: 132.197.158.98 (11 times)
- WRITE the file: $Logfile (8 times)
- READ the file: __init__.pyc (4 times)
- WRITE the file: kickoff.log (3 times)
- START_OUTBOUND the flow: 10.20.0.2 (2 times)
- READ the file: $UsnJrnl:$J (2 times)
- MESSAGE_OUTBOUND the flow: 10.20.0.2 (2 times)
- READ the file: setuptools-0.7.2-py2.7.egg (2 times)
- READ the file: idna-2.5-py2.7.egg (2 times)
- READ the file: enum34-1.1.6-py2.7.egg (2 times)
- READ the file: chardet-3.0.4-py2.7.egg (2 times)
- READ the file: beautifulsoup4-4.6.0-py2.7.egg (2 times)
- MESSAGE_INBOUND the flow: 10.20.4.133 (2 times)
- READ the file: _weakrefset.pyc
- READ the file: abc.pyc
- READ the file: aes.pyc
- READ the file: aliases.pyc
- READ the file: atexit.pyc
- READ the file: autologin.pyc
- READ the file: autologin_windows_worker.pyc
- READ the file: base64.pyc
- CREATE the file: usernames.csv
- DELETE the file: usernames.csv
- READ the file: ConfigParser.pyc
- READ the file: SysClient0201_kickoff.json
- READ the file: UserDict.pyc
- READ the file: __future__.pyc
- READ the file: _abcoll.pyc

### 13:11
- READ the file: beautifulsoup4-4.6.0-py2.7.egg (2 times)
- MESSAGE_OUTBOUND the flow: 10.20.0.2 (2 times)
- MESSAGE_INBOUND the flow: 10.20.4.133 (2 times)
- READ the file: ConfigParser.pyc
- READ the file: SysClient0201_kickoff.json
- READ the file: UserDict.pyc
- READ the file: __future__.pyc
- READ the file: _abcoll.pyc
- READ the file: _endian.pyc
- READ the file: _weakrefset.pyc
- READ the file: abc.pyc
- CREATE the file: SVCHOST.EXE-2562231B.pf
- CREATE the file: usernames.csv
- DELETE the file: usernames.csv
- READ the file: $UsnJrnl:$J
- READ the file: 969252ce11249fdd.customDestinations-ms
- READ the file: file_handler.pyc
- READ the file: file_transport.pyc
- READ the file: fnmatch.pyc
- READ the file: functools.pyc
- READ the file: genericpath.pyc
- READ the file: gettext.pyc
- READ the file: glob.pyc
- READ the file: guestinfo_interface.pyc
- READ the file: datastore.pyc
- READ the file: debug.pyc
- READ the file: decoder.pyc
- READ the file: dep_util.pyc
- READ the file: easy-install.pth
- READ the file: encoder.pyc
- READ the file: errors.pyc
- READ the file: esxi_module_loader.pyc
- READ the file: codecs.pyc
- READ the file: collections.pyc
- READ the file: contextlib.pyc

### 15:11
- READ the file: __init__.pyc (3 times)
- START_OUTBOUND the flow: 10.20.0.2 (2 times)
- READ the file: setuptools-0.7.2-py2.7.egg (2 times)
- READ the file: idna-2.5-py2.7.egg (2 times)
- READ the file: enum34-1.1.6-py2.7.egg (2 times)
- READ the file: beautifulsoup4-4.6.0-py2.7.egg (2 times)
- READ the file: chardet-3.0.4-py2.7.egg (2 times)
- MESSAGE_INBOUND the flow: 10.20.4.133 (2 times)
- MESSAGE_OUTBOUND the flow: 10.20.0.2 (2 times)
- READ the file: base64.pyc
- READ the file: base_module.pyc
- READ the file: blockfeeder.pyc
- READ the file: codecs.pyc
- READ the file: collections.pyc
- READ the file: contextlib.pyc
- READ the file: copy.pyc
- READ the file: copy_reg.pyc
- READ the file: _endian.pyc
- READ the file: _weakrefset.pyc
- READ the file: abc.pyc
- READ the file: aes.pyc
- READ the file: aliases.pyc
- READ the file: atexit.pyc
- READ the file: autologin.pyc
- READ the file: autologin_windows_worker.pyc
- CREATE the file: usernames.csv
- DELETE the file: usernames.csv
- READ the file: $UsnJrnl:$J
- READ the file: ConfigParser.pyc
- READ the file: SysClient0201_kickoff.json
- READ the file: UserDict.pyc
- READ the file: __future__.pyc
- READ the file: _abcoll.pyc

This report summarizes the suspicious activities associated with the `svchost.exe` process, highlighting potential indicators of compromise and the stages of the APT attack. Further investigation is recommended to assess the impact and mitigate any risks associated with these findings.