# Attack Report: Anomalous Activity Detected in optc_anomalous_subgraph_10.csv

## Summary of Attack Behavior

On September 25, 2019, at 13:45, the process `svchost.exe` exhibited suspicious behavior by reading multiple Python compiled files (`*.pyc`) and other files, indicating potential exploitation of Python scripts or libraries. The activity escalated at 13:46, where `svchost.exe` initiated outbound communication to the IP address `10.20.2.66`, which could signify a Command and Control (C2) communication attempt. The process also created threads and continued to read various files, suggesting internal reconnaissance and potential data exfiltration activities.

### APT Stages Identified:
- **Initial Compromise**: Reading of various Python files and libraries.
- **Internal Reconnaissance**: Extensive file reads, indicating exploration of the environment.
- **Command and Control**: Outbound communication to an external IP address.
- **Data Exfiltration**: Reading of files that may contain sensitive information.

## Table of Indicators of Compromise (IoCs)

| IoC                                         | Security Context                                                                                     |
|---------------------------------------------|-----------------------------------------------------------------------------------------------------|
| api.pyc                                     | Commonly used in API interactions; could be exploited for unauthorized access to services.         |
| alert.pyc                                   | May be used for alerting mechanisms; potential for misuse in logging sensitive information.         |
| adapters.pyc                                | Typically used for data handling; could be exploited to manipulate data flows.                     |
| actions                                      | A directory or file that may contain scripts; could be used for executing malicious actions.       |
| action_chains.pyc                           | Used in automation; could be exploited to execute unauthorized commands.                            |
| action_builder.pyc                          | Related to building actions; potential for misuse in creating harmful scripts.                     |
| _win32sysloader.pyd                        | A Windows-specific loader; could be exploited to load malicious modules.                           |
| _fontdata_widths_helvetica.pyc             | Font data handling; unlikely to be malicious but could be part of a larger exploit.                |
| _strptime.pyc                               | Date and time parsing; could be exploited for time-based attacks.                                  |
| _parseaddr.pyc                              | Email address parsing; could be exploited for phishing or spam activities.                         |
| _internal_utils.pyc                         | Internal utility functions; could be exploited to gain unauthorized access to system functions.     |
| _fontdata_widths_zapfdingbats.pyc         | Font data handling; unlikely to be malicious but could be part of a larger exploit.                |
| _fontdata_widths_timesitalic.pyc           | Font data handling; unlikely to be malicious but could be part of a larger exploit.                |
| _fontdata_widths_timesbolditalic.pyc       | Font data handling; unlikely to be malicious but could be part of a larger exploit.                |
| _fontdata_widths_timesbold.pyc             | Font data handling; unlikely to be malicious but could be part of a larger exploit.                |
| _fontdata_widths_symbol.pyc                 | Font data handling; unlikely to be malicious but could be part of a larger exploit.                |
| _fontdata_widths_helveticaoblique.pyc      | Font data handling; unlikely to be malicious but could be part of a larger exploit.                |
| _fontdata_widths_helveticabold.pyc         | Font data handling; unlikely to be malicious but could be part of a larger exploit.                |
| _version.pyc                                | Version control; could be exploited to identify vulnerable versions of software.                   |
| win32gui.pyd                                | Windows GUI interface; could be exploited for unauthorized GUI manipulations.                       |
| win32con.pyc                                | Windows constants; could be exploited for system-level attacks.                                    |
| win32comext                                 | Windows COM extensions; could be exploited for inter-process communication attacks.                 |
| win32api.pyd                                | Windows API interface; could be exploited for unauthorized system calls.                            |
| webelement.pyc                              | Related to web elements; could be exploited for web-based attacks.                                 |
| uuid.pyc                                    | Universally unique identifiers; could be exploited for tracking or identification purposes.        |
| util.pyc                                    | General utility functions; could be exploited for various malicious purposes.                       |
| NodeFilter.pyc                              | Related to DOM manipulation; could be exploited for web-based attacks.                             |
| win32com                                    | Windows COM interface; could be exploited for inter-process communication attacks.                 |
| CONF                                        | Configuration files; could contain sensitive information or settings for exploitation.              |
| MySQL_python-1.2.5-py2.7.egg-info         | Python MySQL library; could be exploited for database access.                                      |
| EGG-INFO                                    | Package metadata; could be exploited to identify vulnerable packages.                               |
| Cookie.pyc                                  | Cookie handling; could be exploited for session hijacking.                                         |
| winerror.pyc                                | Windows error handling; could be exploited for error-based attacks.                                |
| xmlbuilder.pyc                              | XML handling; could be exploited for XML injection attacks.                                        |
| wiredata.pyc                                | Data handling; could be exploited for unauthorized data manipulation.                               |
| hmac.pyc                                    | Hash-based message authentication; could be exploited for bypassing authentication.                |
| hash.pyc                                    | General hashing functions; could be exploited for hash collisions.                                 |
| handlers.pyc                                | Event handling; could be exploited for unauthorized event triggering.                               |
| handler.pyc                                 | General handler functions; could be exploited for various malicious purposes.                       |
| getAttribute.js                             | JavaScript function; could be exploited for web-based attacks.                                     |
| generic.pyc                                 | General-purpose functions; could be exploited for various malicious purposes.                       |
| gencache.pyc                                | Cache generation; could be exploited for performance issues or data manipulation.                   |
| formatter.pyc                               | Data formatting; could be exploited for data manipulation.                                         |
| fonts                                       | Font files; unlikely to be malicious but could be part of a larger exploit.                        |
| bcrypt-3.1.4.dist-info                      | Password hashing library; could be exploited for password cracking.                                 |
| firefox_binary.pyc                          | Firefox browser binary; could be exploited for browser-based attacks.                              |
| firefox                                     | Firefox browser; could be exploited for browser-based attacks.                                     |
| filters.pyc                                 | Data filtering; could be exploited for unauthorized data access.                                    |
| fileshare.csv                               | File sharing data; could contain sensitive information for exfiltration.                           |
| filepost.pyc                                | File posting; could be exploited for unauthorized file uploads.                                    |
| file_detector.pyc                           | File detection; could be exploited for unauthorized file access.                                    |
| fields.pyc                                  | Data fields; could be exploited for unauthorized data access.                                      |
| extension_connection.pyc                    | Extension handling; could be exploited for unauthorized access to browser extensions.               |
| expat.pyc                                   | XML parsing library; could be exploited for XML injection attacks.                                  |
| exception.pyc                               | Exception handling; could be exploited for error-based attacks.                                    |
| flags.pyc                                   | Flag handling; could be exploited for unauthorized access control.                                  |
| mantra.ini                                  | Configuration file; could contain sensitive information or settings for exploitation.               |
| logstash                                    | Log management tool; could be exploited for unauthorized log access.                                |
| logging.ini                                 | Logging configuration; could contain sensitive information or settings for exploitation.            |
| logger.pyc                                  | Logging functions; could be exploited for unauthorized log access.                                  |
| lib                                         | Library files; could be exploited for unauthorized access to library functions.                     |
| keys.pyc                                    | Key handling; could be exploited for unauthorized access to encrypted data.                         |
| key_input.pyc                               | Input handling; could be exploited for unauthorized input capture.                                  |
| isDisplayed.js                              | JavaScript function; could be exploited for web-based attacks.                                     |
| ipv6.pyc                                    | IPv6 handling; could be exploited for network-based attacks.                                       |
| ipv4.pyc                                    | IPv4 handling; could be exploited for network-based attacks.                                       |
| hooks.pyc                                   | Hook functions; could be exploited for unauthorized function interception.                           |
| ipaddress-1.0.22.dist-info                  | IP address handling library; could be exploited for network-based attacks.                          |
| interaction.pyc                             | User interaction handling; could be exploited for unauthorized access.                              |
| integer.pyc                                 | Integer handling; could be exploited for data manipulation.                                        |
| input_device.pyc                            | Input device handling; could be exploited for unauthorized input capture.                           |
| inet.pyc                                    | Internet handling; could be exploited for network-based attacks.                                    |
| incoming.pyc                                | Incoming data handling; could be exploited for unauthorized data access.                            |
| incoming                                    | Directory or file; could be exploited for unauthorized access.                                      |
| ie                                          | Internet Explorer; could be exploited for browser-based attacks.                                    |
| idna.pyc                                    | Internationalized domain names handling; could be exploited for DNS attacks.                       |
| html5                                       | HTML5 handling; could be exploited for web-based attacks.                                          |
| ipaddress.pyc                               | IP address handling; could be exploited for network-based attacks.                                  |
| config.pyc                                  | Configuration file; could contain sensitive information or settings for exploitation.               |
| compat.pyc                                  | Compatibility functions; could be exploited for unauthorized access.                                 |
| compat                                      | Compatibility layer; could be exploited for unauthorized access.                                    |
| common                                      | Common functions; could be exploited for various malicious purposes.                                 |
| command.pyc                                 | Command handling; could be exploited for unauthorized command execution.                            |
| colors.pyc                                  | Color handling; could be exploited for unauthorized UI manipulations.                               |
| client                                      | Client-side scripts; could be exploited for unauthorized access.                                    |
| char.pyc                                    | Character handling; could be exploited for data manipulation.                                       |
| cgi.pyc                                     | Common Gateway Interface; could be exploited for web-based attacks.                                 |
| cffi-1.11.5.dist-info                       | C Foreign Function Interface; could be exploited for unauthorized access to C libraries.            |
| asn1crypto-0.24.0.dist-info                 | ASN.1 encoding/decoding library; could be exploited for data manipulation.                         |
| PyNaCl-1.2.1.dist-info                       | Cryptography library; could be exploited for unauthorized access to encrypted data.                 |
| DESKTOP.INI                                 | Windows desktop configuration; could be exploited for unauthorized access to desktop settings.      |
| __init__.pyc                                | Initialization file; could be exploited for unauthorized module access.                             |
| ncr_utilities-1.0-py2.7.egg                | Utility library; could be exploited for unauthorized access to utility functions.                   |
| mantra_web-2.0-py2.7.egg                   | Web framework; could be exploited for web-based attacks.                                           |
| utils.pyc                                   | General utility functions; could be exploited for various malicious purposes.                       |
| mantra_pdf_documents-3.0-py2.7.egg         | PDF document handling; could be exploited for unauthorized access to PDF files.                     |
| sleekxmpp-1.3.3-py2.7.egg                  | XMPP library; could be exploited for unauthorized messaging.                                        |
| xmpppy-0.5.0rc1-py2.7.egg                  | XMPP library; could be exploited for unauthorized messaging.                                        |
| ncr_status-1.0-py2.7.egg                   | Status reporting library; could be exploited for unauthorized access to status information.         |
| mantra_xmpp-2.0-py2.7.egg                  | XMPP framework; could be exploited for unauthorized messaging.                                       |
| mantra_word_documents-2.0-py2.7.egg         | Word document handling; could be exploited for unauthorized access to Word files.                   |
| mantra_spreadsheets-1.0-py2.7.egg          | Spreadsheet handling; could be exploited for unauthorized access to spreadsheet files.              |
| mantra_rdp-1.0-py2.7.egg                   | RDP handling; could be exploited for unauthorized remote access.                                    |
| mantra_host-2.0-py2.7.egg                  | Host management; could be exploited for unauthorized access to host settings.                       |
| mantra_fileshare-2.0-py2.7.egg              | File sharing handling; could be exploited for unauthorized file access.                             |
| mantra_email-2.0-py2.7.egg                 | Email handling; could be exploited for unauthorized email access.                                   |
| mantra_actors-2.0-py2.7.egg                | Actor management; could be exploited for unauthorized access to actor settings.                     |
| mantra_ssh-2.0-py2.7.egg                   | SSH handling; could be exploited for unauthorized remote access.                                    |
| mantra_presentation_documents-2.0-py2.7.egg| Presentation document handling; could be exploited for unauthorized access to presentation files.   |

## Chronological Log of Actions

### 13:45
- `svchost.exe` READ the file: `dynamic.pyc`
- `svchost.exe` READ the file: `domreg.pyc`
- `svchost.exe` READ the file: `contrib`
- `svchost.exe` READ the file: `dircache.pyc`
- `svchost.exe` READ the file: `desired_capabilities.pyc`
- `svchost.exe` READ the file: `decoder.pyc`
- `svchost.exe` READ the file: `decimal.pyc`
- `svchost.exe` READ the file: `debug.pyc`
- `svchost.exe` READ the file: `dateandtime.pyc`
- `svchost.exe` READ the file: `datatypes.pyc`
- `svchost.exe` READ the file: `cryptography-2.3.1.dist-info`
- `svchost.exe` READ the file: `core.pyc`
- `svchost.exe` READ the file: `cookies.pyc`
- `svchost.exe` READ the file: `dns`
- `svchost.exe` READ the file: `api.pyc`
- `svchost.exe` READ the file: `alert.pyc`
- `svchost.exe` READ the file: `adapters.pyc`
- `svchost.exe` READ the file: `actions`
- `svchost.exe` READ the file: `action_chains.pyc`
- `svchost.exe` READ the file: `action_builder.pyc`
- `svchost.exe` READ the file: `_win32sysloader.pyd`
- `svchost.exe` READ the file: `_fontdata_widths_helvetica.pyc`
- `svchost.exe` READ the file: `_strptime.pyc`
- `svchost.exe` READ the file: `_parseaddr.pyc`
- `svchost.exe` READ the file: `_internal_utils.pyc`
- `svchost.exe` READ the file: `_fontdata_widths_zapfdingbats.pyc`
- `svchost.exe` READ the file: `_fontdata_widths_timesitalic.pyc`
- `svchost.exe` READ the file: `_fontdata_widths_timesbolditalic.pyc`
- `svchost.exe` READ the file: `_fontdata_widths_timesbold.pyc`
- `svchost.exe` READ the file: `_fontdata_widths_symbol.pyc`
- `svchost.exe` READ the file: `_fontdata_widths_helveticaoblique.pyc`
- `svchost.exe` READ the file: `_fontdata_widths_helveticabold.pyc`
- `svchost.exe` READ the file: `_version.pyc`
- `svchost.exe` READ the file: `win32gui.pyd`
- `svchost.exe` READ the file: `win32con.pyc`
- `svchost.exe` READ the file: `win32comext`
- `svchost.exe` READ the file: `win32api.pyd`
- `svchost.exe` READ the file: `webelement.pyc`
- `svchost.exe` READ the file: `uuid.pyc`
- `svchost.exe` READ the file: `util.pyc`
- `svchost.exe` READ the file: `NodeFilter.pyc`
- `svchost.exe` READ the file: `win32com`
- `svchost.exe` READ the file: `CONF`
- `svchost.exe` READ the file: `MySQL_python-1.2.5-py2.7.egg-info`
- `svchost.exe` READ the file: `EGG-INFO`
- `svchost.exe` READ the file: `expat.pyc`
- `svchost.exe` READ the file: `exception.pyc`
- `svchost.exe` READ the file: `flags.pyc`
- `svchost.exe` READ the file: `mantra.ini`
- `svchost.exe` READ the file: `logstash`
- `svchost.exe` READ the file: `logging.ini`
- `svchost.exe` READ the file: `logger.pyc`
- `svchost.exe` READ the file: `lib`
- `svchost.exe` READ the file: `keys.pyc`
- `svchost.exe` READ the file: `key_input.pyc`
- `svchost.exe` READ the file: `isDisplayed.js`
- `svchost.exe` READ the file: `ipv6.pyc`
- `svchost.exe` READ the file: `ipv4.pyc`
- `svchost.exe` READ the file: `hooks.pyc`
- `svchost.exe` READ the file: `ipaddress-1.0.22.dist-info`
- `svchost.exe` READ the file: `interaction.pyc`
- `svchost.exe` READ the file: `integer.pyc`
- `svchost.exe` READ the file: `input_device.pyc`
- `svchost.exe` READ the file: `inet.pyc`
- `svchost.exe` READ the file: `incoming.pyc`
- `svchost.exe` READ the file: `incoming`
- `svchost.exe` READ the file: `ie`
- `svchost.exe` READ the file: `idna.pyc`
- `svchost.exe` READ the file: `html5`
- `svchost.exe` READ the file: `ipaddress.pyc`
- `svchost.exe` READ the file: `config.pyc`
- `svchost.exe` READ the file: `compat.pyc`
- `svchost.exe` READ the file: `compat`
- `svchost.exe` READ the file: `common`
- `svchost.exe` READ the file: `command.pyc`
- `svchost.exe` READ the file: `colors.pyc`
- `svchost.exe` READ the file: `client`
- `svchost.exe` READ the file: `char.pyc`
- `svchost.exe` READ the file: `cgi.pyc`

### 13:46
- `svchost.exe` MESSAGE_OUTBOUND the flow: `10.20.2.66` (4 times)
- `svchost.exe` REMOTE_CREATE a thread (2 times)
- `svchost.exe` START_OUTBOUND the flow: `10.20.2.66`
- `svchost.exe` READ the file: `word_documents_content_instance.json`
- `svchost.exe` READ the file: `word_documents_content.json`
- `svchost.exe` READ the file: `win32com`
- `svchost.exe` READ the file: `util.pyc`
- `svchost.exe` READ the file: `genpy.pyc`
- `svchost.exe` READ the file: `client`
- `svchost.exe` READ the file: `_Font.pyc`
- `svchost.exe` READ the file: `_Document.pyc`
- `svchost.exe` READ the file: `__init__.pyc`
- `svchost.exe` READ the file: `_Application.pyc`
- `svchost.exe` READ the file: `word_documents_content`
- `svchost.exe` READ the file: `Windows.pyc`
- `svchost.exe` READ the file: `Selection.pyc`
- `svchost.exe` READ the file: `Range.pyc`
- `svchost.exe` READ the file: `Paragraphs.pyc`
- `svchost.exe` READ the file: `ParagraphFormat.pyc`

This report summarizes the suspicious activities detected in the logs, highlighting the potential stages of an APT attack and providing a detailed account of the actions taken by the `svchost.exe` process. The identified IoCs serve as critical indicators for further investigation and mitigation efforts.