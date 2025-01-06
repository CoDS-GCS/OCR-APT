# Attack Report: Anomaly Detection from optc_anomalous_subgraph_9.csv

## Summary of Attack Behavior

On September 25, 2019, at 09:20, a series of log events were recorded indicating potential anomalous behavior associated with the execution of Python scripts and the reading of various files. The logs suggest that the Python process was heavily engaged in reading multiple files, including Python bytecode files (.pyc), configuration files, and various libraries, which may indicate an attempt to execute a malicious payload or perform reconnaissance within the system.

The following key events were noted:
- The process `python.exe` was repeatedly invoked to read numerous files, including potentially sensitive files such as `node_id.txt` and `ncr.key`, which could indicate an attempt to gather information for further exploitation.
- The presence of multiple `.egg` files suggests the use of third-party libraries, which could be leveraged for malicious purposes.
- The reading of files related to PDF generation and manipulation (e.g., `pdfutils.pyc`, `pdfmetrics.pyc`) may indicate an attempt to create or manipulate documents for phishing or data exfiltration.

This behavior aligns with the **Internal Reconnaissance** stage of an APT attack, where the attacker seeks to gather information about the environment and potential targets.

## Indicators of Compromise (IoCs)

| IoC                             | Security Context                                                                                     |
|---------------------------------|------------------------------------------------------------------------------------------------------|
| python.exe                      | Legitimate Python executable; high likelihood of exploitation if used to run malicious scripts.     |
| util.pyc                        | Python utility module; could be exploited if it contains malicious code.                            |
| nturl2path.pyc                  | Part of Python's URL handling; potential for exploitation if manipulated.                          |
| node_id.txt                     | Sensitive information file; high risk if accessed by unauthorized users.                            |
| ncr.key                         | Potentially a cryptographic key; critical risk if compromised.                                      |
| napinsp.dll                     | Windows DLL; could be exploited for malicious purposes.                                             |
| namedval.pyc                    | Python module; could be exploited if it contains vulnerabilities.                                   |
| name.pyc                        | Python module; potential for exploitation if malicious code is present.                             |
| pip-18.1.dist-info              | Package management information; legitimate but could be used to install malicious packages.         |
| pdfutils.pyc                    | PDF manipulation library; could be exploited to create malicious PDFs.                              |
| pdfmetrics.pyc                  | PDF metrics handling; potential for exploitation in document manipulation.                          |
| pdfgeom.pyc                     | PDF geometry handling; could be exploited for malicious document creation.                          |
| pdfgen                          | PDF generation library; high risk if used to create phishing documents.                             |
| pdfdoc.pyc                      | PDF document handling; potential for exploitation if manipulated.                                   |
| pdfbase                         | Base library for PDF handling; could be exploited for malicious purposes.                           |
| pdf.pyc                         | Core PDF handling module; high risk if exploited.                                                  |
| pathobject.pyc                  | Python path handling; potential for exploitation if manipulated.                                    |
| namedtype.pyc                   | Python module; could be exploited if it contains vulnerabilities.                                   |
| pagesizes.pyc                   | PDF page size handling; potential for exploitation in document manipulation.                        |
| packages.pyc                    | Python package handling; could be exploited if it contains malicious code.                          |
| packages                         | Directory for Python packages; could be exploited if malicious packages are present.                |
| ordered_dict.pyc                | Python's ordered dictionary implementation; potential for exploitation if manipulated.             |
| options.pyc                     | Configuration options handling; could be exploited if it contains vulnerabilities.                  |
| opera                           | Could refer to the Opera browser; potential for exploitation if used maliciously.                  |
| opcode.pyc                      | Python opcode handling; potential for exploitation if manipulated.                                  |
| octets.pyc                      | Python module; could be exploited if it contains vulnerabilities.                                   |
| numbers.pyc                     | Python number handling; potential for exploitation if manipulated.                                  |
| paramiko-2.4.1.dist-info        | SSH library; could be exploited for unauthorized access.                                            |
| idna.pyc                        | Internationalized Domain Names handling; potential for exploitation if manipulated.                |
| ie                              | Could refer to Internet Explorer; potential for exploitation if used maliciously.                   |
| incoming                         | Could refer to incoming network traffic; potential for exploitation if manipulated.                |
| incoming.pyc                    | Python module; could be exploited if it contains vulnerabilities.                                   |
| inet.pyc                        | Internet handling module; potential for exploitation if manipulated.                                |
| input_device.pyc                | Input device handling; could be exploited if it contains vulnerabilities.                           |
| interaction.pyc                 | User interaction handling; potential for exploitation if manipulated.                               |
| ipaddress-1.0.22.dist-info      | IP address handling library; could be exploited for network reconnaissance.                         |
| ipaddress.pyc                   | Python IP address handling; potential for exploitation if manipulated.                              |
| ipv6.pyc                        | IPv6 handling; could be exploited for network reconnaissance.                                       |
| isDisplayed.js                  | JavaScript file; potential for exploitation if used in a web context.                               |
| key_actions.pyc                 | Key action handling; could be exploited if it contains vulnerabilities.                             |
| key_input.pyc                   | Key input handling; potential for exploitation if manipulated.                                      |
| keys.pyc                        | Key handling module; could be exploited if it contains vulnerabilities.                             |
| phantomjs                       | Headless browser; could be exploited for web scraping or malicious automation.                      |
| log.pyc                         | Logging module; could be exploited if it contains vulnerabilities.                                   |
| logger.pyc                      | Logging functionality; potential for exploitation if manipulated.                                   |
| logging.ini                     | Configuration for logging; could be exploited if it contains sensitive information.                 |
| logstash                        | Log management tool; could be exploited for unauthorized access to logs.                            |
| mantra.ini                      | Configuration file for the Mantra framework; could be exploited if it contains sensitive data.     |
| md5.pyc                         | MD5 hashing module; could be exploited if used for malicious purposes.                              |
| merger.pyc                      | File merging module; potential for exploitation if manipulated.                                     |
| message.pyc                     | Message handling module; could be exploited if it contains vulnerabilities.                         |
| mime                            | MIME type handling; could be exploited for malicious file types.                                    |
| mimetools.pyc                   | MIME tools; potential for exploitation if manipulated.                                             |
| mimetypes.pyc                   | MIME type handling; could be exploited for malicious file types.                                    |
| minidom.pyc                     | XML handling; could be exploited if manipulated.                                                   |
| mobile.pyc                      | Mobile handling module; potential for exploitation if manipulated.                                   |
| models.pyc                      | Model handling in Python; could be exploited if it contains vulnerabilities.                        |
| mouse_button.pyc                | Mouse button handling; potential for exploitation if manipulated.                                   |
| lib                             | Could refer to a library directory; potential for exploitation if malicious libraries are present.  |
| tsig.pyc                        | TSIG handling; could be exploited for DNS manipulation.                                            |
| pyasn1                          | ASN.1 handling library; could be exploited for malicious purposes.                                   |
| xmlbuilder.pyc                  | XML building module; potential for exploitation if manipulated.                                     |
| winrnr.dll                      | Windows DLL; could be exploited for malicious purposes.                                             |
| winerror.pyc                    | Windows error handling; potential for exploitation if manipulated.                                   |
| win32wnet.pyd                   | Windows networking; could be exploited for unauthorized access.                                      |
| cgi.pyc                         | Common Gateway Interface handling; could be exploited for web vulnerabilities.                       |
| bisect.pyc                      | Python bisect module; potential for exploitation if manipulated.                                    |
| certs.pyc                       | Certificate handling; could be exploited if sensitive data is present.                               |
| certifi                         | Certificate authority handling; could be exploited for malicious purposes.                           |
| canvas.pyc                      | Canvas handling in Python; potential for exploitation if manipulated.                               |
| calling.pyc                     | Function calling module; could be exploited if it contains vulnerabilities.                         |
| calendar.pyc                    | Calendar handling; potential for exploitation if manipulated.                                       |
| cElementTree.pyc                | XML handling; could be exploited if manipulated.                                                   |
| by.pyc                          | Python module; could be exploited if it contains vulnerabilities.                                   |
| build.pyc                       | Build handling; potential for exploitation if manipulated.                                          |
| boxstuff.pyc                    | Box handling module; could be exploited if it contains vulnerabilities.                             |
| blockfeeder.pyc                 | Block feeding module; potential for exploitation if manipulated.                                    |
| blackberry                      | Could refer to BlackBerry services; potential for exploitation if used maliciously.                |
| cffi-1.11.5.dist-info           | C Foreign Function Interface; could be exploited for unauthorized access.                           |

## Chronological Log of Actions

### 09:20
- `python.exe` loaded the module `win32api.pyd`.
- `python.exe` read the file `url.pyc`.
- `python.exe` read the file `renderer.pyc`.
- `python.exe` read the file `unicodedata.pyd`.
- `python.exe` read the file `unicode_escape.pyc`.
- `python.exe` read the file `ui.pyc`.
- `python.exe` read the file `type`.
- `python.exe` read the file `ttfonts.pyc`.
- `python.exe` read the file `response.pyc`.
- `python.exe` read the file `resolver.pyc`.
- `python.exe` read the file `requests`.
- `python.exe` read the file `request.pyc`.
- `python.exe` read the file `reportlab`.
- `python.exe` read the file `units.pyc`.
- `python.exe` read the file `_MozillaCookieJar.pyc`.
- `python.exe` read the file `_LWPCookieJar.pyc`.
- `python.exe` read the file `Wbem`.
- `python.exe` read the file `SysClient0051_mantra.json`.
- `python.exe` read the file `StringIO.pyc`.
- `python.exe` read the file `SocketServer.pyc`.
- `python.exe` read the file `PyPDF2`.
- `python.exe` read the file `PyNaCl-1.2.1.dist-info`.
- `python.exe` read the file `PYEXPAT.PYD`.
- `python.exe` read the file `NodeFilter.pyc`.
- `python.exe` read the file `hmac.pyc`.
- `python.exe` read the file `Microsoft-Windows-PeerToPeer-Full-Package~31bf3856ad364e35~amd64~~10.0.15063.0.cat`.
- `python.exe` read the file `Microsoft-Windows-Client-Features-WOW64-Package-AutoMerged-net~31bf3856ad364e35~amd64~~10.0.15063.0.cat`.
- `python.exe` read the file `Microsoft-Windows-Client-Features-WOW64-Package-AutoMerged-ds~31bf3856ad364e35~amd64~~10.0.15063.0.cat`.
- `python.exe` read the file `MANTRA`.
- `python.exe` read the file `ELEMENTPATH.PYC`.
- `python.exe` read the file `EGG-INFO`.
- `python.exe` read the file `Cookie.pyc`.
- `python.exe` read the file `util.pyc` (2 times).
- `python.exe` opened the process `python.exe` (2 times).
- `python.exe` read the file `nturl2path.pyc`.
- `python.exe` read the file `node_id.txt`.
- `python.exe` read the file `ncr.key`.
- `python.exe` read the file `napinsp.dll`.
- `python.exe` read the file `namedval.pyc`.
- `python.exe` read the file `name.pyc`.
- `python.exe` read the file `pip-18.1.dist-info`.
- `python.exe` read the file `pdfutils.pyc`.
- `python.exe` read the file `pdfmetrics.pyc`.
- `python.exe` read the file `pdfgeom.pyc`.
- `python.exe` read the file `pdfgen`.
- `python.exe` read the file `pdfdoc.pyc`.
- `python.exe` read the file `pdfbase`.
- `python.exe` read the file `pdf.pyc`.
- `python.exe` read the file `pathobject.pyc`.
- `python.exe` read the file `namedtype.pyc`.
- `python.exe` read the file `pagesizes.pyc`.
- `python.exe` read the file `packages.pyc`.
- `python.exe` read the file `packages`.
- `python.exe` read the file `ordered_dict.pyc`.
- `python.exe` read the file `options.pyc`.
- `python.exe` read the file `opera`.
- `python.exe` read the file `opcode.pyc`.
- `python.exe` read the file `octets.pyc`.
- `python.exe` read the file `numbers.pyc`.
- `python.exe` read the file `paramiko-2.4.1.dist-info`.
- `python.exe` read the file `idna.pyc`.
- `python.exe` read the file `ie`.
- `python.exe` read the file `incoming`.
- `python.exe` read the file `incoming.pyc`.
- `python.exe` read the file `inet.pyc`.
- `python.exe` read the file `input_device.pyc`.
- `python.exe` read the file `interaction.pyc`.
- `python.exe` read the file `ipaddress-1.0.22.dist-info`.
- `python.exe` read the file `ipaddress.pyc`.
- `python.exe` read the file `ipv6.pyc`.
- `python.exe` read the file `isDisplayed.js`.
- `python.exe` read the file `key_actions.pyc`.
- `python.exe` read the file `key_input.pyc`.
- `python.exe` read the file `keys.pyc`.
- `python.exe` read the file `phantomjs`.
- `python.exe` read the file `log.pyc`.
- `python.exe` read the file `logger.pyc`.
- `python.exe` read the file `logging.ini`.
- `python.exe` read the file `logstash`.
- `python.exe` read the file `mantra.ini`.
- `python.exe` read the file `md5.pyc`.
- `python.exe` read the file `merger.pyc`.
- `python.exe` read the file `message.pyc`.
- `python.exe` read the file `mime`.
- `python.exe` read the file `mimetools.pyc`.
- `python.exe` read the file `mimetypes.pyc`.
- `python.exe` read the file `minidom.pyc`.
- `python.exe` read the file `mobile.pyc`.
- `python.exe` read the file `models.pyc`.
- `python.exe` read the file `mouse_button.pyc`.
- `python.exe` read the file `lib`.
- `python.exe` read the file `tsig.pyc`.
- `python.exe` read the file `pyasn1`.
- `python.exe` read the file `xmlbuilder.pyc`.
- `python.exe` read the file `winrnr.dll`.
- `python.exe` read the file `winerror.pyc`.
- `python.exe` read the file `win32wnet.pyd`.
- `python.exe` read the file `cgi.pyc`.
- `python.exe` read the file `bisect.pyc`.
- `python.exe` read the file `certs.pyc`.
- `python.exe` read the file `certifi`.
- `python.exe` read the file `canvas.pyc`.
- `python.exe` read the file `calling.pyc`.
- `python.exe` read the file `calendar.pyc`.
- `python.exe` read the file `cElementTree.pyc`.
- `python.exe` read the file `by.pyc`.
- `python.exe` read the file `build.pyc`.
- `python.exe` read the file `boxstuff.pyc`.
- `python.exe` read the file `blockfeeder.pyc`.
- `python.exe` read the file `blackberry`.
- `python.exe` read the file `cffi-1.11.5.dist-info`.