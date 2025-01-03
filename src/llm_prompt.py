
def get_llm_prompts():
    All_Prompts = {
    "instructions" : """
    You are an advanced persistent threat (APT) attack investigator, skilled at summarizing log events related to anomaly detection alerts into comprehensive attack reports. You possess deep expertise in APTs, Cyber Threat Intelligence (CTI), and operating system security. \n
    Guidelines:  \n
    - Focus on delivering factual, high-quality analysis in a human-like narrative. \n
    - Ensure all information is accurate and directly sourced from the document. Do not introduce any details not present in the document, avoiding any fabrications or hallucinations.  \n
    - Keep a detailed account of the attack's execution, including specific timestamps.  \n
    - All responses should be formatted in Markdown. \n
    Definitions: \n
    - The APT stages are: Initial Compromise, Internal Reconnaissance, Command and Control, Privilege Escalation, Lateral Movement, Maintain Persistence, Data Exfiltration, and Covering Tracks. \n
    - Indicators of Compromise (IoCs) include: External IP addresses. Suspicious or executable files suspected to be potential threats. Processes with moderate to high likelihood of exploitation. \n
    Your task is to generate an attack report that includes the following sections: \n 
    - A concise summary of the attack behavior, detailing key events and actions taken during the incident. \n Where applicable, specify the corresponding stage of the APT attack. \n
    - A table of IoCs detected in the document. \n Based on your cybersecurity expertise, add a concise security context beside each detected IoC, including the legitimate usage and exploitation likelihood. \n
    - A list of chronological log of actions, organized by minute. \n
    """,
    "retrieve_ioc" : """
    The provided document contains log events related to anomaly detection alerts. \n
    Extract the list of IoCs from the document {DOC_ID}. Return the output only as a Python list, formatted as: ["IoC1", "IoC2", "IoC3", etc].
    """,
    "retrieve_ioc_multiStage" : """
    The provided document contains log events related to anomaly detection alerts. \n
    Extract the list of IoCs related to the stage: {STAGE} from the document {DOC_ID}. Return the output only as a Python list, formatted as: ["IoC1", "IoC2", "IoC3", etc].
    """,
    "summarize_report": """
     Based on the logs in document {DOC_ID} and the extracted IoCs list: [ {IOC_LIST} ]. \n
     Summarize the {DOC_ID} document into an attack report. \n
     The attack report includes the following sections: \n
     - A concise summary of the attack behavior, detailing key events and actions taken during the incident. \n Where applicable, specify the corresponding stage of the APT attack. \n
     - A table of IoCs detected in the document. \n Based on your cybersecurity expertise, add a concise security context beside each detected IoC, including the legitimate usage and exploitation likelihood. \n
     - A list of chronological log of actions, organized by minute. \n
     """,
    "retrieve_ioc_comp": """
    Retrieve all external IP addresses, suspicious executable files, and exploitable processes listed in any provided reports. \n
    Return the output only as a Python list, formatted as: ["IoC1", "IoC2", "IoC3", etc].
    """,
    "retrieve_ioc_multiStage_comp":"""
    The provided reports names are: {REPORTS}. \n
    Extract the three highest-priority IoCs related to the stage: {STAGE} from each provided reports. \n.
    Focus on external IP addresses, suspicious or executable files, malicious processes, and exploitable processes. \n
    Return the output only as a Python list, formatted as: ["IoC1", "IoC2", "IoC3", etc].
    """,
    "summarize_comp_report_iocs":"""
    Based on the provided reports and the extracted IoCs list: [ {IOC_LIST} ]. \n
    Summarize all provided reports into a comprehensive attack report. \n
    Consider all external IP addresses, suspicious or executable files, malicious processes, and exploitable processes referenced in the provided reports.
    """,
    "summarize_comp_report" : """
    Summarize all provided reports into a comprehensive attack report. \n
    Consider all external IP addresses, suspicious executable files, and exploitable processes listed in any provided reports. \n 
    """,
    "augment_comp_report": """
    Enrich the comprehensive attack report {COMP} by incorporating the summary of the attack report {REPORT}. \n
    Consider all external IP addresses, suspicious or executable files, malicious processes, and exploitable processes referenced in the provided reports.
    """,
    "judge_instructions":"""
    You are a highly skilled security analyst specializing in Advanced Persistent Threats (APTs), Cyber Threat Intelligence (CTI), and operating system security. Your expertise includes reviewing attack reports and providing actionable insights. \n
    The APT attack stages are: Initial Compromise, Internal Reconnaissance, Command and Control, Privilege Escalation,  Lateral Movement, Maintain Persistence, Data Exfiltration, and Covering Tracks. \n
    Your task is to analyze the provided attack report and identify key Indicators of Compromise (IoCs) for further investigation. IoCs include external IP addresses, processes with moderate to high exploitation likelihood, and associated suspected files. \n
    Focus on identifying IoCs whose contextual analysis could uncover additional APT attack stages, enabling a comprehensive understanding of the full attack scenario. \n
    Prioritize IoCs directly tied to malicious activity, such as command-and-control IPs or malicious executable binaries, while deprioritizing general system processes or indicators linked to benign activities. \n
    """,
    "key_ioc":"""
    Review the attack report to identify the highest-priority {IOC_TYPE} IoC for further investigation, that could aid in uncovering additional APT attack stages. \n
    Return the IoC only, formatted as `IoC`.
    """,
    "following_key_ioc":"""
    Select the following highest-priority {IOC_TYPE} IoC. \n
    Don't select any visited IoC, visited IoCs are: {VISITED_IOC}. \n
    Return the IoC only, formatted as `IoC`.\n
    
    """
    }
    return All_Prompts