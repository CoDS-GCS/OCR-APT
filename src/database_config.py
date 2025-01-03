import json

def get_attack_time_range(dataset):
    if dataset  == 'cadets':
        attack_time_range = {"BSD1": ["2018-04-06 11:21", "2018-04-06 12:08"],
                                    "BSD2": ["2018-04-11 15:08", "2018-04-11 15:15"],
                                    "BSD3": ["2018-04-12 14:00", "2018-04-12 14:38"],
                                    "BSD4": ["2018-04-13 09:04", "2018-04-13 09:15"]}
    elif dataset == 'theia':
        attack_time_range = {"linux1": ["2018-04-10 09:58", "2018-04-10 14:55"],
                                   "linux2": ["2018-04-12 12:44", "2018-04-12 13:26"],
                                   "linux3": ["2018-04-10 12:28", "2018-04-10 13:42"],
                                   "linux4": ["2018-04-13 09:04", "2018-04-13 09:15"]}
    elif dataset == 'trace':
        attack_time_range = {"linux1": ["2018-04-10 09:46", "2018-04-10 11:09"],
                                   "linux2": ["2018-04-13 12:43", "2018-04-13 12:53"],
                                   "linux3": ["2018-04-10 12:28", "2018-04-10 12:30"],
                                   "linux4": ["2018-04-13 13:50", "2018-04-13 14:28"]}
    elif dataset in ['SysClient0051','SysClient0501','SysClient0201']:
        attack_time_range = {"PlainPowerShell": ["2019-09-23 11:22", "2019-09-23 15:31"],
                             "CustomPowerShell": ["2019-09-24 10:27", "2019-09-25 10:01"],
                             "MaliciousUpgrade": ["2019-09-25 19:28", "2019-09-25 14:25"]}
    elif dataset == 'SimulatedUbuntu':
        attack_time_range = {"Struts2_046": ["2022-03-25 15:16", "2022-03-25 15:19"]}
    elif dataset == 'SimulatedWS12':
        attack_time_range = {"phpstudy": ["2022-03-18 16:45", "2022-03-18 16:48"]}
    elif dataset == 'SimulatedW10':
        attack_time_range = {"APT29": ["2022-04-06 15:09", "2022-04-06 15:17"],
                             "FIN6": ["2022-04-06 14:05", "2022-04-06 14:16"],
                             "Sidewinder": ["2022-04-06 14:30", "2022-04-06 14:34"]}
    else:
        print("The attack period is unknown for dataset: {}".format(dataset))
    return attack_time_range

def get_subgraphs_attributes(dataset):
    attributes = {}
    if dataset == "cadets":
        attributes['process'] = 'NA'
        attributes['pipe'] = 'NA'
        attributes['file'] = 'object_paths'
        attributes['flow'] = 'remote_ip'
    elif dataset == "theia":
        attributes['process'] = 'command_lines'
        attributes['file'] = 'NA'
        attributes['memory'] = 'NA'
        attributes['flow'] = 'remote_ip'
    elif dataset == "trace":
        attributes['process'] = 'command_lines'
        attributes['file'] = 'object_paths'
        attributes['flow'] = 'remote_ip'
        attributes['memory'] = 'NA'
    elif dataset == "optc":
        attributes['process'] = 'image_paths'
        attributes['file'] = 'file_paths'
        attributes['flow'] = 'src_ip'
    else:
        print("Undefined dataset")
    return attributes


def order_x_features(host,edge_types):
    if host == "cadets":
        edge_types = ['execute','unlink','change_principal','modify_file_attributes', 'rename', 'link','write','read',
                      'sendto', 'recvfrom', 'sendmsg','recvmsg', 'modify_process','connect','mmap','fcntl', 'fork',
                      'truncate', 'lseek', 'flows_to','accept', 'create_object','close','exit', 'open', 'bind','signal', 'other']
    elif host == "theia":
        edge_types = ['execute','unlink','modify_file_attributes','write', 'read','sendto','recvfrom', 'sendmsg', 'recvmsg', 'connect',
            'write_socket_params', 'read_socket_params', 'clone', 'mmap','shm', 'mprotect', 'open', 'boot']
    elif host == "trace":
        edge_types = ['execute','unlink','change_principal','modify_file_attributes','update','rename','link','write',
                      'read', 'connect','sendmsg', 'recvmsg','clone','fork','loadlibrary', 'mmap', 'mprotect','truncate',
                      'accept','create_object','close','exit', 'open','unit']
    elif host == 'SysClient0051':
        edge_types = ['DELETE', 'MODIFY', 'RENAME', 'WRITE', 'READ', 'CREATE', 'MESSAGE_OUTBOUND', 'MESSAGE_INBOUND',
                      'LOAD','REMOTE_CREATE', 'OPEN_INBOUND', 'OPEN', 'REMOVE', 'EDIT', 'ADD','START','TERMINATE','START_INBOUND','START_OUTBOUND']
    elif host ==  'SysClient0501':
        edge_types = ['COMMAND','DELETE','MODIFY','RENAME','WRITE','READ','CREATE', 'MESSAGE_OUTBOUND', 'MESSAGE_INBOUND','LOAD',
                    'REMOTE_CREATE','OPEN_INBOUND', 'OPEN','REMOVE', 'EDIT', 'ADD','START','TERMINATE','START_INBOUND','START_OUTBOUND']
    elif host =='SysClient0201':
        edge_types = ['COMMAND','DELETE','MODIFY','RENAME', 'WRITE','READ','CREATE', 'MESSAGE_OUTBOUND', 'MESSAGE_INBOUND','LOAD',
                    'REMOTE_CREATE','OPEN_INBOUND', 'OPEN','REMOVE', 'EDIT', 'ADD','START','TERMINATE','START_INBOUND','START_OUTBOUND']
    elif host == 'SimulatedUbuntu':
        edge_types = ['execve', "rmdir", "chmod", "rename", "write", "writev", "read", "readv", "sendto", "send",
                      "recvfrom", "sendmsg", "recvmsg", "clone", "fork", "pipe", "fcntl"]
    elif host == 'SimulatedW10':
        edge_types = ["Write","Read",'Send','Recv','Start',"Load"]
    elif host == 'SimulatedWS12':
        edge_types = ["Write","Read",'Send','Recv','Start',"Load"]
    else:
        print("Undefined dataset",host)
    sorted_edge_types = []
    for edge in edge_types:
        sorted_edge_types.append("out_" + edge.replace("EVENT_", "").lower())
        sorted_edge_types.append("in_" + edge.replace("EVENT_", "").lower())
    print("Available edges for this host",sorted_edge_types)
    return sorted_edge_types

def rename_node_type(dataset):
    if dataset == "tc3":
        map_node_type = {"SUBJECT_PROCESS":"process","NetFlowObject":"flow","FILE_OBJECT_FILE":"file",
                         "FILE_OBJECT_DIR":"fileDir","FILE_OBJECT_UNIX_SOCKET":"socket","UnnamedPipeObject":"pipe","FILE_OBJECT_BLOCK":"fileBlock",
                         "FILE_OBJECT_CHAR":"fileChar","FILE_OBJECT_LINK":"fileLink","MemoryObject":"memory","SRCSINK_UNKNOWN":"srcsink","SUBJECT_UNIT":"unit"}
    elif dataset == "optc":
        map_node_type = {"PROCESS":"process","FILE":"file","FLOW":"flow","SHELL":"shell","THREAD":"thread","MODULE":"module","REGISTRY":"registry","TASK":"task"}
    elif dataset == "nodlink":
        map_node_type = {"PROCESS":"process","NET":"flow","FILE":"file"}
    return map_node_type