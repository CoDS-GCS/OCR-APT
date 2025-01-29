import pandas as pd
from networkx.readwrite import json_graph
pd.set_option('display.max_colwidth', None)
from IPython.display import Markdown, display
from rich.console import Console
from rich.markdown import Markdown
import glob
import openai
import json
import os
import torch
import re
from copy import deepcopy
import numpy as np
from llama_index.llms.openai import OpenAI
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
from SPARQLWrapper import SPARQLWrapper , JSON
from datetime import datetime, timedelta
from database_config import rename_node_type
import pytz
my_tz = pytz.timezone('America/Nipigon')
import nest_asyncio
nest_asyncio.apply()
import time
from sparql_queries import get_investigation_queries
from llm_prompt import get_llm_prompts
def read_json_graph(filename):
    with open(filename) as f:
        js_graph = json.load(f)
    return json_graph.node_link_graph(js_graph)
from llama_index.embeddings.openai import OpenAIEmbedding
import random
import ast
import argparse
import psutil
from resource import *


def seed_everything(seed: int):
    r"""Sets the seed for generating random numbers in :pytorch:`PyTorch`,
    :obj:`numpy` and Python.

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return

def display_markdown(report):
    console = Console()
    markdown_content = Markdown(report)
    console.print(markdown_content)

def print_memory_usage(message=None):
    print(message)
    print("Memory usage (ru_maxrss) : ",getrusage(RUSAGE_SELF).ru_maxrss/1024," MB")
    print("Memory usage (psutil) : ", psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2), "MB")
    print('The CPU usage is (per process): ', psutil.Process(os.getpid()).cpu_percent(4))
    load1, load5, load15 = psutil.getloadavg()
    cpu_usage = (load15 / os.cpu_count()) * 100
    print("The CPU usage is : ", cpu_usage)
    print('used virtual memory GB:', psutil.virtual_memory().used / (1024.0 ** 3), " percent",
          psutil.virtual_memory().percent)
    return

parser = argparse.ArgumentParser(description='OCR-APT')
parser.add_argument('--dataset', type=str,required=True)
parser.add_argument('--host', type=str,required=True)
parser.add_argument('--root-path', type=str, required=True)
parser.add_argument('--exp-name', type=str, required=True)
parser.add_argument('--inv-exp-name', type=str, required=True)
parser.add_argument('--llm-exp-name', type=str, required=True)
parser.add_argument('--GNN-model-name', type=str, required=True)
parser.add_argument('--llm-model', type=str,default="gpt-4o-mini")
parser.add_argument('--llm-embedding-model', type=str,default="text-embedding-3-large")
parser.add_argument('--abnormality-level', type=str,default="Moderate")
parser.add_argument('--anomalous', type=str, default=None)
parser.add_argument('--load-index', action="store_true", default=False)
parser.add_argument('--runs', type=int, default=1)

args = parser.parse_args()
assert args.dataset in ['tc3', 'optc', 'nodlink']
assert args.host in ['cadets', 'trace', 'theia','SysClient0051','SysClient0501','SysClient0201','SimulatedUbuntu','SimulatedW10','SimulatedWS12']
process = psutil.Process(os.getpid())

if args.dataset == "optc":
    SourceDataset = "DARPA_OPTC"
elif args.dataset == "nodlink":
    SourceDataset = "NODLINK"
else:
    SourceDataset = "DARPA_TC3"

prefix = "https://"+SourceDataset+".graph/" + args.host + "/"
with open("../config.json", "r") as f:
    config = json.load(f)
    if args.dataset == "optc":
        repository_url = config["repository_url_optc"]
    elif args.dataset == "nodlink":
        repository_url = config["repository_url_nodlink"]
    else:
        repository_url = config["repository_url_tc3"]

sparql_queries = get_investigation_queries(args.host,SourceDataset)
All_Prompts = get_llm_prompts()
os.environ["OPENAI_API_KEY"] = config["openai_api_key"]
openai.api_key = os.environ["OPENAI_API_KEY"]
seed = 360
MAX_IOC_CONTEXT_ATTEMPT = 3
llm = OpenAI(model=args.llm_model, temperature=0, seed=seed,timeout=90)
splitter = SentenceSplitter(chunk_size=1024, paragraph_separator="\n")
embed_model = OpenAIEmbedding(model=args.llm_embedding_model)

def init_chat_engine(index,instructions,memory,filter_map=None,k=4):
    if filter_map is not None:
        chat_engine = index.as_chat_engine(
            llm = llm,
            chat_mode="context",
            memory=memory,
            system_prompt=(instructions),
            filters=filter_map,
            similarity_top_k=k,
        )
    else:
        chat_engine = index.as_chat_engine(
            llm = llm,
            chat_mode="context",
            memory=memory,
            system_prompt=(instructions),
            similarity_top_k=k,
        )
    return chat_engine

from typing import Sequence, Any, List
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
)
import logging
from llama_index.core.schema import BaseNode, Document , ObjectType , TextNode


class SafeSemanticSplitter(SemanticSplitterNodeParser):
    safety_chunker : SentenceSplitter = SentenceSplitter(chunk_size=1024,paragraph_separator="\n")
    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        all_nodes : List[BaseNode] = super()._parse_nodes(nodes=nodes,show_progress=show_progress,**kwargs)
        all_good = True
        for node in all_nodes:
            if node.get_type()==ObjectType.TEXT:
                node:TextNode=node
                if self.safety_chunker._token_size(node.text)>self.safety_chunker.chunk_size:
                    logging.info("Chunk size too big after semantic chunking: switching to static chunking")
                    all_good = False
                    break
        if not all_good:
            all_nodes = self.safety_chunker._parse_nodes(nodes,show_progress=show_progress,**kwargs)
        return all_nodes

def parse_name_from_attr(node_attr, node_type):
    if node_attr is None or node_attr in ["", "nan"]:
        return None
    if node_type in ["flow", "netflowobject", "net"]:
        if "->" in node_attr:
            node_attr = node_attr.split('->')[0].split(':')[0] + "->" + node_attr.split('->')[-1].split(':')[0]
        if ":" in node_attr:
            node_attr = node_attr.split(':')[0]
        if "," in node_attr:
            node_attr = node_attr.split(',')[1]
        if " " in node_attr:
            node_attr = node_attr.split(' ')[0]
    else:
        if "/" in node_attr:
            node_attr = node_attr.split('/')[-1]
        if "\\" in node_attr:
            node_attr = node_attr.split('\\')[-1]
    return node_attr

def convert_timestamp_to_datetime(timestamp):
    ## DEBUG: To be verified -- The timestamps mapping has not been mention in the source released dataset or the repective paper. ##
    if args.host == "SimulatedW10":
        base_datetime = datetime(2022, 4, 8, 13, 0, 0 ,0)
    elif args.host == "SimulatedWS12":
        base_datetime = datetime(2022, 3, 16, 13, 28, 4, 0)
    this_datetime = base_datetime + timedelta(seconds=((int(float(timestamp))/1000)))
    return this_datetime.strftime("%Y-%m-%d %H:%M:%S")

def timestamp_in_second(results_df, dataset, host):
    date_format = '%Y-%m-%d %H:%M:%S'
    if dataset == "optc":
        results_df["timestamp"] = results_df["timestamp"].apply(lambda x: x[:19])
    elif host in ["SimulatedW10", "SimulatedWS12"]:
        print("The timestamp format of host", args.host, " is not accurate, need to be fixed")
        results_df["timestamp"] = results_df["timestamp"].apply(lambda x: convert_timestamp_to_datetime(x))
    else:
        results_df["timestamp"] = results_df["timestamp"].apply(
            lambda x: datetime.fromtimestamp(int(x) // 1000000000, tz=pytz.timezone("America/Nipigon"))).dt.floor('S')
        results_df['timestamp'] = results_df['timestamp'].apply(lambda t: t.strftime(date_format))
    return results_df


def get_attack_description_from_df(subgraphs_df):
    subgraphs_df = subgraphs_df.replace({np.nan: None})
    subgraphs_df = subgraphs_df.dropna(subset=['predicate', 'timestamp'])
    subgraphs_df["predicate"] = subgraphs_df['predicate'].str.split("/").str[-1].str.upper().str.replace("EVENT_", "")
    subgraphs_df["subject_type"] = subgraphs_df['subject_type'].str.split("/").str[-1].str.lower()
    subgraphs_df["object_type"] = subgraphs_df['object_type'].str.split("/").str[-1].str.lower()
    subgraphs_df["subject_attr"] = subgraphs_df.apply(
        lambda x: parse_name_from_attr(x["subject_attr"], x["subject_type"]), axis=1)
    subgraphs_df["object_attr"] = subgraphs_df.apply(lambda x: parse_name_from_attr(x["object_attr"], x["object_type"]),
                                                     axis=1)
    subgraphs_df = subgraphs_df.dropna(subset=['subject_attr', 'object_attr'])

    attack_description_df = pd.DataFrame()
    subgraphs_df["description"] = subgraphs_df["subject_attr"] + " " + subgraphs_df['predicate'] + " the " + \
                                  subgraphs_df["object_type"] + " : " + subgraphs_df["object_attr"]
    subgraphs_df = subgraphs_df[["description", "timestamp"]]
    subgraphs_df = subgraphs_df.drop_duplicates()
    print("Total number of triples before dropping duplicated actions (within one second)", len(subgraphs_df))
    subgraphs_df = timestamp_in_second(subgraphs_df, args.dataset, args.host)
    subgraphs_df = subgraphs_df.drop_duplicates()
    print("Total number of triples after dropping duplicated actions (within one second)", len(subgraphs_df))
    return subgraphs_df

def prepare_document(df_id,processed_report):
    processed_report['description'] = processed_report['description'].str.replace("with attribute","")
    map_node_type = rename_node_type(args.dataset)
    for node,mapped_node in map_node_type.items():
        processed_report['description'] = processed_report['description'].str.replace(node,mapped_node, flags=re.I)
    processed_report['description'] = processed_report['description'].str.replace(r'\s{2,}', ' ', regex=True)
    processed_report['timestamp'] = pd.to_datetime(processed_report['timestamp'])
    processed_report = processed_report.sort_values(by='timestamp')
    processed_report = processed_report.groupby(processed_report['timestamp'].dt.floor('T'))['description'].value_counts().reset_index(name='count')
    processed_report['description'] = processed_report.apply(lambda row: row['description'] + (' (' + str(row['count']) + ' times)' if row['count'] > 1 else ''), axis=1)
    processed_report = processed_report[["description","timestamp"]]
    processed_report['timestamp'] = processed_report['timestamp'].dt.strftime("%Y-%m-%d %H:%M")
    document = Document(text="".join([f"{row['description']} on {row['timestamp']}. \n" for _, row in processed_report.iterrows()]),doc_id = df_id,metadata={"file_name":(df_id)})
    return document, processed_report


def map_sparql_query(results_df_tmp):
    results_df_tmp = pd.DataFrame(results_df_tmp['results']['bindings'])
    results_df_tmp = results_df_tmp.map(lambda x: x['value'] if type(x) is dict else x).drop_duplicates()
    return  results_df_tmp

def get_context_of_IOC(IOC,IOC_type,n_hop=1):
    IOC = IOC.lower()
    query_time = time.time()
    read_sparql = SPARQLWrapper(repository_url)
    read_sparql.setReturnFormat(JSON)
    map_node_type = rename_node_type(args.dataset)
    inv_map_node_type = {v: k for k, v in map_node_type.items()}
    if IOC_type in ["flow","file"]:
        if args.anomalous == "sub":
            query = sparql_queries['get_context_of_Object_IOC_anomalous_Subj'].replace("<IOC>",'\"'+IOC+'\"').replace("<HOST>",args.host).replace("<SourceDataset>",SourceDataset).replace("<ObjectType>",inv_map_node_type[IOC_type])
        elif args.anomalous == "subobj":
            query = sparql_queries['get_context_of_Object_IOC_anomalous_SubjObj'].replace("<IOC>",'\"'+IOC+'\"').replace("<HOST>",args.host).replace("<SourceDataset>",SourceDataset).replace("<ObjectType>",inv_map_node_type[IOC_type])
        else:
            query = sparql_queries['get_context_of_Object_IOC'].replace("<IOC>",'\"'+IOC+'\"').replace("<HOST>",args.host).replace("<SourceDataset>",SourceDataset).replace("<ObjectType>",inv_map_node_type[IOC_type])
    elif IOC_type == "process":
        if args.anomalous == "sub":
            query = sparql_queries['get_context_of_anomalous_Subject_IOC'].replace("<IOC>",'\"'+IOC+'\"').replace("<HOST>",args.host).replace("<SourceDataset>",SourceDataset).replace("<ObjectType1>",inv_map_node_type["flow"]).replace("<ObjectType2>",inv_map_node_type["file"]).replace("<SubjectType>",inv_map_node_type["process"])
        elif args.anomalous == "subobj":
            query = sparql_queries['get_context_of_Subject_IOC_anomalous_SubObj'].replace("<IOC>",'\"'+IOC+'\"').replace("<HOST>",args.host).replace("<SourceDataset>",SourceDataset).replace("<ObjectType1>",inv_map_node_type["flow"]).replace("<ObjectType2>",inv_map_node_type["file"]).replace("<SubjectType>",inv_map_node_type["process"])
        else:
            query = sparql_queries['get_context_of_Subject_IOC'].replace("<IOC>",'\"'+IOC+'\"').replace("<HOST>",args.host).replace("<SourceDataset>",SourceDataset).replace("<ObjectType1>",inv_map_node_type["flow"]).replace("<ObjectType2>",inv_map_node_type["file"]).replace("<SubjectType>",inv_map_node_type["process"])
    else:
        print("Unknown IOC type",IOC_type)
        return
    print(query)
    read_sparql.setQuery(query)
    context_df = read_sparql.queryAndConvert()
    context_df = map_sparql_query(context_df)
    display(context_df)
    if n_hop == 2:
        if args.anomalous == "sub":
            query = sparql_queries['get_context_of_FLOW_IOC_2hop_anomalous_Subjects'].replace("<IOC>",'\"'+IOC+'\"').replace("<HOST>",args.host).replace("<SourceDataset>",SourceDataset).replace("<SourceDataset>",SourceDataset).replace("<ObjectType1>",inv_map_node_type["flow"]).replace("<ObjectType2>",inv_map_node_type["file"])
        else:
            query = sparql_queries['get_context_of_FLOW_IOC_2hop'].replace("<IOC>",'\"'+IOC+'\"').replace("<HOST>",args.host).replace("<SourceDataset>",SourceDataset).replace("<SourceDataset>",SourceDataset).replace("<ObjectType1>",inv_map_node_type["flow"]).replace("<ObjectType2>",inv_map_node_type["file"])
        print(query)
        read_sparql.setQuery(query)
        context_df_2 = read_sparql.queryAndConvert()
        context_df_2 = map_sparql_query(context_df_2)
        context_df = pd.concat([context_df, context_df_2], ignore_index=True).drop_duplicates()
        del context_df_2
        display(context_df)
    if len(context_df) == 0 :
        print("The query didn't return any data")
        return None , None, None, None , None
    context_description_df = get_attack_description_from_df(context_df)
    display(context_description_df)
    if IOC_type == "flow":
        doc_id = "context_"+IOC.replace(".","_")
    elif IOC_type == "file":
        doc_id = "context_file_"+IOC.split(".")[0]
    else :
        doc_id = "context_"+IOC.split(".")[0]
    doc, processed_report = prepare_document(doc_id,context_description_df)
    display(processed_report)
    print("prepared context document with ID",doc_id)
    print("querying context times is: ", time.time() - query_time)
    return context_df, context_description_df, processed_report, doc_id, doc

def index_documents(all_documents,vector_index=None,semantic=False):
    if semantic:
        safe_semantic_splitter = SafeSemanticSplitter(
            buffer_size=1, breakpoint_percentile_threshold=95, include_metadata=True, embed_model=embed_model
        )
        nodes = safe_semantic_splitter.get_nodes_from_documents(all_documents)
    else:
        nodes = splitter.get_nodes_from_documents(all_documents)
    print("Number of nodes",len(nodes))
    if vector_index:
        vector_index.insert_nodes(nodes)
    else:
        vector_index = VectorStoreIndex(nodes, embed_model=embed_model)
    del nodes
    return vector_index

def summarize_documents(doc_ids,vector_index,prompt,memory=None):
    generated_reports = {}
    for doc_id in doc_ids:
        respons_time = time.time()
        the_filter_map = MetadataFilters(filters=[MetadataFilter(key= "file_name", value= doc_id,operator="==")])
        if memory is None:
            print("Initialze the memory")
            tmp_memory = ChatMemoryBuffer.from_defaults(token_limit=50000)
            chat_engine = init_chat_engine(vector_index,All_Prompts["instructions"],tmp_memory,filter_map=the_filter_map)
            del tmp_memory
        else:
            chat_engine = init_chat_engine(vector_index,All_Prompts["instructions"],memory,filter_map=the_filter_map)
        print("Summarizing",doc_id)
        generated_reports[doc_id] = prompt_chat_engine(chat_engine, prompt.replace("<DOC_ID>",doc_id))
        print("response times is: ", time.time() - respons_time)
    return generated_reports, memory

def generate_comprehensive_report(generated_reports_index,prompt=All_Prompts["summarize_comp_report"],memory=None,the_filter_map=None):
    respons_time = time.time()
    if memory is None:
        print("Initialze the memory")
        tmp_memory = ChatMemoryBuffer.from_defaults(token_limit=50000)
        chat_engine = init_chat_engine(generated_reports_index,All_Prompts["instructions"],tmp_memory,filter_map=the_filter_map)
        del tmp_memory
    else:
        chat_engine = init_chat_engine(generated_reports_index,All_Prompts["instructions"],memory,filter_map=the_filter_map)
    attack_reports_names = '"' +'", "'.join(list(generated_reports_index.ref_doc_info.keys()))+'"'
    print("Names of provided documents are: ",attack_reports_names)
    print("Prompt:",prompt)
    comprehensive_report = prompt_chat_engine(chat_engine, prompt)
    print("response times is: ", time.time() - respons_time)
    return comprehensive_report, memory, chat_engine

def retrieve_and_generated_comprehensive_report(generated_reports_index,split_by_APT_stages=False,generate_prompt=All_Prompts["summarize_comp_report_iocs"],retrieve_prompt=None,memory=None):
    respons_time = time.time()
    if memory is None:
        print("Initialze the memory")
        tmp_memory = ChatMemoryBuffer.from_defaults(token_limit=50000)
        chat_engine = init_chat_engine(generated_reports_index,All_Prompts["instructions"],tmp_memory)
        del tmp_memory
    else:
        chat_engine = init_chat_engine(generated_reports_index,All_Prompts["instructions"],memory)
    attack_reports_names = '"' + '", "'.join(list(generated_reports_index.ref_doc_info.keys())) + '"'
    print("Names of provided documents are: ", attack_reports_names)
    if split_by_APT_stages:
        if retrieve_prompt is None:
            retrieve_prompt = All_Prompts["retrieve_ioc_multiStage_comp"]
        APT_stages = ['Initial Compromise', 'Internal Reconnaissance', 'Command and Control', 'Privilege Escalation', 'Lateral Movement', 'Maintain Persistence', 'Data Exfiltration', 'Covering Tracks']
        ioc_lst = []
        for stage in APT_stages:
            ioc_lst.extend(retrieve_IOC_list(chat_engine,retrieve_prompt.replace("{STAGE}",stage).replace("{REPORTS}",attack_reports_names),filter_hallucination=False))
    else:
        if retrieve_prompt is None:
            retrieve_prompt = All_Prompts["retrieve_ioc_comp"]
        ioc_lst = retrieve_IOC_list(chat_engine,retrieve_prompt,filter_hallucination=False)

    IOC_LIST = '"' + '", "'.join(ioc_lst) + '"'
    generate_prompt = generate_prompt.replace("{IOC_LIST}",IOC_LIST)
    print("Prompt: ",generate_prompt)
    comprehensive_report = prompt_chat_engine(chat_engine, generate_prompt)
    print("response times is: ", time.time() - respons_time)
    return comprehensive_report, memory, chat_engine

def index_generated_reports(generated_reports,generated_reports_index=None,report_of_interest=None):
    generated_reports_docs = []
    if report_of_interest is None:
        for report_id,report in generated_reports.items():
            generated_reports_docs.append(Document(text=report,doc_id = report_id,metadata={"file_name":report_id}))
    else:
        generated_reports_docs.append(Document(text=generated_reports[report_of_interest],doc_id = report_of_interest,metadata={"file_name":report_of_interest}))
    if (generated_reports_index is None) or (len(generated_reports_docs) > 1):
        generated_reports_index = VectorStoreIndex.from_documents(generated_reports_docs, embed_model=embed_model)
    else:
        generated_reports_index.insert(generated_reports_docs[0])
    del generated_reports_docs
    return generated_reports_index


def prompt_chat_engine(chat_engine,prompt):
    response = chat_engine.chat(prompt)
    display_markdown(response.response)
    return response.response

def retrieve_and_summarize_documents(doc_ids,vector_index,processed_reports,split_by_APT_stages=False,summarize_prompt=All_Prompts["summarize_report"],retrieve_prompt=None,memory=None):
    generated_reports = {}
    filtered_ioc_lsts = {}
    for doc_id in doc_ids:
        respons_time = time.time()
        the_filter_map = MetadataFilters(filters=[MetadataFilter(key="file_name", value=doc_id, operator="==")])
        if memory is None:
            print("Initialze the memory")
            tmp_memory = ChatMemoryBuffer.from_defaults(token_limit=50000)
            chat_engine = init_chat_engine(vector_index,All_Prompts["instructions"],tmp_memory,filter_map=the_filter_map)
            del tmp_memory
        else:
            chat_engine = init_chat_engine(vector_index,All_Prompts["instructions"],memory,filter_map=the_filter_map)
        if split_by_APT_stages:
            if retrieve_prompt is None:
                retrieve_prompt = All_Prompts["retrieve_ioc_multiStage"]
            APT_stages = ['Initial Compromise', 'Internal Reconnaissance', 'Command and Control', 'Privilege Escalation', 'Lateral Movement', 'Maintain Persistence', 'Data Exfiltration', 'Covering Tracks']
            filtered_ioc_lst = []
            for stage in APT_stages:
                filtered_ioc_lst.extend(retrieve_IOC_list(chat_engine,retrieve_prompt.replace("{DOC_ID}",doc_id).replace("{STAGE}",stage),processed_reports[doc_id]))
        else:
            if retrieve_prompt is None:
                retrieve_prompt = All_Prompts["retrieve_ioc"]
            filtered_ioc_lst = retrieve_IOC_list(chat_engine,retrieve_prompt.replace("{DOC_ID}",doc_id),processed_reports[doc_id])
        print("Summarizing",doc_id)
        IOC_LIST = '"' +'", "'.join(filtered_ioc_lst)+'"'
        this_summarize_prompt = summarize_prompt.replace("{DOC_ID}",doc_id).replace("{IOC_LIST}",IOC_LIST)
        print("Prompt: ",this_summarize_prompt)
        generated_reports[doc_id] =prompt_chat_engine(chat_engine, this_summarize_prompt)
        filtered_ioc_lsts[doc_id] = filtered_ioc_lst
        print("response times is: ", time.time() - respons_time)
    return generated_reports, filtered_ioc_lsts ,  memory

def retrieve_IOC_list(chat_engine,retrieve_prompt,processed_report=None,filter_hallucination=True):
    print("Prompt: ",retrieve_prompt)
    iocs_str = prompt_chat_engine(chat_engine, retrieve_prompt)
    iocs_str = iocs_str.strip('```python\n').strip('```')
    # 2. Safely convert the string to a Python list
    try:
        iocs_list = ast.literal_eval(iocs_str)
    except (SyntaxError, ValueError) as e:
        print(f"Error converting output to list: {e}")
    if filter_hallucination == False:
        ioc_lst = deepcopy(iocs_list)
    else:
        ioc_lst = detect_and_filter_hallucination(iocs_list,processed_report)
    return ioc_lst

def detect_and_filter_hallucination(iocs_list,processed_report):
    print("Filter hallucination from IOCs list")
    n_hallucination = 0
    filtered_ioc_lst = deepcopy(iocs_list)
    for ioc in iocs_list:
        matched_ioc = processed_report[processed_report["description"].str.contains(ioc,case=False,regex=False)]
        if len(matched_ioc) == 0:
            print("***** Model Hallucination ********")
            print(ioc,"doesn't exist in the document.\n It will be dropped from the list")
            n_hallucination +=1
            filtered_ioc_lst.remove(ioc)
    print("Number of detected hallucinations is: ", n_hallucination)
    return filtered_ioc_lst


def select_key_ioc(generated_reports, ioc_type="IP", report_of_interest=None, visited_iocs=[]):
    comprehensive_reports_index = index_generated_reports(generated_reports,report_of_interest=report_of_interest)
    memory = ChatMemoryBuffer.from_defaults(token_limit=50000)
    llm_judge = init_chat_engine(comprehensive_reports_index, All_Prompts["judge_instructions"], memory)

    ioc = prompt_chat_engine(llm_judge, All_Prompts["key_ioc"].replace("{IOC_TYPE}", ioc_type))
    ioc = ioc.replace("`", "").lower()
    attempt=1
    while ioc in visited_iocs:
        if attempt > MAX_IOC_CONTEXT_ATTEMPT:
            return None
        visited_iocs_str = '"' +'", "'.join(visited_iocs)+'"'
        ioc = prompt_chat_engine(llm_judge, All_Prompts["following_key_ioc"].replace("{IOC_TYPE}", ioc_type).replace("{VISITED_IOC}",visited_iocs_str))
        ioc = ioc.replace("`", "").lower()
        attempt +=1
    return ioc

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_checkpont(save_path,vector_index_dic, memory,save_reports):
    ensure_dir(save_path)
    torch.save(memory,(save_path+"memory.pt"))
    for vector_index_id , vector_index in  vector_index_dic.items():
        vector_index_id = vector_index_id.split(".")[0]
        vector_index.storage_context.persist(persist_dir=(save_path+"index_"+vector_index_id))
    for report_name, report in save_reports.items():
        with open((save_path+report_name+".md"), 'w') as f:
            f.write(report)

def load_checkpont(load_path):
    storage_context = StorageContext.from_defaults(persist_dir=(load_path+"index_analyzed_log_documents"))
    vector_index = load_index_from_storage(storage_context)
    storage_context = StorageContext.from_defaults(persist_dir=(load_path+"index_generated_reports"))
    generated_reports_index = load_index_from_storage(storage_context)
    memory = torch.load(load_path+"memory.pt")
    generated_reports = {}
    for report_path in glob.glob(load_path+"*.md"):
        report_name = report_path.split("/")[-1].replace(".md","")
        with open(report_path, 'r') as f:
            generated_reports[report_name] = f.read()
    return vector_index,generated_reports_index, memory, generated_reports

def enrich_with_ioc(ioc,ioc_type,processed_reports,generated_reports,last_comp_report=None):
    global vector_index, generated_reports_index
    context_df, context_description_df, context_processed_report, context_doc_id, context_doc = get_context_of_IOC(ioc,IOC_type=ioc_type)
    if context_description_df is None:
        return None,generated_reports,None,None
    processed_reports[context_doc_id] = context_processed_report
    vector_index = index_documents([context_doc], vector_index)
    context_generated_reports, context_filtered_ioc_lsts, memory = retrieve_and_summarize_documents([context_doc_id], vector_index, processed_reports, split_by_APT_stages=True,summarize_prompt=All_Prompts["summarize_report"], retrieve_prompt=All_Prompts["retrieve_ioc_multiStage"])
    generated_reports[context_doc_id] = context_generated_reports[context_doc_id]
    generated_reports_index = index_generated_reports(generated_reports,generated_reports_index, context_doc_id)
    the_filter_map = MetadataFilters(filters=[MetadataFilter(key="file_name", value=last_comp_report, operator="=="),MetadataFilter(key="file_name", value=context_doc_id, operator="==")],condition="or")
    comprehensive_report, memory, chat_engine = generate_comprehensive_report(generated_reports_index,prompt=All_Prompts["augment_comp_report"].replace("{COMP}",last_comp_report).replace("{REPORT}",context_doc_id),the_filter_map=the_filter_map)
    del context_filtered_ioc_lsts,context_df,context_description_df, context_processed_report,context_doc,context_doc_id
    return comprehensive_report,generated_reports,memory, chat_engine

if __name__ == '__main__':
    start_time = time.time()
    print(args)
    seed = 360
    for run in range(args.runs):
        ### Change the seed per run ###
        print("******************************************")
        print("Run number:", run)
        print("Seed: ", seed)
        seed_everything(seed)
        # Get the reprots from original path
        stats_report_path = args.root_path + "results/" + args.exp_name + "/" + args.GNN_model_name.replace(".model","") + "/run"+str(run)+"_" + args.inv_exp_name + "_correlated_subgraphs_statistics.csv"
        inv_reports_path = args.root_path + "investigation/" + args.exp_name + "/" + args.GNN_model_name.replace(".model","") +"/run"+str(run)+"_" + args.inv_exp_name
        subgraphs_stats_df = pd.read_csv(stats_report_path)
        ioc_f = args.root_path + "query_graphs_IOCs.json"
        with open(ioc_f) as f:
            query_graphs_IOCs = json.load(f)
        query_graphs_IOCs_set = set()
        for attack, iocs in query_graphs_IOCs.items():
            query_graphs_IOCs_set.update(iocs["file"])
            query_graphs_IOCs_set.update(iocs["ip"])
        abnormality_order = ['Negligible', 'Minor', 'Moderate', 'Significant', 'Critical']
        abnormality_level_lst = abnormality_order[abnormality_order.index(args.abnormality_level):]
        subgraphs_in_interest_IDs = subgraphs_stats_df[subgraphs_stats_df['severity_level'].isin(abnormality_level_lst)][
            "ID"].tolist()
        if len(subgraphs_in_interest_IDs) ==0:
            print("No subgraphs detected as anomalous to be investigated")
            quit()
        print("subgraphs detected as anomalous IDs:",subgraphs_in_interest_IDs)

        all_documents = []
        processed_reports = {}
        report_names = []
        for report_id in subgraphs_in_interest_IDs:
            report_path = inv_reports_path + "_attack_description_subgraph_"+str(report_id)+".csv"
            report_name = args.dataset + "_anomalous_subgraph_" + str(report_id) + ".csv"
            report_names.append(report_name)
            report = pd.read_csv(report_path)
            doc, processed_report = prepare_document(report_name, report)
            all_documents.append(doc)
            processed_reports[report_name] = processed_report
            del doc, report, processed_report
        if len(report_names) == 0:
            print("No reports serialized from detected anomalous subgraphs")
            quit()
        output_path = args.root_path + "LLM_investigator_reports/" + args.llm_exp_name + "/"
        global vector_index,generated_reports_index
        if args.load_index:
            print("loading the vector index from the path:",output_path)
            vector_index, generated_reports_index, memory, generated_reports = load_checkpont(load_path=output_path)
        else:
            vector_index = index_documents(all_documents)
        generated_reports, filtered_ioc_lsts,  memory =  retrieve_and_summarize_documents(
            report_names,vector_index,processed_reports,summarize_prompt=All_Prompts["summarize_report"],retrieve_prompt=All_Prompts["retrieve_ioc"])
        del filtered_ioc_lsts
        generated_reports_index = index_generated_reports(generated_reports)
        comprehensive_report, memory, chat_engine = retrieve_and_generated_comprehensive_report(generated_reports_index,split_by_APT_stages=True,generate_prompt=All_Prompts["summarize_comp_report_iocs"],retrieve_prompt=All_Prompts["retrieve_ioc_multiStage_comp"])
        del all_documents
        comp_report = 0
        visited_iocs = []
        for ioc_type in ["IP","process","file"]:
            report_of_interest = "comprehensive_report_" + str(comp_report)
            generated_reports[report_of_interest] = comprehensive_report
            generated_reports_index = index_generated_reports(generated_reports, generated_reports_index,report_of_interest)
            vector_index_dic = {"analyzed_log_documents": vector_index, "generated_reports": generated_reports_index}
            save_checkpont(output_path, vector_index_dic, memory, generated_reports)
            print("Enrich the comprehensive_report with the highest-priority priority {} IoC".format(ioc_type))
            for i in range(MAX_IOC_CONTEXT_ATTEMPT):
                ioc = select_key_ioc(generated_reports,report_of_interest=report_of_interest,ioc_type=ioc_type,visited_iocs=visited_iocs)
                if ioc is None:
                    break
                visited_iocs.append(ioc)
                if ioc_type == "IP":
                    comprehensive_report,generated_reports,memory, chat_engine = enrich_with_ioc(ioc, "flow",processed_reports,generated_reports,report_of_interest)
                else:
                    # comprehensive_report, generated_reports, memory, chat_engine = enrich_with_ioc(ioc, ioc_type,processed_reports,generated_reports,report_of_interest)
                    # #### Get context for process and file ####
                    comprehensive_report,generated_reports,memory, chat_engine = enrich_with_ioc(ioc, "process",processed_reports,generated_reports,report_of_interest)
                    if comprehensive_report is not None:
                        comp_report += 1
                        report_of_interest = "comprehensive_report_" + str(comp_report)
                        generated_reports[report_of_interest] = comprehensive_report
                        generated_reports_index = index_generated_reports(generated_reports, generated_reports_index,report_of_interest)
                    comprehensive_report_file,generated_reports,memory, chat_engine = enrich_with_ioc(ioc, "file",processed_reports,generated_reports,report_of_interest)
                    if comprehensive_report_file is not None:
                        comprehensive_report = comprehensive_report_file
                if comprehensive_report is None:
                    print("Couldn't find context of IOC {} in the PG database".format(ioc))
                    print("Enrich the comprehensive_report with the following highest-priority priority {} IoC".format(ioc_type))
                else:
                    break
            if comprehensive_report is None:
                print("Couldn't find context of IOC type {} in the PG database, after trying with {} different IOCs".format(ioc_type,MAX_IOC_CONTEXT_ATTEMPT))
                comprehensive_report = generated_reports[report_of_interest]
            comp_report += 1

        generated_reports["final_comprehensive_report"] = comprehensive_report
        vector_index_dic = {"analyzed_log_documents": vector_index, "generated_reports": generated_reports_index}
        save_checkpont(output_path, vector_index_dic, memory, generated_reports)
        print_memory_usage()
        print("Total time: ", time.time() - start_time, "seconds.")
        seed = np.random.randint(0, 1000)