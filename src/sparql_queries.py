import json
def get_extraction_queries(host,SourceDataset):
    hosts_details = {}
    sparql_queries = {
        'get_annomalies_attributes': """
            PREFIX graph: <https://<SourceDataset>.graph/<DATASET>/>
            SELECT ?node_uuid ?node_attr
            FROM <https://<SourceDataset>.graph/<DATASET>>
            WHERE{
                ?node graph:uuid ?node_uuid .
                ?node graph:node-attribute ?node_attr .
                ?node graph:is_Train "False" .
                ?node graph:anomalies "True" .
            }
            """,
        'get_nodes_attributes': """
                PREFIX graph: <https://<SourceDataset>.graph/<DATASET>/>
                SELECT ?node_uuid ?node_type ?node_attr
                FROM <https://<SourceDataset>.graph/<DATASET>>
                WHERE{
                    ?node graph:uuid ?node_uuid .
                    ?node graph:node-type ?node_type .
                    ?node graph:node-attribute ?node_attr .
                    ?node graph:is_Train "False" .
                }
                """,
        'get_malicious_nodes': """
                PREFIX graph: <https://<SourceDataset>.graph/<DATASET>/>
                SELECT ?node_uuid ?node_type
                FROM <https://<SourceDataset>.graph/<DATASET>>
                WHERE{
                    ?node graph:uuid ?node_uuid .
                    ?node graph:node-type ?node_type .
                    ?node graph:is_malicious "True" .
                    ?node graph:is_Train "False" .
                }
                """,
        'expand_a_node_Forward_via_anomalies_nodes': """
            PREFIX graph: <https://<SourceDataset>.graph/<DATASET>/>
            SELECT DISTINCT ?subject_uuid ?subject_type ?subject_attr ?predicate ?timestamp ?object_uuid ?object_type ?object_attr
            FROM <https://<SourceDataset>.graph/<DATASET>>
            WHERE{
                << ?subject ?predicate ?object >> graph:timestamp ?timestamp .
                ?object graph:is_Train "False" .
                ?subject graph:uuid "<SEED_NODE>" .
                ?object graph:anomalies "True" .
                
                ?subject graph:uuid ?subject_uuid .
                ?subject graph:node-type ?subject_type .
                OPTIONAL{ ?subject graph:node-attribute ?subject_attr . } .

                ?object graph:uuid ?object_uuid .
                ?object graph:node-type ?object_type .
                OPTIONAL{ ?object graph:node-attribute ?object_attr . } .
            } 
            ORDER BY DESC(?timestamp)
            LIMIT <LIMIT>
        """,
        'expand_a_node_Backward_via_anomalies_nodes': """
            PREFIX graph: <https://<SourceDataset>.graph/<DATASET>/>
            SELECT DISTINCT ?subject_uuid ?subject_type ?subject_attr ?predicate ?timestamp ?object_uuid ?object_type ?object_attr
            FROM <https://<SourceDataset>.graph/<DATASET>>
            WHERE{
                << ?subject ?predicate ?object >> graph:timestamp ?timestamp .
                ?object graph:uuid "<SEED_NODE>" .
                ?subject graph:is_Train "False" .
                ?subject graph:anomalies "True" .

                ?subject graph:uuid ?subject_uuid .
                ?subject graph:node-type ?subject_type .
                OPTIONAL{ ?subject graph:node-attribute ?subject_attr . } .

                ?object graph:uuid ?object_uuid .
                ?object graph:node-type ?object_type .
                OPTIONAL{ ?object graph:node-attribute ?object_attr . } .
            }
            ORDER BY DESC(?timestamp)
            LIMIT <LIMIT>
            """,
        'expand_a_node_Forward_via_intermediate_nodes_RR': """
            PREFIX graph: <https://<SourceDataset>.graph/<DATASET>/>
            #3-a forward from all first neighbours  
            SELECT DISTINCT ?subject_uuid ?subject_type ?subject_attr ?predicate ?timestamp ?object_uuid ?object_type ?object_attr ?next_predicate ?next_timestamp ?next_object_uuid ?next_object_type ?next_object_attr
            FROM <https://<SourceDataset>.graph/<DATASET>>
            WHERE {
                << ?subject ?predicate ?object >> graph:timestamp ?timestamp .
                << ?object ?next_predicate ?next_object >> graph:timestamp ?next_timestamp .
                ?subject graph:uuid "<SEED_NODE>" .
                ?next_object graph:anomalies "True" .
                FILTER (?subject != ?next_object) .

                ?subject graph:uuid ?subject_uuid .
                ?subject graph:node-type ?subject_type .
                OPTIONAL{ ?subject graph:node-attribute ?subject_attr . } .

                ?object graph:uuid ?object_uuid .
                ?object graph:node-type ?object_type .
                OPTIONAL{ ?object graph:node-attribute ?object_attr . } .
                
                ?next_object graph:uuid ?next_object_uuid .
                ?next_object graph:node-type ?next_object_type .
                OPTIONAL{ ?next_object graph:node-attribute ?next_object_attr . } .
            }   
            ORDER BY DESC(?timestamp)
            LIMIT <LIMIT>
        """,
        'expand_a_node_Forward_via_intermediate_nodes_RL': """
            PREFIX graph: <https://<SourceDataset>.graph/<DATASET>/>
            #3-a forward from all first neighbours  
            SELECT DISTINCT ?subject_uuid ?subject_type ?subject_attr ?predicate ?timestamp ?object_uuid ?object_type ?object_attr ?next_predicate ?next_timestamp ?next_object_uuid ?next_object_type ?next_object_attr
            FROM <https://<SourceDataset>.graph/<DATASET>>
            WHERE {
                << ?subject ?predicate ?object >> graph:timestamp ?timestamp .
                << ?subject ?next_predicate ?next_object >> graph:timestamp ?next_timestamp .
                ?object graph:uuid "<SEED_NODE>" .
                ?next_object graph:anomalies "True" .
                FILTER (?subject != ?next_object) .

                ?subject graph:uuid ?subject_uuid .
                ?subject graph:node-type ?subject_type .
                OPTIONAL{ ?subject graph:node-attribute ?subject_attr . } .

                ?object graph:uuid ?object_uuid .
                ?object graph:node-type ?object_type .
                OPTIONAL{ ?object graph:node-attribute ?object_attr . } .

                ?next_object graph:uuid ?next_object_uuid .
                ?next_object graph:node-type ?next_object_type .
                OPTIONAL{ ?next_object graph:node-attribute ?next_object_attr . } .
            }   
            ORDER BY DESC(?timestamp)
            LIMIT <LIMIT>
            """,
        'expand_a_node_Forward_via_intermediate_nodes_LR': """
            PREFIX graph: <https://<SourceDataset>.graph/<DATASET>/>
            #3-a forward from all first neighbours  
            SELECT DISTINCT ?subject_uuid ?subject_type ?subject_attr ?predicate ?timestamp ?object_uuid ?object_type ?object_attr ?next_predicate ?next_timestamp ?next_subject_uuid ?next_subject_type ?next_subject_attr
            FROM <https://<SourceDataset>.graph/<DATASET>>
            WHERE {
                << ?subject ?predicate ?object >> graph:timestamp ?timestamp .
                << ?next_subject ?next_predicate ?object >> graph:timestamp ?next_timestamp .
                ?subject graph:uuid "<SEED_NODE>" .
                ?next_subject graph:anomalies "True" .
                FILTER (?subject != ?next_subject) .

                ?subject graph:uuid ?subject_uuid .
                ?subject graph:node-type ?subject_type .
                OPTIONAL{ ?subject graph:node-attribute ?subject_attr . } .

                ?object graph:uuid ?object_uuid .
                ?object graph:node-type ?object_type .
                OPTIONAL{ ?object graph:node-attribute ?object_attr . } .

                ?next_subject graph:uuid ?next_subject_uuid .
                ?next_subject graph:node-type ?next_subject_type .
                OPTIONAL{ ?next_subject graph:node-attribute ?next_subject_attr . } .
            }   
            ORDER BY DESC(?timestamp)
            LIMIT <LIMIT>
            """,
        'expand_a_node_Forward_via_intermediate_nodes_LL': """
            PREFIX graph: <https://<SourceDataset>.graph/<DATASET>/>
            #3-a forward from all first neighbours  
            SELECT DISTINCT ?subject_uuid ?subject_type ?subject_attr ?predicate ?timestamp ?object_uuid ?object_type ?object_attr ?next_predicate ?next_timestamp ?next_subject_uuid ?next_subject_type ?next_subject_attr
            FROM <https://<SourceDataset>.graph/<DATASET>>
            WHERE {
                << ?subject ?predicate ?object >> graph:timestamp ?timestamp .
                << ?next_subject ?next_predicate ?subject >> graph:timestamp ?next_timestamp .
                ?object graph:uuid "<SEED_NODE>" .
                ?next_subject graph:anomalies "True" .
                FILTER (?subject != ?next_subject) .

                ?subject graph:uuid ?subject_uuid .
                ?subject graph:node-type ?subject_type .
                OPTIONAL{ ?subject graph:node-attribute ?subject_attr . } .

                ?object graph:uuid ?object_uuid .
                ?object graph:node-type ?object_type .
                OPTIONAL{ ?object graph:node-attribute ?object_attr . } .

                ?next_subject graph:uuid ?next_subject_uuid .
                ?next_subject graph:node-type ?next_subject_type .
                OPTIONAL{ ?next_subject graph:node-attribute ?next_subject_attr . } .
            }   
            ORDER BY DESC(?timestamp)
            LIMIT <LIMIT>            
            """,
                    'construct_subgraphs_0_hop': """
            PREFIX graph: <https://<SourceDataset>.graph/<DATASET>/>
            SELECT DISTINCT ?subject_uuid ?subject_type ?predicate ?timestamp ?object_uuid ?object_type
            FROM <https://<SourceDataset>.graph/<DATASET>>
            WHERE{
                << ?subject ?predicate ?object >> graph:timestamp ?timestamp .
                ?subject graph:is_Train "False" .
                ?object graph:is_Train "False" .
                ?subject graph:anomalies "True" .
                ?object graph:anomalies "True" .
                ?subject graph:uuid ?subject_uuid .
                ?subject graph:node-type ?subject_type .
                ?object graph:uuid ?object_uuid .
                ?object graph:node-type ?object_type .
            }
            """,
        'construct_subgraphs_1_hop_Forward': """
            PREFIX graph: <https://<SourceDataset>.graph/<DATASET>/>
            SELECT DISTINCT ?subject_uuid ?subject_type ?predicate ?timestamp ?object_uuid ?object_type
            FROM <https://<SourceDataset>.graph/<DATASET>>
            WHERE{
                << ?subject ?predicate ?object >> graph:timestamp ?timestamp .
                ?subject graph:is_Train "False" .
                ?object graph:is_Train "False" .
                ?subject graph:anomalies "True" .
                ?subject graph:uuid ?subject_uuid .
                ?subject graph:node-type ?subject_type .
                ?object graph:uuid ?object_uuid .
                ?object graph:node-type ?object_type .
            }
            """,
        'construct_subgraphs_1_hop_Backwards_process': """
            PREFIX graph: <https://<SourceDataset>.graph/<DATASET>/>
            SELECT DISTINCT ?subject_uuid ?subject_type ?predicate ?timestamp ?object_uuid ?object_type
            FROM <https://<SourceDataset>.graph/<DATASET>>
            WHERE{
                << ?subject ?predicate ?object >> graph:timestamp ?timestamp .
                ?subject graph:is_Train "False" .
                ?object graph:is_Train "False" .
                ?object graph:anomalies "True" .

                # {?subject graph:node-type graph:PROCESS } UNION {?subject graph:anomalies "True" } .
                {?subject graph:node-type graph:SUBJECT_PROCESS } UNION {?subject graph:anomalies "True" } .

                ?subject graph:uuid ?subject_uuid .
                ?subject graph:node-type ?subject_type .
                ?object graph:uuid ?object_uuid .
                ?object graph:node-type ?object_type .
            }
            """,
        'construct_subgraphs_1_hop_Forward_process': """
                PREFIX graph: <https://<SourceDataset>.graph/<DATASET>/>
                SELECT DISTINCT ?subject_uuid ?subject_type ?predicate ?timestamp ?object_uuid ?object_type
                FROM <https://<SourceDataset>.graph/<DATASET>>
                WHERE{
                    << ?subject ?predicate ?object >> graph:timestamp ?timestamp .
                    ?subject graph:is_Train "False" .
                    ?object graph:is_Train "False" .
                    ?subject graph:anomalies "True" .

                    # {?object graph:node-type graph:PROCESS } UNION {?object graph:anomalies "True" } .
                    {?object graph:node-type graph:SUBJECT_PROCESS } UNION {?object graph:anomalies "True" } .

                    ?subject graph:uuid ?subject_uuid .
                    ?subject graph:node-type ?subject_type .
                    ?object graph:uuid ?object_uuid .
                    ?object graph:node-type ?object_type .
                }
                """,
        'construct_subgraphs_1_hop_Backwards': """
                PREFIX graph: <https://<SourceDataset>.graph/<DATASET>/>
                SELECT DISTINCT ?subject_uuid ?subject_type ?predicate ?timestamp ?object_uuid ?object_type
                FROM <https://<SourceDataset>.graph/<DATASET>>
                WHERE{
                    << ?subject ?predicate ?object >> graph:timestamp ?timestamp .
                    ?subject graph:is_Train "False" .
                    ?object graph:is_Train "False" .
                    ?object graph:anomalies "True" .
                    ?subject graph:uuid ?subject_uuid .
                    ?subject graph:node-type ?subject_type .
                    ?object graph:uuid ?object_uuid .
                    ?object graph:node-type ?object_type .
                }
                """,

        'construct_subgraphs_2_hop_FF': """
        PREFIX graph: <https://<SourceDataset>.graph/<DATASET>/>
        #3-a forward from all first neighbours  
        SELECT DISTINCT ?subject_uuid ?subject_type ?subject_attr ?predicate ?timestamp ?object_uuid ?object_type ?object_attr ?next_predicate ?next_timestamp ?next_object_uuid ?next_object_type ?next_object_attr
        FROM <https://<SourceDataset>.graph/<DATASET>>
        WHERE {
            << ?subject ?predicate ?object >> graph:timestamp ?timestamp .
            << ?object ?next_predicate ?next_object >> graph:timestamp ?next_timestamp .
            ?subject graph:anomalies "True" .
            # FILTER (?subject != ?next_object) .

            ?subject graph:uuid ?subject_uuid .
            ?subject graph:node-type ?subject_type .
            OPTIONAL{ ?subject graph:node-attribute ?subject_attr . } .

            ?object graph:uuid ?object_uuid .
            ?object graph:node-type ?object_type .
            OPTIONAL{ ?object graph:node-attribute ?object_attr . } .

            ?next_object graph:uuid ?next_object_uuid .
            ?next_object graph:node-type ?next_object_type .
            OPTIONAL{ ?next_object graph:node-attribute ?next_object_attr . } .
        }   
    """,
        'construct_subgraphs_2_hop_BF': """
        PREFIX graph: <https://<SourceDataset>.graph/<DATASET>/>
        #3-a forward from all first neighbours  
        SELECT DISTINCT ?subject_uuid ?subject_type ?subject_attr ?predicate ?timestamp ?object_uuid ?object_type ?object_attr ?next_predicate ?next_timestamp ?next_object_uuid ?next_object_type ?next_object_attr
        FROM <https://<SourceDataset>.graph/<DATASET>>
        WHERE {
            << ?subject ?predicate ?object >> graph:timestamp ?timestamp .
            << ?subject ?next_predicate ?next_object >> graph:timestamp ?next_timestamp .
            ?object graph:anomalies "True" .
            FILTER (?subject != ?next_object) .

            ?subject graph:uuid ?subject_uuid .
            ?subject graph:node-type ?subject_type .
            OPTIONAL{ ?subject graph:node-attribute ?subject_attr . } .

            ?object graph:uuid ?object_uuid .
            ?object graph:node-type ?object_type .
            OPTIONAL{ ?object graph:node-attribute ?object_attr . } .

            ?next_object graph:uuid ?next_object_uuid .
            ?next_object graph:node-type ?next_object_type .
            OPTIONAL{ ?next_object graph:node-attribute ?next_object_attr . } .
        }   
        """,
        'construct_subgraphs_2_hop_FB': """
        PREFIX graph: <https://<SourceDataset>.graph/<DATASET>/>
        #3-a forward from all first neighbours  
        SELECT DISTINCT ?subject_uuid ?subject_type ?subject_attr ?predicate ?timestamp ?object_uuid ?object_type ?object_attr ?next_predicate ?next_timestamp ?next_subject_uuid ?next_subject_type ?next_subject_attr
        FROM <https://<SourceDataset>.graph/<DATASET>>
        WHERE {
            << ?subject ?predicate ?object >> graph:timestamp ?timestamp .
            << ?next_subject ?next_predicate ?object >> graph:timestamp ?next_timestamp .
            ?subject graph:anomalies "True" .
            FILTER (?subject != ?next_subject) .

            ?subject graph:uuid ?subject_uuid .
            ?subject graph:node-type ?subject_type .
            OPTIONAL{ ?subject graph:node-attribute ?subject_attr . } .

            ?object graph:uuid ?object_uuid .
            ?object graph:node-type ?object_type .
            OPTIONAL{ ?object graph:node-attribute ?object_attr . } .

            ?next_subject graph:uuid ?next_subject_uuid .
            ?next_subject graph:node-type ?next_subject_type .
            OPTIONAL{ ?next_subject graph:node-attribute ?next_subject_attr . } .
        }   
        """,
        'construct_subgraphs_2_hop_BB': """
        PREFIX graph: <https://<SourceDataset>.graph/<DATASET>/>
        #3-a forward from all first neighbours  
        SELECT DISTINCT ?subject_uuid ?subject_type ?subject_attr ?predicate ?timestamp ?object_uuid ?object_type ?object_attr ?next_predicate ?next_timestamp ?next_subject_uuid ?next_subject_type ?next_subject_attr
        FROM <https://<SourceDataset>.graph/<DATASET>>
        WHERE {
            << ?subject ?predicate ?object >> graph:timestamp ?timestamp .
            << ?next_subject ?next_predicate ?subject >> graph:timestamp ?next_timestamp .
            ?object graph:anomalies "True" .
            FILTER (?subject != ?next_subject) .

            ?subject graph:uuid ?subject_uuid .
            ?subject graph:node-type ?subject_type .
            OPTIONAL{ ?subject graph:node-attribute ?subject_attr . } .

            ?object graph:uuid ?object_uuid .
            ?object graph:node-type ?object_type .
            OPTIONAL{ ?object graph:node-attribute ?object_attr . } .

            ?next_subject graph:uuid ?next_subject_uuid .
            ?next_subject graph:node-type ?next_subject_type .
            OPTIONAL{ ?next_subject graph:node-attribute ?next_subject_attr . } .
        }            
        """,
        'construct_subgraphs_1_hop_anomalies_1': """
            PREFIX graph: <https://<SourceDataset>.graph/<DATASET>/>
            SELECT ?subject_uuid ?subject_type ?predicate ?timestamp ?object_uuid ?object_type
            FROM <https://<SourceDataset>.graph/<DATASET>>
            WHERE{
                << ?subject ?predicate ?object >> graph:timestamp ?timestamp .
                << ?subject2 ?predicate2 ?object >> graph:timestamp ?timestamp .
                ?subject graph:is_Train "False" .
                ?object graph:is_Train "False" .
                ?subject2 graph:is_Train "False" .
                
                ?subject graph:anomalies "True" .
                ?subject2 graph:anomalies "True" .
                ?subject graph:uuid ?subject_uuid .
                ?subject graph:node-type ?subject_type .
                ?object graph:uuid ?object_uuid .
                ?object graph:node-type ?object_type .
            }
            """,
        'construct_subgraphs_1_hop_anomalies_2': """
                PREFIX graph: <https://<SourceDataset>.graph/<DATASET>/>
                SELECT ?subject_uuid ?subject_type ?predicate ?timestamp ?object_uuid ?object_type
                FROM <https://<SourceDataset>.graph/<DATASET>>
                WHERE{
                    << ?subject ?predicate ?object >> graph:timestamp ?timestamp .
                    << ?subject ?predicate2 ?object2 >> graph:timestamp ?timestamp .
                    ?subject graph:is_Train "False" .
                    ?object graph:is_Train "False" .
                    ?object2 graph:is_Train "False" .
                    
                    ?object graph:anomalies "True" .
                    ?object2 graph:anomalies "True" .
                    ?subject graph:uuid ?subject_uuid .
                    ?subject graph:node-type ?subject_type .
                    ?object graph:uuid ?object_uuid .
                    ?object graph:node-type ?object_type .
                }
                """,
        'construct_subgraphs_1_hop_anomalies_3': """
                PREFIX graph: <https://<SourceDataset>.graph/<DATASET>/>
                SELECT ?subject_uuid ?subject_type ?predicate ?timestamp ?object_uuid ?object_type
                FROM <https://<SourceDataset>.graph/<DATASET>>
                WHERE{
                    << ?subject ?predicate ?object >> graph:timestamp ?timestamp .
                    << ?object ?predicate2 ?object2 >> graph:timestamp ?timestamp .
                    ?subject graph:is_Train "False" .
                    ?object graph:is_Train "False" .
                    ?object2 graph:is_Train "False" .
                    
                    ?subject graph:anomalies "True" .
                    ?object2 graph:anomalies "True" .
                    ?subject graph:uuid ?subject_uuid .
                    ?subject graph:node-type ?subject_type .
                    ?object graph:uuid ?object_uuid .
                    ?object graph:node-type ?object_type .
                }
                """,
        'construct_subgraphs_1_hop_anomalies_4': """
                PREFIX graph: <https://<SourceDataset>.graph/<DATASET>/>
                SELECT ?subject_uuid ?subject_type ?predicate ?timestamp ?object_uuid ?object_type
                FROM <https://<SourceDataset>.graph/<DATASET>>
                WHERE{
                    << ?subject ?predicate ?object >> graph:timestamp ?timestamp .
                    << ?subject2 ?predicate2 ?subject >> graph:timestamp ?timestamp .
                    ?subject graph:is_Train "False" .
                    ?object graph:is_Train "False" .                    
                    ?subject2 graph:is_Train "False" . 
                    
                    ?subject2 graph:anomalies "True" .
                    ?object graph:anomalies "True" .
                    ?subject graph:uuid ?subject_uuid .
                    ?subject graph:node-type ?subject_type .
                    ?object graph:uuid ?object_uuid .
                    ?object graph:node-type ?object_type .
                }
                """,
        'Delete_Anomalies_Labels': """
        PREFIX graph: <https://<SourceDataset>.graph/<DATASET>/>
        WITH <https://<SourceDataset>.graph/<DATASET>>
        DELETE { ?s graph:anomalies "True" .}
        WHERE {
            ?s graph:anomalies "True" .
        }
        """,
        'Insert_Anomalies_Labels': """
            PREFIX graph: <https://<SourceDataset>.graph/<DATASET>/> 
            PREFIX node: <https://<SourceDataset>.graph/<DATASET>/node/>
            INSERT DATA 
            {  GRAPH <https://<SourceDataset>.graph/<DATASET>> {
                <TRIPLES>
                }
            }
        """,
        'Count_Anomalies_Nodes': """
            PREFIX graph: <https://<SourceDataset>.graph/<DATASET>/>
            SELECT (count(distinct ?s) as ?count_anomalies) 
            FROM <https://<SourceDataset>.graph/<DATASET>>
            WHERE {
                ?s graph:anomalies "True" .
            }
        """

    }
    for sparql_name, sparql_query in sparql_queries.items():
        sparql_queries[sparql_name] = sparql_query.replace("<DATASET>", host).replace("<SourceDataset>", SourceDataset)
    return sparql_queries

def get_investigation_queries(host,SourceDataset):
    sparql_queries = {
        "get_context_of_Object_IOC": """
            PREFIX graph: <https://<SourceDataset>.graph/<HOST>/>

            SELECT DISTINCT ?subject_type ?subject_attr ?predicate ?object_type ?object_attr ?timestamp
            FROM <https://<SourceDataset>.graph/<HOST>>
            WHERE{
                << ?subject ?predicate ?object >> graph:timestamp ?timestamp .
                ?subject graph:node-type ?subject_type .
                ?object graph:node-type graph:<ObjectType> .
                BIND(graph:<ObjectType> AS ?object_type) . 
                ?subject graph:node-attribute ?subject_attr .
                ?object graph:node-attribute ?object_attr .
                ?subject graph:is_Train "False" .
                ?object graph:is_Train "False" .
                FILTER(CONTAINS(lcase(?object_attr), <IOC>)) .
                }
        """,
        "get_context_of_FLOW_IOC_2hop": """
            PREFIX graph: <https://<SourceDataset>.graph/<HOST>/>

            SELECT DISTINCT ?subject_type ?subject_attr ?predicate ?object_type ?object_attr ?timestamp
            FROM <https://<SourceDataset>.graph/<HOST>>
            WHERE{

                << ?subject ?prev_predicate ?flow_object >> graph:timestamp ?prev_timestamp .
                << ?subject ?predicate ?object >> graph:timestamp ?timestamp .

                ?flow_object graph:node-attribute ?flow_object_attr .
                ?flow_object graph:is_Train "False" .
                FILTER(CONTAINS(lcase(?object_attr), <IOC>)) .

                ?subject graph:node-type ?subject_type .
                OPTIONAL{?subject graph:node-attribute ?subject_attr} .
                ?subject graph:is_Train "False" .

                ?object graph:node-type ?object_type . 
                ?object graph:node-attribute ?object_attr .
                ?object graph:is_Train "False" .
                #FILTER(?object_type IN (graph:<ObjectType1> , graph:<ObjectType2>)).
                }
        """,
        "get_context_of_Subject_IOC": """
            PREFIX graph: <https://<SourceDataset>.graph/<HOST>/>

            SELECT DISTINCT ?subject_type ?subject_attr ?predicate ?object_type ?object_attr ?timestamp
            FROM <https://<SourceDataset>.graph/<HOST>>
            WHERE{
                << ?subject ?predicate ?object >> graph:timestamp ?timestamp .
                ?subject graph:uuid ?subject_uuid .
                ?object graph:uuid ?object_uuid . 
                ?subject graph:node-type ?subject_type .
                ?object graph:node-type ?object_type . 
                ?subject graph:node-attribute ?subject_attr .
                ?object graph:node-attribute ?object_attr .
                ?subject graph:is_Train "False" .
                ?object graph:is_Train "False" .
                # FILTER(CONTAINS(?subject_attr, <IOC>) || CONTAINS(?object_attr, <IOC>)) .
                FILTER(CONTAINS(lcase(?subject_attr), <IOC>)) .
                FILTER(?object_type IN (graph:<ObjectType1> , graph:<ObjectType2>)).
                FILTER(?subject_type IN (graph:<SubjectType>)).
                }
        """,
        "get_context_of_Object_IOC_anomalous_Subj": """
                PREFIX graph: <https://<SourceDataset>.graph/<HOST>/>
                SELECT DISTINCT ?subject_type ?subject_attr ?predicate ?object_type ?object_attr ?timestamp
                FROM <https://<SourceDataset>.graph/<HOST>>
                WHERE{
                    << ?subject ?predicate ?object >> graph:timestamp ?timestamp .
                    ?subject graph:anomalies ?subject_isAnomalous .
                    ?subject graph:node-type ?subject_type .
                    # ?object graph:node-type ?object_type .
                    ?object graph:node-type graph:<ObjectType> .
                    BIND(graph:<ObjectType> AS ?object_type) . 
                    ?subject graph:node-attribute ?subject_attr .
                    ?object graph:node-attribute ?object_attr .
                    ?subject graph:is_Train "False" .
                    ?object graph:is_Train "False" .
                    FILTER(CONTAINS(lcase(?object_attr), <IOC>)) .
                    }
            """,
        "get_context_of_Object_IOC_anomalous_SubjObj":"""
            PREFIX graph: <https://<SourceDataset>.graph/<HOST>/>
            SELECT DISTINCT ?subject_type ?subject_attr ?predicate ?object_type ?object_attr ?timestamp
            FROM <https://<SourceDataset>.graph/<HOST>>
            WHERE{
                << ?subject ?predicate ?object >> graph:timestamp ?timestamp .
                ?subject graph:node-type ?subject_type .
                ?subject graph:node-attribute ?subject_attr .
                ?subject graph:is_Train "False" .
                OPTIONAL {?subject graph:anomalies ?subject_isAnomalous .}
                
                # ?object graph:node-type ?object_type .
                ?object graph:node-type graph:<ObjectType> .
                BIND(graph:<ObjectType> AS ?object_type) . 
                ?object graph:node-attribute ?object_attr .
                ?object graph:is_Train "False" .
                OPTIONAL {?object graph:anomalies ?object_isAnomalous .}
                
                FILTER(CONTAINS(lcase(?object_attr), <IOC>)) .
                FILTER(BOUND(?subject_isAnomalous) || BOUND(?object_isAnomalous)).
                }
        """,
        "get_context_of_FLOW_IOC_2hop_anomalous_Subjects": """
                PREFIX graph: <https://<SourceDataset>.graph/<HOST>/>
                SELECT DISTINCT ?subject_type ?subject_attr ?predicate ?object_type ?object_attr ?timestamp
                FROM <https://<SourceDataset>.graph/<HOST>>
                WHERE{

                    << ?subject ?prev_predicate ?flow_object >> graph:timestamp ?prev_timestamp .
                    << ?subject ?predicate ?object >> graph:timestamp ?timestamp .
                    ?flow_object graph:node-attribute ?flow_object_attr .
                    ?flow_object graph:is_Train "False" .
                    FILTER(CONTAINS(lcase(?object_attr), <IOC>)) .

                    ?subject graph:node-type ?subject_type .
                    OPTIONAL{?subject graph:node-attribute ?subject_attr} .
                    ?subject graph:is_Train "False" .

                    ?object graph:node-type ?object_type . 
                    ?object graph:node-attribute ?object_attr .
                    ?object graph:is_Train "False" .
                    }
            """,
        "get_context_of_anomalous_Subject_IOC": """
                PREFIX graph: <https://<SourceDataset>.graph/<HOST>/>

                SELECT DISTINCT ?subject_type ?subject_attr ?predicate ?object_type ?object_attr ?timestamp
                FROM <https://<SourceDataset>.graph/<HOST>>
                WHERE{
                    << ?subject ?predicate ?object >> graph:timestamp ?timestamp .
                    ?subject graph:anomalies ?subject_isAnomalous .
                    ?subject graph:node-type ?subject_type .
                    ?object graph:node-type ?object_type . 
                    ?subject graph:node-attribute ?subject_attr .
                    ?object graph:node-attribute ?object_attr .
                    ?subject graph:is_Train "False" .
                    ?object graph:is_Train "False" .
                    # FILTER(CONTAINS(?subject_attr, <IOC>) || CONTAINS(?object_attr, <IOC>)) .
                    FILTER(CONTAINS(lcase(?subject_attr), <IOC>)) .
                    FILTER(?object_type IN (graph:<ObjectType1> , graph:<ObjectType2>)).
                    FILTER(?subject_type IN (graph:<SubjectType>)).
                    }
            """,
        "get_context_of_Subject_IOC_anomalous_SubObj": """
        PREFIX graph: <https://<SourceDataset>.graph/<HOST>/>

        SELECT DISTINCT ?subject_type ?subject_attr ?predicate ?object_type ?object_attr ?timestamp
        FROM <https://<SourceDataset>.graph/<HOST>>
        WHERE {
            << ?subject ?predicate ?object >> graph:timestamp ?timestamp .
            
            # Subject properties
            ?subject graph:node-type ?subject_type .
            ?subject graph:node-attribute ?subject_attr .
            ?subject graph:is_Train "False" .
            OPTIONAL {?subject graph:anomalies ?subject_isAnomalous .}
        
            # Object properties
            ?object graph:node-type ?object_type .
            ?object graph:node-attribute ?object_attr .
            ?object graph:is_Train "False" .
            OPTIONAL { ?object graph:anomalies ?object_isAnomalous .}
        
            # Filter: Include triples with at least one anomalous node
            FILTER(BOUND(?subject_isAnomalous) || BOUND(?object_isAnomalous)).
            
            # IOC filter: Match on subject attributes
            FILTER(CONTAINS(lcase(?subject_attr), <IOC>)) .
        
            # Type filtering
            FILTER(?object_type IN (graph:<ObjectType1>, graph:<ObjectType2>)).
            FILTER(?subject_type IN (graph:<SubjectType>)).
        }
"""
    }
    for sparql_name, sparql_query in sparql_queries.items():
        sparql_queries[sparql_name] = sparql_query.replace("<DATASET>", host).replace("<SourceDataset>", SourceDataset)
    return sparql_queries

########################### DRAFT ######################################

# """,
# 'construct_subgraphs': """
# PREFIX graph: <https://DARPA_TC3.graph/<DATASET>/>
# SELECT ?s_uuid ?predicate ?o_uuid ?timestamp
# FROM <https://DARPA_TC3.graph/<DATASET>>
# WHERE{
#     << ?subject ?predicate ?object >> graph:timestamp ?timestamp .
#     ?subject graph:uuid ?s_uuid .
#     ?object graph:uuid ?o_uuid .
#     FILTER(?s_uuid IN <ANOMALIES_LIST> ) .
#     FILTER(?o_uuid IN <ANOMALIES_LIST> ) .
# }
# """,
# ##########################################################33