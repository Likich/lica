
#coded lemmas is EXTERNAL file, it was created outside using external GPU
def make_graph_llm(method):
    import ast
    import pandas as pd
    import json
    import warnings
    import random
    from qualcode import combine_data
    from qualcode_preload import process_doc_topics
    combine_data('df_raw.csv', 'clusters_documents.csv')
    
    warnings.filterwarnings('ignore')
    # data = pd.read_csv("coded_lemmas_new.csv")
    doc_topic = pd.read_csv('doc_topics.csv')

    code_dict = {}
    topics = doc_topic['Topic'].unique()

    number_of_colors = 1000
    color_pallette = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                for i in range(number_of_colors)]


    if method == 'Llama3':
        process_doc_topics('doc_topics.csv', 'coded_lemmas_new.csv', model_name = "Likich/llama3-finetune-qualcoding_1000_prompt1_dot")
        data = pd.read_csv("coded_lemmas_new.csv")
        data_filtered = data[data['Extracted codes from lemmas'].str.len() <= 30]
        for topic in topics:
            topic_data = data_filtered[data_filtered['Topic'] == topic]
            code_counts = topic_data['Extracted codes from lemmas'].explode().value_counts().to_dict()
            code_list = list(code_counts.items())
            code_dict[topic] = code_list

        print(code_list)
        nodes = []
        links = []
        for key, values in code_dict.items():
            if key == -2:
                pass
            else:
                for tupl in values:
                    nodes.append({'name': tupl[0]})
        nodes_str = []
        for i in nodes:
            nodes_str.append(str(i))
        nodes_str = list(set(nodes_str))
        clear_nodes = []
        for i in nodes_str:
            clear_nodes.append(ast.literal_eval(i))
        node_namings_only = []
        for node in clear_nodes:
            for key, node_name in node.items():
                node_namings_only.append(node_name)
        links = []

        for node in clear_nodes:
            for key, node_name in node.items():
                c=random.choice(color_pallette)
                for k,v in code_dict.items():
                    if k == -1:
                        pass
                    else:
                        for tupl_id in range(len(v)):
                            if  v[0][0] == node_name:
                                links.append( {"source":node_namings_only.index(v[0][0]), 'target': node_namings_only.index(v[tupl_id][0]), 'color':c})         

        clear_links = []
        for i in range(len(links)):
            compare = []
            for k,v in links[i].items():
                compare.append(v)
            if compare[0] == compare[1]:
                pass
            else:
                clear_links.append(links[i])

        dict_json = {'links': clear_links, 'nodes': clear_nodes}
        with open('concept_graph.json', 'w', encoding='utf-8') as file:
            json.dump(dict_json, file, ensure_ascii=False)

    elif method == 'Falcon':
        process_doc_topics('doc_topics.csv', 'coded_lemmas_new.csv', model_name = "Likich/falcon-finetune-qualcoding_1000_prompt1_dot")
        data = pd.read_csv("coded_lemmas_new.csv")
        data_filtered = data[data['Extracted codes from lemmas'].str.len() <= 30]
        for topic in topics:
            topic_data = data_filtered[data_filtered['Topic'] == topic]
            code_counts = topic_data['Extracted codes from lemmas'].explode().value_counts().to_dict()
            code_list = list(code_counts.items())
            code_dict[topic] = code_list

        print(code_list)
        nodes = []
        links = []
        for key, values in code_dict.items():
            if key == -2:
                pass
            else:
                for tupl in values:
                    nodes.append({'name': tupl[0]})
        nodes_str = []
        for i in nodes:
            nodes_str.append(str(i))
        nodes_str = list(set(nodes_str))
        clear_nodes = []
        for i in nodes_str:
            clear_nodes.append(ast.literal_eval(i))
        node_namings_only = []
        for node in clear_nodes:
            for key, node_name in node.items():
                node_namings_only.append(node_name)
        links = []

        for node in clear_nodes:
            for key, node_name in node.items():
                c=random.choice(color_pallette)
                for k,v in code_dict.items():
                    if k == -1:
                        pass
                    else:
                        for tupl_id in range(len(v)):
                            if  v[0][0] == node_name:
                                links.append( {"source":node_namings_only.index(v[0][0]), 'target': node_namings_only.index(v[tupl_id][0]), 'color':c})         

        clear_links = []
        for i in range(len(links)):
            compare = []
            for k,v in links[i].items():
                compare.append(v)
            if compare[0] == compare[1]:
                pass
            else:
                clear_links.append(links[i])

        dict_json = {'links': clear_links, 'nodes': clear_nodes}
        with open('concept_graph.json', 'w', encoding='utf-8') as file:
            json.dump(dict_json, file, ensure_ascii=False)       
    
    print('Everything is ready. Please run the command: python3 -m http.server and go to your localhost.')
    return clear_nodes, clear_links


  
def translate_to_eng(clear_nodes, clear_links):
    from googletrans import Translator
    translator = Translator()
    import json

        #Translation module        
    eng_nodes = []
    for tupl in clear_nodes:
        for key, items in tupl.items():
            smol_dict = {'name': translator.translate(items).text}
            eng_nodes.append(smol_dict)

    dict_json = {'links': clear_links, 'nodes': eng_nodes}
    with open('concept_graph.json', 'w', encoding='utf-8') as file:
        json.dump(dict_json, file, ensure_ascii=False)
     
    print('Everything is ready. Please run the command: python3 -m http.server and go to your localhost.')
  
       