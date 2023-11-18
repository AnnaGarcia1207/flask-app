from elasticsearch import Elasticsearch, exceptions
from indexMapping import indexMapping
import pandas as pd
import ast

import numpy as np
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import torch



def main():
    
    try:
        elastic_search = Elasticsearch("https://localhost:9200",
        basic_auth=("elastic", "3_q9X3DGNLs7_9jT3U-h"),
        ca_certs="C:/Users/annaa/elasticsearch-8.10.4-windows-x86_64/elasticsearch-8.10.4/config/certs/http_ca.crt")
    except ConnectionError as e:
        print("Connection Error ElasticSearch: ", e)
    
    if elastic_search.ping():
        None
    else:
        print("Check if elasticsearch is up and running")
    

    # read the csv
    df = pd.read_csv("C:/Users/annaa/workspace/flask/embeddingsv6.csv")
    # convert the str type vectors to array
    df['text_embeddings'] = df['text_embeddings'].apply(string_to_list)
    df['img_embeddings'] = df['img_embeddings'].apply(string_to_list)

    df['text_embeddings'] = df['text_embeddings'].apply(flatten_embedding)
    df['img_embeddings'] = df['img_embeddings'].apply(flatten_embedding)
    # print(df.columns)

    
    # create index in elastic search
    index_name = "scifig-pilot"

    # # Delete the index if it already exists
    if elastic_search.indices.exists(index=index_name):
        elastic_search.indices.delete(index=index_name)
        print(f"Index '{index_name}' deleted.")


    elastic_search.indices.create(index=index_name, mappings=indexMapping)

    # # turn dataframe into dictionary
    data_dict = df.to_dict("records")
    # print(data_dict[0])

    try:
        for data in data_dict:
            # print(data['id'])
            elastic_search.index(index=index_name, document=data)
    except Exception as e:
        print(f"An error occurred while indexing the document: {e}")
    
    print(elastic_search.count(index=index_name))

def string_to_list(embedding_str):
    try:
        return ast.literal_eval(embedding_str)
    except (ValueError, SyntaxError):  # catching errors like malformatted strings
        return embedding_str   
    
def flatten_embedding(embedding):
    if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
        return embedding[0]
    return embedding



if __name__ == "__main__":
    main()
    print("Data is loaded!")
    # Set the device
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # model_ID = "openai/clip-vit-base-patch32"

    # model, processor, tokenizer = get_model_info(model_ID, device)

    # index_name = "scifig-pilot"

    # es = connect()

    # process_text_query(es, "bar graph")