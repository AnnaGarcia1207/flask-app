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


# def connect():
#     try:
#         elastic_search = Elasticsearch("https://localhost:9200",
#         basic_auth=("elastic", "3_q9X3DGNLs7_9jT3U-h"),
#         ca_certs="C:/Users/annaa/elasticsearch-8.10.4-windows-x86_64/elasticsearch-8.10.4/config/certs/http_ca.crt")
#     except ConnectionError as e:
#         print("Connection Error ElasticSearch: ", e)
    
#     if elastic_search.ping():
#         return elastic_search
#     else:
#         print("Check if elasticsearch is up and running")

# def get_model_info(model_ID, device):
#       # save the model to device
#   model = CLIPModel.from_pretrained(model_ID).to(device)

#   # Get the processor
#   processor = CLIPProcessor.from_pretrained(model_ID)

#   # get the tokenizer
#   tokenizer = CLIPTokenizer.from_pretrained(model_ID)

#   return model, processor, tokenizer

# def search_text_embeddings(elastic_search, embedding_vector, k):
#     source_fields = ['id', 'label', 'location']
#     query = {
#         "field": "text_embeddings",
#         "query_vector": embedding_vector,
#         "k": 2,
#         "num_candidates": 10
#     }
#     try:
#         response = elastic_search.knn_search(index="scifig-pilot", knn=query, source=source_fields)
#         print(response)
#         return response['hits']['hits']

#     except exceptions.RequestError as e:
#         print(e.message)
#         print("\n")
#         print(e)
#         print(f"Error: {e.info['error']['root_cause'][0]['reason']}")
    

# def process_text_query(es, input_query):
#    inputs = tokenizer(input_query, return_tensors="pt").to(device)
#    text_embeddings = model.get_text_features(**inputs)
#    embeddings_as_numpy = text_embeddings.cpu().detach().numpy().reshape(-1)

#    search_results = search_text_embeddings(es, embeddings_as_numpy,2)


#    return search_results


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