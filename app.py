from flask import Flask, render_template, request
from elasticsearch import Elasticsearch, exceptions
import pandas as pd
import numpy as np
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import torch
import json

app = Flask(__name__)

elastic_search = Elasticsearch(
    "https://localhost:9200",
    basic_auth=("elastic", "3_q9X3DGNLs7_9jT3U-h"),
    ca_certs="C:/Users/annaa/elasticsearch-8.10.4-windows-x86_64/elasticsearch-8.10.4/config/certs/http_ca.crt"
)

def get_model_info(model_ID, device):
      # save the model to device
  model = CLIPModel.from_pretrained(model_ID).to(device)

  # Get the processor
  processor = CLIPProcessor.from_pretrained(model_ID)

  # get the tokenizer
  tokenizer = CLIPTokenizer.from_pretrained(model_ID)

  return model, processor, tokenizer

# Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"

model_ID = "openai/clip-vit-base-patch32"

model, processor, tokenizer = get_model_info(model_ID, device)

index_name = "scifig-pilot"

def search_text_embeddings(embedding_vector, k):
    source_fields = ['id', 'label', 'location']
    query = {
        "field": "text_embeddings",
        "query_vector": embedding_vector,
        "k": k,
        "num_candidates": 100
    }
    try:
        response = elastic_search.knn_search(index="scifig-pilot", knn=query, source=source_fields)
        print(response)
        return response['hits']['hits']

    except exceptions.RequestError as e:
        print(f"Error: {e.info['error']['root_cause'][0]['reason']}")
    


def process_text_query(input_query, k):
   inputs = tokenizer(input_query, return_tensors="pt").to(device)
   text_embeddings = model.get_text_features(**inputs)
   embeddings_as_numpy = text_embeddings.cpu().detach().numpy().reshape(-1)

   # Dictionary containing : ['id', 'label', 'location']
   search_results = search_text_embeddings(embeddings_as_numpy,k)

#    results_json = json.dumps(search_results)

   results_map = {}

   for hit in search_results:
        hit_id = hit['_id']
        score = hit['_score'] *100
        label = hit['_source']['label']
        location = hit['_source']['location']
        name = location.split('/')[-1]


        results_map[hit_id] = {
            "score": f'{score:.2f}%',
            "name": name,
            "label": label,
            "location": location 
        }

   return results_map
   
   

@app.route('/', methods=['GET','POST'])
def index():
    # grab the text query
    results_dict = None
    if request.method =='GET':
        text_query = request.form.get('searchQuery')
        k = request.form.get('k')

        # passed into the process_text_query and k
        results_dict = process_text_query(text_query, k)
    
    return render_template("index.html", results=results_dict)

if __name__=="__main__":
    app.run(debug=True)