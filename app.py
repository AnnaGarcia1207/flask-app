from flask import Flask, render_template, request
from elasticsearch import Elasticsearch, exceptions
import pandas as pd
import numpy as np
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import torch
import json
from PIL import Image

app = Flask(__name__)

config_file_path = "config.json"

with open(config_file_path, 'r') as file:
    config = json.load(file)

CLOUD_ID = config['elasticsearch']['cloud_id']
ELASTIC_USERNAME = config['elasticsearch']['username']
ELASTIC_PASSWORD = config['elasticsearch']['password']
CERT_FINGERPRINT = config['elasticsearch']['cert_fingerprint']


try:
    elastic_search = Elasticsearch(
        cloud_id=CLOUD_ID,  
        basic_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD),
        ssl_assert_fingerprint=CERT_FINGERPRINT
        )
except ConnectionError as e:
    print("Connection Error ElasticSearch: ", e)

if elastic_search.ping():
    print("Connected to elasticsearch successfully!")
else:
    print("EXCEPTION: Check if elasticsearch is up and running")


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

def create_results_map(search_results):
    results = {}
    for hit in search_results:
        
        hit_id = hit['_id']
        score = hit['_score'] *100
        label = hit['_source']['label']
        location = hit['_source']['location']
        name = location.split('/')[-1]


        results[hit_id] = {
            "score": f'{score:.2f}%',
            "name": name,
            "label": label,
            "location": location 
        }
    
    return results

def search_embeddings(embedding_vector, k, embedding_type):
    source_fields = ['id', 'label', 'location']
    query = {
        "field": embedding_type,
        "query_vector": embedding_vector,
        "k": k,
        "num_candidates": 1001
    }
    try:
        response = elastic_search.knn_search(index="scifig-pilot", knn=query, source=source_fields)
        print(response)
        return response['hits']['hits']

    except exceptions.RequestError as e:
        print(f"Error: {e.info['error']['root_cause'][0]['reason']}")
    


def process_text_query(input_query, k=99):
   inputs = tokenizer(input_query, return_tensors="pt").to(device)
   text_embeddings = model.get_text_features(**inputs)
   embeddings_as_numpy = text_embeddings.cpu().detach().numpy().reshape(-1)

   # Dictionary containing : ['id', 'label', 'location']
   search_results = search_embeddings(embeddings_as_numpy,k, "text_embeddings")

   results_map = create_results_map(search_results)

   return results_map

def process_image_query(input, k=99):
    input_image = Image.open(input).convert("RGB")
    image = processor(
    text = None,
    images = input_image,
    return_tensors="pt"
    )["pixel_values"].to(device)

    embeddings = model.get_image_features(image)

    numpy_embeddings = embeddings.cpu().detach().numpy().reshape(-1)


    search_results = search_embeddings(numpy_embeddings,k, "img_embeddings")
    results_dict = create_results_map(search_results)

    return results_dict

   


@app.route('/', methods=['GET','POST'])
def index():
    # grab the text query
    results_dict = None
    if request.method =='POST':
        text_query = request.form.get('searchQuery')
        k = 99 #request.form.get('k')

        # passed into the process_text_query and k
        results_dict = process_text_query(text_query, k)
    
    return render_template("index.html", results=results_dict)


@app.route('/upload', methods=['POST'])
def upload():
    results_dict = None
    if 'imageUpload' in request.files:
        image_query = request.files['imageUpload']

        results_dict = process_image_query(image_query, 99)
    
    return render_template("index.html", results=results_dict)

if __name__=="__main__":
    app.run(debug=True)