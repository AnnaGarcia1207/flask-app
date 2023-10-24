import requests
import json

URL = "http://localhost:9200/scifig-pilot/_search"

# Structuring the query to match documents where the label field is "algorithms"
q = {
    "query": {
        "match": {
            "label": "algorithms"
        }
    }
}

headers = {
    'Content-Type': 'application/json'
}

# Sending a POST request with the query
r = requests.post(url=URL, headers=headers, json=q)

data = r.json()

# Printing the response data
print(json.dumps(data, indent=4))
