indexMapping = { 
    "properties": {
        "id": {
            "type": "integer"
        },
        "label": {
            "type": "text"
        },
        "sci_fig": {
            "type": "keyword"
        },
        "text_embeddings": {
            "type": "dense_vector",
            "dims": 512,
            "index": "true",
            "similarity": "cosine"
        },
        "img_embeddings": {
            "type": "dense_vector",
            "dims": 512,
            "index": "true",
            "similarity": "cosine"
        },
        "location": {
            "type": "text"
        },
        "img_loc": {
            "type": "text"
        }
    }
}
