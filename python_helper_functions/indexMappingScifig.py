indexMappingScifig = { 
    "properties": {
        "Id": {
            "type": "keyword"
        },
        "filename": {
            "type": "text",
            "index": "true"
        },
        "label": {
            "type": "text",
            "index": "true"
        },
        "scores": {
            "type": "keyword"
        },
        "confidence": {
            "type": "keyword"
        },
        "image_path": {
            "type": "keyword"
        },
        "external_drive_path": {
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
        }
    }
}
