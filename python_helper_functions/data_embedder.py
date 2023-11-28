import torch
from PIL import Image
import pandas as pd
import numpy as np
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import time

def get_model_info(model_ID, device):
    # Load the model to the specified device
    model = CLIPModel.from_pretrained(model_ID).to(device)

    # Get the processor and tokenizer
    processor = CLIPProcessor.from_pretrained(model_ID)
    tokenizer = CLIPTokenizer.from_pretrained(model_ID)

    return model, processor, tokenizer

def get_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image

def get_single_image_embeddings(input_image, index):
    image = processor(
        text=None,
        images=input_image,
        return_tensors="pt"
    )["pixel_values"].to(device)

    embeddings = model.get_image_features(image)
    numpy_embeddings = embeddings.cpu().detach().numpy()

    if index % 1000 == 0:
        # Print an update with a timestamp when the index is a multiple of 1000
        current_time = time.strftime("%Y-%m-%d %H:%M", time.localtime())
        print(f"Processed {index} images at {current_time}")

    return numpy_embeddings

def get_multiple_image_embeddings(df, custom_col_name):
    df["img_embeddings"] = df.apply(lambda row: get_single_image_embeddings(row[custom_col_name], row.name), axis=1)
    df["img_embeddings"] = df["img_embeddings"].apply(lambda x: x.tolist())
    return df

def main():
    df = pd.read_csv("C:/Users/annaa/workspace/flask/merged_df_with_text_embeddings.csv")

    df_img_embeddings = df[['filename', 'localImagePath']]

    df_img_embeddings['image'] = df_img_embeddings['localImagePath'].apply(get_image)
    print("Finished converting Images into RGB\n")

    df_img_embeddings = get_multiple_image_embeddings(df_img_embeddings, 'image')
    df_img_embeddings.to_csv("df_img_embeddings.csv")
    print("Finished Embeddings of Images\n")

if __name__ == "__main__":
    # Set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_ID = "openai/clip-vit-base-patch32"

    model, processor, tokenizer = get_model_info(model_ID, device)
    main()
