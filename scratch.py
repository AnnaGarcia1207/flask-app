from PIL import Image
import pandas as pd
import os
from elasticsearch import Elasticsearch, exceptions
import shutil

import pandas as pd
import shutil
import datetime

# Load your DataFrame with image paths
df = pd.read_csv("dataframes\scifig_embeddings_scores.csv")

# Specify the destination folder where you want to copy the images
destination_folder = "D:\images\png"

batch_size = 100

for batch_start in range(0, len(df), batch_size):
    batch_df = df.iloc[batch_start:batch_start + batch_size]
    for index, row in batch_df.iterrows():
        source_path = row['external_drive_path']
        destination_path = os.path.join(destination_folder, os.path.basename(source_path))
        
        # Copy the image from source to destination
        shutil.copy(source_path, destination_path)
    
    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:")
    
    print(f"[{timestamp}]: Processed {batch_start + len(batch_df)} rows")


# import pandas as pd
# import shutil
# import datetime
# import os
# import zipfile

# # Load your DataFrame with image paths
# df = pd.read_csv("D:\\images\\dataframes\\scifig_embeddings_scores.csv")

# # Specify the destination folder where you want to copy the images
# destination_folder = "D:\\images\\imgs"
# batch_size = 1000

# def zip_files(files, zip_name):
#     with zipfile.ZipFile(zip_name, 'w') as zipf:
#         for file in files:
#             zipf.write(file, os.path.basename(file))

# for batch_start in range(0, len(df), batch_size):
#     batch_df = df.iloc[batch_start:batch_start + batch_size]
#     files_to_zip = []

#     for index, row in batch_df.iterrows():
#         source_path = row['external_drive_path']
#         destination_path = os.path.join(destination_folder, os.path.basename(source_path))
#         shutil.copy(source_path, destination_path)
#         files_to_zip.append(destination_path)
    
#     # Create a zip file for the batch
#     zip_name = f"{destination_folder}\\batch_{batch_start // batch_size + 1}.zip"
#     zip_files(files_to_zip, zip_name)

#     # Optional: Delete the images after zipping
#     # for file in files_to_zip:
#     #     os.remove(file)

#     timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     print(f"[{timestamp}]: Processed and zipped batch {batch_start // batch_size + 1}")

# Remember to handle any exceptions and errors in real scenarios.



# df = pd.read_csv("dataframes\scifig_embeddings_scores.csv")

# result = df.head(1)

# destination_folder = "D:\images\png"

# source_path = "D:\Scifig-datasets\SciFig\png\P14-1071.pdf-Figure4.png" #result['external_drive_path']

# destination_path = os.path.join(destination_folder, os.path.basename(source_path))

# # Copy the image from source to destination
# shutil.copy(source_path, destination_path)

# print("DONE")


# result.to_csv("text.txt", index=False, sep='\t')

# def file_exists(file_path):
#     return os.path.exists(file_path)


# path = "C:/Users/annaa/workspace/flask/localImagePath.csv"

# df = pd.read_csv(path)

# df['file_exists'] = df['localImagePath'].apply(file_exists)

# df.to_csv("df_1.csv", index=False)

# print(df.columns)

# path = "D:\Scifig-datasets\SciFig\png\O04-1016.pdf-Figure2.png"

# img = Image.open(path)
# print(img)

# ---------------------------------------------------------
# import os
# import shutil

# # Source directory (D:/Documents/png)
# source_directory = "D:\\Scifig-datasets\\SciFig\\png"

# # Destination directory (C:/Documents/png)
# destination_directory = "C:\\Users\\annaa\\Documents\\ODU\\CS734\\png"

# # Number of images to move
# num_images_to_move = 10

# # Ensure the source directory exists
# if not os.path.exists(source_directory):
#     print(f"Source directory '{source_directory}' does not exist.")
# else:
#     # List files in the source directory
#     files = os.listdir(source_directory)

#     # Filter the list to include only image files (you can adjust the extensions as needed)
#     image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

#     # Ensure the destination directory exists, create it if necessary
#     if not os.path.exists(destination_directory):
#         os.makedirs(destination_directory)

#     # Move the specified number of images
#     for i, image_file in enumerate(image_files[:num_images_to_move]):
#         source_path = os.path.join(source_directory, image_file)
#         destination_path = os.path.join(destination_directory, image_file)
#         shutil.move(source_path, destination_path)
#         print(f"Moved image {i + 1}/{num_images_to_move} from '{source_path}' to '{destination_path}'.")

# print("Done")


# ---------------------------------------------------------------------

# config_file_path = "config.json"

# with open(config_file_path, 'r') as file:
#     config = json.load(file)

# CLOUD_ID = config['elasticsearch']['cloud_id']
# ELASTIC_USERNAME = config['elasticsearch']['username']
# ELASTIC_PASSWORD = config['elasticsearch']['password']
# CERT_FINGERPRINT = config['elasticsearch']['cert_fingerprint']
# api_creds=(config['elasticsearch']['id'], config['elasticsearch']['api_key'])

# es = Elasticsearch(['https://xxxxxxxxxxxxxxxxxxxxxxxxxxx.us-east-1.aws.found.io:9243'], api_key=(api['id'],api['api_key']))
# try:    
#     client = Elasticsearch(
#         cloud_id=CLOUD_ID,
#         basic_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD),
#         ssl_assert_fingerprint=CERT_FINGERPRINT
#     )
#     print(client.info())
# except ConnectionError as e:
#      print("Connection Error ElasticSearch: ", e)

# if client.ping():
#       print("DOOONEEE!")
# else:
#       print("NOOOOOOOOOOOOO Check if elasticsearch is up and running")

# ---------------------------------------------------------------------

