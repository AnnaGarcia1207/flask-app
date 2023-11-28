import os
import fnmatch
import pandas as pd

# Specify the directory where you want to collect image paths
folder_path = 'D:\Scifig-datasets\SciFig\png'

# Function to collect all image file paths in the specified folder
def find_image_files(directory):
    image_data = []

    for root, _, files in os.walk(directory):
        for filename in files:
            # Check if the file has a common image file extension (you can add more if needed)
            if filename.lower().endswith(('.png')):
                local_image_path = os.path.join(root, filename)
                image_path = 'images/png/' + filename
                image_data.append((filename, local_image_path, image_path))

    return image_data

# Call the function to get a list of image filename and path tuples
image_data = find_image_files(folder_path)

# Create a DataFrame from the image data
df = pd.DataFrame(image_data, columns=['filename', 'localImagePath', 'image_path'])

# Export the DataFrame to a CSV file
csv_file_path = 'image_data.csv'
df.to_csv(csv_file_path, index=False)

print(f'DataFrame saved to {csv_file_path}')

