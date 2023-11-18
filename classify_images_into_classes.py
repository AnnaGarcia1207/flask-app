import torch
import clip
from PIL import Image
from datetime import datetime
import pandas as pd

    
def save_results_to_csv(df, batch_number):
    csv_filename = f"csvs/img_predictions_batch_{batch_number}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Saved results for batch {batch_number} to {csv_filename}")

def predict(df, categories, batch_size=100, save_every=1000):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load clip model
    model, preprocess = clip.load("ViT-B/32", device=device)

    text = clip.tokenize(categories).to(device)

    predicted_labels = []
    predicted_scores = []

    num_images = len(df)
    num_batches = (num_images + batch_size - 1) // batch_size

    processed_images = 0  # Counter for processed images

    batch_images_filenames = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_images)

        batch_images = []
        

        for i in range(start_idx, end_idx):
            local_img_path = df.loc[i, 'localImagePath']
            filename = df.loc[i, 'filename']

            try:
                image = preprocess(Image.open(local_img_path)).to(device)
                batch_images.append(image)
                batch_images_filenames.append(filename)

            except Exception as e:
                print(f"Error opening image: {df.loc[i, 'filename']} from {local_img_path}: {str(e)}")
                continue  # Skip this image and continue with the next one

        if not batch_images:
            continue  # Skip processing the batch if there are no valid images

        batch_images = [img for img in batch_images if img is not None]  # Remove None placeholders

        if not batch_images:
            continue  # Skip processing the batch if there are no valid images left

        batch_images = torch.stack(batch_images)

        with torch.no_grad():
            logits_per_image, logits_per_text = model(batch_images, text)

        predicted_indices = logits_per_image.argmax(dim=1)
        predicted_labels.extend([categories[idx] if idx < len(categories) else "N/A" for idx in predicted_indices])
        predicted_scores.extend(logits_per_image.tolist())

        processed_images += len(batch_images)  # Increment the counter

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        print(f"[{current_time}] Processed batch {batch_idx + 1}/{num_batches} (size: {len(batch_images)})")

        # Check if it's time to save results
        if processed_images >= save_every:
            save_results_to_csv(pd.DataFrame({'filename': batch_images_filenames, 'predicted_class': predicted_labels, 'scores': predicted_scores}), batch_idx + 1)
            # Reset counters
            processed_images = 0

    return predicted_labels, predicted_scores



def main():
    class_prediction = ['a photo of an algorithm',
                    'a photo of an architecture diagram',
                    'a photo of a bar chart',
                    'a photo of a boxplot',
                    'a photo of a confusion matrix',
                    'a photo of a graph',
                    'a photo of a line chart',
                    'a photo of a map',
                    'a photo of a pareto',
                    'a photo of a venn diagram',
                    'a photo of a word cloud',
                    'a photo of a natural image',
                    'a photo of a neural networks',
                    'a photo of an NLP (natural language processing) grammar',
                    'a photo of a pie chart',
                    'a photo of a scatter plot',
                    'a photo of a screenshot',
                    'a photo of a tables',
                    'a photo of a tree graph']

    csv = "image_data.csv"
    df = pd.read_csv(csv)

    predicted_labels, predicted_scores = predict(df, class_prediction, 100, 1000)

    # Add lists to the dataframe
    df['predicted_class'] = predicted_labels
    df['scores'] = predicted_scores

    df.to_csv("img_predictions.csv", index=False)

    print("done")


if __name__ == "__main__":
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("Starting Time: " + start_time)
    main()
    end_time = start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("Ending Time : " + end_time)


