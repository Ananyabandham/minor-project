import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Define paths
image_folder_path = 'D:\minor-project\images\images_small'
csv_file_path = 'D:\minor-project\captions_small.csv'

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file_path)

# Dictionary to hold image filenames and their corresponding captions
image_captions = {}

# Iterate over the rows of the DataFrame
for index, row in df.iterrows():
    image_name = row['img_name'].strip()
    caption = row['caption'].strip()
    if image_name in image_captions:
        image_captions[image_name].append(caption)
    else:
        image_captions[image_name] = [caption]

# Function to load images and preprocess them
def load_images(image_captions, image_folder_path, target_size=(224, 224)):
    images = []
    captions = []
    for image_name, caption_list in image_captions.items():
        # Construct the full path to the image
        image_path = os.path.join(image_folder_path, image_name)
        try:
            # Load the image
            img = load_img(image_path, target_size=target_size)
            # Convert the image to a numpy array
            img_array = img_to_array(img)
            # Normalize the image data to 0-1 range
            img_array = img_array / 255.0
            images.append(img_array)
            captions.append(caption_list)
        except FileNotFoundError:
            print(f"Image {image_name} not found in {image_folder_path}. Skipping.")
        except OSError as e:
            print(f"Error loading image {image_name}: {e}")
    return np.array(images), captions

# Load and preprocess images and get corresponding captions
image_data, captions = load_images(image_captions, image_folder_path)

# Check the shape of the loaded data
# print(image_data)
# print(f"Loaded {image_data.shape[0]} images with shape {image_data.shape[1:]}")
# for i, caption_list in enumerate(captions):
#     print(f"Image {i} has {len(caption_list)} captions.")

print(image_captions)