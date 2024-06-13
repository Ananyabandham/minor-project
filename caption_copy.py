import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, add, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
import pickle

# Define paths
image_folder_path = 'D:/minor-project/images/images_small'
csv_file_path = 'D:/minor-project/captions_small.csv'

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
images, captions = load_images(image_captions, image_folder_path)

# Tokenize the captions
tokenizer = Tokenizer()
all_captions = [caption for sublist in captions for caption in sublist]
tokenizer.fit_on_texts(['<start> <end>'] + all_captions)
vocab_size = len(tokenizer.word_index) + 1


tokenizer_path = 'D:/minor-project/tokenizer.pkl'  # Path to save the tokenizer
with open(tokenizer_path, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Convert captions to sequences
num_samples = len(images)
input_sequences = []
target_sequences = []
image_ids = []

for img_id, caption_list in enumerate(captions):
    for caption in caption_list:
        sequence = tokenizer.texts_to_sequences([caption])[0]
        if len(sequence) == 0:  # Skip empty sequences
            continue
        for i in range(1, len(sequence)):
            input_sequences.append(sequence[:i])
            target_sequences.append(sequence[i])
            image_ids.append(img_id)

# Check if input_sequences and target_sequences are not empty
if not input_sequences or not target_sequences:
    raise ValueError("No valid sequences found. Please check your captions and tokenization.")

# Pad input sequences
# max_length=15
max_length =max(len(seq) for seq in input_sequences)
# Save max_length to a file
max_length_path = 'D:/minor-project/max_length.pkl'
with open(max_length_path, 'wb') as handle:
    pickle.dump(max_length, handle, protocol=pickle.HIGHEST_PROTOCOL)

# print(max_length)
padded_input_sequences = pad_sequences(input_sequences, maxlen=max_length, padding='post')
# print(max_length)
# One-hot encode target sequences
one_hot_targets = np.zeros((len(target_sequences), vocab_size))
for i, target in enumerate(target_sequences):
    if target is not None:  # Ensure target is not None
        one_hot_targets[i, target] = 1

# Convert to numpy arrays
padded_input_sequences = np.array(padded_input_sequences)
image_features = np.array([images[i] for i in image_ids])


base_model = VGG16(include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze the layers
image_model = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))

# Extract features using the image model
extracted_image_features = image_model.predict(image_features)

# Ensure the shapes are consistent for training
assert extracted_image_features.shape[0] == padded_input_sequences.shape[0] == one_hot_targets.shape[0], "Mismatch in the number of samples among inputs and targets."

# Define the image input and features
image_input = Input(shape=(extracted_image_features.shape[1],))
image_features = Dense(512, activation='relu')(image_input)

# Define the caption model
caption_input = Input(shape=(max_length,))
caption_embedding = Embedding(vocab_size, 512)(caption_input)
caption_lstm = LSTM(512)(caption_embedding)

# Combine image features and caption model
decoder_input = add([image_features, caption_lstm])
decoder_hidden = Dense(512, activation='relu')(decoder_input)
output = Dense(vocab_size, activation='softmax')(decoder_hidden)

# Define the complete model
model = Model(inputs=[image_input, caption_input], outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# # Summary of the model
# model.summary()

# # Train the model
model.fit([extracted_image_features, padded_input_sequences], one_hot_targets, epochs=100)

model_path = 'D:/minor-project/trained_model.h5'  # You can choose a different path or filename
model.save(model_path)


