import os
import numpy as np
from PIL import Image
import pandas as pd
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, add, Dropout, GlobalAveragePooling2D
# Load the dataset (images and captions)
# Assuming you have a folder 'dataset' with images and a file 'captions.txt' containing image filenames and captions

data = pd.read_csv('D:/minor-project/captions_small.csv')

# Split the data into images and captions
images = []
captions = []
for index, row in data.iterrows():
    img_name = row['img_name']
    caption = row['caption']
    img_path = os.path.join('images/images_small', img_name)
    img = Image.open(img_path)
    img = img.resize((224, 224), Image.ANTIALIAS)
    img = np.array(img) / 255.0
    images.append(img)
    captions.append(caption)

images = np.array(images)

# Tokenize captions
tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(caption.split()) for caption in captions)

# Convert captions to sequences
sequences = tokenizer.texts_to_sequences(captions)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Load the VGG16 model
vgg16 = VGG16(include_top=False, weights='imagenet')
vgg16.trainable = False  # Freeze the layers


# Create a new model by removing the final classification layer
feature_extractor= Model(inputs=vgg16.input, outputs=GlobalAveragePooling2D()(vgg16.output))

# Extract features from images
image_features = feature_extractor.predict(images, verbose=1)



# Define the input shapes
inputs1 = Input(shape=(1,))
inputs = Dense(512, activation='relu')(inputs1)
inputs2 = Input(shape=(512,))

# Embedding layer
emb = Embedding(vocab_size, 512, mask_zero=True)(inputs1)

# LSTM layer
lstm = LSTM(512)(emb)
lstm = Dropout(0.5)(lstm)

# Merge image features and LSTM output
merged = add([lstm, inputs2])

# Dense layer for prediction
output = Dense(vocab_size, activation='softmax')(merged)

# Create the model
model = Model(inputs=[inputs1, inputs2], outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# One-hot encode the padded sequences
one_hot_targets = np.zeros((padded_sequences.shape[0], max_length, vocab_size))
for i, seq in enumerate(padded_sequences):
    for j, word_idx in enumerate(seq):
        if word_idx > 0:
            one_hot_targets[i, j, word_idx] = 1

# Train the model
model.fit([padded_sequences, image_features], one_hot_targets, epochs=10, batch_size=32)

def generate_caption(image_path, max_length=max_length):
    # Preprocess the image
    img = Image.open(image_path)
    img = img.resize((224, 224), Image.ANTIALIAS)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Extract image features
    image_features = feature_extractor.predict(img)

    # Start token for caption generation
    start_token = tokenizer.word_index['startseq']
    caption = [start_token]

    for _ in range(max_length):
        # Prepare the input sequence
        seq = pad_sequences([caption], maxlen=max_length, padding='post')

        # Predict the next word
        yhat = model.predict([seq, image_features])
        yhat = np.argmax(yhat)

        # Map the prediction back to the word
        word = tokenizer.index_word.get(yhat, 'endseq')

        # Stop if the 'endseq' token is predicted
        if word == 'endseq':
            break

        # Append the predicted word to the caption
        caption.append(yhat)

    # Remove the 'startseq' token
    caption = caption[1:]

    # Convert the caption back to text
    caption = ' '.join([tokenizer.index_word.get(idx, '?') for idx in caption])

    return caption

# Example usage
image_path = 'images/images_small/667626_18933d713e.jpg'
caption = generate_caption(image_path,max_length)
print(f"Generated caption: {caption}")