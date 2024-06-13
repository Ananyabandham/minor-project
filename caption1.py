import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, add, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
import pickle

# def preprocess_image(image_path, target_size=(224, 224)):
#     img = load_img(image_path, target_size=target_size)
#     img_array = img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = preprocess_input(img_array) / 255.0
#     return img_array


# def generate_caption(model, tokenizer, image, genre, max_length,method='greedy'):
#     # img = preprocess_image(image_path)
#     # img_features = image_model.predict(img)
#     caption = '<start>'
    
#     for _ in range(max_length):
#         seq = tokenizer.texts_to_sequences([caption])[0]
#         seq = pad_sequences([seq], maxlen=max_length, padding='post')
#         genre_seq=np.array([genre])
#         yhat = model.predict([image, seq,genre_seq], verbose=0)
#         if method == 'greedy':
#             yhat = np.argmax(yhat)
#         elif method == 'sample':
#             yhat = np.random.choice(range(len(yhat[0])), p=yhat[0])
#         word = tokenizer.index_word.get('yhat',None)
#         if word is None:
#             break
#         caption += ' ' + word
#         if word == 'end':
#             break
#     final_caption = caption.split()
#     final_caption = final_caption[1:-1]  # Remove 'startseq' and 'endseq'
#     return ' '.join(final_caption)


# tokenizer_path = 'D:/minor-project/tokenizer.pkl'  # Update this path if necessary
# with open(tokenizer_path, 'rb') as handle:
#     tokenizer = pickle.load(handle)

# # Load the trained model
# model_path = 'D:/minor-project/trained_model.h5'  # Update this path if necessary
# model = load_model(model_path)
# model.compile(optimizer='adam', loss='categorical_crossentropy')

# base_model = VGG16(include_top=False, weights='imagenet')
# base_model.trainable = False  # Freeze the layers
# image_model = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))



# # Load max_length from the file
# max_length_path = 'D:/minor-project/max_length.pkl'
# with open(max_length_path, 'rb') as handle:
#     max_length = pickle.load(handle)

# genre_to_int_path= 'D:/minor-project/genre_to_int.pkl'
# with open(genre_to_int_path, 'rb') as handle:
#     genre_to_int = pickle.load(handle)

# # max_length=24


# def caption_image(image_path,genre_label):
#     # Preprocess the image
#     image = preprocess_image(image_path)
#     # Extract features using the image model
#     image_features = image_model.predict(image, verbose=0)
#     genre=genre_to_int[genre_label]
#     # Generate caption using the trained model
#     caption = generate_caption(model, tokenizer, image_features, genre, max_length,method='sample')
#     return caption

# # Example usage
# image_path = 'D:/minor-project/images/images_small/12830823_87d2654e31.jpg'  # Update this path
# genre_label='blog'
# caption = caption_image(image_path,genre_label)
# print("Generated Captions:", caption)


def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array) / 255.0
    return img_array

def generate_caption(model, tokenizer, image, genre, max_length,num_captions=5,temperature=1.0):
    # img = preprocess_image(image_path)
    # img_features = image_model.predict(img)
    def sample(preds,temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds + 1e-8) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)
    
    captions=[]
    for _ in range(num_captions):
        caption = '<start>'
        for _ in range(max_length):
            seq = tokenizer.texts_to_sequences([caption])[0]
            seq = pad_sequences([seq], maxlen=max_length, padding='post')
            genre_seq=np.array([genre])
            yhat = model.predict([image, seq, genre_seq], verbose=0)[0]
            yhat = sample(yhat,temperature)
            word = tokenizer.index_word.get(yhat,None)
            if word is None:
                break
            caption += ' ' + word
            if word == 'end':
                break
        final_caption = caption.split()
        final_caption = final_caption[1:-1]  # Remove 'startseq' and 'endseq'
        captions.append(' '.join(final_caption))
    return captions
    # return caption


tokenizer_path = 'D:/minor-project/tokenizer.pkl'  # Update this path if necessary
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the trained model
model_path = 'D:/minor-project/trained_model.h5'  # Update this path if necessary
model = load_model(model_path)
model.compile(optimizer='adam', loss='categorical_crossentropy')

base_model = VGG16(include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze the layers
image_model = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))



# Load max_length from the file
max_length_path = 'D:/minor-project/max_length.pkl'
with open(max_length_path, 'rb') as handle:
    max_length = pickle.load(handle)

genre_to_int_path= 'D:/minor-project/genre_to_int.pkl'
with open(genre_to_int_path, 'rb') as handle:
    genre_to_index = pickle.load(handle)

# max_length=24


def caption_image(image_path,genre_label,num_captions=5,temperature=1.0):
    # Preprocess the image
    image = preprocess_image(image_path)
    # Extract features using the image model
    image_features = image_model.predict(image, verbose=0)
    genre=genre_to_index[genre_label]
    # Generate caption using the trained model
    captions = generate_caption(model, tokenizer, image_features, genre, max_length,num_captions,temperature)
    return captions

# Example usage
image_path = 'D:/minor-project/images/images_small/667626_18933d713e.jpg'  # Update this path
genre_label='fun'
captions = caption_image(image_path,genre_label,num_captions=5,temperature=0.7)
for i,caption in enumerate(captions):
    print(f"Generated Caption {i+1}: {caption}")