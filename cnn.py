from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16

# Load the VGG16 model pre-trained on ImageNet and remove the top layer
image_model = VGG16(include_top=False, weights='imagenet')
image_model.trainable = False  # Freeze the layers
image_model = Model(inputs=image_model.input, outputs=GlobalAveragePooling2D()(image_model.output))

# Define the image feature extractor model
image_input = Input(shape=(224, 224, 3))
image_features = image_model(image_input)

# Define the caption model
caption_input = Input(shape=(max_length,))
caption_embedding = Embedding(vocab_size, 256)(caption_input)
caption_lstm = LSTM(256)(caption_embedding)

# Combine image features and caption model
decoder_input = add([image_features, caption_lstm])
decoder_hidden = Dense(256, activation='relu')(decoder_input)
output = Dense(vocab_size, activation='softmax')(decoder_hidden)

# Define the complete model
model = Model(inputs=[image_input, caption_input], outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Summary of the model
model.summary()