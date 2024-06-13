def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def generate_caption(model, tokenizer, image_path, max_length):
    img = preprocess_image(image_path)
    img_features = image_model.predict(img)

    caption = 'startseq'
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([caption])[0]
        seq = pad_sequences([seq], maxlen=max_length, padding='post')
        yhat = model.predict([img_features, seq], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word[yhat]
        if word is None:
            break
        caption += ' ' + word
        if word == 'endseq':
            break
    return caption

# Example usage
image_path = 'D:\minor-project\images\images_small\images\images_small\56494233_1824005879.jpg'
caption = generate_caption(model, tokenizer, image_path, max_length)
print("Generated Caption:", caption)