num_samples = len(image_data)
dummy_image_data = image_data[:num_samples]
dummy_caption_data = padded_sequences[:num_samples]
dummy_target_data = np.zeros((num_samples, vocab_size))  # Replace with actual target data
for i in range(num_samples):
    dummy_target_data[i, np.random.randint(0, vocab_size)] = 1
# Train the model
model.fit([dummy_image_data, dummy_caption_data], dummy_target_data, epochs=1)
