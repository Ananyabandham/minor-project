from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class CaptionGenerator:
    def __init__(self):
        self.model = AutoModelForSequenceToSequenceLM.from_pretrained('t5-base')
        self.tokenizer = AutoTokenizer.from_pretrained('t5-base')

    def generate_caption(self, image_id, genre, additional_info=None):
        # Prepare the input
        input_text = f"Generate a caption for image {image_id} in the style of {genre}"
        if additional_info:
            input_text += f" with {additional_info}"

        # Tokenize the input
        inputs = self.tokenizer.encode_plus(
            input_text,
            max_length=50,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Generate the caption
        outputs = self.model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        caption = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return caption