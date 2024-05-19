from flask import Flask, request, render_template
from model import CaptionGenerator
from dataset import Flickr8kDataset

app = Flask(__name__)

# Load the Flickr8k dataset
# dataset = Flickr8kDataset()

# Load the Hugging Face model
model = CaptionGenerator()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the user's input
        # image_url = request.form['image_url']
        # genre = request.form['genre']
        # additional_info = request.form.get('additional_info')

        # Preprocess the image URL
        # image_id = dataset.get_image_id(image_url)
        image="images/Images/57417274_d55d34e93e.jpg"
        # Generate the caption
        caption = model.generate_caption(image)

        # Render the result page
        return render_template('result.html', caption=caption)
    else:
        # Render the input page
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)