import pandas as pd

class Flickr8kDataset:
    def __init__(self):
        self.dataset = pd.read_csv('flickr8k.csv')

    def get_image_id(self, image_url):
        # Assume the image URL is in the format 'https://example.com/image_<id>.jpg'
        image_id = image_url.split('_')[-1].split('.')[0]
        return int(image_id)