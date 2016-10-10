import numpy as np
from nn import models
from PIL import Image


class Predictor:
    def __init__(self, file_name='./cnn_params_2c.pkl', width=66, height=98):
        self.file_name = file_name
        self.model = None
        self.width = width
        self.height = height

    def load_model(self):
        self.model = models.CNN(input_dim=(3, self.width, self.height), output_size=2)
        self.model.load_params(file_name=self.file_name)

    def predict(self, image_file_name):
        image = Image.open(image_file_name).convert('RGB')
        resize_image = image.resize((self.height, self.width))
        image_array = np.asarray(resize_image).transpose(2, 0, 1)
        image_array = image_array.astype(np.float32)
        image_array /= 255.0
        image_array = image_array.reshape((1, image_array.shape[0], image_array.shape[1], image_array.shape[2]))
        result = self.model.predict(image_array)

        if result.argmax(axis=1)[0] == 0:
            return 'other'
        else:
            return 'human'
