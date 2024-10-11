import numpy as np
class ImageProcessor:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.image_8bit = np.zeros((height, width), dtype=np.uint8)
        self.image_float = np.zeros((height, width), dtype=np.float32)

    def load_8bit_image(self, image_data):
        if image_data.shape != (self.height, self.width):
            raise ValueError("Image dimensions do not match.")
        self.image_8bit = image_data.astype(np.uint8)

    def convert_8bit_to_float(self):
        self.image_float = self.image_8bit.astype(np.float32) / 255.0

    def convert_float_to_8bit(self):
        self.image_8bit = (self.image_float * 255).clip(0, 255).astype(np.uint8)

    def get_8bit_image(self):
        return self.image_8bit

    def get_float_image(self):
        return self.image_float