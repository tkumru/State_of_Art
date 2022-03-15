import numpy as np
from imageio import imread, imwrite
from PIL import Image

class Process:
    image_width = 800
    image_height = 600
    color_channels = 3
    noise_ratio = 0.6
    mean_values = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))

    def __init__(self) -> None:
        pass
    
    def resize_image(self, image_path, output_path):
        image = Image.open(image_path)
        image = image.resize((800, 600))
        image.save(output_path)

    def generate_noise_image(self, content_image, noise_ratio=noise_ratio):
        noise_image = np.random.uniform(-20, 20, 
        (1, Process.image_height, Process.image_width, Process.color_channels)).astype('float32')

        return noise_image * noise_ratio + content_image * (1 - noise_ratio)
    
    def load_image(self, path):
        image = imread(path)
        image = np.reshape(image, ((1,) + image.shape))
        
        return image - Process.mean_values

    def save_image(self, path, image):
        image += Process.mean_values
        image = image[0]
        image = np.clip(image, 0, 255).astype('uint8')

        imwrite(path, image)
