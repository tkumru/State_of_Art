from imageio import imread, mimsave
from os import listdir, getcwd

class GIF:
    def __init__(self, file_path, set_gifname) -> None:
        self.set_gifname = set_gifname
        self.file_path = file_path

        self.images = list()
        self.image_path = getcwd() + '/output/' + self.file_path
        self.image_filenames = listdir(self.image_path)

    def make_gif(self):
        images = self.images
        image_filenames = self.image_filenames
        image_path = self.image_path

        for file_name in image_filenames:
            images.append(imread(image_path + '/' + file_name))

        mimsave(getcwd() + f'/gif/{self.file_path}/{self.set_gifname}.gif', images, duration=0.75)
