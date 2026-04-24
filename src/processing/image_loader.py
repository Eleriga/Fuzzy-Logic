import cv2 # usato per le immagini
import numpy as np
from pathlib import Path # usato per gestire i percorsi dei file 

class ImageLoader:

    def __init__(self, target_size=None):
        self.target_size = target_size

    def load(self, filepath, as_gray=True):
        
        filepath = str(filepath)
        
        flags = cv2.IMREAD_GRAYSCALE if as_gray else cv2.IMREAD_COLOR
        image = cv2.imread(filepath, flags)

        if image is None:
            raise FileNotFoundError(f"Error: Impossible to load the image in {filepath}")

        if self.target_size:
            image = cv2.resize(image, self.target_size)

        return image

    def preprocess(self, image, normalize=True, flatten=False):
        
        processed_image = image.astype(np.float64)
        original_shape = processed_image.shape

        if normalize:
            processed_image /= 255.0

        if flatten:
            processed_image = processed_image.reshape(-1, 1)

        return processed_image, original_shape