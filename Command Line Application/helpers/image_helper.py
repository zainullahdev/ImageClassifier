import numpy as np
import tensorflow as tf
from PIL import Image

def load_and_process_img(image_name):
    im = Image.open(image_name)
    test_image = np.asarray(im)
    processed_image = process_image(test_image)
    return processed_image

def process_image(image):
    IMG_SIZE = 224
    image = tf.cast(image,tf.float32)
    image = tf.image.resize(image,(IMG_SIZE,IMG_SIZE))
    image /= 255
    return image.numpy()