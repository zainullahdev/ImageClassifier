import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from helpers.arguments_helper import user_arguments as arguments 
from helpers.label_mapper import map_labels as mapper
from helpers.image_helper import load_and_process_img as img_processor

user_args = arguments()
image = user_args.image
model_name = user_args.model_path
top_k = user_args.top_k
category_names = user_args.category_names

if top_k is None:
    top_k = 1

if category_names is None:
    category_names = 'label_map.json'

class_names = mapper(category_names)
processed_img = img_processor(image)

model = tf.keras.models.load_model(model_name,custom_objects={'KerasLayer':hub.KerasLayer})

batched_processed_image = np.expand_dims(processed_img, axis=0)
prediction = model.predict(batched_processed_image)
values,indices = tf.math.top_k(prediction,k=top_k,sorted=False)
indices_numpy = indices.numpy().squeeze()
values_numpy = values.numpy().squeeze()
predicted_probabilites = {}
for n in range(top_k):
    if top_k == 1:
        class_index = str(indices_numpy+1)
        prob = values_numpy
    else:
        class_index = str(indices_numpy[n]+1)
        prob = values_numpy[n]
    class_name = class_names[class_index]
    predicted_probabilites[class_name] = float(prob)

print('Predicted Probabilites are:\n', predicted_probabilites)