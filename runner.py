# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 14:07:16 2024

@author: swaro
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

tpath = r"C:\Users\swaro\Desktop\dehazer\Dense_Haze_NTIRE19\hazy\34_hazy.png"

IMG_HEIGHT, IMG_WIDTH = 256, 256
model_path = r"C:\Users\swaro\Desktop\dehazer\dehazeractual\modelok.keras"
model = load_model(model_path)

def read_and_resize_image(image_path, size=(256, 256)):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    image = np.array([image]) / 255.0
    return image


hazy_image = read_and_resize_image(tpath)
dehazed_image = model.predict(hazy_image)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Hazy Image')
plt.imshow(hazy_image[0])
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title('Dehazed Image')
plt.imshow(dehazed_image[0])
plt.axis('off')
plt.tight_layout()
plt.show()    


