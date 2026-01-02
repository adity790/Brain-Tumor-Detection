import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
import os

IMAGE_SIZE = 128

# Load model
model = load_model("brain_tumor_vgg16.h5")

# Load class names (same order as training)
CLASS_NAMES = sorted(os.listdir("Data/Training"))

def predict_image(image_path):
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    return CLASS_NAMES[class_index], confidence
