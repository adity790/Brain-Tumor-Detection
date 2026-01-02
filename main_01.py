# Libraries and Modules
import os
import numpy as np
import random
from PIL import Image, ImageEnhance

# Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from sklearn.utils import shuffle

train = 'Data/Training'
test = 'Data/Testing'

# Load and Shuffle train Data

train_paths = []
train_labels = []

for label in os.listdir(train):
  # print(label)
  for image in os.listdir(os.path.join(train, label)):
    train_paths.append(os.path.join(train, label, image))
    train_labels.append(label)

train_paths, train_labels = shuffle(train_paths, train_labels)

# train_paths

# Load and Shuffle Test Data
test_paths = []
test_labels = []

for label in os.listdir(test):
  # print(label)
  for image in os.listdir(os.path.join(test, label)):
    test_paths.append(os.path.join(test, label, image))
    test_labels.append(label)

test_paths, test_labels = shuffle(test_paths, test_labels)

# test_paths

import random
import matplotlib.pyplot as plt
from PIL import Image

# Select random indices
random_index = random.sample(range(len(train_paths)), 10)

# Create figure with 2 rows and 5 columns
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()

# Loop through random indices and display images
for i, idx in enumerate(random_index):
    img_path = train_paths[idx]
    label = train_labels[idx]

    img = Image.open(img_path)
    img = img.resize((128, 128))

    axes[i].imshow(img)
    axes[i].axis("off")
    axes[i].set_title(f"Label: {label}", fontsize=12)

plt.tight_layout()
# plt.show()

#Image Augmentation
def augment_image(image):
  image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
  image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))

  image = np.array(image)/255.0
  return image

# Load Image and apply Image Augmentation
def open_image(paths):
  images = []

  for path in paths:
    img = load_img(path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img = augment_image(img)
    images.append(img)

  return np.array(images)

# Encoder labels (Convert labels into integers)
def encode_labels(labels):
  unique_labels = sorted(os.listdir(train))
  return np.array([unique_labels.index(label) for label in labels])


# Data Geerator for Batching
def data_gen(paths, labels, batch_size=12):
    while True:   # infinite generator
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i+batch_size]
            batch_images = open_image(batch_paths)
            batch_labels = labels[i:i+batch_size]
            batch_labels = encode_labels(batch_labels)
            yield batch_images, batch_labels

from IPython.core import history
# Image Architecture (128, 128, 3) 3 here indicates RGB color
IMAGE_SIZE = 128
base_model = VGG16(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights='imagenet')

# Freeze all layers of the VGG16  base model
for layer in base_model.layers:
  layer.trainable = False

# Set only the last few layers
base_model.layers[-2].trainable = True
base_model.layers[-3].trainable = True
base_model.layers[-4].trainable = True

# Build Model
model = Sequential()
model.add(Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))) # Input Layer
model.add(base_model) # VGG16 Model
model.add(Flatten()) # Flatten layer
model.add(Dropout(0.3)) # Dropout layer
model.add(Dense(128, activation='relu')) # Dense Layer

model.add(Dropout(0.2)) # Dropout Layer

model.add(Dense(len(os.listdir(train)), activation='softmax')) # Output Layer

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

# Parameters
batch_size = 20
steps = int(len(train_paths)/batch_size)
epochs = 5

# train model
history = model.fit(
    data_gen(train_paths, train_labels, batch_size),
    steps_per_epoch=steps,
    epochs=epochs
)

# Save the trained model
model.save("brain_tumor_vgg16.h5")
print("Model saved successfully!")