import numpy as np
import argparse
import joblib
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Define data generators

train_dir = '/content/drive/MyDrive/archive/train'
val_dir = '/content/drive/MyDrive/archive/test'

num_train = 28709
num_val = 7178
batch_size = 64
num_epoch = 50

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
target_size=(48,48),
batch_size=batch_size,
color_mode="grayscale",
class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
val_dir,
target_size=(48,48),
batch_size=batch_size,
color_mode="grayscale",
class_mode='categorical'
)




# Creating the model

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))




# Load and preprocess your image
image = cv2.imread('/content/drive/MyDrive/archive/test/angry/im100.png', cv2.IMREAD_GRAYSCALE)
resized_image = cv2.resize(image, (48, 48))
normalized_image = resized_image / 255.0

# Add batch dimension and channel dimension
input_data = np.expand_dims(np.expand_dims(normalized_image, axis=0), axis=-1)

# Now 'input_data' has the shape (48, 48, 1) and can be fed into your model




# Make prediction
prediction = model.predict(input_data)

# Access the predicted class
predicted_class = np.argmax(prediction)

# Print the result
print(prediction)
print("Predicted class:", predicted_class)




# Save the trained model to a file
model_filename = '/content/drive/MyDrive/facial_emotion_recognition_model.joblib'
joblib.dump(model, model_filename)
