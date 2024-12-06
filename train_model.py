#TRAINING
# Importing necessary libraries
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import cv2
# Function to load and preprocess the dataset
def load_dataset(dataset_path):
    data = []
    labels = []

    for folder in os.listdir(dataset_path):
        for file in os.listdir(os.path.join(dataset_path, folder)):
            image_path = os.path.join(dataset_path, folder, file)
            image = cv2.imread(image_path)

            # Check if the image is loaded successfully
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                image = cv2.resize(image, (64, 64))  # Resize the image to a fixed size
                data.append(image)
                labels.append(folder)
            else:
                print(f"Warning: Unable to load image {image_path}")

    # Convert lists to numpy arrays
    data = np.array(data, dtype="float32") / 255.0  # Normalize pixel values to range [0, 1]
    labels = np.array(labels)

    # Perform one-hot encoding on the labels
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    labels = to_categorical(labels, 2)

    return data, labels

# Load the dataset
dataset_path = r"C:\Users\ASUS\Desktop\Smile_Detection-master\Smile_Detection-master\dataset\SMILEsmileD\SMILEs"
data, labels = load_dataset(dataset_path)

# Split the dataset into training and testing sets
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(trainX, trainY, validation_data=(testX, testY), epochs=10, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(testX, testY)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Save the trained model
model.save(r"C:\Users\ASUS\Desktop\Smile_Detection-master\Smile_Detection-master\smile_detection_model.h5")
