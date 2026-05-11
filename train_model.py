import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical

# ---------------- LOAD DATASET ----------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# ---------------- PREPROCESS ----------------
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Reshape for CNN
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# ---------------- BUILD MODEL ----------------
model = Sequential([

    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),

    Flatten(),

    Dense(256, activation='relu'),
    Dropout(0.5),

    Dense(10, activation='softmax')
])

# ---------------- COMPILE ----------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ---------------- TRAIN ----------------
history = model.fit(
    x_train,
    y_train,
    validation_split=0.1,
    epochs=12,
    batch_size=64
)

# ---------------- EVALUATE ----------------
loss, accuracy = model.evaluate(x_test, y_test)

print(f"\n✅ Test Accuracy: {accuracy*100:.2f}%")

# ---------------- SAVE MODEL ----------------
os.makedirs("model", exist_ok=True)

model.save("model/mnist_model.keras")

print("✅ Model saved successfully!")

# ---------------- PLOT ----------------
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training Accuracy")

plt.show()