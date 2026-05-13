from tensorflow.keras.models import load_model

# Load existing model
model = load_model("model/mnist_model.keras", compile=False)

# Save in old compatible format
model.save("model/mnist_model.h5")

print("Model converted successfully!")