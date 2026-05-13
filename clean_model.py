from tensorflow.keras.models import load_model

# Load old model
model = load_model("model/mnist_model.keras", compile=False)

# Save clean model
model.save("model/clean_mnist_model.h5")