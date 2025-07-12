import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import matplotlib.pyplot as plt

# ✅ Paths
data_dir = "data/chest_xray_small"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
model_path = "models/pneumonia_cnn.h5"
plot_path = "outputs/accuracy_plot.png"

# ✅ Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ✅ Image preprocessing
train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir, target_size=(64, 64), color_mode='grayscale',
    batch_size=16, class_mode='binary'
)

val_data = val_gen.flow_from_directory(
    val_dir, target_size=(64, 64), color_mode='grayscale',
    batch_size=16, class_mode='binary'
)

# ✅ Build CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D(2, 2),
    Dropout(0.2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.2),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ✅ Train
history = model.fit(train_data, validation_data=val_data, epochs=10)

# ✅ Save model
model.save(model_path)
print(f"✅ Model saved at: {model_path}")

# ✅ Plot accuracy
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(plot_path)
print(f"📊 Accuracy plot saved at: {plot_path}")
