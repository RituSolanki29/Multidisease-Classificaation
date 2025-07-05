#try3
# Import libraries
import pennylane as qml
from pennylane import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from pennylane.qnn.keras import KerasLayer
import matplotlib.pyplot as plt
import time
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns # For better confusion matrix visualization

# Step 1: Data Preprocessing
img_size = 64
batch_size = 32

# Enhanced Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    validation_split=0.3
)

train_gen = datagen.flow_from_directory(
    'C:/Users/mahav/Downloads/Dataset',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    'C:/Users/mahav/Downloads/Dataset',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Step 2: Define the Quantum Circuit
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# QNN layer with PennyLane
weight_shapes = {"weights": (3, n_qubits)}
qlayer = KerasLayer(quantum_circuit, weight_shapes, output_dim=n_qubits, dtype="float32")

# Custom layer to cast to float32 if using QNN (necessary for some TensorFlow ops)
class CastToFloat32(layers.Layer):
    def call(self, inputs):
         return tf.cast(inputs, tf.float32)

# Step 3: Hybrid CNN + QNN Model
def create_hybrid_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        # Uncomment these lines to integrate QNN:
        layers.Dense(n_qubits),
        qlayer,
        CastToFloat32(), # Needed if the QNN output type is not already float32
        layers.Dense(train_gen.num_classes, activation='softmax') # Use train_gen.num_classes for output
    ])
    return model

model = create_hybrid_model()

# Step 4: Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 5: Set callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True) # Increased patience
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5) # Increased patience

# Step 6: Train the model and record time
print("Starting model training...")
start_time = time.time()

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50, # You might need to adjust this based on your dataset and early stopping
    callbacks=[early_stop, reduce_lr]
)

end_time = time.time()
execution_time = end_time - start_time
print(f"\nTotal Training Time: {execution_time:.2f} seconds")

# Step 7: Evaluate the model and get detailed metrics
print("\nEvaluating the model...")
loss, accuracy = model.evaluate(val_gen)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Predict probabilities on the validation set
val_predictions = model.predict(val_gen)
val_predicted_classes = np.argmax(val_predictions, axis=1)
val_true_classes = val_gen.classes[val_gen.index_array] # Correctly get true labels

# Get class names
class_names = list(train_gen.class_indices.keys())

# Generate classification report
print("\n--- Classification Report ---")
print(classification_report(val_true_classes, val_predicted_classes, target_names=class_names))

# Generate Confusion Matrix
print("\n--- Confusion Matrix ---")
cm = confusion_matrix(val_true_classes, val_predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Step 8: Plotting the graphs
plt.figure(figsize=(14, 6))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy', color='green')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Plot Loss vs Epoch (combined)
plt.figure(figsize=(7, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Loss vs Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot Accuracy vs Epoch (combined)
plt.figure(figsize=(7, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy', color='green')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
plt.title('Accuracy vs Epoch')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()