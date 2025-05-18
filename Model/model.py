import os
import numpy as np
import pandas as pd
from keras._tf_keras.keras.applications import EfficientNetB0
from keras._tf_keras.keras.models import Model, load_model
from keras._tf_keras.keras.layers import Input, Dense, Dropout, Flatten, Concatenate, Conv2D
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import cv2
import json

# Define paths
metadata_file = "full_df.csv"
image_folder = "ODIR-5K\ODIR-5K\Training Images"  # Folder containing images 
progress_file = "training_progress.json"  # To track training progress (epoch)
checkpoint_file = "model_checkpoint.keras"  # Checkpoint file

# Load metadata
data = pd.read_csv(metadata_file)

# Preprocess metadata
metadata = data[["Patient Age", "Patient Sex"]]
metadata["Patient Sex"] = metadata["Patient Sex"].map({"Male": 0, "Female": 1})
metadata = metadata.to_numpy()
labels = data[["N", "D", "G", "C", "A", "H", "M", "O"]].to_numpy()

# Load images and preprocess
images = []
filtered_labels = []
filtered_metadata = []
skipped_files = []

for idx, row in data.iterrows():
    left_image_path = os.path.join(image_folder, row["Left-Fundus"])
    right_image_path = os.path.join(image_folder, row["Right-Fundus"])
    
    left_image = cv2.imread(left_image_path)
    right_image = cv2.imread(right_image_path)
    
    if left_image is not None and right_image is not None:
        try:
            left_image = cv2.resize(left_image, (224, 224)) / 255.0
            right_image = cv2.resize(right_image, (224, 224)) / 255.0
            combined_image = np.concatenate((left_image, right_image), axis=2)
            images.append(combined_image)
            filtered_labels.append(labels[idx])          # Append the corresponding label
            filtered_metadata.append(metadata[idx])      # Append the corresponding metadata
        except Exception as e:
            skipped_files.append((row["Left-Fundus"], row["Right-Fundus"], str(e)))
    else:
        skipped_files.append((row["Left-Fundus"], row["Right-Fundus"], "File not found or corrupted"))

images = np.array(images)
filtered_labels = np.array(filtered_labels)
filtered_metadata = np.array(filtered_metadata)

# Log skipped files if any
if skipped_files:
    with open("skipped_files.log", "w") as log_file:
        for left_file, right_file, error in skipped_files:
            log_file.write(f"Left: {left_file}, Right: {right_file}, Error: {error}\n")

# Check lengths
print(f"Images: {len(images)}, Labels: {len(filtered_labels)}, Metadata: {len(filtered_metadata)}")
# Split data using the filtered arrays
X_train_images, X_val_images, y_train, y_val, X_train_metadata, X_val_metadata = train_test_split(
    images, filtered_labels, filtered_metadata, test_size=0.2, random_state=42
)

# Check if a checkpoint exists
if os.path.exists(checkpoint_file):
    print("Loading model from the last checkpoint...")
    model = load_model(checkpoint_file)
else:
    # Define the input shape for 6 channels
    image_input = Input(shape=(224, 224, 6))

    # Apply a Conv2D layer to convert the 6 channels to 3 channels
    x = Conv2D(3, (1, 1), padding="same", activation="linear")(image_input)  # Keep the image shape but reduce channels to 3


    # Initialize EfficientNetB0 without ImageNet weights
    base_model = EfficientNetB0(include_top=False, weights=None, input_tensor=x)

    x = Flatten()(base_model.output)

    # Define metadata input
    metadata_input = Input(shape=(2,))
    metadata_dense = Dense(32, activation="relu")(metadata_input)

    # Combine image and metadata features
    combined = Concatenate()([x, metadata_dense])
    x = Dense(128, activation="relu")(combined)
    x = Dropout(0.5)(x)
    output = Dense(8, activation="sigmoid")(x)

    # Create the model
    model = Model(inputs=[image_input, metadata_input], outputs=output)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),  # Use a higher learning rate at the start, the learning rate will lower auomatically during training
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Ensure progress file exists
if not os.path.exists(progress_file):
    # Create a progress file with default epoch 0 if it doesn't exist
    with open(progress_file, "w") as f:
        json.dump({"epoch": 0}, f)

# Determine starting epoch
with open(progress_file, "r") as f:
    try:
        training_progress = json.load(f)
        initial_epoch = training_progress.get("epoch", 0)
        print(f"Resuming training from epoch {initial_epoch + 1}")
    except json.JSONDecodeError:
        print("Progress file is corrupted or empty. Starting from epoch 0.")
        initial_epoch = 0

# Callbacks
checkpoint = ModelCheckpoint(
    checkpoint_file, save_weights_only=False, save_freq="epoch", verbose=1
)
early_stopping = EarlyStopping(
    monitor="val_accuracy", patience=5, verbose=1, restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.1, patience=3, verbose=1
)

# Train the model
history = model.fit(
    [X_train_images, X_train_metadata], y_train,
    validation_data=([X_val_images, X_val_metadata], y_val),
    epochs=50,
    batch_size=32,
    callbacks=[checkpoint, early_stopping, reduce_lr],
    initial_epoch=initial_epoch
)

# Save training progress after training
try:
    with open(progress_file, "w") as f:
        json.dump({"epoch": history.epoch[-1]}, f)
        print(f"Training progress saved: Epoch {history.epoch[-1]}")
except Exception as e:
    print(f"Error saving training progress: {e}")

# Save the model
model.save("ocular_disease_model.keras")

