from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.applications import MobileNetV2
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import Dense, Flatten, Dropout
from keras._tf_keras.keras.optimizers import Adam

# Paths
dataset_path = "C:/Users/jasar/Desktop/OcularDiseaseRecognition/dataset"  
model_save_path = "C:/Users/jasar/Desktop/OcularDiseaseRecognition/retina_validation_model.keras"

# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# Data generators
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2,  # 80-20 train-validation split
    horizontal_flip=True,
    rotation_range=15,
    zoom_range=0.2,
)

train_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
)

val_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
)

# Base model (MobileNetV2)
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
base_model.trainable = False

# Add custom classification layers
x = Flatten()(base_model.output)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    steps_per_epoch=train_gen.samples // BATCH_SIZE,
    validation_steps=val_gen.samples // BATCH_SIZE,
    verbose=1,
)

# Save the model
model.save(model_save_path)
print(f"Model saved to {model_save_path}")
