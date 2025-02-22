from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


# Paths
train_dir = "C:/Users/Aaditya/Desktop/skin_cancer_detection/augmented_dataset/train"
val_dir = "C:/Users/Aaditya/Desktop/skin_cancer_detection/augmented_dataset/validation"

# Image Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Data Generators
train_datagen = ImageDataGenerator(rescale=1.0/255)
val_datagen = ImageDataGenerator(rescale=1.0/255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Load VGG16 Model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze Base Layers
for layer in base_model.layers:
    layer.trainable = False

# Add Custom Layers
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(7, activation='softmax')(x)  # 7 classes

model = Model(inputs=base_model.input, outputs=output)

# Compile the Model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define the checkpoint callback
checkpoint = ModelCheckpoint(
    "best_vgg16_model.keras",  # Save in .keras format
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Train the Model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,  # Adjust as needed
    steps_per_epoch=(train_data.samples + BATCH_SIZE - 1) // BATCH_SIZE,
    validation_steps=(val_data.samples + BATCH_SIZE - 1) // BATCH_SIZE,
    callbacks=[checkpoint]  # Include the checkpoint callback
)

# Save the Final Model
model.save("vgg16_skin_cancer_model.h5")
print("Model training completed and saved!")

