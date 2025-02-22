import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Load your trained model
model_path = "C:/Users/Aaditya/Desktop/skin_cancer_detection/vgg16_skin_cancer_model.h5"
model = tf.keras.models.load_model(model_path)

# Define your class labels
class_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# Load test dataset
test_dir = r"C:\Users\Aaditya\Desktop\skin_cancer_detection\augmented_dataset\test"
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=1,  # Process one image at a time
    class_mode='categorical',
    shuffle=False
)

# Get true labels and predicted labels
true_labels = test_generator.classes  # True labels
predictions = model.predict(test_generator)  # Raw model predictions
predicted_labels = np.argmax(predictions, axis=1)  # Convert to class indices

# ✅ Debugging Step: Print counts
print("True label distribution:", np.bincount(true_labels))
print("Predicted label distribution:", np.bincount(predicted_labels))

# Generate confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Print classification report
print("Classification Report:\n", classification_report(true_labels, predicted_labels, target_names=class_labels))
