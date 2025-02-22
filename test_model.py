import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Load trained model
model_path = r"C:\Users\Aaditya\Desktop\skin_cancer_detection\vgg16_skin_cancer_model.h5"
model = tf.keras.models.load_model(model_path)

# Class names in the same order as training
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# Function to preprocess image
def preprocess_image(image_path):
    img_size = (224, 224)  
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0  
    return img_array

# Predict function
def predict_image(image_path):
    img_array = preprocess_image(image_path)
    probabilities = model.predict(img_array)
    predicted_index = np.argmax(probabilities[0])
    return predicted_index

# Test images (3 classes only)
test_images = [
    r"C:\Users\Aaditya\Desktop\skin_cancer_detection\organized_dataset\mel\ISIC_0024482.jpg",
    r"C:\Users\Aaditya\Desktop\skin_cancer_detection\organized_dataset\bcc\bcc_0_24.jpg",
    r"C:\Users\Aaditya\Desktop\skin_cancer_detection\organized_dataset\nv\ISIC_0024322.jpg"
]

# True labels (index of the actual class)
y_true = [4, 1, 5]  # mel=4, bcc=1, nv=5

# Predictions
y_pred = [predict_image(img) for img in test_images]

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

# Classification Report (Includes Sensitivity & Specificity)
report = classification_report(y_true, y_pred, target_names=['mel', 'bcc', 'nv'])

# Print Confusion Matrix
print("\nConfusion Matrix:\n", cm)

# Print Classification Report
print("\nClassification Report:\n", report)

# Plot Confusion Matrix
plt.figure(figsize=(5, 5))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks(ticks=[0, 1, 2], labels=['mel', 'bcc', 'nv'])
plt.yticks(ticks=[0, 1, 2], labels=['mel', 'bcc', 'nv'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# Add text annotations
for i in range(len(cm)):
    for j in range(len(cm[i])):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='red')

plt.show()
