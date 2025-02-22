import os
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# Paths
model_path = r"C:\Users\Aaditya\Desktop\skin_cancer_detection\vgg16_skin_cancer_model.h5"
test_labels_path = "C:/Users/Aaditya/Desktop/skin_cancer_detection/augmented_dataset/test_labels.csv"
class_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']  # 7 Classes

# Load model
model = tf.keras.models.load_model(model_path)

# Load test labels
test_labels_df = pd.read_csv(test_labels_path)
print("Test Labels CSV Loaded Successfully")

# Ensure test labels match class labels
test_labels_df = test_labels_df[test_labels_df['label'].isin(class_labels)]
unique_classes = sorted(test_labels_df['label'].unique())
print(f"Unique classes in dataset: {unique_classes}")
print(f"Length of unique classes: {len(unique_classes)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']
    file_path = os.path.join("static/uploads", file.filename)
    file.save(file_path)

    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]

    # Compute evaluation metrics
    y_true = test_labels_df['label'].values
    y_pred = [class_labels[np.argmax(model.predict(np.expand_dims(image.img_to_array(
        image.load_img(os.path.join("C:/Users/Aaditya/Desktop/skin_cancer_detection/augmented_dataset/test", filename), target_size=(224, 224))) / 255.0, axis=0)))] for filename in test_labels_df['filename']]

    report = classification_report(y_true, y_pred, output_dict=True, target_names=class_labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)

    # Save Confusion Matrix Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig("static/confusion_matrix.png")
    plt.close()

    return render_template('result.html', prediction=predicted_class, report=report, image_path=file_path, cm_path="static/confusion_matrix.png")

if __name__ == '__main__':
    app.run(debug=True)
