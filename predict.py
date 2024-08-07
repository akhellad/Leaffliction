
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import zipfile
from tensorflow.keras.models import load_model
import tensorflow as tf
import logging

tf.get_logger().setLevel(logging.ERROR)

sys.stdout.reconfigure(encoding='utf-8')

def gaussian_blur(image):
    return cv2.GaussianBlur(image, (15, 15), 0)

def mask_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    return mask

def roi_objects(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    roi_image = image.copy()
    cv2.drawContours(roi_image, contours, -1, (0, 255, 0), 3)
    return roi_image

def analyze_object(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    analyze_image = image.copy()
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(analyze_image, (cx, cy), 5, (0, 0, 255), -1)
    return analyze_image

def pseudolandmarks(image):
    landmarks_image = image.copy()
    height, width, _ = image.shape
    for i in range(5, width, 50):
        for j in range(5, height, 50):
            cv2.circle(landmarks_image, (i, j), 5, (255, 0, 0), -1)
    return landmarks_image

def histogram_equalization(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_output

def predict_and_display(image_path, model, class_names):
    image = cv2.imread(image_path)
    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transformations = {
        'Original': original_image,
        'Gaussian Blur': gaussian_blur(image),
        'Mask': mask_image(image),
        'Roi Objects': roi_objects(image),
        'Analyze Object': analyze_object(image),
        'Pseudolandmarks': pseudolandmarks(image),
        'Histogram Equalization': histogram_equalization(image)
    }

    fig, axes = plt.subplots(1, 7, figsize=(15, 4), facecolor='lightgrey')

    for ax, (title, transformed_image) in zip(axes, transformations.items()):
        ax.imshow(transformed_image, cmap='gray' if title == 'Mask' else None)
        ax.set_title(title)
        ax.axis('off')

    image_resized = cv2.resize(image, (128, 128))
    image_resized = image_resized.astype('float32') / 255.0
    image_resized = np.expand_dims(image_resized, axis=0)

    prediction = model.predict(image_resized)
    predicted_index = np.argmax(prediction)
    predicted_label = class_names[predicted_index]

    plt.figtext(0.5, 0.05, f"The predicted disease is: {predicted_label}", ha="center", fontsize=40, color='green')
    plt.tight_layout()
    plt.show()

zip_path = "output_model.zip"
if len(sys.argv) != 2:
    print("Please provide the image path as an argument.")
    sys.exit(1)

image_path = sys.argv[1]

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("extracted_model")

model = load_model('extracted_model/model.h5')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

data_dir = "images"
class_names = sorted(set(d.split('_', 1)[1] for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))))

predict_and_display(image_path, model, class_names)