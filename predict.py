import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import os
import warnings
import argparse

warnings.filterwarnings("ignore", category=UserWarning,
                        module='tensorflow')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import zipfile
from tensorflow.keras.models import load_model
import tensorflow as tf
import logging

tf.get_logger().setLevel(logging.ERROR)

sys.stdout.reconfigure(encoding='utf-8')

def gaussian_blur(image):
    """Apply Gaussian blur."""
    return cv2.GaussianBlur(image, (15, 15), 0)


def mask_image(image):
    """Create binary mask."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def roi_objects(image):
    """Draw ROI contours."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)
    roi_image = image.copy()
    cv2.drawContours(roi_image, contours, -1, (0, 255, 0), 3)
    return roi_image


def analyze_object(image):
    """Analyze objects and mark centroids."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)
    analyze_image = image.copy()
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(analyze_image, (cx, cy), 5, (0, 0, 255), -1)
    return analyze_image


def pseudolandmarks(image):
    """Create pseudo-landmarks."""
    landmarks_image = image.copy()
    height, width, _ = image.shape
    for i in range(5, width, 50):
        for j in range(5, height, 50):
            cv2.circle(landmarks_image, (i, j), 5, (255, 0, 0), -1)
    return landmarks_image


def histogram_equalization(image):
    """Apply histogram equalization."""
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_output

def predict_and_display(image_path, model, class_names, image_size=(128, 128)):
    """
    Predict disease from image and display transformations.
    Image size must match training size (128x128).
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return None

    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transformations = {
        'Original': original_image,
        'Gaussian Blur': cv2.cvtColor(gaussian_blur(image),
                                       cv2.COLOR_BGR2RGB),
        'Mask': cv2.cvtColor(mask_image(image), cv2.COLOR_BGR2RGB),
        'Roi Objects': cv2.cvtColor(roi_objects(image), cv2.COLOR_BGR2RGB),
        'Analyze Object': cv2.cvtColor(analyze_object(image),
                                       cv2.COLOR_BGR2RGB),
        'Pseudolandmarks': cv2.cvtColor(pseudolandmarks(image),
                                        cv2.COLOR_BGR2RGB),
        'Histogram Equalization': cv2.cvtColor(histogram_equalization(image),
                                               cv2.COLOR_BGR2RGB)
    }

    fig, axes = plt.subplots(1, 7, figsize=(21, 4), facecolor='white')

    for ax, (title, transformed_image) in zip(axes,
                                                transformations.items()):
        ax.imshow(transformed_image)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')

    # Preprocess image for prediction (must match training size)
    image_resized = cv2.resize(image, image_size)
    image_resized = image_resized.astype('float32') / 255.0
    image_resized = np.expand_dims(image_resized, axis=0)

    # Make prediction
    print("\nMaking prediction...")
    prediction = model.predict(image_resized, verbose=0)
    predicted_index = np.argmax(prediction)
    predicted_label = class_names[predicted_index]
    confidence = prediction[0][predicted_index] * 100

    # Display prediction
    prediction_text = (f"Predicted disease: {predicted_label} "
                       f"(confidence: {confidence:.2f}%)")
    plt.figtext(0.5, 0.02, prediction_text, ha="center",
                fontsize=14, color='darkgreen', fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()

    print(f"\n=== PREDICTION RESULT ===")
    print(f"Disease: {predicted_label}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"\nTop 3 predictions:")
    top_3_idx = np.argsort(prediction[0])[-3:][::-1]
    for idx in top_3_idx:
        print(f"  {class_names[idx]}: {prediction[0][idx]*100:.2f}%")

    return predicted_label

def get_class_names_from_zip(zip_path):
    """
    Extract class names from sample images in the model zip.
    """
    class_names = set()
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for name in zip_ref.namelist():
            if name.startswith('samples/') and '/' in name[8:]:
                # Extract directory name like "Apple_healthy"
                dir_name = name.split('/')[1]
                if '_' in dir_name:
                    disease = dir_name.split('_', 1)[1]
                    class_names.add(disease)
    return sorted(class_names)


def get_class_names_from_directory(data_dir):
    """
    Get class names from a data directory.
    """
    try:
        class_names = sorted(set(
            d.split('_', 1)[1]
            for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d)) and '_' in d
        ))
        return class_names
    except Exception as e:
        print(f"Error reading directory {data_dir}: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Predict leaf disease from an image',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py image.jpg
  python predict.py image.jpg -m output_model.zip
  python predict.py image.jpg -m model.zip -d Apple
        """
    )
    parser.add_argument('image_path', help='Path to the image to predict')
    parser.add_argument('-m', '--model',
                        default='output_model.zip',
                        help='Path to model zip file (default: output_model.zip)')
    parser.add_argument('-d', '--data-dir',
                        help='Data directory for class names (optional)')

    args = parser.parse_args()

    # Validate arguments
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
        sys.exit(1)

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)

    print("=== LOADING MODEL ===")
    print(f"Extracting model from {args.model}...")

    # Extract model
    with zipfile.ZipFile(args.model, 'r') as zip_ref:
        zip_ref.extractall("extracted_model")

    # Load model
    model = load_model('extracted_model/model.h5', compile=False)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print("Model loaded successfully!")

    # Get class names
    print("\n=== LOADING CLASS NAMES ===")
    if args.data_dir and os.path.isdir(args.data_dir):
        class_names = get_class_names_from_directory(args.data_dir)
        print(f"Loaded {len(class_names)} classes from {args.data_dir}")
    else:
        class_names = get_class_names_from_zip(args.model)
        print(f"Loaded {len(class_names)} classes from model zip")

    if not class_names:
        print("Error: Could not determine class names")
        sys.exit(1)

    print(f"Classes: {', '.join(class_names)}")

    # Make prediction
    print(f"\n=== ANALYZING IMAGE ===")
    print(f"Image: {args.image_path}")
    predict_and_display(args.image_path, model, class_names,
                        image_size=(128, 128))