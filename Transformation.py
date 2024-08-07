import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

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

def color_histogram(image):
    chans = cv2.split(image)
    colors = ("b", "g", "r")
    plt.figure()
    plt.title("Color Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()

def histogram_equalization(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_output

def transform_image(image_path):
    image = cv2.imread(image_path)
    
    transformations = {
        'Original': image,
        'Gaussian Blur': gaussian_blur(image),
        'Mask': mask_image(image),
        'Roi Objects': roi_objects(image),
        'Analyze Object': analyze_object(image),
        'Pseudolandmarks': pseudolandmarks(image),
        'Histogram Equalization': histogram_equalization(image)
    }
    
    plt.figure(figsize=(15, 10))
    for i, (title, transformed_image) in enumerate(transformations.items(), 1):
        plt.subplot(2, 4, i)
        plt.imshow(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def save_transformed_images(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    for filename in os.listdir(src_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(src_dir, filename)
            image = cv2.imread(image_path)
            
            transformations = {
                'Gaussian_Blur': gaussian_blur(image),
                'Mask': mask_image(image),
                'Roi_Objects': roi_objects(image),
                'Analyze_Object': analyze_object(image),
                'Pseudolandmarks': pseudolandmarks(image),
                'Histogram_Equalization': histogram_equalization(image)
            }
            
            for aug_name, transformed_image in transformations.items():
                output_path = os.path.join(dst_dir, f"{os.path.splitext(filename)[0]}_{aug_name}.jpg")
                cv2.imwrite(output_path, transformed_image)
                print(f"Saved transformed image: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply image transformations")
    parser.add_argument("image_or_dir", help="Path to an image or a directory of images")
    parser.add_argument("-dst", "--destination", help="Destination directory to save transformed images")
    args = parser.parse_args()
    
    if os.path.isfile(args.image_or_dir):
        transform_image(args.image_or_dir)
        color_histogram(cv2.imread(args.image_or_dir))
    elif os.path.isdir(args.image_or_dir) and args.destination:
        save_transformed_images(args.image_or_dir, args.destination)
    else:
        print("Invalid input. Provide either a path to an image or a directory with -dst for destination.")
        parser.print_help()
