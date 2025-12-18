import os
import cv2
import matplotlib
import matplotlib.pyplot as plt
import argparse
matplotlib.use('TkAgg')


def gaussian_blur(image):
    """Apply Gaussian blur to reduce noise."""
    return cv2.GaussianBlur(image, (15, 15), 0)


def mask_image(image):
    """Create a binary mask from the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    return mask


def roi_objects(image):
    """Detect and draw contours (ROI) on the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    roi_image = image.copy()
    cv2.drawContours(roi_image, contours, -1, (0, 255, 0), 3)
    return roi_image


def analyze_object(image):
    """Analyze objects and mark their centroids."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    analyze_image = image.copy()
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(analyze_image, (cx, cy), 5, (0, 0, 255), -1)
    return analyze_image


def pseudolandmarks(image):
    """Create pseudo-landmarks on the image grid."""
    landmarks_image = image.copy()
    height, width, _ = image.shape
    for i in range(5, width, 50):
        for j in range(5, height, 50):
            cv2.circle(landmarks_image, (i, j), 5, (255, 0, 0), -1)
    return landmarks_image


def histogram_equalization(image):
    """Apply histogram equalization to improve contrast."""
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_output


def transform_image(image_path):
    """
    Display 6 image transformations + original image.
    As required by the subject.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return

    transformations = {
        'Original': image,
        'Gaussian Blur': gaussian_blur(image),
        'Mask': mask_image(image),
        'Roi Objects': roi_objects(image),
        'Analyze Object': analyze_object(image),
        'Pseudolandmarks': pseudolandmarks(image),
        'Histogram Equalization': histogram_equalization(image)
    }

    plt.figure(figsize=(18, 8))
    for i, (title, transformed_image) in enumerate(
            transformations.items(), 1):
        plt.subplot(2, 4, i)
        if len(transformed_image.shape) == 2:  # Grayscale
            plt.imshow(transformed_image, cmap='gray')
        else:  # Color
            plt.imshow(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))
        plt.title(title, fontsize=11, fontweight='bold')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def save_transformed_images(src_dir, dst_dir, transform_type=None):
    """
    Save all transformations from a source directory to destination.
    If transform_type is specified, only that transformation is saved.
    """
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
        print(f"Created directory: {dst_dir}")

    image_files = [f for f in os.listdir(src_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"No images found in {src_dir}")
        return

    print(f"Processing {len(image_files)} images...")

    for filename in image_files:
        image_path = os.path.join(src_dir, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Warning: Could not read {filename}")
            continue

        transformations = {
            'Gaussian_Blur': gaussian_blur(image),
            'Mask': mask_image(image),
            'Roi_Objects': roi_objects(image),
            'Analyze_Object': analyze_object(image),
            'Pseudolandmarks': pseudolandmarks(image),
            'Histogram_Equalization': histogram_equalization(image)
        }

        # Filter by transform_type if specified
        if transform_type and transform_type in transformations:
            transformations = {transform_type: transformations[transform_type]}

        for aug_name, transformed_image in transformations.items():
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(dst_dir,
                                       f"{base_name}_{aug_name}.jpg")
            cv2.imwrite(output_path, transformed_image)

    print(f"Saved all transformations to {dst_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply image transformations for leaf disease analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Transformation.py image.jpg
  python Transformation.py -src images/ -dst output/
  python Transformation.py -src images/ -dst output/ -mask
        """
    )
    parser.add_argument("image_path", nargs='?',
                        help="Path to a single image")
    parser.add_argument("-src", "--source",
                        help="Source directory with images")
    parser.add_argument("-dst", "--destination",
                        help="Destination directory for transformed images")
    parser.add_argument("-mask", action="store_true",
                        help="Apply only mask transformation")
    parser.add_argument("-blur", action="store_true",
                        help="Apply only Gaussian blur")
    parser.add_argument("-h-help", action="store_true",
                        help="Show this help message")

    args = parser.parse_args()

    # Single image display
    if args.image_path and os.path.isfile(args.image_path):
        transform_image(args.image_path)
    # Batch processing
    elif args.source and args.destination:
        transform_type = None
        if args.mask:
            transform_type = 'Mask'
        elif args.blur:
            transform_type = 'Gaussian_Blur'

        if os.path.isdir(args.source):
            save_transformed_images(args.source, args.destination,
                                    transform_type)
        else:
            print(f"Error: {args.source} is not a valid directory")
    else:
        print("Usage:")
        print("  Display transformations: python Transformation.py <image>")
        print("  Save transformations: python Transformation.py "
              "-src <dir> -dst <dir>")
        parser.print_help()
