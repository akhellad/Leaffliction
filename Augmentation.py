import os
import sys
from PIL import Image, ImageOps
import random
import matplotlib.pyplot as plt
import numpy as np


def flip_image(image):
    return ImageOps.mirror(image)


def rotate_image(image):
    return image.rotate(random.choice([90, 180, 270]))


def skew_image(image):
    width, height = image.size
    xshift = width * random.uniform(-0.2, 0.2)
    new_width = width + abs(xshift)
    image = image.transform((int(new_width), height),
                            Image.AFFINE,
                            (1, xshift / width, 0, 0, 1, 0),
                            Image.BICUBIC)
    return image


def shear_image(image):
    width, height = image.size
    m = random.uniform(-0.3, 0.3)
    xshift = abs(m) * width
    new_width = width + int(round(xshift))
    image = image.transform((new_width, height),
                            Image.AFFINE,
                            (1, m, -xshift if m > 0 else 0, 0, 1, 0),
                            Image.BICUBIC)
    return image


def crop_image(image):
    width, height = image.size
    left = random.randint(0, width // 4)
    top = random.randint(0, height // 4)
    right = random.randint(3 * width // 4, width)
    bottom = random.randint(3 * height // 4, height)
    return image.crop((left, top, right, bottom))


def distort_image(image):
    width, height = image.size
    coeffs = find_coeffs(
        [(random.uniform(0, width * 0.2),
          random.uniform(0, height * 0.2)),
         (width - random.uniform(0, width * 0.2),
          random.uniform(0, height * 0.2)),
         (width - random.uniform(0, width * 0.2),
          height - random.uniform(0, height * 0.2)),
         (random.uniform(0, width * 0.2),
          height - random.uniform(0, height * 0.2))],
        [(0, 0), (width, 0), (width, height), (0, height)]
    )
    return image.transform((width, height),
                           Image.PERSPECTIVE,
                           coeffs,
                           Image.BICUBIC)


def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0],
                       p1[1],
                       1,
                       0,
                       0,
                       0,
                       -p2[0] * p1[0],
                       -p2[0] * p1[1]])
        matrix.append([0,
                       0,
                       0,
                       p1[0],
                       p1[1],
                       1,
                       -p2[1] * p1[0],
                       -p2[1] * p1[1]])

    A = np.matrix(matrix, dtype=float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)


def augment_image(image_path, output_dir):
    print(f"Augmenting image: {image_path}")
    image = Image.open(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    augmentations = {
        'Flip': flip_image,
        'Rotate': rotate_image,
        'Skew': skew_image,
        'Shear': shear_image,
        'Crop': crop_image,
        'Distort': distort_image
    }

    for aug_name, aug_func in augmentations.items():
        augmented_image = aug_func(image)
        augmented_image_path = os.path.join(output_dir,
                                            f"{base_name}_{aug_name}.JPG")
        augmented_image.save(augmented_image_path)


def display_augmented_images(image_path, output_dir):
    print(f"Displaying augmented images for: {image_path}")
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    augmentations = [
        'Original', 'Flip', 'Rotate', 'Skew', 'Shear', 'Crop', 'Distort'
    ]
    fig, axs = plt.subplots(1, 7, figsize=(20, 4))
    for ax, aug_name in zip(axs, augmentations):
        if aug_name == 'Original':
            augmented_image_path = image_path
        else:
            augmented_image_path = os.path.join(output_dir,
                                                f"{base_name}_{aug_name}.JPG")
        if os.path.exists(augmented_image_path):
            augmented_image = Image.open(augmented_image_path)
            ax.imshow(augmented_image)
            ax.set_title(aug_name)
            ax.axis('off')
        else:
            print(f"Warning: {augmented_image_path} does not exist.")
            ax.set_title(f"{aug_name} (missing)")
            ax.axis('off')
    plt.show()


def analyze_dataset(directory):
    plant_types = {}
    for subdir in os.listdir(directory):
        path = os.path.join(directory, subdir)
        if os.path.isdir(path):
            plant_type, disease = subdir.split('_', 1)
            if plant_type not in plant_types:
                plant_types[plant_type] = {}
            num_images = len(os.listdir(path))
            plant_types[plant_type][disease] = num_images
    return plant_types


def balance_dataset(directory):
    categories = analyze_dataset(directory)

    for plant_type, diseases in categories.items():
        max_images = max(diseases.values())
        print(f"Balancing {plant_type}")
        print(f"Target is {max_images} images per category")
        for disease, count in diseases.items():
            path = os.path.join(directory, f"{plant_type}_{disease}")
            print(f"Processing {plant_type}_{disease} with {count} images")

            if count < max_images:
                images = [os.path.join(path, file) for
                          file in
                          os.listdir(path)]
                while count < max_images:
                    for image_path in images:
                        augment_image(image_path, path)
                        count += 6
                        if count >= max_images:
                            break
                print(f"Balanced {plant_type}_{disease} to {count} images")
            else:
                print(f"No need to balance {plant_type}_{disease}")
        for disease, count in diseases.items():
            path = os.path.join(directory, f"{plant_type}_{disease}")
            print(f"Processing {plant_type}_{disease} with {count} images")

            if count < max_images:
                images = [os.path.join(path, file) for
                          file in
                          os.listdir(path)]
                while count < max_images:
                    for image_path in images:
                        augment_image(image_path, path)
                        count += 6
                        if count >= max_images:
                            break
                print(f"Balanced {plant_type}_{disease} to {count} images")
            else:
                print(f"No need to balance {plant_type}_{disease}")
        for disease, count in diseases.items():
            path = os.path.join(directory, f"{plant_type}_{disease}")
            print(f"Processing {plant_type}_{disease} with {count} images")

            if count < max_images:
                images = [os.path.join(path, file) for
                          file in
                          os.listdir(path)]
                while count < max_images:
                    for image_path in images:
                        augment_image(image_path, path)
                        count += 6
                        if count >= max_images:
                            break
                print(f"Balanced {plant_type}_{disease} to {count} images")
            else:
                print(f"No need to balance {plant_type}_{disease}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Augmentation.py <path>")
        sys.exit(1)
    path = sys.argv[1]
    if os.path.isfile(path):
        output_dir = os.path.dirname(path)
        augment_image(path, output_dir)
        display_augmented_images(path, output_dir)
    elif os.path.isdir(path):
        balance_dataset(path)
    else:
        print("Invalid path provided")
        sys.exit(1)
