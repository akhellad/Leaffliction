import os
import sys
from PIL import Image, ImageOps
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use('TkAgg')


def flip_image(image):
    return ImageOps.mirror(image)


def rotate_image(image):
    return image.rotate(random.choice([90, 180, 270]))


def skew_image(image):
    width, height = image.size
    # Réduit pour être moins agressif
    xshift = width * random.uniform(-0.1, 0.1)
    new_width = width + abs(xshift)
    image = image.transform((int(new_width), height),
                            Image.AFFINE,
                            (1, xshift / width, 0, 0, 1, 0),
                            Image.BICUBIC)
    return image


def shear_image(image):
    width, height = image.size
    m = random.uniform(-0.2, 0.2)  # Réduit pour être moins agressif
    xshift = abs(m) * width
    new_width = width + int(round(xshift))
    image = image.transform((new_width, height),
                            Image.AFFINE,
                            (1, m, -xshift if m > 0 else 0, 0, 1, 0),
                            Image.BICUBIC)
    return image


def crop_image(image):
    width, height = image.size
    left = random.randint(0, width // 8)  # Moins agressif
    top = random.randint(0, height // 8)
    right = random.randint(7 * width // 8, width)
    bottom = random.randint(7 * height // 8, height)
    return image.crop((left, top, right, bottom))


def distort_image(image):
    width, height = image.size
    coeffs = find_coeffs(
        [(random.uniform(0, width * 0.1),  # Réduit l'intensité
          random.uniform(0, height * 0.1)),
         (width - random.uniform(0, width * 0.1),
          random.uniform(0, height * 0.1)),
         (width - random.uniform(0, width * 0.1),
          height - random.uniform(0, height * 0.1)),
         (random.uniform(0, width * 0.1),
          height - random.uniform(0, height * 0.1))],
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


def augment_image(image_path, output_dir, num_augmentations_needed=1):
    """
    Apply augmentations to an image.
    num_augmentations_needed: number of augmentations to generate
    """
    print(f"Augmenting image: {image_path}")
    image = Image.open(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 6 types of augmentation as required by the subject
    augmentations = [
        ('Flip', flip_image),
        ('Rotate', rotate_image),
        ('Skew', skew_image),
        ('Shear', shear_image),
        ('Crop', crop_image),
        ('Distortion', distort_image),
    ]

    # Select random augmentations to apply
    selected_augmentations = random.sample(
        augmentations,
        min(num_augmentations_needed, len(augmentations))
    )

    augmented_count = 0
    for aug_name, aug_func in selected_augmentations:
        if augmented_count >= num_augmentations_needed:
            break
        try:
            augmented_image = aug_func(image)
            augmented_image_path = os.path.join(
                output_dir,
                f"{base_name}_{aug_name}_{random.randint(1000, 9999)}.JPG"
            )
            augmented_image.save(augmented_image_path)
            augmented_count += 1
            print(f"  Created: {os.path.basename(augmented_image_path)}")
        except Exception as e:
            print(f"  Error with {aug_name}: {e}")

    return augmented_count


def display_augmented_images(image_path, output_dir):
    """
    Display the original image and 6 augmented versions.
    """
    print(f"Displaying augmented images for: {image_path}")
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 6 augmentation types + original (7 total as shown in subject)
    augmentations = [
        'Original', 'Flip', 'Rotate', 'Skew', 'Shear', 'Crop', 'Distortion'
    ]

    fig, axs = plt.subplots(1, len(augmentations), figsize=(21, 3))

    for ax, aug_name in zip(axs, augmentations):
        if aug_name == 'Original':
            augmented_image_path = image_path
        else:
            # Find first image with this augmentation name
            matching_files = [
                f for f in os.listdir(output_dir)
                if f.startswith(f"{base_name}_{aug_name}")
            ]
            if matching_files:
                augmented_image_path = os.path.join(
                    output_dir,
                    matching_files[0])
            else:
                augmented_image_path = None

        if augmented_image_path and os.path.exists(augmented_image_path):
            augmented_image = Image.open(augmented_image_path)
            ax.imshow(augmented_image)
            ax.set_title(aug_name, fontsize=10, fontweight='bold')
            ax.axis('off')
        else:
            ax.set_title(f"{aug_name} (missing)", fontsize=9)
            ax.axis('off')

    plt.tight_layout()
    plt.show()


def analyze_dataset(directory):
    """Analyse le dataset et retourne les statistiques par classe"""
    plant_types = {}
    for subdir in os.listdir(directory):
        path = os.path.join(directory, subdir)
        if os.path.isdir(path):
            try:
                plant_type, disease = subdir.split('_', 1)
                if plant_type not in plant_types:
                    plant_types[plant_type] = {}
                num_images = len([
                    f for f in os.listdir(path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ])
                plant_types[plant_type][disease] = num_images
            except ValueError:
                msg = f"Ignoring directory {subdir}"
                msg += " (doesn't follow plant_disease format)"
                print(msg)
    return plant_types


def balance_dataset(directory):
    """
    Équilibre le dataset en augmentant les classes minoritaires.
    Pour atteindre le même nombre d'images que la classe majoritaire
    """
    print("=== ANALYSE INITIALE ===")
    categories = analyze_dataset(directory)

    if not categories:
        print("Aucune catégorie trouvée!")
        return

    for plant_type, diseases in categories.items():
        print(f"\n=== ÉQUILIBRAGE DE {plant_type.upper()} ===")

        # Trouve le nombre max d'images pour ce type de plante
        max_images = max(diseases.values())
        target_images = max_images

        print(f"Nombre max d'images: {max_images}")
        msg = f"Cible pour équilibrage: {target_images}"
        msg += " (= classe majoritaire)"
        print(msg)

        # Équilibre chaque maladie
        for disease, count in diseases.items():
            path = os.path.join(directory, f"{plant_type}_{disease}")
            msg = f"\nTraitement de {plant_type}_{disease}: {count} images"
            print(msg)

            if count < target_images:
                needed = target_images - count
                print(f"  Besoin de {needed} images supplémentaires")

                # Get all existing original images (not augmented ones)
                original_images = [
                    os.path.join(path, file)
                    for file in os.listdir(path)
                    if file.lower().endswith(('.png', '.jpg', '.jpeg'))
                    and not any(aug in file for aug in [
                        '_Flip', '_Rotate', '_Skew', '_Crop',
                        '_Shear', '_Distortion'
                    ])
                ]

                if not original_images:
                    print(f"  Aucune image originale trouvée dans {path}")
                    continue

                msg = f"  {len(original_images)} images originales disponibles"
                print(msg)

                # Génère les augmentations nécessaires
                generated = 0
                attempts = 0
                max_attempts = needed * 2  # Évite les boucles infinies

                while generated < needed and attempts < max_attempts:
                    # Sélectionne une image au hasard
                    image_path = random.choice(original_images)

                    # Détermine combien d'augmentations créer
                    remaining = needed - generated
                    # Max 3 par image
                    augmentations_to_create = min(3, remaining)

                    try:
                        created = augment_image(
                            image_path, path, augmentations_to_create
                        )
                        generated += created
                        base = os.path.basename(image_path)
                        msg = f"  Généré {created} images depuis {base}"
                        msg += f" (total: {generated})"
                        print(msg)
                    except Exception as e:
                        base = os.path.basename(image_path)
                        print(f"  Erreur avec {base}: {e}")

                    attempts += 1

                msg = f"  ✓ Terminé: {generated} nouvelles images créées"
                print(msg)
            else:
                msg = f"  ✓ Déjà équilibré ({count} >= {target_images})"
                print(msg)

    print("\n=== ANALYSE FINALE ===")
    final_categories = analyze_dataset(directory)
    for plant_type, diseases in final_categories.items():
        print(f"\n{plant_type}:")
        for disease, count in diseases.items():
            print(f"  {disease}: {count} images")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Augmentation.py <path>")
        print("  - For a single image: displays 6 augmentation types")
        print("  - For a directory: balances the dataset")
        sys.exit(1)
    path = sys.argv[1]
    if os.path.isfile(path):
        output_dir = os.path.dirname(path)
        # Create 6 augmentations as required by the subject
        augment_image(path, output_dir, 6)
        display_augmented_images(path, output_dir)
    elif os.path.isdir(path):
        balance_dataset(path)
    else:
        print("Invalid path provided")
        sys.exit(1)
