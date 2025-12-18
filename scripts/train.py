import os
import sys
import zipfile
import time
import cv2
import numpy as np
import matplotlib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Input, BatchNormalization
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import Sequence
matplotlib.use('TkAgg')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

sys.stdout.reconfigure(encoding='utf-8')


def load_dataset(data_dir):
    """Charge le dataset de manière optimisée"""
    image_paths = []
    labels = []

    print("Chargement du dataset...")
    start_time = time.time()

    for subdir in os.listdir(data_dir):
        path = os.path.join(data_dir, subdir)
        if os.path.isdir(path):
            try:
                label = subdir.split('_', 1)[1]
                files = [
                    f for f in os.listdir(path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]

                for file in files:
                    image_path = os.path.join(path, file)
                    image_paths.append(image_path)
                    labels.append(label)

                print(f"  {subdir}: {len(files)} images")
            except ValueError:
                print(f"  Ignoré: {subdir} (format incorrect)")

    load_time = time.time() - start_time
    msg = f"Dataset chargé en {load_time:.2f}s"
    msg += f" - Total: {len(image_paths)} images"
    print(msg)
    return image_paths, labels


def split_dataset(image_paths, labels, split_ratio=0.8):
    """Split stratifié pour équilibrer train/val"""
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []

    classes = set(labels)
    print(f"Classes trouvées: {len(classes)}")

    for cls in classes:
        cls_indexes = [i for i, label in enumerate(labels) if label == cls]
        cls_image_paths = [image_paths[i] for i in cls_indexes]
        cls_labels = [labels[i] for i in cls_indexes]

        split_index = int(len(cls_indexes) * split_ratio)

        train_paths.extend(cls_image_paths[:split_index])
        train_labels.extend(cls_labels[:split_index])
        val_paths.extend(cls_image_paths[split_index:])
        val_labels.extend(cls_labels[split_index:])

        val_count = len(cls_indexes) - split_index
        print(f"  {cls}: {split_index} train, {val_count} val")

    return train_paths, train_labels, val_paths, val_labels


class OptimizedDataGenerator(Sequence):
    """Optimized data generator for speed and accuracy"""
    def __init__(self, image_paths, labels, batch_size=32,
                 image_size=(128, 128), n_classes=10, shuffle=True,
                 label_encoder=None):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        # Increased to 128x128 for better accuracy
        self.image_size = image_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        # Use provided label encoder (shared between train/val) or create new
        if label_encoder is not None:
            self.labels_encoder = label_encoder
        else:
            self.labels_encoder = LabelEncoder()
            self.labels_encoder.fit(labels)
        super().__init__()
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        image_paths_temp = [self.image_paths[k] for k in indexes]
        labels_temp = [self.labels[k] for k in indexes]
        X, y = self.__data_generation(image_paths_temp, labels_temp)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, image_paths_temp, labels_temp):
        X = np.empty((self.batch_size, *self.image_size, 3),
                     dtype=np.float32)
        y = np.empty((self.batch_size), dtype=int)

        for i, (image_path, label) in enumerate(
                zip(image_paths_temp, labels_temp)):
            try:
                # Optimized reading and resizing
                image = cv2.imread(image_path)
                if image is None:
                    msg = f"Warning: Unable to read image {image_path}"
                    print(msg)
                    # Use black image as default
                    image = np.zeros((*self.image_size, 3), dtype=np.uint8)
                else:
                    image = cv2.resize(image, self.image_size,
                                       interpolation=cv2.INTER_AREA)

                # Faster normalization
                X[i,] = image.astype(np.float32) / 255.0
                y[i] = self.labels_encoder.transform([label])[0]
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                # Default image in case of error
                X[i,] = np.zeros((*self.image_size, 3), dtype=np.float32)
                y[i] = 0

        return X, to_categorical(y, num_classes=self.n_classes)


def preprocess_labels(labels):
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    return labels, le


def build_optimized_model(input_shape, num_classes):
    """
    Optimized CNN model for high accuracy (>90%).
    Enhanced architecture with more filters and deeper layers.
    """
    model = Sequential([
        Input(shape=input_shape),

        # First block - basic feature extraction
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Second block - more complex features
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Third block - advanced features
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Fourth block - high-level features
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        # Classification layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])

    # Optimizer with adaptive learning rate
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def train_model(train_generator, val_generator, epochs=30):
    """
    Training with callbacks to optimize speed and avoid overfitting.
    Increased epochs to 30 for better convergence.
    """
    print("=== MODEL CONSTRUCTION ===")

    input_shape = train_generator.image_size + (3,)
    num_classes = train_generator.n_classes

    print(f"Input shape: {input_shape}")
    print(f"Number of classes: {num_classes}")

    model = build_optimized_model(input_shape, num_classes)

    # Display model summary
    model.summary()

    # Callbacks to optimize training
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=8,  # Increased patience for better training
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,  # Increased patience
            min_lr=0.00001,
            verbose=1
        )
    ]

    print("\n=== TRAINING START ===")
    start_time = time.time()

    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    training_time = time.time() - start_time
    msg = f"\nTraining completed in {training_time:.2f}s"
    msg += f" ({training_time/60:.1f} min)"
    print(msg)

    print("\n=== FINAL EVALUATION ===")
    loss, accuracy = model.evaluate(val_generator, verbose=0)
    print(f"Final accuracy: {accuracy * 100:.2f}%")
    print(f"Final loss: {loss:.4f}")

    return model


def save_model_and_images(model, data_dir, output_zip):
    """Sauvegarde optimisée du modèle"""
    print(f"\nSauvegarde du modèle dans {output_zip}...")
    start_time = time.time()

    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Sauvegarde du modèle
        model_file = 'model.h5'
        model.save(model_file)
        zipf.write(model_file)
        os.remove(model_file)

        # Sauvegarde d'un échantillon d'images
        total_images = 0
        for subdir in os.listdir(data_dir):
            path = os.path.join(data_dir, subdir)
            if os.path.isdir(path):
                files = [f for f in os.listdir(path) if
                         f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                # Prend seulement les 10 premières images de chaque classe
                for file in files[:10]:
                    image_path = os.path.join(path, file)
                    zipf.write(image_path, f"samples/{subdir}/{file}")
                    total_images += 1

        save_time = time.time() - start_time
        print(f"Sauvegarde terminée en {save_time:.2f}s"
              f"({total_images} images échantillons)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train.py <data_directory> [epochs]")
        print("Example: python train.py Apple 30")
        sys.exit(1)

    data_dir = sys.argv[1]
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    output_zip = "output_model.zip"

    print("=== CHARGEMENT DES DONNÉES ===")
    image_paths, labels = load_dataset(data_dir)

    if not image_paths:
        print("Aucune image trouvée!")
        sys.exit(1)

    print("\n=== SÉPARATION TRAIN/VALIDATION ===")
    train_paths, train_labels, val_paths, val_labels = split_dataset(
                                                                image_paths,
                                                                labels)

    print("\n=== GENERATOR CREATION ===")
    # Get number of unique classes from the labels
    all_labels = train_labels + val_labels
    n_classes = len(set(all_labels))
    print(f"Number of classes: {n_classes}")

    # Create a SHARED label encoder for both train and val generators
    # This ensures consistent label mapping between train and validation
    shared_label_encoder = LabelEncoder()
    shared_label_encoder.fit(all_labels)
    label_map = dict(zip(shared_label_encoder.classes_, range(n_classes)))
    print(f"Label mapping: {label_map}")

    train_generator = OptimizedDataGenerator(
        train_paths,
        train_labels,
        batch_size=32,
        image_size=(128, 128),  # Increased for better accuracy
        n_classes=n_classes,
        label_encoder=shared_label_encoder  # Pass shared encoder
    )

    val_generator = OptimizedDataGenerator(
        val_paths,
        val_labels,
        batch_size=32,
        image_size=(128, 128),  # Increased for better accuracy
        n_classes=n_classes,
        shuffle=False,
        label_encoder=shared_label_encoder  # Pass same shared encoder
    )

    print(f"Training: {len(train_paths)} images")
    print(f"Validation: {len(val_paths)} images")
    print("Batch size: 32")
    print("Image size: 128x128")

    # Entraînement
    model = train_model(train_generator, val_generator, epochs)

    # Sauvegarde
    save_model_and_images(model, data_dir, output_zip)

    print(f"\n✓ Training completed! Model saved in {output_zip}")
