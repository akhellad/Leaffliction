import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import zipfile
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Input, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import Sequence
import time

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
                files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                for file in files:
                    image_path = os.path.join(path, file)
                    image_paths.append(image_path)
                    labels.append(label)
                
                print(f"  {subdir}: {len(files)} images")
            except ValueError:
                print(f"  Ignoré: {subdir} (format incorrect)")
    
    load_time = time.time() - start_time
    print(f"Dataset chargé en {load_time:.2f}s - Total: {len(image_paths)} images")
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
        
        print(f"  {cls}: {split_index} train, {len(cls_indexes)-split_index} val")
    
    return train_paths, train_labels, val_paths, val_labels

class OptimizedDataGenerator(Sequence):
    """Générateur de données optimisé pour la vitesse"""
    def __init__(self, image_paths, labels, batch_size=32, image_size=(96, 96), n_classes=10, shuffle=True):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.image_size = image_size  # Réduit de 128 à 96 pour plus de vitesse
        self.n_classes = n_classes
        self.shuffle = shuffle
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
        X = np.empty((self.batch_size, *self.image_size, 3), dtype=np.float32)
        y = np.empty((self.batch_size), dtype=int)
        
        for i, (image_path, label) in enumerate(zip(image_paths_temp, labels_temp)):
            try:
                # Lecture et redimensionnement optimisés
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Unable to read image {image_path}")
                    # Utilise une image noire par défaut
                    image = np.zeros((*self.image_size, 3), dtype=np.uint8)
                else:
                    image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)
                
                # Normalisation plus rapide
                X[i,] = image.astype(np.float32) / 255.0
                y[i] = self.labels_encoder.transform([label])[0]
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                # Image par défaut en cas d'erreur
                X[i,] = np.zeros((*self.image_size, 3), dtype=np.float32)
                y[i] = 0
        
        return X, to_categorical(y, num_classes=self.n_classes)

def preprocess_labels(labels):
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    return labels, le

def build_optimized_model(input_shape, num_classes):
    """Modèle CNN optimisé pour la vitesse et la performance"""
    model = Sequential([
        Input(shape=input_shape),
        
        # Premier bloc - extraction de features basiques
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.15),
        
        # Deuxième bloc - features plus complexes
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.15),
        
        # Troisième bloc - features avancées
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        
        # Classification
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    # Optimiseur avec learning rate adaptative
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train_model(train_generator, val_generator, epochs=15):
    """Entraînement avec callbacks pour optimiser la vitesse et éviter l'overfitting"""
    print("=== CONSTRUCTION DU MODÈLE ===")
    
    input_shape = train_generator.image_size + (3,)
    num_classes = train_generator.n_classes
    
    print(f"Shape d'entrée: {input_shape}")
    print(f"Nombre de classes: {num_classes}")
    
    model = build_optimized_model(input_shape, num_classes)
    
    # Callbacks pour optimiser l'entraînement
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    print("\n=== DÉBUT DE L'ENTRAÎNEMENT ===")
    start_time = time.time()
    
    history = model.fit(
        train_generator, 
        validation_data=val_generator, 
        epochs=epochs, 
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"\nEntraînement terminé en {training_time:.2f}s ({training_time/60:.1f} min)")
    
    print("\n=== ÉVALUATION FINALE ===")
    loss, accuracy = model.evaluate(val_generator, verbose=0)
    print(f"Précision finale: {accuracy * 100:.2f}%")
    print(f"Perte finale: {loss:.4f}")
    
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
        
        # Sauvegarde d'un échantillon d'images (pas toutes pour économiser l'espace)
        total_images = 0
        for subdir in os.listdir(data_dir):
            path = os.path.join(data_dir, subdir)
            if os.path.isdir(path):
                files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                # Prend seulement les 10 premières images de chaque classe
                for file in files[:10]:
                    image_path = os.path.join(path, file)
                    zipf.write(image_path, f"samples/{subdir}/{file}")
                    total_images += 1
        
        save_time = time.time() - start_time
        print(f"Sauvegarde terminée en {save_time:.2f}s ({total_images} images échantillons)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train.py <data_directory> [epochs]")
        print("Exemple: python train.py images 20")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    output_zip = "output_model.zip"
    
    print("=== CHARGEMENT DES DONNÉES ===")
    image_paths, labels = load_dataset(data_dir)
    
    if not image_paths:
        print("Aucune image trouvée!")
        sys.exit(1)
    
    labels, le = preprocess_labels(labels)
    
    print("\n=== SÉPARATION TRAIN/VALIDATION ===")
    train_paths, train_labels, val_paths, val_labels = split_dataset(image_paths, labels)
    
    print("\n=== CRÉATION DES GÉNÉRATEURS ===")
    train_generator = OptimizedDataGenerator(
        train_paths,
        train_labels,
        batch_size=32,
        image_size=(96, 96),  # Réduit pour plus de vitesse
        n_classes=len(le.classes_)
    )
    
    val_generator = OptimizedDataGenerator(
        val_paths,
        val_labels,
        batch_size=32,
        image_size=(96, 96),
        n_classes=len(le.classes_),
        shuffle=False
    )
    
    print(f"Entraînement: {len(train_paths)} images")
    print(f"Validation: {len(val_paths)} images")
    print(f"Batch size: 32")
    print(f"Taille images: 96x96")
    
    # Entraînement
    model = train_model(train_generator, val_generator, epochs)
    
    # Sauvegarde
    save_model_and_images(model, data_dir, output_zip)
    
    print(f"\n🎉 Entraînement terminé! Modèle sauvé dans {output_zip}")