import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import zipfile
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Input
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import Sequence

sys.stdout.reconfigure(encoding='utf-8')

def load_dataset(data_dir):
    image_paths = []
    labels = []
    
    for subdir in os.listdir(data_dir):
        path = os.path.join(data_dir, subdir)
        if os.path.isdir(path):
            label = subdir.split('_', 1)[1]
            for file in os.listdir(path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(path, file)
                    image_paths.append(image_path)
                    labels.append(label)
    
    print(f"Total images found: {len(image_paths)}")
    return image_paths, labels

def split_dataset(image_paths, labels, split_ratio=0.8):
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []

    classes = set(labels)
    
    for cls in classes:
        cls_indexes = [i for i, label in enumerate(labels) if label == cls]
        cls_image_paths = [image_paths[i] for i in cls_indexes]
        cls_labels = [labels[i] for i in cls_indexes]

        split_index = int(len(cls_indexes) * split_ratio)

        train_paths.extend(cls_image_paths[:split_index])
        train_labels.extend(cls_labels[:split_index])
        val_paths.extend(cls_image_paths[split_index:])
        val_labels.extend(cls_labels[split_index:])
    
    return train_paths, train_labels, val_paths, val_labels

class DataGenerator(Sequence):
    def __init__(self, image_paths, labels, batch_size=32, image_size=(128, 128), n_classes=10, shuffle=True):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.image_size = image_size
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
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Unable to read image {image_path}")
                continue
            image = cv2.resize(image, self.image_size)
            image = image.astype('float32') / 255.0
            X[i,] = image
            y[i] = self.labels_encoder.transform([label])[0]
        
        return X, to_categorical(y, num_classes=self.n_classes)

def preprocess_labels(labels):
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    return labels, le

def build_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(train_generator, val_generator):
    print("Starting the training process...")
    
    input_shape = train_generator.image_size + (3,)
    num_classes = train_generator.n_classes
    
    print("Building the CNN model...")
    
    model = build_model(input_shape, num_classes)
    
    print("Fitting the model with training data...")
    history = model.fit(train_generator, validation_data=val_generator, epochs=10, verbose=1)
    print("Model training completed.")
    
    print("Evaluating the model...")
    loss, accuracy = model.evaluate(val_generator)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
    
    return model

def save_model_and_images(model, data_dir, output_zip):
    with zipfile.ZipFile(output_zip, 'w') as zipf:
        model_file = 'model.h5'
        model.save(model_file)
        zipf.write(model_file)
        os.remove(model_file)

        for subdir in os.listdir(data_dir):
            path = os.path.join(data_dir, subdir)
            if os.path.isdir(path):
                for file in os.listdir(path):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(path, file)
                        zipf.write(image_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the data directory as a command line argument.")
        sys.exit(1)
    data_dir = sys.argv[1]
    output_zip = "output_model.zip"
    
    image_paths, labels = load_dataset(data_dir)
    labels, le = preprocess_labels(labels)
    
    train_paths, train_labels, val_paths, val_labels = split_dataset(image_paths, labels)
    
    train_generator = DataGenerator(train_paths,
                                    train_labels,
                                    batch_size=32,
                                    image_size=(128, 128),
                                    n_classes=len(le.classes_))
    val_generator = DataGenerator(val_paths,
                                  val_labels,
                                  batch_size=32,
                                  image_size=(128, 128),
                                  n_classes=len(le.classes_),
                                  shuffle=False)
    
    print(f"Training on {len(train_paths)} images, validating on {len(val_paths)} images")
    model = train_model(train_generator, val_generator)
    save_model_and_images(model, data_dir, output_zip)
