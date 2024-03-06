import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Sequential
import os
from os import path
import csv
import cv2
import textwrap
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Model

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as sklearn_shuffle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from keras.utils import to_categorical

original_dir = '/Dataset/dataset_patch_raw_ver3/original'
denoised_dir = '/Dataset/dataset_patch_raw_ver3/denoised'
csv_path     = '/Dataset/patch_label_median.csv'
result_file_path = "/Dataset/Results/GS_Overall_results.csv"



def extract_y_channel_from_yuv_with_patch_numbers(yuv_file_path: str, width: int, height: int):
    y_size = width * height
    patches, patch_numbers = [], []

    if not os.path.exists(yuv_file_path):
        print(f"Warning: File {yuv_file_path} does not exist.")
        return [], []

    with open(yuv_file_path, 'rb') as f:
        y_data = f.read(y_size)

    if len(y_data) != y_size:
        print(f"Warning: Expected {y_size} bytes, got {len(y_data)} bytes.")
        return [], []

    y_channel = np.frombuffer(y_data, dtype=np.uint8).reshape((height, width))
    patch_number = 0

    for i in range(0, height, 224):
        for j in range(0, width, 224):
            patch = y_channel[i:i+224, j:j+224]
            if patch.shape[0] < 224 or patch.shape[1] < 224:
                patch = np.pad(patch, ((0, 224 - patch.shape[0]), (0, 224 - patch.shape[1])), 'constant')
            patches.append(patch)
            patch_numbers.append(patch_number)
            patch_number += 1

    return patches, patch_numbers


def load_data_from_csv(csv_path, original_dir, denoised_dir):
    df = pd.read_csv(csv_path)
    
    all_original_patches = []
    all_denoised_patches = []
    all_scores = []
    
    for _, row in df.iterrows():
        
        original_file_name = f"original_{row['original_image_name']}.raw"
        denoised_file_name = f"denoised_{row['original_image_name']}.raw"

        original_path = os.path.join(original_dir, original_file_name)
        denoised_path = os.path.join(denoised_dir, denoised_file_name)
        
        original_patches, original_patch_numbers = extract_y_channel_from_yuv_with_patch_numbers(original_path, row['width'], row['height'])
        denoised_patches, denoised_patch_numbers = extract_y_channel_from_yuv_with_patch_numbers(denoised_path, row['width'], row['height'])

        all_original_patches.extend(original_patches)
        all_denoised_patches.extend(denoised_patches)
    
        scores = np.array([0 if float(score) == 0 else 1 for score in row['patch_score'].split(',')])
        if len(scores) != len(original_patches) or len(scores) != len(denoised_patches):
            print(f"Error: Mismatch in number of patches and scores for {row['original_image_name']}")
            continue
        all_scores.extend(scores)

    return all_original_patches, all_denoised_patches, all_scores


def calculate_difference(original, ghosting):
    return [np.abs(ghost.astype(np.int16) - orig.astype(np.int16)).astype(np.uint8) for orig, ghost in zip(original, ghosting)]


def prepare_data(data, labels):
    data = np.array(data).astype('float32') / 255.0
    lbl = np.array(labels)
    return data, lbl


def create_model(kernel_size, activation, dropout_rate, optimizer, learning_rate):
    input_shape = (224, 224, 1)
    model = Sequential()
    
    # Choose activation function
    if activation == 'LeakyReLU':
        activation_layer = tf.keras.layers.LeakyReLU(alpha=0.1)
    else:
        activation_layer = activation

    model.add(Conv2D(32, kernel_size=kernel_size, input_shape=input_shape))
    model.add(activation_layer if activation == 'LeakyReLU' else Activation(activation))
    
    model.add(Conv2D(32, kernel_size=kernel_size))
    model.add(activation_layer if activation == 'LeakyReLU' else Activation(activation))
    
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(BatchNormalization())
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    
    model.add(Conv2D(64, kernel_size=kernel_size))
    model.add(activation_layer if activation == 'LeakyReLU' else Activation(activation))
    
    model.add(Conv2D(64, kernel_size=kernel_size))
    model.add(activation_layer if activation == 'LeakyReLU' else Activation(activation))
    
    model.add(Conv2D(128, kernel_size=kernel_size))
    model.add(activation_layer if activation == 'LeakyReLU' else Activation(activation))
    
    model.add(Conv2D(128, kernel_size=kernel_size))
    model.add(activation_layer if activation == 'LeakyReLU' else Activation(activation))

    model.add(BatchNormalization())
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    
    model.add(Flatten())
    model.add(Dense(128))
    model.add(activation_layer if activation == 'LeakyReLU' else Activation(activation))
    
    model.add(Dense(2, activation='softmax'))

    # Configure the optimizer
    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = SGD(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


kernel_sizes = [(3, 3)]
learning_rates = [1e-2, 1e-3, 1e-4]
epochs = [20] 
dropout_rates = [0.0, 0.25, 0.5]
activations = ['relu', 'elu', 'LeakyReLU']
optimizers = ['adam', 'sgd', 'rmsprop']


original_patches, denoised_patches, labels = load_data_from_csv(csv_path, original_dir, denoised_dir)
diff_patches = calculate_difference(original_patches, denoised_patches)
diff_patches_np, labels_np = prepare_data(diff_patches, labels)

X_train, X_test, y_train, y_test = train_test_split(diff_patches_np, labels_np, test_size=0.2, random_state=42)
y_train = keras.utils.to_categorical(y_train, 2)
y_test = keras.utils.to_categorical(y_test, 2)

# Simplified grid search
for kernel_size in kernel_sizes:
    for learning_rate in learning_rates:
        for epoch in epochs:
            for activation in activations:
                for dropout_rate in dropout_rates:
                    for optimizer in optimizers:
                        model = create_model( kernel_size=kernel_size,activation=activation, dropout_rate=dropout_rate, optimizer=optimizer, learning_rate=learning_rate)
                        model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, validation_split=0.2, verbose=0)
                        _, accuracy = model.evaluate(X_test, y_test, verbose=2)
                        print(f"Accuracy: {accuracy:.4f} | Params: ks={kernel_size}, lr={learning_rate}, eppchs={epoch}, Activation={activation}, dr={dropout_rate}, opt={optimizer}")

# Note: Running this code as-is will take a significant amount of time due to the number of combinations.
