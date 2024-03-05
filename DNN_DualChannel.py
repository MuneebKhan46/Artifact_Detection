import tensorflow as tf
import numpy as np
import os
from os import path
import csv
import cv2
import textwrap
import pandas as pd


from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as sklearn_shuffle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score


models = []
class_1_accuracies = []

original_dir = '/Dataset/dataset_patch_raw_ver3/original'
denoised_dir = '/Dataset/dataset_patch_raw_ver3/denoised'
csv_path     = '/Dataset/patch_label_median.csv'
result_file_path = "/Dataset/Results/Overall_results.csv"



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
    denoised_image_names = []
    all_patch_numbers = []

    for _, row in df.iterrows():
        
        original_file_name = f"original_{row['original_image_name']}.raw"
        denoised_file_name = f"denoised_{row['original_image_name']}.raw"

        original_path = os.path.join(original_dir, original_file_name)
        denoised_path = os.path.join(denoised_dir, denoised_file_name)
        
        original_patches, original_patch_numbers = extract_y_channel_from_yuv_with_patch_numbers(original_path, row['width'], row['height'])
        denoised_patches, denoised_patch_numbers = extract_y_channel_from_yuv_with_patch_numbers(denoised_path, row['width'], row['height'])
        
        all_original_patches.extend(original_patches)
        all_denoised_patches.extend(denoised_patches)
        
        dual_maps = [generate_absolute_dual_map(orig, ghost) for orig, ghost in zip(all_original_patches, all_denoised_patches)]
        denoised_image_names.extend([row['image_name']] * len(denoised_patches))  
        
        all_patch_numbers.extend(denoised_patch_numbers) 

        scores = np.array([0 if float(score) == 0 else 1 for score in row['patch_score'].split(',')])
        if len(scores) != len(original_patches) or len(scores) != len(denoised_patches):
            print(f"Error: Mismatch in number of patches and scores for {row['original_image_name    ']}")
            continue
        all_scores.extend(scores)

    return dual_maps, all_scores, denoised_image_names, all_patch_numbers



def generate_absolute_dual_map(original_patch, ghosting_patch):
    original = np.expand_dims(original_patch, axis=2)
    ghosting = np.expand_dims(ghosting_patch, axis=2)
    patch_dual = np.concatenate((original, ghosting), axis=2)
    return patch_dual

def prepare_data(data, labels):
    data = np.array(data).astype('float32') / 255.0
    lbl = np.array(labels)
    return data, lbl


def save_metric_details(model_name, technique, feature_name, test_acc, weighted_precision, weighted_recall, weighted_f1_score, test_loss, accuracy_0, accuracy_1, result_file_path):
    
    if path.exists(result_file_path):
    
        df_existing = pd.read_csv(result_file_path)
        df_new_row = pd.DataFrame({
            'Model': [model_name],
            'Technique' : [technique],
            'Feature Map' : [feature_name],
            'Overall Accuracy': [test_acc],
            'Precision': [weighted_precision],
            'Recall': [weighted_recall],
            'F1-Score': [weighted_f1_score],
            'Loss': [test_loss],
            'Non-Ghosting Artifacts Accuracy': [accuracy_0],
            'Ghosting Artifacts Accuracy': [accuracy_1]
        })
        df_metrics = pd.concat([df_existing, df_new_row], ignore_index=True)
    else:
    
        df_metrics = pd.DataFrame({
            'Model': [model_name],
            'Technique' : [technique],            
            'Feature Map' : [feature_name],
            'Overall Accuracy': [test_acc],
            'Precision': [weighted_precision],
            'Recall': [weighted_recall],
            'F1-Score': [weighted_f1_score],
            'Loss': [test_loss],
            'Non-Ghosting Artifacts Accuracy': [accuracy_0],
            'Ghosting Artifacts Accuracy': [accuracy_1]
        })

    
    df_metrics.to_csv(result_file_path, index=False)


def create_dnn_model(input_shape=(224,224, 1)):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(512, activation='relu'),
        Dense(512, activation='relu'),
        
        Dense(256, activation='relu'),
        Dense(256, activation='relu'),
        Dense(256, activation='relu'),
        
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),             
        
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        
        Dense(2, activation='softmax')
    ])
     
    return model


dual_maps = []
dual_maps, labels, denoised_image_names, all_patch_numbers = load_data_from_csv(csv_path, original_dir, denoised_dir)

diff_patches_np, labels_np = prepare_data(dual_maps, labels)


combined = list(zip(diff_patches_np, labels_np, denoised_image_names, all_patch_numbers))
combined = sklearn_shuffle(combined)


ghosting_artifacts = [item for item in combined if item[1] == 1]
non_ghosting_artifacts = [item for item in combined if item[1] == 0]


num_ghosting_artifacts = 3502
num_non_ghosting_artifacts = 27944
num_test_ghosting = 502
num_test_non_ghosting = 502


num_train_ghosting = num_ghosting_artifacts - num_test_ghosting
num_train_non_ghosting = num_non_ghosting_artifacts - num_test_non_ghosting


train_ghosting = ghosting_artifacts[num_test_ghosting:]
test_ghosting = ghosting_artifacts[:num_test_ghosting]

train_non_ghosting = non_ghosting_artifacts[num_test_non_ghosting:]
test_non_ghosting = non_ghosting_artifacts[:num_test_non_ghosting]


train_dataset = train_ghosting + train_non_ghosting
test_dataset = test_ghosting + test_non_ghosting


train_patches, train_labels, train_image_names, train_patch_numbers = zip(*train_dataset)
test_patches, test_labels, test_image_names, test_patch_numbers = zip(*test_dataset)


train_patches = np.array(train_patches)
train_labels = np.array(train_labels)


X_train, X_test, y_train, y_test = train_test_split(train_patches, train_labels, test_size=0.2, random_state=42)
y_train = keras.utils.to_categorical(y_train, 2)
y_test = keras.utils.to_categorical(y_test, 2)


## Without Class Weight

dnn_wcw_model = create_dnn_model()
dnn_wcw_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


wcw_model_checkpoint = ModelCheckpoint(filepath='/Dataset/Model/DNN_DualChannel_wCW.h5', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1 )
wcw_history = dnn_wcw_model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), callbacks=[wcw_model_checkpoint])


## With Class Weight

ng = 27442
ga = 3000

total = ng + ga
imbalance_ratio = ng / ga  

weight_for_0 = (1 / ng) * (total / 2.0)
weight_for_1 = (1 / ga) * (total / 2.0)
class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0 (Non-ghosting): {:.2f}'.format(weight_for_0))
print('Weight for class 1 (Ghosting): {:.2f}'.format(weight_for_1))

dnn_cw_model = create_dnn_model()
dnn_cw_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


cw_model_checkpoint = ModelCheckpoint(filepath='/Dataset/Model/DNN_DualChannel_CW.h5', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1 )
cw_history = dnn_cw_model.fit(X_train, y_train, epochs=20, class_weight=class_weight, validation_data=(X_test, y_test), callbacks=[cw_model_checkpoint])


## With Balance Class Training


combined = list(zip(train_patches, train_labels, train_image_names, train_patch_numbers))
combined = sklearn_shuffle(combined)


ghosting_artifacts = [item for item in combined if item[1] == 1]
non_ghosting_artifacts = [item for item in combined if item[1] == 0]


num_ghosting_artifacts = 3000
num_non_ghosting_artifacts = 27442
num_train_val_ghosting = 2700
num_train_val_non_ghosting = 2700


num_test_ghosting = num_ghosting_artifacts - num_train_val_ghosting
num_test_non_ghosting = num_non_ghosting_artifacts - num_train_val_non_ghosting


train_val_ghosting = ghosting_artifacts[:num_train_val_ghosting]
test_ghosting = ghosting_artifacts[num_train_val_ghosting:]
train_val_non_ghosting = non_ghosting_artifacts[:num_train_val_non_ghosting]
test_non_ghosting = non_ghosting_artifacts[num_train_val_non_ghosting:]


cb_train_dataset = train_val_ghosting + train_val_non_ghosting
cb_test_dataset = test_ghosting + test_non_ghosting
print(len(cb_train_dataset))
print(len(cb_test_dataset))


cb_train_patches, cb_train_labels, cb_train_img_names, cb_train_patch_numbers = zip(*cb_train_dataset)
cb_test_patches, cb_test_labels, cb_test_img_names, cb_test_patch_numbers = zip(*cb_test_dataset)
cb_train_patches = np.array(cb_train_patches)
cb_train_labels = np.array(cb_train_labels)
cb_test_patches = np.array(cb_test_patches)
cb_test_labels = np.array(cb_test_labels)


cb_train_labels = keras.utils.to_categorical(cb_train_labels, 2)
cb_test_labels = keras.utils.to_categorical(cb_test_labels, 2)


dnn_cb_model = create_dnn_model()
dnn_cb_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


cb_model_checkpoint = ModelCheckpoint(filepath='/Dataset/Model/DNN_DualChannel_CB.h5', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1 )
cb_history = dnn_cb_model.fit(cb_train_patches, cb_train_labels, epochs=20, class_weight=class_weight, validation_data=(cb_test_patches, cb_test_labels), callbacks=[cb_model_checkpoint])


## Testing


test_patches = np.array(test_patches)
test_labels = np.array(test_labels)
test_labels = keras.utils.to_categorical(test_labels, 2)


## Without Class Weight


test_loss, test_acc = dnn_wcw_model.evaluate(test_patches, test_labels)
test_acc  = test_acc *100

predictions = dnn_wcw_model.predict(test_patches)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_labels, axis=-1)


report = classification_report(true_labels, predicted_labels, output_dict=True, target_names=["Non-Ghosting Artifact", "Ghosting Artifact"])

misclass_wCW_csv_path = '/Dataset/CSV/DNN_DualChannel_wCW_misclassified_patches.csv'
misclassified_indexes = np.where(predicted_labels != true_labels)[0]
misclassified_data = []

for index in misclassified_indexes:
    denoised_image_name = test_image_names[index]
    patch_number = test_patch_numbers[index]
    true_label = true_labels[index]
    predicted_label = predicted_labels[index]
    probability_non_ghosting = predictions[index, 0]
    probability_ghosting = predictions[index, 1]
    
    misclassified_data.append([
        denoised_image_name, patch_number, true_label, predicted_label,
        probability_non_ghosting, probability_ghosting
    ])

misclassified_df = pd.DataFrame(misclassified_data, columns=[
    'Denoised Image Name', 'Patch Number', 'True Label', 'Predicted Label', 
    'Probability Non-Ghosting', 'Probability Ghosting'
])

misclassified_df.to_csv(misclass_wCW_csv_path, index=False)

conf_matrix = confusion_matrix(true_labels, predicted_labels)
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]
TP = conf_matrix[1, 1]

total_class_0 = TN + FP
total_class_1 = TP + FN
correctly_predicted_0 = TN
correctly_predicted_1 = TP


accuracy_0 = (TN / total_class_0) * 100
accuracy_1 = (TP / total_class_1) * 100

precision_0 = TN / (TN + FN) if (TN + FN) > 0 else 0
recall_0 = TN / (TN + FP) if (TN + FP) > 0 else 0
precision_1 = TP / (TP + FP) if (TP + FP) > 0 else 0
recall_1 = TP / (TP + FN) if (TP + FN) > 0 else 0


weighted_precision = (precision_0 * total_class_0 + precision_1 * total_class_1) / (total_class_0 + total_class_1)
weighted_recall = (recall_0 * total_class_0 + recall_1 * total_class_1) / (total_class_0 + total_class_1)

if weighted_precision + weighted_recall > 0:
    weighted_f1_score = 2 * (weighted_precision * weighted_recall) / (weighted_precision + weighted_recall)
else:
    weighted_f1_score = 0

weighted_f1_score  = weighted_f1_score*100
weighted_precision = weighted_precision*100
weighted_recall    = weighted_recall*100


model_name = "DNN"
feature_name = "Dual Channel Map"
technique = "Without Class Weight"
save_metric_details(model_name, technique, feature_name, test_acc, weighted_precision, weighted_recall, weighted_f1_score, test_loss, accuracy_0, accuracy_1, result_file_path)


class_1_precision = report['Ghosting Artifact']['precision']
models.append(dnn_wcw_model)
class_1_accuracies.append(class_1_precision)


## With Class Weight

test_loss, test_acc = dnn_cw_model.evaluate(test_patches, test_labels)
test_acc  = test_acc *100

predictions = dnn_cw_model.predict(test_patches)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_labels, axis=-1)

report = classification_report(true_labels, predicted_labels, output_dict=True, target_names=["Non-Ghosting Artifact", "Ghosting Artifact"])

misclass_CW_csv_path  = '/Dataset/CSV/DNN_DualChannel_CW_misclassified_patches.csv'    

misclassified_indexes = np.where(predicted_labels != true_labels)[0]
misclassified_data = []
for index in misclassified_indexes:
    denoised_image_name = test_image_names[index]
    patch_number = test_patch_numbers[index]
    true_label = true_labels[index]
    predicted_label = predicted_labels[index]
    probability_non_ghosting = predictions[index, 0]
    probability_ghosting = predictions[index, 1]
    
    misclassified_data.append([
        denoised_image_name, patch_number, true_label, predicted_label,
        probability_non_ghosting, probability_ghosting
    ])

misclassified_df = pd.DataFrame(misclassified_data, columns=[
    'Denoised Image Name', 'Patch Number', 'True Label', 'Predicted Label', 
    'Probability Non-Ghosting', 'Probability Ghosting'
])

misclassified_df.to_csv(misclass_CW_csv_path, index=False)

conf_matrix = confusion_matrix(true_labels, predicted_labels)
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]
TP = conf_matrix[1, 1]

total_class_0 = TN + FP
total_class_1 = TP + FN
correctly_predicted_0 = TN
correctly_predicted_1 = TP


accuracy_0 = (TN / total_class_0) * 100
accuracy_1 = (TP / total_class_1) * 100

precision_0 = TN / (TN + FN) if (TN + FN) > 0 else 0
recall_0 = TN / (TN + FP) if (TN + FP) > 0 else 0
precision_1 = TP / (TP + FP) if (TP + FP) > 0 else 0
recall_1 = TP / (TP + FN) if (TP + FN) > 0 else 0


weighted_precision = (precision_0 * total_class_0 + precision_1 * total_class_1) / (total_class_0 + total_class_1)
weighted_recall = (recall_0 * total_class_0 + recall_1 * total_class_1) / (total_class_0 + total_class_1)

if weighted_precision + weighted_recall > 0:
    weighted_f1_score = 2 * (weighted_precision * weighted_recall) / (weighted_precision + weighted_recall)
else:
    weighted_f1_score = 0

weighted_f1_score  = weighted_f1_score*100
weighted_precision = weighted_precision*100
weighted_recall    = weighted_recall*100


model_name = "DNN"
feature_name = "Dual Channel Map"
technique = "Class Weight"
save_metric_details(model_name, technique, feature_name, test_acc, weighted_precision, weighted_recall, weighted_f1_score, test_loss, accuracy_0, accuracy_1, result_file_path)


class_1_precision = report['Ghosting Artifact']['precision']
models.append(dnn_cw_model)
class_1_accuracies.append(class_1_precision)

## With Class Balance

test_loss, test_acc = dnn_cb_model.evaluate(test_patches, test_labels)
test_acc  = test_acc *100

predictions = dnn_cb_model.predict(test_patches)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_labels, axis=-1)

report = classification_report(true_labels, predicted_labels, output_dict=True, target_names=["Non-Ghosting Artifact", "Ghosting Artifact"])

misclass_CB_csv_path  = '/Dataset/CSV/DNN_DualChannel_CB_misclassified_patches.csv'    
misclassified_indexes = np.where(predicted_labels != true_labels)[0]
misclassified_data = []

for index in misclassified_indexes:
    denoised_image_name = test_image_names[index]
    patch_number = test_patch_numbers[index]
    true_label = true_labels[index]
    predicted_label = predicted_labels[index]
    probability_non_ghosting = predictions[index, 0]
    probability_ghosting = predictions[index, 1]
    
    misclassified_data.append([
        denoised_image_name, patch_number, true_label, predicted_label,
        probability_non_ghosting, probability_ghosting
    ])

misclassified_df = pd.DataFrame(misclassified_data, columns=[
    'Denoised Image Name', 'Patch Number', 'True Label', 'Predicted Label', 
    'Probability Non-Ghosting', 'Probability Ghosting'
])

misclassified_df.to_csv(misclass_CB_csv_path, index=False)


conf_matrix = confusion_matrix(true_labels, predicted_labels)
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]
TP = conf_matrix[1, 1]


total_class_0 = TN + FP
total_class_1 = TP + FN
correctly_predicted_0 = TN
correctly_predicted_1 = TP


accuracy_0 = (TN / total_class_0) * 100
accuracy_1 = (TP / total_class_1) * 100

precision_0 = TN / (TN + FN) if (TN + FN) > 0 else 0
recall_0 = TN / (TN + FP) if (TN + FP) > 0 else 0
precision_1 = TP / (TP + FP) if (TP + FP) > 0 else 0
recall_1 = TP / (TP + FN) if (TP + FN) > 0 else 0


weighted_precision = (precision_0 * total_class_0 + precision_1 * total_class_1) / (total_class_0 + total_class_1)
weighted_recall = (recall_0 * total_class_0 + recall_1 * total_class_1) / (total_class_0 + total_class_1)

if weighted_precision + weighted_recall > 0:
    weighted_f1_score = 2 * (weighted_precision * weighted_recall) / (weighted_precision + weighted_recall)
else:
    weighted_f1_score = 0

weighted_f1_score  = weighted_f1_score*100
weighted_precision = weighted_precision*100
weighted_recall    = weighted_recall*100


model_name = "DNN"
feature_name = "Dual Channel Map"
technique = "Class Balance"
save_metric_details(model_name, technique, feature_name, test_acc, weighted_precision, weighted_recall, weighted_f1_score, test_loss, accuracy_0, accuracy_1, result_file_path)


class_1_precision = report['Ghosting Artifact']['precision']
models.append(dnn_cb_model)
class_1_accuracies.append(class_1_precision)


## ENSEMBLE 

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, log_loss

weights = np.array(class_1_accuracies) / np.sum(class_1_accuracies)
predictions = np.array([model.predict(test_patches)[:, 1] for model in models])
weighted_predictions = np.tensordot(weights, predictions, axes=([0], [0]))
predicted_classes = (weighted_predictions > 0.5).astype(int)

test_acc = accuracy_score(true_labels, predicted_classes)

weighted_precision, weighted_recall, weighted_f1_score, _ = precision_recall_fscore_support(true_labels, predicted_classes, average='weighted')
test_loss = log_loss(true_labels, weighted_predictions)

conf_matrix = confusion_matrix(true_labels, predicted_classes)
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]
TP = conf_matrix[1, 1]

total_class_0 = TN + FN  
total_class_1 = TP + FP  
correctly_predicted_0 = TN  
correctly_predicted_1 = TP

test_acc = test_acc *100
weighted_precision = weighted_precision * 100
weighted_recall   = weighted_recall * 100
weighted_f1_score = weighted_f1_score * 100

accuracy_0 = (TN / total_class_0) * 100
accuracy_1 = (TP / total_class_1) * 100


model_name = "DNN"
feature_name = "Dual Channel Map"
technique = "Ensemble"

save_metric_details(model_name, technique, feature_name, test_acc, weighted_precision, weighted_recall, weighted_f1_score, test_loss, accuracy_0, accuracy_1, result_file_path)