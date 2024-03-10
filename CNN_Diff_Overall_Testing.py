
import tensorflow as tf
import numpy as np
import os
from os import path
import csv
import cv2
import textwrap
import pandas as pd

from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, log_loss
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as sklearn_shuffle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score


models = []
class_1_accuracies = []

original_dir = '/Dataset/dataset_patch_raw_ver3/original'
denoised_dir = '/Dataset/dataset_patch_raw_ver3/denoised'
csv_path     = '/Dataset/patch_label_median.csv'



# In[2]:


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
        denoised_image_names.extend([row['original_image_name']] * len(denoised_patches))
        all_patch_numbers.extend(denoised_patch_numbers)

        scores = np.array([0 if float(score) == 0 else 1 for score in row['patch_score'].split(',')])
        if len(scores) != len(original_patches) or len(scores) != len(denoised_patches):
            print(f"Error: Mismatch in number of patches and scores for {row['original_image_name']}")
            continue
        all_scores.extend(scores)

    return all_original_patches, all_denoised_patches, all_scores, denoised_image_names, all_patch_numbers


def calculate_difference(original, ghosting):
    return [ghost.astype(np.int16) - orig.astype(np.int16) for orig, ghost in zip(original, ghosting)]



def prepare_data(data, labels):
    data = np.array(data).astype('float32') / 255.0
    lbl = np.array(labels)
    return data, lbl



original_patches, denoised_patches, labels, denoised_image_names, all_patch_numbers = load_data_from_csv(csv_path, original_dir, denoised_dir)
diff_patches = calculate_difference(original_patches, denoised_patches)
diff_patches_np, labels_np = prepare_data(diff_patches, labels)



test_patches = np.array(diff_patches_np)
test_labels = np.array(labels_np)
test_labels = keras.utils.to_categorical(test_labels, 2)


cnn_wcw_model= tf.keras.models.load_model('/Dataset/Model/CNN_AbsDiff_wCW.h5')


# In[6]:


test_loss, test_acc = cnn_wcw_model.evaluate(test_patches, test_labels)
test_acc  = test_acc *100

predictions = cnn_wcw_model.predict(test_patches)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_labels, axis=-1)


report = classification_report(true_labels, predicted_labels, output_dict=True, target_names=["Non-Ghosting Artifact", "Ghosting Artifact"])

class_1_precision = report['Ghosting Artifact']['precision']
models.append(cnn_wcw_model)
class_1_accuracies.append(class_1_precision)




# ## With Class Weight


cnn_cw_model= tf.keras.models.load_model('/Dataset/Model/CNN_Diff_CW.h5')


test_loss, test_acc = cnn_cw_model.evaluate(test_patches, test_labels)
test_acc  = test_acc *100

predictions = cnn_cw_model.predict(test_patches)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_labels, axis=-1)

report = classification_report(true_labels, predicted_labels, output_dict=True, target_names=["Non-Ghosting Artifact", "Ghosting Artifact"])

class_1_precision = report['Ghosting Artifact']['precision']
models.append(cnn_cw_model)
class_1_accuracies.append(class_1_precision)



# ## With Class Balance

cnn_cb_model= tf.keras.models.load_model('/Dataset/Model/CNN_Diff_CB.h5')


test_loss, test_acc = cnn_cb_model.evaluate(test_patches, test_labels)
test_acc  = test_acc *100

predictions = cnn_cb_model.predict(test_patches)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_labels, axis=-1)

report = classification_report(true_labels, predicted_labels, output_dict=True, target_names=["Non-Ghosting Artifact", "Ghosting Artifact"])


class_1_precision = report['Ghosting Artifact']['precision']
models.append(cnn_cb_model)
class_1_accuracies.append(class_1_precision)




# # Ensemble

# In[11]:


true_labels = np.argmax(test_labels, axis=-1)


weights = np.array(class_1_accuracies) / np.sum(class_1_accuracies)
predictions = np.array([model.predict(test_patches)[:, 1] for model in models])
weighted_predictions = np.tensordot(weights, predictions, axes=([0], [0]))
predicted_classes = (weighted_predictions > 0.5).astype(int)

misclassified_indexes = np.where(predicted_classes != true_labels)[0]
# print(f"Misclassified indexes: {misclassified_indexes}")


# In[12]:


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


model_name = "CNN"
feature_name = "Difference Map"
technique = "Ensemble"


# In[13]:


# save_metric_details(model_name, technique, feature_name, test_acc, weighted_precision, weighted_recall, weighted_f1_score, test_loss, accuracy_0, accuracy_1, result_file_path)

print(f"Accuracy: {test_acc:.4f} | precision: {weighted_precision:.4f}, Recall={weighted_recall:.4f}, F1-score={weighted_f1_score:.4f}, Loss={test_loss:.4f}, N.G.A Accuracy={accuracy_0:.4f}, G.A Accuracy={accuracy_1:.4f}")



misclass_En_csv_path = '/Dataset/CSV/Overall Testing_CNN_Diff_misclassified_patches.csv'

misclassified_data = []

for index in misclassified_indexes:
    denoised_image_name = denoised_image_names[index]
    patch_number = all_patch_numbers[index]
    true_label = true_labels[index]
    predicted_label = predicted_classes[index]

    probability_non_ghosting = 1 - weighted_predictions[index]
    probability_ghosting = weighted_predictions[index]
    
    misclassified_data.append([
        denoised_image_name, patch_number, true_label, predicted_label,
        probability_non_ghosting, probability_ghosting
    ])

misclassified_df = pd.DataFrame(misclassified_data, columns=[
    'Denoised Image Name', 'Patch Number', 'True Label', 'Predicted Label', 
    'Probability Non-Ghosting', 'Probability Ghosting'
])
misclassified_df.to_csv(misclass_En_csv_path, index=False)
