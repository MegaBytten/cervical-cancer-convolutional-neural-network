## LIBRARIES
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from contextlib import redirect_stdout
import os
import datetime

START = datetime.datetime.now()
DATE = datetime.datetime.now().strftime("%d%m%Y")
TIME = datetime.datetime.now().strftime("%H%M%S")
FILEPATH = f"results/externalvalidation_{DATE}_{TIME}"
if not os.path.exists(FILEPATH):
    os.makedirs(FILEPATH)


## INFO

"""
Using MASSIVE dataset from Kaggle: ~ 9GB download - 600MB images
https://www.kaggle.com/datasets/prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed/data?select=im_Superficial-Intermediate
Cervical Cancer largest dataset (SipakMed)
The SIPaKMeD Database consists of 4049 images of isolated cells that have been manually cropped from 966 cluster cell images of Pap smear slides.
These images were acquired through a CCD camera adapted to an optical microscope.
The cell images are divided into five categories containing normal, abnormal and benign cells.

https://www.researchgate.net/publication/327995161_Sipakmed_A_New_Dataset_for_Feature_and_Image_Based_Classification_of_Normal_and_Pathological_Cervical_Cells_in_Pap_Smear_Images
Dyskeratotic and Koilocytotic groups are considered abnormal. 
Parabasal and Superficial-Intermediate are considered normal.
Metaplastic is considered Benign.
"""

# # # # # # # # # # LOADING A MODEL
# used so that other develops can 'load' and use the model
# also means can be plugged into an API
model = tf.keras.models.load_model('cervicalcancer_model.keras')

# Show the model architecture
with open(f"{FILEPATH}/ROC.txt", 'a') as f:
    with redirect_stdout(f):
        model.summary() # 11 million parameter model
f.close()

# TESTING ON EXTERNAL DATA
# load image: Dyskeratotic + Koilocytotic = abnormal | Parabasal + Superficial-Intermediate = normal
MAX_SAMPLE_LENGTH = 50
USE_MAX_SAMPLE = False

abnormal_sample_imgs = []
normal_sample_imgs = []

abnormal_folders = ['im_Dyskeratotic', 'im_Koilocytotic']
normal_folders = ['im_Superficial-Intermediate', 'im_Parabasal']

def resize_img(img):
    return cv2.resize(img, (256,256)) # RESIZING imgs

if USE_MAX_SAMPLE:
    for folder in abnormal_folders:
        for file in os.listdir(f"./cervical_cnn_training_dataset_clean/{folder}/")[:MAX_SAMPLE_LENGTH]:
            img = cv2.imread(f"./cervical_cnn_training_dataset_clean/{folder}/{file}")
            if img is not None:  # Ensure the image was read successfully
                img = resize_img(img)
                abnormal_sample_imgs.append(img)
    
    for folder in normal_folders:
        for file in os.listdir(f"./cervical_cnn_training_dataset_clean/{folder}/")[:MAX_SAMPLE_LENGTH]:
            img = cv2.imread(f"./cervical_cnn_training_dataset_clean/{folder}/{file}")
            if img is not None:  # Ensure the image was read successfully
                img = img = resize_img(img)
                normal_sample_imgs.append(img)
else:
    for folder in abnormal_folders:
        for file in os.listdir(f"./cervical_cnn_training_dataset_clean/{folder}/"):
            img = cv2.imread(f"./cervical_cnn_training_dataset_clean/{folder}/{file}")
            if img is not None:  # Ensure the image was read successfully
                img = resize_img(img)
                abnormal_sample_imgs.append(img)
    
    for folder in normal_folders:
        for file in os.listdir(f"./cervical_cnn_training_dataset_clean/{folder}/"):
            img = cv2.imread(f"./cervical_cnn_training_dataset_clean/{folder}/{file}")
            if img is not None:  # Ensure the image was read successfully
                img = img = resize_img(img)
                normal_sample_imgs.append(img)

# testing model
# Value 0 = high risk / carcinoma
# Value 1 = Normal cell
# Each model prediction takes ~25-35ms
THRESHOLDS = [x * 0.1 for x in range(0, 11)]
THRESHOLDS = list(map(lambda x: round(x, 1), THRESHOLDS))

# Thresholds + False positive + true positive
# v + normal classified abnormal (yhat < 0.5) + abnormal classified as abnormal (yhat < 0.5)
df_rows = []
# Running external validation for every threshold and collecting 
for threshold in THRESHOLDS:
    
    # Testing positive images - getting true positive rate
    abnormal_counter = 0
    for img in abnormal_sample_imgs:
        yhat = model.predict(np.expand_dims(img/255, 0))
        if yhat <= threshold: abnormal_counter+=1
    
    # Testing control images - getting false positive rate
    false_abnormal_counter = 0
    for img in normal_sample_imgs:
        yhat = model.predict(np.expand_dims(img/255, 0))
        if yhat <= threshold: false_abnormal_counter+=1
    
    
    true_positive_rate = abnormal_counter/len(abnormal_sample_imgs)
    false_positive_rate = false_abnormal_counter/len(normal_sample_imgs)
    row_values = {
        "Threshold":threshold,
        "True Positive Rate": true_positive_rate,
        "False Positive Rate": false_positive_rate
    }
    df_rows.append(row_values)


df = pd.DataFrame(df_rows)
df.to_csv("roc_data.csv") # SAVING TO CSV


# PLOTTING ROC CURVE
plt.figure()
plt.plot(df["False Positive Rate"], df["True Positive Rate"], marker='o', color='blue', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
#plt.show()
plt.savefig(f'{FILEPATH}/cervical_model_roc.png') # SAVING ROC IMAGE


# Writing ROC results to roc.txt
roc_auc = np.trapz(df["True Positive Rate"], df["False Positive Rate"])
f = open(f"{FILEPATH}/ROC.txt", "a")
f.write(f'AUC score: {roc_auc}')
f.close()

# Completed - record duration
END = datetime.datetime.now()
f = open(f"{FILEPATH}/ROC.txt", "a")
f.write(f"Script duration in h:m:s.ms was {END - START}")
f.close()