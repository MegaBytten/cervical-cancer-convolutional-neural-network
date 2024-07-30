import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from contextlib import redirect_stdout
import datetime
import os

START = datetime.datetime.now()
DATE = datetime.datetime.now().strftime("%d%m%Y")
TIME = datetime.datetime.now().strftime("%H%M%S")
FILEPATH = f"results/training_{DATE}_{TIME}"
if not os.path.exists(FILEPATH):
    os.makedirs(FILEPATH)

# Getting Data: Data Sourced from DTU/HERLEV DATABASE: https://mde-lab.aegean.gr/index.php/downloads/

# loading data
# tensorflow has data API --> better for building data pipelines: No need to read it in-memory
# tf.data.Dataset --> is an API tf.data.Dataset.list_files
# also accessible through keras --> 


# uses tensorflow data API through keras
# builds img dataset for you - no need to do preprocessing/processing like image labelling, resizing images, classes
# operates the images in batches of 32, shuffles, creates a validation split
# data = a *generator* not a dataset/frame
# reads 2 directories from directory -> produces img labels 1 or 0 based on which folder images come from
"""
Label 0: high_dys_or_carcinoma
Label 1: normal_cells
"""

data_dir = "model_data"

data = tf.keras.utils.image_dataset_from_directory(
    directory = data_dir,
    labels='inferred', # default
    batch_size = 32, #default
    image_size = (256, 256), #default
    validation_split = None # default
)  # import from data folder

# Recording class names to see which directory data has been assigned what labels
f = open(f"{FILEPATH}/training_results.txt", "a")
f.write("Class names and their corresponding labels:\n")

for i, class_name in enumerate(data.class_names):
    f.write(f"Label {i}: {class_name}\n")
f.close()


# # # # # # # # # # PRE-PROCESSING STAGE
# - Scaling (normalizing data)
# - Splitting data (feature training, classification, validation)


# normalize - deep learning models love smaller numbers for enhanced optimisation, decrease integer arithmatic and binary read/writes
# need to use data object to perform operation while data is being loaded, because we're using the generator
# data returns: [ batch[0], batch[1] ] = img RGB values, labels
# therefore need to normalize RGB values between 0-1 and leave labels as is
# LOT of other tensorflow transofmriations you can do: https://www.tensorflow.org/guide/data
data = data.map(lambda x,y: (x/255, y))


# Partitioning data
# print(len(data)) # prints how many batches of data we have
# look for data.split or some similar method to best practice split, bc lower sucks.
train_size = int( round(len(data)*0.7) ) # 70% of data length cast to int - train deep learning model
val_size = int( round(len(data)*0.2) ) # validate to make sure no overfitting - while we're training 
test_size = int( round(len(data)*0.1) ) # test model - not having to see until final evaluation

if train_size + val_size + test_size != len(data):
    exit(f"Train, Val, test = {train_size} + {val_size} + {test_size}\nMust Manually adjust batch sizes / splits!")

# actually partitioning - make sure data is shuffled before we do this
train = data.take(train_size) # takes train_size number of batches from data
val = data.skip(train_size).take(val_size) # takes val_size batches after train_size batches
test = data.skip(train_size).skip(val_size).take(test_size) # skips train and val size before taking test_size batches


# # # # # # # # # # MODELLING STAGE

#import some deps
# Sequential is great neural network for single input --> output mapping
# Functional is more powerful for more inputs/outputs

# Flatten used to convert convolutional, higher dimensional data back to array of features

# creating model
# adding layers - architectural decisions: number filters, filter size, stride
# look at recent publications to see what their architectural decisions where
model = Sequential()
 # create convolution layer for feature extraction, using 16 filters
model.add(Conv2D(
    32, # 32 filters applied for element-wise feature detection
    (3,3), # filter size
    1, # stride = 1 (moves by 1 pixel)
    activation='relu', # add non-linearity activation to filter negative and preserve positive values, allows detection of non-linear patterns, # another popular activation = sigmoid
    input_shape=(256,256,3)
))
model.add(MaxPooling2D()) # Pool results for complexity reduction, return Max. Default = 2,2 (halves data)

model.add(Conv2D(64, (3,3), 1, activation='relu')) # 32 filters applied onto extracted features
model.add(MaxPooling2D()) # Pool results for complexity reduction

model.add(Conv2D(32, (3,3), 1, activation='relu')) # 32 features
model.add(MaxPooling2D()) # Pool results for complexity reduction


model.add(Flatten())

model.add(Dense(256, activation='relu')) # 256-neuron fully connected neuron network
model.add(Dense(1, activation='sigmoid')) # binary classification for 0-1 based on our earliest classes


# need to compile
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy']) # use adam optimiser, tf.optimizers.

# Writing Model Summary
with open(f"{FILEPATH}/training_results.txt", 'a') as f:
    with redirect_stdout(f):
        model.summary()
f.close()


# # # # # # # # # # TRAINING and VISUALISATIONS
logdir = 'logs'
#os.mkdir(logdir)

# tensorboard allows us to check model progress while its happening
# a monitoring tool
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# model.fit() takes training data, 1 epoch = 1 iteration over entire training data set
# whilst training after batches will evaluate against validation_data 
# also give tensorboard_callback
# can now access all information from training and validation in hist
hist = model.fit(train, epochs=50, validation_data=val, callbacks=[tensorboard_callback])

#print(hist.history) # testing loss/accuracy and validation loss/accuracy

# plotting model training/validation LOSS
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc='upper left')
#plt.show()
plt.savefig(f'{FILEPATH}/model_loss_training.png') # SAVING IMAGE


# INTERPRETATION: both training and validation loss should both be steadily decreasing
# IF validation loss rises while training loss decreases - suggests OVERFITTING, decrease filters!
# Might need deeper higher quality data or more filters! or REGULARIZE!

# plotting model training/validation ACCURACY
fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc='upper left')
#plt.show()
plt.savefig(f'{FILEPATH}/model_accuracy_training.png') # SAVING IMAGE



# # # # # # # # # # TRAINING
# using Precision, Recall and Binary Accuracy prediction - maybe do confusion matrix?
# Precision = 
# Recall = 
# Binary Accuracy = 
pre = Precision()
re = Recall()
acc = BinaryAccuracy()

# iterate through batches in test
for batch in test.as_numpy_iterator():
    x, y = batch
    yhat = model.predict(x) # this is how we make predictions
    pre.update_state(y, yhat) # compare predicted (yhat) to actual (y)
    re.update_state(y, yhat) # compare predicted (yhat) to actual (y)
    acc.update_state(y, yhat) # compare predicted (yhat) to actual (y)

f = open(f"{FILEPATH}/training_results.txt", "a")
f.write(f'Precision:{pre.result().numpy()}; Recall:{re.result().numpy()}; Accuraccy:{acc.result().numpy()}\n')
f.close()

# SAVING MODEL
# model.save('imageclassificationmodel.h5') # serialization - similar to ZIP/RAR, compativle with load_model() function # LEGACY H5 FORMAT
model.save('cervicalcancer_classifier.keras')


# Timestamp
END = datetime.datetime.now()
f = open(f"{FILEPATH}/training_results.txt", "a")
f.write(f"Script duration in h:m:s.ms was {END - START}")
f.close()