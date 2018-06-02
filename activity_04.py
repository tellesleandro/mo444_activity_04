import cv2
import numpy as np
import pandas as pd
import random
from keras.layers import *
from keras.models import *
from keras.applications import *
from keras.optimizers import *
from keras.applications.inception_v3 import preprocess_input
from keras.callbacks import EarlyStopping
import os

from pdb import set_trace as bp

#path is the path that we save data and dpath the path from where we read data
#paths and be path can be the same or not
path='/tmp/'
dpath='/home/leandro/.kaggle/competitions/dog-breed-identification/'

labelsdf = pd.read_csv(dpath+'labels.csv')
ssdf = pd.read_csv(dpath+'sample_submission.csv')

# import sys
# print("Python version %s.%s.%s" % sys.version_info[:3])
# print("Tensorflow version %s" % tf.__version__)
# print("Keras version %s" % keras.__version__)

#Read train images and labels
train_images_len = labelsdf.shape[0]
breed = ssdf.columns[1:]
num_class = len(breed)

width = 299
x_train = np.zeros((train_images_len, width, width, 3), dtype=np.uint8)
y_train = np.zeros((train_images_len, num_class), dtype=np.uint8)
for i in range(train_images_len):
    x_train[i] = cv2.resize(cv2.imread(dpath+'train/%s.jpg' % labelsdf['id'][i]), (width, width), interpolation = cv2.INTER_AREA)
    y_train[i][np.where(breed ==  labelsdf['breed'][i])[0][0] ] = 1

#BGR to RGB
x_train=x_train[...,::-1]

def get_features(MODEL, data=x_train):
    #Extract Features Function from GlobalAveragePooling2D layer
    pretrained_model = MODEL(include_top=False, input_shape=(width, width, 3), weights='imagenet', pooling='avg')

    inputs = Input((width, width, 3))
    x = Lambda(preprocess_input, name = 'preprocessing')(inputs)
    outputs = pretrained_model(x)
    model = Model(inputs, outputs)

    features = model.predict(data, batch_size=64, verbose=1)
    return features

bp()

# Extract features from the original images:
x_xception = get_features(Xception, x_train)
x_inception = get_features(InceptionV3, x_train)
features_train1 = np.concatenate([x_xception, x_inception], axis=-1)

# Extract features from flipped images:
x_xception = get_features(Xception, np.flip(x_train,axis=2))
x_inception = get_features(InceptionV3, np.flip(x_train,axis=2))
features_train2 = np.concatenate([x_xception, x_inception], axis=-1)

del x_train
import gc
gc.collect()

#Read test images
test_images_len = len(ssdf)
x_test = np.zeros((test_images_len, width, width, 3), dtype=np.uint8)
for i in range(test_images_len):
    x_test[i] = cv2.resize(cv2.imread(dpath+'test/%s.jpg' % ssdf['id'][i]), (width, width))


#BGR to RGB
x_test=x_test[...,::-1]


#Extract Features
x_test_xception = get_features(Xception, x_test)
x_test_inception = get_features(InceptionV3, x_test)
features_test1 = np.concatenate([x_test_xception, x_test_inception], axis=-1)
#Features from flipped images
x_test_xception = get_features(Xception, np.flip(x_test,axis=2))
x_test_inception = get_features(InceptionV3, np.flip(x_test,axis=2))
features_test2 = np.concatenate([x_test_xception, x_test_inception], axis=-1)

#5 fold predictions with horizontal image flips
allrows=([x for x in range(features_train1.shape[0])])
trainpreds=np.zeros((features_train1.shape[0],120))
testpreds=np.zeros((features_test1.shape[0],120))
callbacks = [EarlyStopping(monitor='val_loss',  patience=3, verbose=1)]

input_shape = features_train1.shape[1:]
for i in range(5):
    inputs = Input(input_shape)
    x = Dropout(0.5)(inputs)
    outputs = Dense(num_class, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='Adagrad', loss='categorical_crossentropy', metrics=['accuracy'])

    inputs = Input(input_shape)
    x = Dropout(0.5)(inputs)
    outputs = Dense(num_class, activation='softmax')(x)
    model2 = Model(inputs, outputs)
    model2.compile(optimizer='Adagrad', loss='categorical_crossentropy', metrics=['accuracy'])
    if i !=4:
        valrows=([x for x in range(2044*i,2044*(i+1))])
        trainrows=([x for x in allrows if x not in valrows])

    else:
        valrows=([x for x in range(2044*i,features_train1.shape[0])])
        trainrows=np.array([x for x in allrows if x not in valrows])

    h = model.fit(features_train1[trainrows,:], y_train[trainrows,:], batch_size=128, epochs=50,
              validation_data=(features_train1[valrows,:], y_train[valrows,:]), callbacks=callbacks)
    h2 = model2.fit(features_train2[trainrows,:], y_train[trainrows,:], batch_size=128, epochs=50,
              validation_data=(features_train2[valrows,:], y_train[valrows,:]), callbacks=callbacks)

    y_pred_train = model.predict(features_train1[valrows,:], batch_size=128)
    y_pred_train += model2.predict(features_train2[valrows,:], batch_size=128)
    trainpreds[valrows,:]=y_pred_train/2
    y_pred = model.predict(features_test1, batch_size=128)
    y_pred2 = model2.predict(features_test2, batch_size=128)
    testpreds += (y_pred+y_pred2)/2
testpreds/=5

#Create submission file
for b in ssdf.columns[1:]:
    ssdf[b] = testpreds[:,np.where(breed == b)[0][0]]
ssdf.to_csv('XceptInceptSubmissionWithFlips.csv', index=None)
