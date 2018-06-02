import gc
import sys
import pickle
from keras.layers import *
from keras.models import *
from keras.applications import *
from keras.applications.inception_v3 import preprocess_input
from keras.callbacks import EarlyStopping
from keras.models import load_model
from collections import defaultdict

from pdb import set_trace as bp

import logging

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', \
                    level=logging.INFO, \
                    datefmt='%Y-%m-%d %H:%M:%S')

train_path = sys.argv[1]
val_path = sys.argv[2]
test_path = sys.argv[3]

images_width = images_height = 299

def calculate_features(X):

    logging.info('Loading pre-trained InceptionV3 weights')
    pretrained_model = InceptionV3( \
                                    include_top = False, \
                                    input_shape = ( \
                                        images_width, \
                                        images_height, \
                                        3
                                    ), \
                                    weights = 'imagenet', \
                                    pooling = 'avg' \
                                    )

    logging.info('Creating input layer')
    inputs = Input((images_width, images_height, 3))

    logging.info('Creating pre-process layer')
    preprocess_layer = Lambda(preprocess_input, name = 'preprocessing')(inputs)

    logging.info('Creating output layer')
    outputs = pretrained_model(preprocess_layer)

    logging.info('Creating the transfer learning model')
    model = Model(inputs, outputs)

    logging.info('Generating features matrix')
    features = model.predict(X, batch_size=64, verbose=1)

    return features

saved_model_filename = 'model.h5'

if os.path.isfile(saved_model_filename):

    logging.info('Loading saved model')
    model = load_model(saved_model_filename)

else:

    logging.info('Loading train features file')
    x_train = np.load(train_path + 'x_train.npy')
    x_train_flipped = np.load(train_path + 'x_train_flipped.npy')
    y_train = np.load(train_path + 'y_train.npy')
    X_train = np.concatenate((x_train, x_train_flipped), axis = 0)
    Y_train = np.concatenate((y_train, y_train), axis = 0)
    # X_train = x_train # delete this row
    # Y_train = y_train # delete this row
    num_classes = Y_train.shape[1]

    logging.info('Free up memory')
    del x_train
    del x_train_flipped
    del y_train
    gc.collect()

    logging.info('Calculating train features')
    train_features = calculate_features(X_train)

    logging.info('Loading validation features file')
    x_val = np.load(val_path + 'x_val.npy')
    x_val_flipped = np.load(val_path + 'x_val_flipped.npy')
    y_val = np.load(val_path + 'y_val.npy')
    X_val = np.concatenate((x_val, x_val_flipped), axis = 0)
    Y_val = np.concatenate((y_val, y_val), axis = 0)
    # X_val = x_val # delete this row
    # Y_val = y_val # delete this row

    logging.info('Free up memory')
    del x_val
    del x_val_flipped
    del y_val
    gc.collect()

    logging.info('Calculating val features')
    val_features = calculate_features(X_val)

    logging.info('Creating the breed model')
    input_shape = train_features.shape[1:]
    inputs = Input(input_shape)
    x = Dropout(0.5)(inputs)
    outputs = Dense(num_classes, activation='sigmoid')(x)
    model = Model(inputs, outputs)

    logging.info('Compiling the breed model')
    model.compile(optimizer='Adagrad', loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [EarlyStopping(monitor='val_loss',  patience=3, verbose=1)]

    logging.info('Training the breed model')
    model.fit( \
                train_features, \
                Y_train, \
                validation_data = ( \
                    val_features, \
                    Y_val \
                ), \
                callbacks = callbacks \
            )

    logging.info('Saving the model')
    model.save(saved_model_filename)

logging.info('Loading test features file')
x_test = np.load(test_path + 'x_test.npy')
y_test = np.load(test_path + 'y_test.npy')

logging.info('Calculating test features')
test_features = calculate_features(x_test)

logging.info('Predicting breeds')
predictions = model.predict(test_features, batch_size=128)
actual_classes = [np.argmax(x) for x in y_test]
predicted_classes = [np.argmax(x) for x in predictions]

logging.info('Predictions')
logging.info('True classes: ' + str(actual_classes))
logging.info('Predicted classes: ' + str(predicted_classes))

correct_predictions = {}
correct_predictions = defaultdict(lambda: 0, correct_predictions)
total_predictions = {}
total_predictions = defaultdict(lambda: 0, total_predictions)
for idx, val in enumerate(actual_classes):
    if actual_classes[idx] == predicted_classes[idx]:
        correct_predictions[val] += 1
    total_predictions[val] += 1

accuracies = {}
accuracies = defaultdict(lambda: 0, accuracies)
accuracy = 0
for key, value in total_predictions.items():
    if key in correct_predictions:
        accuracies[key] = correct_predictions[key] / total_predictions[key]
    else:
        accuracies[key] = 0
    logging.info('Accuracy for class ' + str(key) + ': ' + str(accuracies[key]))
    accuracy += accuracies[key]

accuracy /= len(accuracies)
logging.info('Normalized accuracy: ' + str(accuracy))
