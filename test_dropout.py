'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import numpy as np
from keras.models import model_from_yaml


class VotingModel(object):

    def __init__(self, model_list, voting='hard',
                 weights=None, nb_classes=None):
        """(Weighted) majority vote model for a given list of Keras models.
        Parameters
        ----------
        model_list: An iterable of Keras models.
        voting: Choose 'hard' for straight-up majority vote of highest model probilities or 'soft'
            for a weighted majority vote. In the latter, a weight vector has to be specified.
        weights: Weight vector (numpy array) used for soft majority vote.
        nb_classes: Number of classes being predicted.
        Returns
        -------
        A voting model that has a predict method with the same signature of a single keras model.
        """
        self.model_list = model_list
        self.voting = voting
        self.weights = weights
        self.nb_classes = nb_classes

        if voting not in ['hard', 'soft']:
            raise 'Voting has to be either hard or soft'

        if weights is not None:
            if len(weights) != len(model_list):
                raise ('Number of models {0} and length of weight vector {1} has to match.'
                       .format(len(weights), len(model_list)))

    def predict(self, X, batch_size=128, verbose=1):
        predictions = list(map(lambda model: model.predict(X, batch_size, verbose), self.model_list))
        nb_preds = len(X)

        if self.voting == 'hard':
            for i, pred in enumerate(predictions):
                pred = list(map(
                    lambda probas: np.argmax(probas, axis=-1), pred
                ))
                predictions[i] = np.asarray(pred).reshape(nb_preds, 1)
            argmax_list = list(np.concatenate(predictions, axis=1))
            votes = np.asarray(list(
                map(lambda arr: max(set(arr)), argmax_list)
            ))
        if self.voting == 'soft':
            for i, pred in enumerate(predictions):
                pred = list(map(lambda probas: probas * self.weights[i], pred))
                predictions[i] = np.asarray(pred).reshape(nb_preds, self.nb_classes, 1)
            weighted_preds = np.concatenate(predictions, axis=2)
            weighted_avg = np.mean(weighted_preds, axis=2)
            votes = np.argmax(weighted_avg, axis=1)

        return votes


def voting_model_from_yaml(yaml_list, voting='hard', weights=None):
    model_list = map(lambda yml: model_from_yaml(yml), yaml_list)
    return VotingModel(model_list, voting, weights)

batch_size = 128
num_classes = 10
epochs = 2

class TestTimeDropout(Dropout):
    def __init__(self, p):
        super(TestTimeDropout, self).__init__(p)

    def call(self, inputs, training=None):
        noise_shape = self._get_noise_shape(inputs)

        return K.dropout(inputs, self.rate, noise_shape,
                             seed=self.seed)

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

_ytest = y_test
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(TestTimeDropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(TestTimeDropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(TestTimeDropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

_ml = (model for i in range(10))

ensamble = VotingModel(list(_ml), nb_classes=num_classes)

print('\ntest accuracy ensamble:', np.mean(ensamble.predict(x_test) == _ytest))
print(model.predict(x_test))
print('\ntest accuracy single:', np.mean(model.predict(x_test) == _ytest))

model.save("dropout_model.keras")
