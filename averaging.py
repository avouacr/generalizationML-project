
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Dense, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.models import Sequential
from keras.callbacks import EarlyStopping, Callback


def load_cifar_data():
    """Loads the CIFAR-10 dataset using Keras and preprocess for training."""
    # Download / Load the datasets
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize the data
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Label vector
    labels = [
        'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
        'frog', 'horse', 'ship', 'truck'
    ]

    return x_train, y_train, x_test, y_test, labels


def cnn_model(input_shape, num_classes, optimizer):
    """Compile CNN model for a given optimizer."""
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])
    return model


def train_model(model, x, y, batch_size=32, epochs=10, file_name=None, callbacks=[]):
    """
    Trains the model on the given data.
    """

    stop_early = EarlyStopping(monitor='val_acc', min_delta=0, patience=2, mode='auto')
    callbacks = [stop_early] + callbacks

    ret = model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_split=0.2,
                    shuffle=True, callbacks=callbacks, verbose=1)

    if file_name:
        model.save(file_name)

    return ret


class Averaging(Callback):

    def __init__(self, share_avg):
        super(Averaging, self).__init__()
        self.share_avg = share_avg
        self.weights_to_avg = []

    def on_train_begin(self, logs={}):
        self.n_epochs = self.params['epochs']
        self.epoch_start_avg = int((1 - self.share_avg) * self.n_epochs)
        n_epochs_avg = self.n_epochs - self.epoch_start_avg
        print(f'Averaging SGD weights of last {n_epochs_avg} epochs.')
        print(self.epoch_start_avg)

    def on_epoch_end(self, epoch, logs={}):
        # Keep record of all weights after the iteration chosen to start avg
        if epoch > self.epoch_start_avg:
            wgt = np.array(self.model.get_weights())
            self.weights_to_avg.append(wgt)

    def on_train_end(self, logs={}):
        # Output average of stored weights
        averaged_weights = np.mean(self.weights_to_avg, axis=0)
        self.model.set_weights(averaged_weights)
        print('Final model parameters set to averaged weights.')


x_train, y_train, x_test, y_test, labels = load_cifar_data()

sgd = SGD(lr=0.1, decay=1.0e-6)
cnn_sgd = cnn_model(input_shape=x_train.shape[1:], num_classes=10, optimizer=sgd)

avg_callback = Averaging(share_avg=0.5)
cnn_sgd_trained = train_model(cnn_sgd, x_train, y_train, batch_size=32, epochs=4,
                              callbacks=[avg_callback])
