import tensorflow as tf
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras import backend as K
from utils import loss_mae, loss_function


def forward_network(units_layer1=300, units_layer2=800, optimizer='rmsprop'):
    inp = Input(shape=(12,), name='forward_input')
    x = inp
    x = Dense(units_layer1)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(units_layer2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    out = Dense(30, activation=None)(x)
    model = Model(inputs=inp, outputs=out)

    if optimizer == 'sgd':
        optimizer = SGD(lr=0.001, momentum=0.9, nesterov=True)
    elif optimizer == 'adam':
        optimizer = Adam(learning_rate=0.001)
    elif optimizer == 'rmsprop':
        optimizer = RMSprop(learning_rate=0.001)

    model.compile(loss=loss_function, optimizer=optimizer, metrics=[loss_mae])
    return model


def inverse_network(units_layer1=128, units_layer2=600, optimizer='rmsprop'):
    inp = Input(shape=(30,), name='forward_input')
    x = inp
    x = Dense(units_layer1)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(units_layer2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    out1 = Dense(4, activation=None)(x)
    out2 = Dense(4, activation=tf.keras.activations.softmax)(x)
    out3 = Dense(4, activation=tf.keras.activations.softmax)(x)
    out = K.concatenate((out1, out2, out3), axis=-1)
    model = Model(inputs=inp, outputs=out)

    if optimizer == 'sgd':
        optimizer = SGD(lr=0.001, momentum=0.9, nesterov=True)
    elif optimizer == 'adam':
        optimizer = Adam(learning_rate=0.001)
    elif optimizer == 'rmsprop':
        optimizer = RMSprop(learning_rate=0.001)

    model.compile(loss=loss_function, optimizer=optimizer, metrics=[loss_mae])
    return model
