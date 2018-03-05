import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

import os
import os.path as path

import numpy as np
from PIL import Image as im

MODEL_NAME = 'DevnagariIdentifierKerasModel'
EPOCHS = 5
BATCH_SIZE = 36

image_resource_path = 'resource/consonants/'
resized_image_resource_path = 'resized_resource/'

# get labels from resource
labels = os.listdir(image_resource_path)
if '.DS_Store' in labels: labels.remove('.DS_Store')

filenames = []

imagevector = []    #will eventually be an array of shape (7380, 1297)

no_of_data_per_label = 205

# get data from resource
for label in labels:
    print('obtaining data from folder : :  ' + label )
    filenames.append(os.listdir(image_resource_path+'/'+label+'/'))
    if '.BridgeSort' in filenames[0]: filenames[0].remove('.BridgeSort')
    for file in filenames[0]:
        imagevector.append(np.insert(np.reshape(np.asarray(im.open(image_resource_path + label + '/' + file), dtype=float),
                                       (-1, 1296)), 0, [float(label)]))

    filenames = []

print(imagevector[0])
print(np.shape(imagevector))


# resize resource and save in resized_resource however all images have been resized to 36x36


def build_model():
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, \
            padding='same', activation='relu', \
            input_shape=[36, 36, 1]))

    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))

    model.add(Conv2D(filters=128, kernel_size=3, strides=1, \
            padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))

    model.add(Conv2D(filters=256, kernel_size=3, strides=1, \
            padding='same', activation='relu'))

    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(36, activation='softmax'))
    return model


# train


def train(model, x_train, y_train, x_test, y_test):
    model.compile(loss=keras.losses.categorical_crossentropy, \
                  optimizer=keras.optimizers.Adadelta(), \
                  metrics=['accuracy'])

    model.fit(x_train, y_train, \
              batch_size=BATCH_SIZE, \
              epochs=EPOCHS, \
              verbose=1, \
              validation_data=(x_test, y_test))

# test

# export


def export_model(saver, model, input_node_names, output_node_name):

    tf.train.write_graph(K.get_session().graph_def, 'out', \
        MODEL_NAME + '_graph.pbtxt')

    saver.save(K.get_session(), 'out/' + MODEL_NAME + '.chkp')

    freeze_graph.freeze_graph('out/' + MODEL_NAME + '_graph.pbtxt', None, \
        False, 'out/' + MODEL_NAME + '.chkp', output_node_name, \
        "save/restore_all", "save/Const:0", \
        'out/frozen_' + MODEL_NAME + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + MODEL_NAME + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_' + MODEL_NAME + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("graph saved!")





def load_data(vectordata):
    # splitting vector data into training and test, label and data
    x_train = vectordata[:5900, 1:]
    print(np.shape(x_train))
    y_train = vectordata[:5900, :1]
    print(np.shape(y_train))
    x_test = vectordata[5900:, 1:]
    print(np.shape(x_test))
    y_test = vectordata[5900:, :1]
    print(np.shape(y_test))
    y_train = (np.reshape(y_train, (5900,))).astype(int)-1
    y_test = (np.reshape(y_test, (1480,))).astype(int)-1
    x_train = x_train.reshape(x_train.shape[0], 36, 36, 1)
    x_test = x_test.reshape(x_test.shape[0], 36, 36, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = keras.utils.to_categorical(y_train, 36)
    y_test = keras.utils.to_categorical(y_test, 36)
    return x_train, y_train, x_test, y_test


# main runner

def main():
    if not path.exists('out'):
        os.mkdir('out')

    #resize_from_folder() dataset already resized to 36x36
    npimagevector = np.asarray(imagevector)

    print('before shuffle:  ')
    print(npimagevector[:1, :])
    np.random.shuffle(npimagevector)
    print('after shuffle:  ')
    print(npimagevector[:1, :])
    x_train, y_train, x_test, y_test = load_data(npimagevector)

    model = build_model()

    train(model, x_train, y_train, x_test, y_test)

    export_model(tf.train.Saver(), model, ["conv2d_1_input"], "dense_2/Softmax")


if __name__ == '__main__':
    main()
