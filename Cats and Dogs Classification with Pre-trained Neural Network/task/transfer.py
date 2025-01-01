import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import requests
from zipfile import ZipFile
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Model

if __name__ == '__main__':

    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    # if not os.path.exists('../ImageData'):
    #     os.mkdir('../ImageData')

    if not os.path.exists('../SavedModels'):
        os.mkdir('../SavedModels')

    if not os.path.exists('../SavedHistory'):
        os.mkdir('../SavedHistory')

    # Download data if it is unavailable.
    if 'cats-and-dogs-images.zip' not in os.listdir('../Data'):
        sys.stderr.write("[INFO] Image dataset is loading.\n")
        url = "https://www.dropbox.com/s/jgv5zpw41ydtfww/cats-and-dogs-images.zip?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/cats-and-dogs-images.zip', 'wb').write(r.content)
        sys.stderr.write("[INFO] Loaded.\n")

        sys.stderr.write("\n[INFO] Extracting files.\n")
        with ZipFile('../Data/cats-and-dogs-images.zip', 'r') as zip:
            zip.extractall(path="../Data")
            sys.stderr.write("[INFO] Completed.\n")


    image_height = image_width = 200
    batch_size = 64
    learning_rate = 1e-5
    epochs = 5

    data_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_data_gen = data_gen.flow_from_directory(directory='../Data/train',
                                                  target_size=(image_height, image_width),
                                                  batch_size=batch_size,
                                                  class_mode='categorical')
    valid_data_gen = data_gen.flow_from_directory(directory='../Data/valid',
                                                  target_size=(image_height, image_width),
                                                  batch_size=batch_size,
                                                  class_mode='categorical')
    test_data_gen = data_gen.flow_from_directory(directory='../Data/',
                                                 target_size=(image_height, image_width),
                                                 batch_size=batch_size,
                                                 classes=['test'],
                                                 shuffle=False)

    model = load_model('../SavedModels/stage_four_model.h5')

    n = 1

    for layer in model.layers[0].layers[-n:]:
        layer.trainable = True

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_data_gen,
                        epochs=epochs,
                        validation_data=valid_data_gen,
                        steps_per_epoch=len(train_data_gen),
                        validation_steps=len(valid_data_gen),
                        verbose=1)


    pred = model.predict(test_data_gen).argmax(axis=1)

    with open('../SavedHistory/stage_five_history', 'wb') as file:
        pickle.dump(pred, file)
