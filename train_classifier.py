
import numpy as np
import os
import PIL
from PIL import Image
import pandas as pd

import tensorflow
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAvgPool2D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array


def create_model(num_class):
    backbone = ResNet50(input_shape=(224,224,3), include_top=False)
    x = backbone.output
    x = GlobalAvgPool2D()(x)
    x = Dense(1024, activation='relu')(x)
    output = Dense(num_class, activation='sigmoid')(x)
    opt = Adam(learning_rate=0.0001)
    model = Model(inputs=backbone.input, outputs=output)
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=opt)
    return model


def generate_samples(samples_location):
    datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    train_generator = datagen.flow_from_directory(
            samples_location,
            target_size=(224, 224),
            batch_size=1,
            class_mode='binary')
    return train_generator


model = create_model(1)
model.summary()

samples = input("Masukkan Lokasi Samples: ")
data = generate_samples(samples)
data.class_indices

model.fit_generator(train_generator, epochs=100, verbose=1, validation_data=train_generator, workers=4, shuffle=True)
model.save('/ResNet50_classifier_model.h5')