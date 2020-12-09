from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, add, Input
from tensorflow.keras.layers import Conv2D, MaxPool2D, ZeroPadding2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

# from tensorflow.keras.models import model_from_json
from tensorflow.keras import backend as K
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import random
from shutil import copyfile, rmtree
import os
import time
# import cv2

# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt

root_dir = "./Q5_Image"

TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


def Dataset_Preproscess():
    cat_files = os.listdir(f"{root_dir}/Cat")
    dog_files = os.listdir(f"{root_dir}/Dog")
    cat_train, cat_valid = train_test_split(cat_files, test_size=0.1)
    dog_train, dog_valid = train_test_split(dog_files, test_size=0.1)
    
    train_dir = f'{root_dir}/train'
    valid_dir = f'{root_dir}/valid'
    
    if os.path.exists(train_dir):
        rmtree(train_dir, ignore_errors=True)
    timeout = 0.001
    while True:
        try:
            os.mkdir(train_dir)
            os.mkdir(f'{train_dir}/cat')
            os.mkdir(f'{train_dir}/dog')
            break
        except PermissionError as e:
            if e.winerror != 5 or timeout >= 2:
                raise
            time.sleep(timeout)
            timeout *= 2

    for file_name in cat_train:
        copyfile(f'{root_dir}/Cat/{file_name}', f'{train_dir}/cat/{file_name}')
    for file_name in dog_train:
        copyfile(f'{root_dir}/Dog/{file_name}', f'{train_dir}/dog/{file_name}')

    if os.path.exists(valid_dir):
        rmtree(valid_dir, ignore_errors=True)
    timeout = 0.001
    while True:
        try:
            os.mkdir(valid_dir)
            os.mkdir(f'{valid_dir}/cat')
            os.mkdir(f'{valid_dir}/dog')
            break
        except PermissionError as e:
            if e.winerror != 5 or timeout >= 2:
                raise
            time.sleep(timeout)
            timeout *= 2

    for file_name in cat_valid:
        copyfile(f'{root_dir}/Cat/{file_name}', f'{valid_dir}/cat/{file_name}')
    for file_name in dog_valid:
        copyfile(f'{root_dir}/Dog/{file_name}', f'{valid_dir}/dog/{file_name}')
    

def identity_block(input_tensor, kernel_size, filters, stage, block):
    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_data_format() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters=nb_filter1, kernel_size=(1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=nb_filter2, kernel_size=(kernel_size, kernel_size), padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=nb_filter3, kernel_size=(1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):

    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_data_format() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters=nb_filter1, kernel_size=(1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=nb_filter2, kernel_size=(kernel_size, kernel_size), padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=nb_filter3, kernel_size=(1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters=nb_filter3, kernel_size=(1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x

def ResNet50():
    
    img_input = Input(shape=(224, 224, 3)) # image size is 224x224

    x = ZeroPadding2D(padding=(3, 3))(img_input)
    x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    base_model = Model(img_input, x)

    x = AveragePooling2D((7, 7), name='avg_pool')(base_model.output)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs=base_model.input, outputs=x)

    top_num = 4
    for layer in model.layers[:-top_num]:
        layer.trainable = False

    for layer in model.layers[-top_num:]:
        layer.trainable = True
    print(model.summary())

    return model

def Training():
    
    image_size = (224, 224)
    
    train_datagen = ImageDataGenerator(rescale=1.0/255)
    train_generator = train_datagen.flow_from_directory('Q5_Image/train',  # this is the target directory
                                                        target_size=image_size,  # all images will be resized to 224x224
                                                        batch_size=50,
                                                        class_mode='binary')
    
    validation_datagen = ImageDataGenerator(rescale=1.0/255)
    validation_generator = validation_datagen.flow_from_directory('Q5_Image/valid',  # this is the target directory
                                                                  target_size=image_size,  # all images will be resized to 224x224
                                                                  batch_size=50,
                                                                  class_mode='binary')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model = ResNet50()
    model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    model_path = f"{root_dir}/model"
    if os.path.exists(model_path):
        os.mkdir(model_path)
    filepath = os.path.join(model_path ,"model_{epoch:03d}_{val_accuracy:.3f}.hdf5")
    checkpoint = ModelCheckpoint(filepath, verbose=1, monitor='val_accuracy', save_best_only=True)
    History = model.fit(train_generator,
                        shuffle=True,
                        steps_per_epoch=2048,
                        epochs=40,
                        batch_size=500,
                        validation_data=validation_generator,
                        validation_steps=1024,
                        callbacks=[checkpoint, TensorBoard(log_dir=f"{root_dir}/logs", histogram_freq=1)])
    # History = full_model.fit([train_re, train_re_org, train_qp, train_cu], train_label,
    #                          shuffle=True,
    #                          epochs=steps, 
    #                          batch_size=batch_size, 
    #                          validation_split=0.2,
    #                          callbacks=[TensorBoard(log_dir='./logs', histogram_freq=1),best_model])


if __name__ == "__main__":
    # For generate train & valid data, only need to do once
    # Dataset_Preproscess()
    # For training
    Training()
    
    

