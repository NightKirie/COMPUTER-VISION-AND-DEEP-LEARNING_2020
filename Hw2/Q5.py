from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, add, Input
from tensorflow.keras.layers import Conv2D, MaxPool2D, ZeroPadding2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers

from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing import image
from shutil import copyfile, rmtree
import os
import time
import cv2
import random

import numpy as np
import tensorflow as tf

ROOT_DIR = "Q5_Image"

IMAGE_SIZE = (224, 224)
NUM_CLASSES = 2
BATCH_SIZE = 16
FREEZE_LAYERS = 2
NUM_EPOCHS = 20

def Get_Random_Eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        if input_img.ndim == 3:
            img_h, img_w, img_c = input_img.shape
        elif input_img.ndim == 2:
            img_h, img_w = input_img.shape

        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            if input_img.ndim == 3:
                c = np.random.uniform(v_l, v_h, (h, w, img_c))
            if input_img.ndim == 2:
                c = np.random.uniform(v_l, v_h, (h, w))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w] = c

        return input_img

    return eraser


def Dataset_Preproscess():
    cat_files = os.listdir(f"{ROOT_DIR}/Cat")
    dog_files = os.listdir(f"{ROOT_DIR}/Dog")
    cat_train, cat_valid = train_test_split(cat_files, test_size=0.2)
    dog_train, dog_valid = train_test_split(dog_files, test_size=0.2)
    
    train_dir = f'{ROOT_DIR}/train'
    valid_dir = f'{ROOT_DIR}/valid'
    
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
        copyfile(os.path.join(ROOT_DIR, 'Cat', file_name), os.path.join(train_dir, 'cat', file_name))
    for file_name in dog_train:
        copyfile(os.path.join(ROOT_DIR, 'Dog', file_name), os.path.join(train_dir, 'dog', file_name))
    print("Finish create train files")

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
        copyfile(os.path.join(ROOT_DIR, 'Cat', file_name), os.path.join(valid_dir, 'Cat', file_name))
    for file_name in dog_valid:
        copyfile(os.path.join(ROOT_DIR, 'Dog', file_name), os.path.join(valid_dir, 'cat', file_name))
    print("Finish create valid files")
    

def identity_block(input_tensor, kernel_size, filters, stage, block):
    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):

    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, kernel_size=(1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

def MyResNet50():
    img_input = Input(shape=(224, 224, 3)) # image size is 224x224

    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPool2D((3, 3), strides=(2, 2))(x)

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
    TF_WEIGHTS_PATH_NO_TOP = './Q5_Image/pre_train_weight/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'  
    base_model.load_weights(TF_WEIGHTS_PATH_NO_TOP)

    x = AveragePooling2D((7, 7), name='avg_pool')(base_model.output)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)
   
    model = Model(inputs=base_model.input, outputs=x)
    
    for layer in model.layers[:-FREEZE_LAYERS]:
        layer.trainable = False

    for layer in model.layers[-FREEZE_LAYERS:]:
        layer.trainable = True
    print(model.summary())

    return model

def Training(is_rand_erase):    
    train_datagen = ImageDataGenerator(rotation_range=40,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True,
                                        fill_mode='nearest',
                                        preprocessing_function=(Get_Random_Eraser(v_l=0, v_h=1, pixel_level=False) if is_rand_erase else None))
    train_generator = train_datagen.flow_from_directory(os.path.join(ROOT_DIR, "train"),  # this is the target directory
                                                        target_size=IMAGE_SIZE,  # all images will be resized to 224x224
                                                        batch_size=BATCH_SIZE,
                                                        interpolation='bicubic',
                                                        class_mode='categorical',
                                                        seed=42,
                                                        shuffle=True)
    
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = validation_datagen.flow_from_directory(os.path.join(ROOT_DIR, "valid"),  # this is the target directory
                                                                  target_size=IMAGE_SIZE,  # all images will be resized to 224x224
                                                                  batch_size=BATCH_SIZE,
                                                                  interpolation='bicubic',
                                                                  class_mode='categorical',
                                                                  seed=42,
                                                                  shuffle=False)
    



    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model_path = os.path.join(ROOT_DIR, f"result_{NUM_EPOCHS}_{BATCH_SIZE}")
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    checkpoint_dir = os.path.join(ROOT_DIR, "checkpoint")
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    log_dir = os.path.join(ROOT_DIR, "log")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    tbCallBack = TensorBoard(log_dir=log_dir,  
                             histogram_freq=0,  
                             #BATCH_SIZE=600,     
                             write_graph=True, 
                             write_grads=True, 
                             write_images=True,
                             embeddings_freq=0, 
                             embeddings_layer_names=None, 
                             embeddings_metadata=None)
    
    model = MyResNet50()
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=1e-3), metrics=['accuracy'])
    
    filepath = os.path.join(model_dir ,"model_{epoch:03d}_{val_accuracy:.3f}.hdf5")
    checkpoint = ModelCheckpoint(filepath, verbose=1, monitor='val_accuracy', save_best_only=True)
    model.fit_generator(train_generator,
                                  epochs=NUM_EPOCHS,
                                  steps_per_epoch=train_generator.samples // BATCH_SIZE,
                                  validation_data=validation_generator,
                                  validation_steps=validation_generator.samples // BATCH_SIZE,
                                  callbacks=[tbCallBack, checkpoint])
    model.save(os.path.join(model_path, "model.h5"))

def Show_Train_Result():
    img_1 = cv2.imread("./Q5_Image/5.1/1.jpg")
    img_2 = cv2.imread("./Q5_Image/5.1/2.jpg")
    cv2.imshow("train result 1", img_1)
    cv2.imshow("train result 2", img_2)

def Show_Tensorboard():
    img = cv2.imread("./Q5_Image/5.2/1.jpg")
    cv2.imshow("tensorboard", img)

def Classify_Random_Picture():
    model = MyResNet50()
    model.load_weights("./Q5_Image/model/model_normal.h5")
    cat_or_dog = random.randint(0, 1) # 0 as cat, 1 as dog
    while True:
        img_idx = random.randint(0, 12499)
        img_path = f"./Q5_Image/{'Cat' if cat_or_dog == 0 else 'Dog'}/{img_idx}.jpg"
        img = image.load_img(img_path, target_size=(224, 224))
        if img is None:
            continue
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        pred = model.predict(x)[0]
        top_inds = pred.argsort()[::-1][:5]
        cv2_img = cv2.imread(img_path)
        cv2.imshow(f"{'Cat' if top_inds[0] == 0 else 'dog'}", cv2_img)
        for i in top_inds:
            print('    {:.3f}  {}'.format(pred[i], i))
        break

def Show_Acc_Diff():
    img = cv2.imread("./Q5_Image/5.4/1.jpg")
    cv2.imshow("Acc Compare", img)

if __name__ == "__main__":
    # For generate train & valid data, only need to do once
    # Dataset_Preproscess()
    # For training normally
    # Training(False)
    # For training with random eraser
    # Training(True)
    


