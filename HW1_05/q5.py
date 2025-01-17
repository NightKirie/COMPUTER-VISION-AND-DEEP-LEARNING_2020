from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.utils import get_file
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


LABEL_NAME = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

class LearningRateScheduler(tf.keras.callbacks.Callback):
    """Learning rate scheduler which sets the learning rate according to schedule.

    Arguments:
        schedule: a function that takes an epoch index
            (integer, indexed from 0) and current learning rate
            as inputs and returns a new learning rate as output (float).
    """

    def __init__(self, schedule):
        super(LearningRateScheduler, self).__init__()
        self.schedule = schedule

    def on_train_begin(self, logs=None):
        print("Training start, remove previous values")
        if os.path.exists("Q5_data/train_acc.txt"):
            os.remove(("Q5_data/train_acc.txt"))
        
        if os.path.exists("Q5_data/test_acc.txt"):
            os.remove(("Q5_data/test_acc.txt"))

        if os.path.exists("Q5_data/train_loss.txt"):
            os.remove(("Q5_data/train_loss.txt"))

    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch+1} end, store values")
        with open("Q5_data/train_acc.txt", "a+") as f:
            f.write(str(logs['accuracy']) + "\n")
            f.close()

        with open("Q5_data/test_acc.txt", "a+") as f:
            f.write(str(logs['val_accuracy']) + "\n")
            f.close()

        with open("Q5_data/train_loss.txt", "a+") as f:
            f.write(str(logs['loss']) + "\n")
            f.close()


class cifar10vgg:
    def __init__(self, batch_size, maxepoches, learning_rate, train, load):
        self.num_classes = 10
        self.weight_decay = 0.0005
        self.x_shape = [32,32,3]
        self.batch_size = batch_size
        self.maxepoches = maxepoches
        self.learning_rate = learning_rate
        self.loss_list = []
        self.accuracy_list = []
        self.model = self.build_model()
        if train:
            if load:
                self.model.load_weights('Q5_data/cifar10vgg.h5')
            self.model = self.train(self.model)
        else:
            if os.path.exists("Q5_data/cifar10vgg.h5"):
                self.model.load_weights('Q5_data/cifar10vgg.h5')
            else:
                print("No pre-train model found")


    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

        model = Sequential()
        weight_decay = self.weight_decay

        model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=self.x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))
        return model


    def normalize(self,X_train,X_test):
        # this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(X_train,axis=(0,1,2,3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7)
        return X_train, X_test

    def normalize_production(self,x):
        # this function is used to normalize instances in production according to saved training set statistics
        # Input: X - a training set
        # Output X - a normalized training set according to normalization constants.

        # these values produced during first training and are general for the standard cifar10 training set normalization
        mean = 120.707
        std = 64.15
        return (x-mean)/(std+1e-7)

    def predict(self,x,normalize=True,batch_size=50):
        if normalize:
            x = self.normalize_production(x)
        return self.model.predict(x,self.batch_size)

    def train(self,model):
        # training parameters
        lr_decay = 1e-6
        lr_drop = 20

        # The data, shuffled and split between train and test sets:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train, x_test = self.normalize(x_train, x_test)

        y_train = tf.keras.utils.to_categorical(y_train, self.num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, self.num_classes)

        def lr_scheduler(epoch):
            return self.learning_rate * (0.5 ** (epoch // lr_drop))
        #reduce_lr = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

        # data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # optimization details
        sgd = optimizers.SGD(lr=self.learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])


        # training process in a for loop with learning rate drop every 25 epoches.
        historytemp = model.fit(
            datagen.flow(x_train, y_train, batch_size=self.batch_size),
            steps_per_epoch=x_train.shape[0] // self.batch_size,
            epochs=self.maxepoches,
            validation_data=(x_test, y_test),
            callbacks=[LearningRateScheduler(lr_scheduler)],
            verbose=2)

        model.save_weights('Q5_data/cifar10vgg.h5')
        return model
    
    def show_random_10_img(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        fig=plt.figure()
        for i in range(1, 11):
            rand = random.randint(0, len(x_train))
            fig.add_subplot(1, 10, i)
            plt.imshow(x_train[rand])
            plt.xlabel(LABEL_NAME[y_train[rand][0]])
        plt.show()


    def show_model_structure(self):
        print(self.model.summary())

    def show_hyperparameters(self):
        print("hyperparameters:")
        print(f"batch size: {self.batch_size}")
        print(f"learning rate: {self.learning_rate}")
        print(f"optimizer: SGD")

def train_model():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    train_acc_list = []
    train_loss_list = []
    test_acc_list = []


    model = cifar10vgg(
        batch_size=128, 
        maxepoches=100, 
        learning_rate=0.1, 
        train=True,
        load=False)

    plot_acc_loss()

def plot_acc_loss():
    train_acc_list = []
    test_acc_list = []
    train_loss_list = []

    with open("Q5_data/train_acc.txt", "r") as f:
        train_acc_list = [eval(i) for i in f.read().splitlines()]
        f.close()

    with open("Q5_data/test_acc.txt", "r") as f:
        test_acc_list = [eval(i) for i in f.read().splitlines()]
        f.close()

    with open("Q5_data/train_loss.txt", "r") as f:
        train_loss_list = [eval(i) for i in f.read().splitlines()]
        f.close()

    x = [i for i in range(1, len(train_acc_list)+1)]
    plt.subplot(2, 1, 1)
    plt.plot(x, train_acc_list, label="training")
    plt.plot(x, test_acc_list, label="testing")
    plt.legend()
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel("1 = 100%")
    plt.subplot(2, 1, 2)
    plt.plot(x, train_loss_list)
    plt.title('loss')
    plt.xlabel('epoch')
    plt.ylabel("loss")
    plt.savefig("Q5_data/acc_loss.png")
    

def show_train_image():
    model = cifar10vgg(
        batch_size=128, 
        maxepoches=100, 
        learning_rate=0.1, 
        train=False,
        load=False)
    model.show_random_10_img()
    

def show_hyperparameters():
    model = cifar10vgg(
        batch_size=128, 
        maxepoches=100, 
        learning_rate=0.1, 
        train=False,
        load=False)
    model.show_hyperparameters()

def show_model_structure():
    model = cifar10vgg(
        batch_size=128, 
        maxepoches=100, 
        learning_rate=0.1, 
        train=False,
        load=False)
    model.show_model_structure()

def show_accuracy():
    plot_acc_loss()
    plt.clf()
    img = mpimg.imread('Q5_data/acc_loss.png')
    plt.imshow(img)
    plt.show()

def test(idx):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_test_origin = x_test
    x_test = x_test.astype('float32')
    
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    model = cifar10vgg(
        batch_size=128, 
        maxepoches=100, 
        learning_rate=0.1, 
        train=False,
        load=False)

    predicted_x = model.predict(x_test[idx].reshape(-1,32,32,3))
    plt.subplot(2, 1, 1)
    plt.imshow(x_test_origin[idx])
    plt.title(f"Image {idx}")
    plt.subplot(2, 1, 2)
    plt.bar(LABEL_NAME, predicted_x[0])
    plt.xlabel("class")
    plt.ylabel("1 = 100%")
    plt.show()
