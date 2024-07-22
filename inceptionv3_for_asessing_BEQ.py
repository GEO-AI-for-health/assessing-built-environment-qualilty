# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 00:46:31 2023

@author: 15641
"""


import matplotlib
matplotlib.use('Agg')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.model_selection import KFold
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
import h5py
import numpy as np
from keras.callbacks import TensorBoard
import keras
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input, Dense, Activation, Reshape, GlobalAveragePooling2D
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import  Callback

# set L2 regularizer
l2_reg = 1e-7#-5?

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
batch_size = 64

def r2score(y_true, y_pred):
    ss_res =  K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - ss_res/(ss_tot + K.epsilon()))



# MixUP
class MixupGenerator():
    def __init__(self, X_train, y_train, batch_size=64, alpha=0.2, shuffle=True, datagen=None):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(X_train)
        self.datagen = datagen

    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                X, y = self.__data_generation(batch_ids)

                yield X, y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_train.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]
        X = X1 * X_l + X2 * (1 - X_l)

        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        if isinstance(self.y_train, list):
            y = []

            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1 - y_l))
        else:
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1 - y_l)

        return X, y


def build_transfer_learning_model(input_shape=(512, 512, 3), out_dims=4):
    # load InceptionV3
    base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=input_shape,pooling=max)
    # get mixed4
    x = base_model.get_layer('mixed4').output
    # GlobalAveragePooling
    x = GlobalAveragePooling2D()(x)
    # Dropout
    x = tf.keras.layers.Dropout(0.4)(x)

    # fc
    x = Dense(512, activation='relu',kernel_regularizer=l2(l2_reg))(x)#,kernel_regularizer=l2(l2_reg)
    x = Dense(128, activation='relu',kernel_regularizer=l2(l2_reg))(x)  # ,kernel_regularizer=l2(l2_reg)
    x = Dense(32, activation='relu', kernel_regularizer=l2(l2_reg))(x)#,kernel_regularizer=l2(l2_reg)
    fc2 = Dense(out_dims, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=fc2)

    return model

# loss-acc

class SGDLearningRateTracker(Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.r2scores = {'batch': [], 'epoch': []}
        self.val_losses = {'batch': [], 'epoch': []}
        self.val_r2scores = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.r2scores['batch'].append(logs.get('r2score'))
        self.val_losses['batch'].append(logs.get('val_loss'))
        self.val_r2scores['batch'].append(logs.get('val_r2score'))

    def on_epoch_end(self, epoch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.r2scores['epoch'].append(logs.get('r2score'))
        self.val_losses['epoch'].append(logs.get('val_loss'))
        self.val_r2scores['epoch'].append(logs.get('val_r2score'))
        self.plot_losses('epoch')
        self.plot_r2scores('epoch')

    def plot_losses(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        plt.plot(iters, self.losses[loss_type], 'g', label='training loss')
        if loss_type == 'epoch':
            plt.plot(iters, self.val_losses[loss_type], 'k', label='validation loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('Loss')
        plt.legend(loc='best')
        plt.savefig(f'C:/Files/built environment quality/loss/{n_fold}_loss_{loss_type}.png', dpi=500)
        plt.close()

    def plot_r2scores(self, loss_type):
        iters = range(len(self.r2scores[loss_type]))
        plt.figure()
        plt.plot(iters, self.r2scores[loss_type], 'r', label='training r2 score')
        if loss_type == 'epoch':
            plt.plot(iters, self.val_r2scores[loss_type], 'b', label='validation r2 score')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('R2 Score')
        plt.legend(loc='best')
        plt.savefig(f'C:/Files/built environment quality/loss/{n_fold}_r2score_{loss_type}.png', dpi=500)
        plt.close()

#cosine learning rate decay
def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
    if total_steps >= warmup_steps:
        learning_rate = 0.5 * learning_rate_base * (1 + np.cos(np.pi *
            (global_step - warmup_steps - hold_base_rate_steps) / float(total_steps - warmup_steps - hold_base_rate_steps)))

        if hold_base_rate_steps > 0:
            learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                     learning_rate, learning_rate_base)
        if warmup_steps > 0:
            if learning_rate_base < warmup_learning_rate:
                raise ValueError('learning_rate_base must be larger or equal to '
                                 'warmup_learning_rate.')
            #
            slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
            warmup_rate = slope * global_step + warmup_learning_rate
            #
            learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                     learning_rate)
        return np.where(global_step > total_steps, 0.0, learning_rate)
    raise ValueError('total_steps must be larger or equal to '
                     'warmup_steps.')


class WarmUpCosineDecayScheduler(keras.callbacks.Callback):
    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 verbose=0):
        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.global_step = global_step_init
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        self.verbose = verbose
        self.learning_rates = []
    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)
    def on_batch_begin(self, batch, logs=None):
        lr = cosine_decay_with_warmup(global_step=self.global_step,
                                      learning_rate_base=self.learning_rate_base,
                                      total_steps=self.total_steps,
                                      warmup_learning_rate=self.warmup_learning_rate,
                                      warmup_steps=self.warmup_steps,
                                      hold_base_rate_steps=self.hold_base_rate_steps)
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_step + 1, lr))


with h5py.File(
        'C:/Files/built environment quality/train_data.h5') as hf:
    train_X = hf['X'][:]
    train_Y = hf['Y'][:]

print("train_X.shape:", train_X.shape)
print("len(train_Y):", len(train_Y))

n_splits = 5
k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=7)
n_fold = 0
for train_index, val_index in k_fold.split(train_X, train_Y):
    print("n_flod:{}.".format(n_fold))
    X_train = train_X[train_index]
    y_train = train_Y[train_index]
    X_val = train_X[val_index]
    y_val = train_Y[val_index]
    m = build_transfer_learning_model(input_shape=(512, 512, 3),
                                      out_dims=4, weights_path=None)

    sample_count = X_train.shape[0]
    # Total epochs to train.
    epochs = 150
    # Number of warmup epochs.
    warmup_epoch = 30
    # Base learning rate after warmup.
    learning_rate_base = 1e-5

    total_steps = int(epochs * sample_count / batch_size)

    # Compute the number of warmup batches.
    warmup_steps = int(warmup_epoch * sample_count / batch_size)

    # Compute the number of warmup batches.
    warmup_batches = warmup_epoch * sample_count / batch_size

    # Create the Learning rate scheduler.
    warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                            total_steps=total_steps,
                                            warmup_learning_rate=4e-06,
                                            warmup_steps=warmup_steps,
                                            hold_base_rate_steps=5,
                                            verbose=0,
                                            )
    optimizer = RMSprop()#Adam()#lr=0.0001
    # optimizer=SGD(lr=0.0001,momentum=0.9)
    loss = tf.keras.losses.Huber()
    m.compile(optimizer=optimizer, loss=loss, metrics=[r2score])
    m.summary()

    early_stop = EarlyStopping(monitor="val_loss", min_delta=0, patience=30, mode="auto")
    save_best_model = keras.callbacks.ModelCheckpoint(
        'C:/Files/built environment quality/models/'
        + str(n_fold) + '_'
        + 'weights.{epoch:02d}-{r2score:.4f}-{val_r2score:.4f}.hdf5',
        monitor='val_r2score', verbose=0, save_best_only=True,
        save_weights_only=False, mode='max', period=1)

    datagen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)

    training_generator = MixupGenerator(X_train, y_train, batch_size=batch_size, alpha=0.3,
                                        datagen=datagen)()
    history = m.fit(x=training_generator,
                    steps_per_epoch=X_train.shape[0] // batch_size,
                    validation_data=(X_val, y_val),
                    epochs=epochs, verbose=1,
                    callbacks=[early_stop, save_best_model, warm_up_lr,SGDLearningRateTracker()])

    print(history.history.keys())
    n_fold += 1
    break
