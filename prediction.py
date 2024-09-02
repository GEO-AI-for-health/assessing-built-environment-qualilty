# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 00:46:31 2023

@author: wanglinsen
"""

import matplotlib
matplotlib.use('Agg')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2

import os
import cv2
import numpy as np
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense,  GlobalAveragePooling2D

# set L2 regularizer
l2_reg = 1e-7#-5?
#set batchsize
bsz=64

#def model
def build_transfer_learning_model(input_shape=(256, 256, 3), out_dims=4):
    base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=input_shape,pooling=max)
    x = base_model.get_layer('mixed4').output
    x = GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = Dense(512, activation='relu',kernel_regularizer=l2(l2_reg))(x)
    x = Dense(128, activation='relu',kernel_regularizer=l2(l2_reg))(x)
    x = Dense(32, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    fc2 = Dense(out_dims, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=fc2)

    return model

#def a function to get images
def sample_iter(x, bsz, path):
    range_l, range_r = 0, bsz
    while range_l<len(x):
        X = x[range_l: range_r]
        X = [cv2.resize(cv2.imread("%s/%s" % (path, _)), (256, 256)) for _ in X]
        X = np.array(X, dtype = np.float32) / 255.0
        range_l = range_r
        range_r += bsz
        yield X
# %%
img_path = r'C:/Files/built environment quality/prediction/imges' #change ur file path
save_path = r'C:/Files/built environment quality/prediction' #change ur file path

for folder in os.listdir(img_path):
    pre_img_path = img_path + '/' + folder
    save_file = save_path + '/' + folder + '_.csv'
    m = build_transfer_learning_model(input_shape=(256, 256, 3),
                                      out_dims=4)
    m.load_weights(
        'C:/Files/built environment quality/finetuning_inceptionv3/new_optim(RMSprop)/0_weights.54-0.2757-0.2553.hdf5')# change ur weights file path

    pre_samples = os.listdir(pre_img_path)  # pre_img_path

    pre_y = []
    for smaple in sample_iter(pre_samples, bsz, pre_img_path):
        pre_y.append(m.predict(smaple, verbose=1))
    pre_yy = np.concatenate(pre_y, axis=0)

    with open(save_file, 'w') as wf:
        wf.write(','.join(['lon', 'lat', 'liveliness', 'safety', 'beauty', 'uncleanness']))
        wf.write('\n')
        for i, j in zip(pre_samples, pre_yy):
            wf.write(','.join(i[:-4].split('_')) + ',' + ','.join([str(_) for _ in j]))
            wf.write('\n')
            x = [str(_) for _ in j]

            x = []
            for _ in j:
                x.append(_)
