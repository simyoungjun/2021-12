#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COVID-DETECT test code implementation.


** The code for the Capsule Network implementation is adopted from https://keras.io/examples/cifar10_cnn_capsule/.
"""

from __future__ import print_function
from keras import backend as K
from keras.layers import Layer
from keras import activations
from keras import utils
from keras.models import Model
from keras.layers import *
from keras.utils.training_utils import multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras import optimizers
import keras
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import pydicom
import cv2
from lungmask import mask  # lung segmentation model
import SimpleITK as sitk
import pandas as pd

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Set the path based on your data directory
data_path = "./ICASSP_SPGC2021_TestData/SPGC-Test1"
# "./ICASSP_SPGC2021_TestData/SPGC-Test2"
# "./ICASSP_SPGC2021_TestData/SPGC-Test3"

K.set_image_data_format('channels_last')


def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (1 + s_squared_norm)
    return scale * x


def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)


def margin_loss(y_true, y_pred):
    lamb, margin = 0.5, 0.1
    return K.sum((y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (
            1 - y_true) * K.square(K.relu(y_pred - margin))), axis=-1)


class Capsule(Layer):

    def __init__(self,
                 num_capsule,
                 dim_capsule,
                 routings=3,
                 share_weights=True,
                 activation='squash',
                 **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_capsule': self.num_capsule,
            'dim_capsule': self.dim_capsule,
            'routings': self.routings,
            'share_weight': self.share_weights,

        })
        return config

    def build(self, input_shape):
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(1, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(input_num_capsule, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)

    def call(self, inputs):

        if self.share_weights:
            hat_inputs = K.conv1d(inputs, self.kernel)
        else:
            hat_inputs = K.local_conv1d(inputs, self.kernel, [1], [1])

        batch_size = K.shape(inputs)[0]
        input_num_capsule = K.shape(inputs)[1]
        hat_inputs = K.reshape(hat_inputs,
                               (batch_size, input_num_capsule,
                                self.num_capsule, self.dim_capsule))
        hat_inputs = K.permute_dimensions(hat_inputs, (0, 2, 1, 3))

        b = K.zeros_like(hat_inputs[:, :, :, 0])
        for i in range(self.routings):
            c = softmax(b, 1)
            o = self.activation(keras.backend.batch_dot(c, hat_inputs, [2, 2]))
            if i < self.routings - 1:
                b = keras.backend.batch_dot(o, hat_inputs, [2, 3])
                if K.backend() == 'theano':
                    o = K.sum(o, axis=1)

        return o

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


# normalization function
def normalize_image(x):  # normalize image pixels between 0 and 1
    if np.max(x) - np.min(x) == 0 and np.max(x) == 0:
        return x
    elif np.max(x) - np.min(x) == 0 and np.max(x) != 0:
        return x / np.max(x)
    else:
        return (x - np.min(x)) / (np.max(x) - np.min(x))


def segment_lung(mask, model, volume_path):

    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(volume_path):
        for filename in fileList:
            if ".dcm" in filename.lower():  # check whether the file's DICOM
                lstFilesDCM.append(os.path.join(dirName, filename))

    dataset = pydicom.dcmread(lstFilesDCM[0])  # a sample image
    slice_numbers = len(lstFilesDCM)  # number of slices
    # print('Slices:',slice_numbers)

    if 'PixelData' in dataset:
        rows = int(dataset.Rows)
        cols = int(dataset.Columns)
        # print('Image size:',rows,cols)

    slice_z_locations = []
    for filenameDCM in lstFilesDCM:
        ds = pydicom.dcmread(filenameDCM)
        slice_z_locations.append(ds.get('SliceLocation'))

    # sorting slices based on z locations
    slice_locations = list(zip(lstFilesDCM, slice_z_locations))
    sorted_slice_locations = sorted(slice_locations, key=lambda x: x[1])[-1::-1]

    # Saving Slices in a numpy array
    ArrayDicom = np.zeros((slice_numbers, rows, cols))
    lung_mask = np.uint8(np.zeros((slice_numbers, rows, cols)))
    # loop through all the DICOM files
    i = 0
    for filenameDCM, z_location in sorted_slice_locations:
        # read the file
        ds = sitk.ReadImage(filenameDCM)
        segmentation = mask.apply(ds, model)
        lung_mask[i, :, :] = np.uint8(((segmentation > 0) * 1)[0])
        ArrayDicom[i, :, :] = sitk.GetArrayFromImage(ds)
        i = i + 1

    lungs = np.zeros((ArrayDicom.shape[0], 256, 256, 1))
    # resizing the data
    for i in range(ArrayDicom.shape[0]):
        ct = normalize_image(ArrayDicom[i, :, :])
        mask_l = lung_mask[i, :, :]
        seg = mask_l * ct  # apply mask on the image
        img = cv2.resize(seg, (256, 256))
        img = normalize_image(img)
        lungs[i, :, :, :] = np.expand_dims(img, axis=-1)
    return lung_mask, ArrayDicom, lungs


def test_one_dicom(model, X_test):
    # Test
    X_test_normal = np.zeros(X_test.shape)
    for i in range(X_test.shape[0]):
        X_test_normal[i, :, :, :] = normalize_image(X_test[i, :, :, :])

    predict = model.predict([X_test_normal])
    print("predict 1 : ", predict, "\n")

    predict = np.argmax(predict, axis=1)
    print("predict 2 : ", predict, "\n")
    print("len(predict2)", len(predict))
    print('Abnormal slices:', sum(predict))
    return X_test_normal, predict, sum(predict)


def stage_one_output(model, X_test):
    X_test, predict_init, abnormal_slices = test_one_dicom(model, X_test)  # predict classes for all slices
    sum_seg = np.sum(np.sum(X_test, axis=1), axis=1)
    a = np.where(sum_seg[:, 0] != 0)


    print("predict before predict_init[a]:", len(predict_init))
    # to find out if lung exists or not
    predict = predict_init[a]
    infected_slices = sum(predict)  # save the number of infected slices
    case_slices = len(predict)
    print('Total Slices:', case_slices)

    X_numpy_inf = np.zeros((infected_slices, 256, 256, 1))
    X_numpy_hlt = np.zeros((case_slices - infected_slices, 256, 256, 1))
    hlt_ind = 0
    inf_ind = 0
    for i in range(X_test.shape[0]):
        if sum_seg[i, 0] != 0:  # if lung exists
            if predict_init[i] == 0:  # save healthy slices
                X_numpy_hlt[hlt_ind, :, :, :] = X_test[i, :, :, :]
                hlt_ind = hlt_ind + 1
            if predict_init[i] == 1:  # save healthy slices
                X_numpy_inf[inf_ind, :, :, :] = X_test[i, :, :, :]
                inf_ind = inf_ind + 1

    return X_numpy_hlt, X_numpy_inf, len(predict_init), infected_slices, case_slices, abnormal_slices


def stage_two_output(x_test, normal_thresh, model2, cutoff):

    pred = model2.predict(x_test)
    prob_one = sum(pred[:, 1]) / len(pred[:, 1])
    pred_final = (prob_one >= cutoff) * 1  # cut-off probability
    return prob_one, pred_final

def enum_folder_only(dirname):
    for filename in os.listdir(dirname):
        file_path = os.path.join(dirname, filename)
        if os.path.isdir(file_path):
            if filename[-1].isdigit():
                People.append([filename, file_path])
            enum_folder_only(file_path)

def compare(result_list):
    true = 0
    false = 0
    for i in result_list:
        if (i[0][0] == 'c'):  # is it non-covid(CAP)?
            if (i[1][0] == 'n'):
                true += 1
            else:
                false += 1
        elif (i[0][0] == 'P'):  # is it COVID-19?
            if (i[1][0] == 'C'):
                true += 1
            else:
                false += 1
        elif (i[0][0] == 'n'):  # is it Normal?
            if (i[1][0] == 'N'):
                true += 1
            else:
                false += 1

        if (true == 0):
            accuracy = 0
        else:
            accuracy = true / (true + false)
    return accuracy


input_image = Input(shape=(None, None, 1))
x = Conv2D(32, (3, 3), activation='relu', trainable=True)(input_image)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                       gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones',
                       beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(x)
x = Conv2D(32, (3, 3), activation='relu', trainable=True)(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', trainable=True)(x)
x = Conv2D(64, (3, 3), activation='relu', trainable=True)(x)
x = Dropout(0.2)(x)
# # # # # # # # # # # # # # #
x = Reshape((-1, 64))(x)
x = Capsule(16, 8, 1, True)(x)
capsule = Capsule(2, 16, 1, True)(x)
output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule)

model2 = Model(inputs=[input_image], outputs=[output])

model2.summary()
# model2.load_weights("/home/ubuntu/COVID19/code_jaeyeon/0310_best_model2.h5")

model = mask.get_model('unet', 'R231CovidWeb')

# %% Testing


# load each peope make list of people
People = []
enum_folder_only(data_path)
print(People)
print('test start...')

result=[]

for i in People:
    print(i[1])
    lung_mask, ArrayDicom, lung = segment_lung(mask, model,i[1])

    model2.load_weights("./weight_for_stage1.h5")
    # Stage 1
    X_numpy_hlt, X_numpy_inf, slice_size, infected_slices, case_slices, abnormal_slices = stage_one_output(
        model2, lung)
    if infected_slices <= int(case_slices*0.3):
        prediction = 'Normal'
    else:
        # Stage 2
        model2.load_weights("./weight_for_stage2.h5")
        normal_thresh = slice_size
        x_test = X_numpy_inf
        cutoff = 0.6  # cut-off probability
        prob_one, pred_final = stage_two_output(x_test, normal_thresh, model2, cutoff)
        print(prob_one)
        if pred_final == 1:
            prediction = 'COVID-19'
        else:
            prediction = 'non-COVID(CAP)'
    print('prediction: ', prediction)
    print('\n')

    result.append([i[0], prediction])

# %%

print(result)



result = pd.DataFrame(result, columns=['Patient', 'Class'])
result = result.sort_values(by=['Patient'], axis=0)
print(result)
#
# result.to_csv(os.path.join(data_path, 'result.csv'), index=False,
#               encoding='cp949')
result.to_csv('SPGC-Test'+data_path[-1]+'_result.csv', index=False,
              encoding='cp949')