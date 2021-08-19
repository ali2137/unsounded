# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 16:37:51 2021

@author: 11486
"""

from __future__ import absolute_import,division, print_function

import tensorflow as tf
import os,sys
import numpy as np
from contextlib import redirect_stdout
from tensorflow import keras

from tensorflow.keras import layers,Model
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping


import matplotlib.pyplot as plt

sys.path.append('F:\\aicomposer\\models\\sequential_lstm')
from tsp import data_process_int,data_process_map
from subclass import att_model
tf.random.set_seed(123)

network_input, network_output = data_process_int('data.csv',seq_len=20,trained_level=0.7,training=True,)       
eps = 300
seq_len=20
n = Input(shape = (seq_len,1),name = 'pitch_in')
d = Input(shape = (seq_len,1),name = 'duration_in')

notes = att_model(seq_len=20).model(n)
beats = att_model(seq_len=20).model(d)
notes_out=Dense(20, activation = 'relu',kernel_initializer='RandomNormal',
                                        bias_initializer='RandomNormal',name = 'pitch_out')(notes)
durations_out=Dense(20, activation = 'relu',kernel_initializer='RandomNormal',
                                        bias_initializer='RandomNormal',name = 'durations_out')(beats)

model = Model([n, d], [notes_out, durations_out])        
plot_model(
    model, to_file='model.png', show_shapes=True, show_layer_names=True,
    rankdir='TB', expand_nested=True, dpi=96
)

plot_model(
    model, to_file='model_simplified.png', show_shapes=False, show_layer_names=False,
    rankdir='TB', expand_nested=True, dpi=96
)
model.summary()
with open('modelsummary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()
opti = keras.optimizers.Adam(lr = 0.008)
model.compile(loss=['mse', 'mse'],metrics=['accuracy'], optimizer=opti)
history=model.fit(network_input, network_output, validation_split=0.3,validation_data=(network_input, network_output),
                  epochs=eps,batch_size=1000)

model.save('muti_attmodel')

reall_eps = len(history.history['loss'])
plt.figure(figsize=(10.24, 7.68))
plt.subplot(2,1 , 1)
plt.plot(range(reall_eps), history.history['pitch_out_accuracy'], label='Training Accuracy')
plt.plot(range(reall_eps), history.history['val_pitch_out_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy freq')

plt.subplot(2,1 , 2)
plt.plot(range(reall_eps), history.history['pitch_out_loss'], label='Training Loss')
plt.plot(range(reall_eps), history.history['val_pitch_out_loss'], label='Validation loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss freq')
    
plt.savefig('accuracy_loss_freq_oo_att.png')

plt.figure(figsize=(10.24, 7.68))
plt.subplot(2,1 , 1)
plt.plot(range(reall_eps), history.history['durations_out_accuracy'], label='Training Accuracy')
plt.plot(range(reall_eps), history.history['val_durations_out_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy beat')

plt.subplot(2,1 , 2)
plt.plot(range(reall_eps), history.history['durations_out_loss'], label='Training Loss')
plt.plot(range(reall_eps), history.history['val_durations_out_loss'], label='Validation loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss beat')
    
plt.savefig('accuracy_loss_beat_oo_att.png')