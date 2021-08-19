from __future__ import absolute_import,division, print_function

import tensorflow as tf
import os,sys
import numpy as np
from contextlib import redirect_stdout
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Input,Dense,Conv1D,GlobalAveragePooling1D,Permute,SeparableConv1D
from tensorflow.keras.layers import TimeDistributed,Reshape,Multiply,Lambda,Activation
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping

from tensorflow.keras.optimizers.schedules import InverseTimeDecay

import matplotlib.pyplot as plt


tf.random.set_seed(123)
'''
'''
out=20
seq_len=20
filterss=20
filters=20
network_input, network_output = data_process('data.csv',seq_len,trained_level=0.7,training=True,scale='None') #from another script for data processing      
eps = 200

#Core model design
n = Input(shape = (seq_len,1),name = 'pitch_in')
d = Input(shape = (seq_len,1),name = 'duration_in')

durations= Conv1D(
    filters=filterss,
    kernel_size=8,activation='relu',kernel_initializer='random_normal',
    bias_initializer='random_normal', 
    padding='same')(d)

notes= Conv1D(
    filters=filterss,
    kernel_size=8,activation='relu',kernel_initializer='random_normal',
    bias_initializer='random_normal', 
    padding='same')(n)

# attention1
d_s=GlobalAveragePooling1D(data_format='channels_last')(durations)
n_s=GlobalAveragePooling1D(data_format='channels_last')(notes)

dalpha_repeated = Permute([2, 1])(TimeDistributed(Dense(seq_len, activation='relu',
                    kernel_initializer='RandomNormal',
                    bias_initializer='RandomNormal',))(Reshape([filterss,1])(d_s)))
nalpha_repeated = Permute([2, 1])(TimeDistributed(Dense(seq_len, activation='relu',
                    kernel_initializer='RandomNormal',
                    bias_initializer='RandomNormal',))(Reshape([filterss,1])(n_s)))

natt = Lambda(lambda xin: keras.backend.sum(xin, axis=1), 
                     output_shape=(seq_len,))(Multiply()([notes, Activation('softmax')(nalpha_repeated)]))
datt = Lambda(lambda xin: keras.backend.sum(xin, axis=1), 
                     output_shape=(seq_len,))(Multiply()([durations, Activation('softmax')(dalpha_repeated)]))

dt1 = TimeDistributed(Dense(seq_len, 
                    kernel_initializer='RandomNormal',
                    bias_initializer='RandomNormal',))(Reshape([filterss,1])(datt))

nt1 = TimeDistributed(Dense(seq_len, 
                    kernel_initializer='RandomNormal',
                    bias_initializer='RandomNormal',))(Reshape([filterss,1])(natt))

durations= Conv1D(filters=filterss,kernel_size=8,activation='relu',kernel_initializer='random_normal',
    bias_initializer='random_normal', padding='same')(Multiply()([dt1,d]))


notes= Conv1D(filters=filterss,kernel_size=8,activation='relu',kernel_initializer='random_normal',
    bias_initializer='random_normal', padding='same')(Multiply()([nt1,n]))


d_s=GlobalAveragePooling1D(data_format='channels_last')(durations)
n_s=GlobalAveragePooling1D(data_format='channels_last')(notes)

d_mix= Multiply()([dalpha_repeated,Reshape([filterss,1])(d_s)])

n_mix= Multiply()([nalpha_repeated,Reshape([filterss,1])(n_s)])

dalpha_repeated = Permute([2, 1])(TimeDistributed(Dense(filterss, activation='relu',
                    kernel_initializer='RandomNormal',
                    bias_initializer='RandomNormal',))(d_mix))

nalpha_repeated = Permute([2, 1])(TimeDistributed(Dense(filterss, activation='relu',
                    kernel_initializer='RandomNormal',
                    bias_initializer='RandomNormal',))(n_mix))

natt = Lambda(lambda xin: keras.backend.sum(xin, axis=1), 
                     output_shape=(seq_len,))(Multiply()([Permute([2, 1])(n), Activation('softmax')(nalpha_repeated)]))
datt = Lambda(lambda xin: keras.backend.sum(xin, axis=1), 
                     output_shape=(seq_len,))(Multiply()([Permute([2, 1])(d), Activation('softmax')(dalpha_repeated)]))

notes_out = Dense(out, activation = 'relu',kernel_initializer='RandomNormal',bias_initializer='RandomNormal',name = 'pitch_out')(natt)
durations_out = Dense(out, activation = 'relu',kernel_initializer='RandomNormal',bias_initializer='RandomNormal',name = 'duration_out')(datt)

model = Model([n, d], [notes_out, durations_out])

tf.keras.utils.plot_model(
    model, to_file='model.png', show_shapes=True, show_layer_names=True,
    rankdir='TB', expand_nested=True, dpi=96
)

tf.keras.utils.plot_model(
    model, to_file='model_simplified.png', show_shapes=False, show_layer_names=False,
    rankdir='TB', expand_nested=True, dpi=96
)
model.summary()
with open('modelsummary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()
# lr_schedule = InverseTimeDecay(
#   initial_learning_rate=0.0015,
#   decay_steps=1.0,
#   decay_rate=0.0001,
#   staircase=False)
opti = keras.optimizers.Adam(lr = 0.008)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_pitch_out_accuracy', patience=20)

model.compile(loss=['mse', 'mse'],metrics=['accuracy'], optimizer=opti)
history=model.fit(network_input, network_output, validation_split=0.3,validation_data=(network_input, network_output),
                  epochs=eps,batch_size=2400,verbose=2)


model.save('cnn_att')
# # model.save_weights("cnn_nightly_short_cut.h5")


# reall_eps = len(history.history['loss'])
# plt.figure(figsize=(16, 9))
# # plt.subplot(2,1 , 1)
# plt.yticks(np.arange(0, 1, 0.05))
# plt.plot(range(reall_eps), history.history['pitch_out_accuracy'], label='Training Accuracy')
# plt.plot(range(reall_eps), history.history['val_pitch_out_accuracy'], label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy freq')

# # plt.subplot(2,1 , 2)
# # plt.plot(range(reall_eps), history.history['pitch_out_loss'], label='Training Loss')
# # plt.plot(range(reall_eps), history.history['val_pitch_out_loss'], label='Validation loss')
# # plt.legend(loc='upper right')
# # plt.title('Training and Validation Loss freq')
    
# plt.savefig('accuracy_loss_freq_oo_att.png')

# plt.figure(figsize=(16, 9))
# # plt.subplot(2,1 , 1)
# plt.yticks(np.arange(0, 1, 0.05))
# plt.plot(range(reall_eps), history.history['duration_out_accuracy'], label='Training Accuracy')
# plt.plot(range(reall_eps), history.history['val_duration_out_accuracy'], label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy beat')

# # plt.subplot(2,1 , 2)
# # plt.plot(range(reall_eps), history.history['duration_out_loss'], label='Training Loss')
# # plt.plot(range(reall_eps), history.history['val_duration_out_loss'], label='Validation loss')
# # plt.legend(loc='upper right')
# # plt.title('Training and Validation Loss beat')
    
# plt.savefig('accuracy_loss_beat_oo_att.png')
