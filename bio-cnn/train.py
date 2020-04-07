from preprocess import *
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM, Activation
from keras.utils import to_categorical
import wandb
from wandb.keras import WandbCallback
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
import sklearn.metrics as metrics
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTETomek
import tensorflow as tf

def plot_2d_space(X, y, label='Classes'):   
    colors = ['#111111', '#222222', '#333333', '#444444', '#555555', '#666666', '#777777', '#888888']
    markers = ['o', 's', 'o', 's','o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()

def run_sampler( X, y, sampler ) :
    print(X.shape)
    print(y.shape)

    X_samples, _, _, _ = X.shape

    d2_X = X.reshape((X_samples,config.buckets*config.max_len*channels))

    X_s, y_s = sampler.fit_sample(d2_X, y)

    #plot_2d_space(X_rus, y_rus, 'Random under-sampling')

    X_s = X_s.reshape((X_s.shape[0], config.buckets, config.max_len, channels))
    print("X_s", X_s.shape)
    print("Y_s", y_s.shape)
    
    return X_s, y_s

hyperparameter_defaults = dict(
    max_len = 32,
    buckets = 128,
    epochs = 7,
    batch_size = 100,
    layer_one = 21,
    layer_two = 48,
    layer_three = 48,
    layer_four = 64,
    dropout_one = 0.5,
    dropout_two = 0.5,
    sampler = 'none'
    )

wandb.init( config=hyperparameter_defaults )
config = wandb.config


# Save data to array file first
save_data_to_array(max_len=config.max_len, n_mfcc=config.buckets)

labels=np.array(["BAM", "BBI", "BIN", 
                 "OPI", "OQU"])

# Loading train/test set
X_train, X_test, X_val, y_train, y_test, y_val = get_train_test()

# Setting channels to 1 to generalize stereo sound to 1 channel
channels = 1

# Number of classes
num_classes = labels.shape[0]
print(X_train.shape)

# Reshape X_train and X_test to include a 4th dimension (channels)
X_train = X_train.reshape(X_train.shape[0], config.buckets, config.max_len, channels)
X_test = X_test.reshape(X_test.shape[0], config.buckets, config.max_len, channels)
X_val = X_val.reshape(X_val.shape[0], config.buckets, config.max_len, channels)

under_sampler = RandomUnderSampler()
over_sampler = RandomOverSampler()
smote_sampler = SMOTETomek()

if(config.sampler == 'under') :
    sampler = under_sampler
    X_train, y_train = run_sampler( X_train, y_train, sampler )
    X_test, y_test = run_sampler( X_test, y_test, sampler )
    X_val, y_val = run_sampler( X_val, y_val, sampler )
    
if(config.sampler == 'over') :
    sampler = over_sampler
    X_train, y_train = run_sampler( X_train, y_train, sampler )
    X_test, y_test = run_sampler( X_test, y_test, sampler )
    X_val, y_val = run_sampler( X_val, y_val, sampler )
    
if(config.sampler == 'smote') :
    sampler = smote_sampler
    X_train, y_train = run_sampler( X_train, y_train, sampler )
    X_test, y_test = run_sampler( X_test, y_test, sampler )
    X_val, y_val = run_sampler( X_val, y_val, sampler )

# Getting vector number where each number corresponds to a label
y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)
y_val_hot = to_categorical(y_val)

# Building the model
model = Sequential()

input_shape= (config.buckets, config.max_len, channels)

model.add(Conv2D( config.layer_one, (3, 3), strides=(1, 1), input_shape=input_shape))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Activation('relu'))

model.add(Conv2D( config.layer_two, (3, 3), padding="valid"))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Activation('relu'))

model.add(Conv2D( config.layer_three, (3, 1), padding="valid"))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dropout(rate=config.dropout_one))

model.add(Dense( config.layer_four ))
model.add(Activation('relu'))
model.add(Dropout(rate=config.dropout_two))

model.add(Dense(len(labels)))
model.add(Activation('softmax'))
model.summary()

# Configure CNN for training
model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=['accuracy', tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives(), tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives()])

wandb.init()
# Train the CNN model
#    X_train: Input data
#    y_train_hot: Target data
model.fit(X_train, y_train_hot, epochs=config.epochs, validation_data=(X_val, y_val_hot), callbacks=[WandbCallback(data_type="image", labels=labels)])

# Save the keras model
model.save("ant_cnn_model.h5")
print("Model has been saved.")
