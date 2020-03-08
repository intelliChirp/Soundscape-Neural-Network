from keras.models import load_model
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM, Activation
from keras.utils import to_categorical
import wandb
from wandb.keras import WandbCallback
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import librosa
import numpy as np

# load the models
ant_model = load_model('ant_cnn_model.h5')
bio_model = load_model('bio_cnn_model.h5')
#geo_model = load_model('.\\geo_cnn\\geo_cnn_model.h5')


## Running the models

n_mfcc = 50 # bucket size !!SUBJECT TO CHANGE!!
max_len = 21 # max_len size !!SUBJECT TO CHANGE!!
channels = 1 # channels !!SUBJECT TO CHANGE!!

# convert file to wav2mfcc
# Mel-frequency cepstral coefficients
file_path = "./prediction/nature_sc.wav"
big_wave, sr = librosa.load(file_path, mono=True, sr=None)
#print(wave.shape, sr)

ant_classification = []
bio_classification = []
geo_classification = []

for sec_index in range( int(big_wave.shape[0] / sr) ) :
    start_sec = sec_index
    end_sec = sec_index + 1

    sec_to_trim = np.array( [ float(start_sec), float(end_sec) ] )
    sec_to_trim = np.ceil( sec_to_trim * sr )

    wave = big_wave[int(sec_to_trim[0]) : int(sec_to_trim[1])]

    wave = np.asfortranarray(wave[::3])
    mfcc = librosa.feature.mfcc(wave, sr=16000, n_mfcc=n_mfcc)

    # If maximum length exceeds mfcc lengths then pad the remaining ones
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # Else cutoff the remaining parts
    else:
        mfcc = mfcc[:, :max_len]

    # Convert wav to MFCC
    #prediction_data = wav2mfcc('./prediction/nature_sc.wav')
    prediction_data = mfcc

    # Reshape to 4 dimensions
    prediction_data = prediction_data.reshape(1, n_mfcc, max_len, channels)

    # Run the model on the inputted file
    ant_predicted = ant_model.predict(prediction_data)
    bio_predicted = bio_model.predict(prediction_data)
    # geo_predicted = geo_model.predict(prediction_data


    # Output the prediction values for each class
    print ('PREDICTED VALUES')
    labels_indices = range(len(labels))
    max_value = 0
    max_value_index = 0
    for index in labels_indices:
        print("\n", labels[index], ": ", '%.08f' % predicted[0,index])
        if predicted[0,index] > max_value:
            max_value_index = index
            max_value = predicted[0,index]

    # Output the prediction
    if max_value < 0.5:
        print("GUESS: Nothing")
        classification.append( { "class" : "Nothing", "timestamp" : start_sec } )
    else:
        print('\n\nGUESS: ', labels[max_value_index])
        classification.append( { "class" : labels[max_value_index], "timestamp" : start_sec } )

print(classification)
