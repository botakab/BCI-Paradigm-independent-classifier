#!/usr/bin/env python
# coding: utf-8

# Loading dependecies
import scipy.io
import scipy.signal
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# Extracts n segments of raw EEG from the provided data.
# 
# To do so, the function selects n values from `data.t` array, which contains timestamps of the trial beginning. If shuffle=True, then the function randomly selects n trials, otherwise first n values from `data.t` are selected.
# 
# For each timestamp, function extracts corresponding block 900 values of raw EEG data with 32 channels and the trial label (1 or 0 - target or non-target).
# 
# The function returns `segments(n, 900, 32), y(n), t(n)`.

# In[ ]:


def raw_eeg_segments(raw_eeg, t):

    selected_channels = [12, 16, 33, 34, 28, 7, 29, 44, 41, 25, 11, 42, 14, 5, 17, 21]
    for i in range(len(selected_channels)):
        selected_channels[i] -= 1

    raw_eeg = raw_eeg[:, selected_channels]

    n = t.shape[0]
    
    segments = np.zeros((n, 900, raw_eeg.shape[1]))
    for i in range(n):
        segments[i] = raw_eeg[t[i]-100:t[i]+800]

    return segments


# This function filters the given segments with 5-th order Butterworth bandpass filter between 0.5 - 40 Hz.

# In[ ]:


def filter(segments, show=False):
    sos = scipy.signal.butter(5, (0.5, 40), btype='bandpass', output='sos', fs=1000)
    filtered = scipy.signal.sosfilt(sos, segments, axis=1)

    if show:
        plt.figure(figsize=(12, 8))
        plt.plot(range(-100, 800), segments[0,:,0])
        plt.plot(range(-100, 800), filtered[0,:,0])
        plt.legend(['Original', 'Filtered'])
        plt.grid('on')
        plt.show()

    return filtered


# As you remember, the `raw_eeg_segments()` was extracting segments with 900 values for 32 channels. First 100 values are used to perform baseline average for the rest 800 samples. This function subtracts the average value of first 100 elements from the rest 800 elements in each segment. The segment size is reduced from 900 to 800.

# In[ ]:


def baseline_correct(filtered, show=False):
    corrected = np.empty((filtered.shape[0], 800, filtered.shape[2]))

    for i in range(filtered.shape[0]):
        corrected[i] = filtered[i,100:,:] - filtered[i,:100,:].mean(axis=0)
    
    if show:
        plt.figure(figsize=(12, 8))
        plt.plot(range(-100, 800), filtered[0,:,0])
        plt.plot(range(0, 800), corrected[0,:,0])
        plt.grid('on')
        plt.legend(['Filtered', 'Filtered and baseline corrected'])
        plt.show()

    return corrected


# The segments are downsampled from 800 to 8 samples in this function. Each block of 100 samples in the segment is averaged. 

# In[ ]:


def average(corrected, show=False):

    x = np.empty((corrected.shape[0], 8, corrected.shape[2]))

    for i in range(corrected.shape[0]):
        for j in range(corrected.shape[1] // 100):
            x[i, j] = corrected[i, j*100:(j+1)*100].mean(axis=0)

    if show:
        plt.figure(figsize=(12, 8))
        plt.plot(range(0, 800), corrected[0,:,0])
        plt.plot(range(50,800, 100), x[0,:,0], marker='o')
        plt.grid('on')
        plt.legend(['Filtered and baseline corrected',
                    'Filtered, corrected, averaged'],
                loc='best')
        plt.title('ERP EEG preprocessing')
        plt.xlabel('Time (ms)')
        plt.xticks(ticks=range(-100,801, 100))
        plt.savefig('/content/ERP_EEG_preprocessing.png')
        plt.show()

    return x


# All of the above operations are combined in this function. It takes data from `scipy.io.loadmat(filename)['EEG_ERP_train']` to extract `n` samples of data. The function returns two numpy arrays: `x` with the shape (n, 256) and `y` with the shape (n, 1).

# In[ ]:


def get_erp_features(x, t, show=False):
    segments = raw_eeg_segments(x, t)
    filtered = filter(segments)
    corrected = baseline_correct(filtered)
    x = average(corrected)

    x = x[:, 1:7, :]

    if show:
        plt.figure(figsize=(12, 8))
        plt.plot(range(-100, 800), segments[0, :, 0])
        plt.plot(range(-100, 800), filtered[0, :, 0])
        plt.plot(range(0, 800), corrected[0,:,0])
        plt.plot(range(150,700, 100), x[0,:,0], marker='o')
        plt.grid('on')
        plt.legend(['Raw EEG',
                    'Bandpass filtered'
                    'Filtered and baseline corrected',
                    'Filtered, corrected, averaged'],
                loc='best')
        plt.title('ERP EEG preprocessing (one channel)')
        plt.xlabel('Time (ms)')
        plt.xticks(ticks=range(-100,801, 100))
        plt.savefig('/content/0. ERP_EEG_preprocessing.png')
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.plot(range(-100, 800), segments[0, :, :])
        plt.grid('on')
        plt.title('Raw ERP EEG')
        plt.xlabel('Time (ms)')
        plt.xticks(ticks=range(-100,801, 100))
        plt.savefig('/content/1. ERP_EEG_raw_eeg.png')
        plt.show()
        
        plt.figure(figsize=(12, 8))
        plt.plot(range(-100, 800), filtered[0, :, :])
        plt.grid('on')
        plt.title('Bandpass Filtered ERP EEG')
        plt.xlabel('Time (ms)')
        plt.xticks(ticks=range(-100,801, 100))
        plt.savefig('/content/2. ERP_EEG_bandpass.png')
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.plot(range(0, 800), corrected[0,:,:])
        plt.grid('on')
        plt.title('Baseline corrected ERP EEG')
        plt.xlabel('Time (ms)')
        plt.xticks(ticks=range(-100,801, 100))
        plt.savefig('/content/3. ERP_EEG_baselineCorrected.png')
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.plot(range(150,700, 100), x[0,:,:], marker='o')
        plt.grid('on')
        plt.title('ERP EEG features')
        plt.xlabel('Time (ms)')
        plt.xticks(ticks=range(-100,801, 100))
        plt.savefig('/content/4. ERP_EEG_features.png')
        plt.show()

    x = x.reshape(t.shape[0], -1)

    return x

