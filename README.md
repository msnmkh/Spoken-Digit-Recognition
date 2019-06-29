<h1 align="center">
  <img src="media/spoken.png" width="15%"><br/>Spoken Digit Recognition
</h1>

<h4 align="center">
  🎙️ Spoken Digit Recognition with HMM
</h4>



## Overview
This project use mfcc feature extractor and Hidden Markove Model classification algorithm to  recognition  0 - 9 digit of  Kaggle dataset.

## Dataset
* Dataset of Kaggle : [https://www.kaggle.com/divyanshu99/spoken-digit-dataset](https://www.kaggle.com/divyanshu99/spoken-digit-dataset)

## Classification Algorithm
* Hidden Markove Model

## Feature Extractor Algorithm
* Mel-frequency Cepstrum


## Code Requirements
This code is written in python. To use it you will need:
* python3
* hmmlearn
* librosa.feature
* numpy
* librosa
* random
Use [pip](https://pypi.org/project/pip/) to install any missing dependencies


## General Steps
* Split train and test data
* Feature extract each audio using mfcc
* Transpose each audio signal matrix
* Vstack each transpose audio signal matrix
* Create Hidden Markov Modle
* Test Audio signal and predict them


## Accuracy
94 %


## Usage
Run python SDR.py


