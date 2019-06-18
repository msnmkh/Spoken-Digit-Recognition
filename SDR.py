import warnings
import os
from hmmlearn import hmm
import numpy as np
from librosa.feature import mfcc
import librosa
import random

def extract_mfcc(full_audio_path):
    wave, sample_rate =  librosa.load(full_audio_path)
    mfcc_features = mfcc(wave, sample_rate )
    return mfcc_features

def buildDataSet(dir,rte):
    # Filter out the wav audio files under the dir
    fileList = [f for f in os.listdir(dir) if os.path.splitext(f)[1] == '.wav']
    train_dataset = {}
    test_dataset = {}
    cnt=1
    nm = int(rte*50)
    rnd = random.sample(range(0,50), nm)

    for fileName in fileList:
        tmp = fileName.split('.')[0]
        label = tmp.split('_')[1]
        feature = extract_mfcc(dir+fileName).T
        if cnt in rnd:
            if label not in test_dataset.keys():
                test_dataset[label] = []
                test_dataset[label].append(feature)
            else:
                exist_feature = test_dataset[label]
                exist_feature.append(feature)
                test_dataset[label] = exist_feature
        else:
            if label not in train_dataset.keys():
                train_dataset[label] = []
                train_dataset[label].append(feature)
            else:
                exist_feature = train_dataset[label]
                exist_feature.append(feature)
                train_dataset[label] = exist_feature
        if (cnt == 50):
            cnt = 1
            rnd = random.sample(range(0, 50), 12)
        else:
            cnt += 1

    return train_dataset,test_dataset

def train_GMMHMM(dataset):
    GMMHMM_Models = {}
    for label in dataset.keys():
        model = hmm.GMMHMM()
        trainData = dataset[label]
        trData = np.vstack(trainData)
        model.fit(trData)
        GMMHMM_Models[label] = model
    return GMMHMM_Models

def main():
    ### ignore warning message
    warnings.filterwarnings('ignore')

    ### Step.1 Loading data
    trainDir = 'spoken_digit/'
    print('Step.1 data loading...')
    trainDataSet,testDataSet = buildDataSet(trainDir,rte=0.3)
    print("Finish prepare the data")


    ### Step.2 Training
    print('Step.2 Training model...')
    hmmModels = train_GMMHMM(trainDataSet)
    print("Finish training of the GMM_HMM models for digits 0-9")


    ### Step.3 predict test data
    score_cnt = 0
    for label in testDataSet.keys():
        feature = testDataSet[label]
        scoreList = {}
        for model_label in hmmModels.keys():
            model = hmmModels[model_label]
            score = model.score(feature[0])
            scoreList[model_label] = score
        predict = max(scoreList, key=scoreList.get)
        if predict == label:
            score_cnt+=1
    accuracy = 100.0*score_cnt/len(testDataSet.keys())
    print("\n##########################################################################")
    print("##################### A-C-C-U-R-A-C-Y ####################################")
    print("#####################      ",accuracy,"%","     #################################")
    print("##########################################################################")


if __name__ == '__main__':
    main()
