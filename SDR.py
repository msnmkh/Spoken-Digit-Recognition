import warnings
import os
from hmmlearn import hmm
import numpy as np
from librosa.feature import mfcc
import librosa
import random

def buildDataSet(dir,rte):
    # Filter out the wav audio files under the dir
    fileList = [f for f in os.listdir(dir) if os.path.splitext(f)[1] == '.wav']
    train_dataset = {}
    test_dataset = {}
    cnt=1
    # Calculate percent of each train and test
    nm = int(rte*50)
    rnd = random.sample(range(0,50), nm)

    for fileName in fileList:
        label = fileName.split('_')[0]
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


def extract_mfcc(full_audio_path):
    wave, sample_rate =  librosa.load(full_audio_path)
    mfcc_features = mfcc(wave, sample_rate )
    return mfcc_features


### Gussian HMM
def train_HMM(dataset):
    Models = {}
    for label in dataset.keys():
        model = hmm.GMMHMM(n_components=10)
        trainData = dataset[label]
        trData = np.vstack(trainData)
        model.fit(trData)
        Models[label] = model
    return Models

def main():
    ### ignore warning message of readfile
    warnings.filterwarnings('ignore')

    ### Step.1 Loading data
    trainDir = 'new_spoken_digit/'
    print('Step.1 data loading...')
    trainDataSet,testDataSet = buildDataSet(trainDir,rte=0.25)
    print("Finish prepare the data")


    ### Step.2 Training
    print('Step.2 Training model...')
    hmmModels = train_HMM(trainDataSet)
    print("Finish training of the GMM_HMM models for digits 0-9")

    ### Step.3 predict test data
    acc_count = 0
    all_data_count = 0
    for label in testDataSet.keys():
        feature = testDataSet[label]
        for index in range(len(feature)):
            all_data_count+=1
            scoreList = {}
            for model_label in hmmModels.keys():
                model = hmmModels[model_label]
                score = model.score(feature[index])
                scoreList[model_label] = score
            predict = max(scoreList, key=scoreList.get)
            if predict == label:
                acc_count+=1

    accuracy = round(((acc_count/all_data_count)*100.0),3)

    print("\n##########################################################################")
    print("######################## A-C-C-U-R-A-C-Y #################################")
    print("########################    ",accuracy,"%","   #################################")
    print("##########################################################################")


if __name__ == '__main__':
    main()
