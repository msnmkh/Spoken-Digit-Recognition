import numpy as np
import os
from hmmlearn import hmm
from python_speech_features import mfcc
from utills import Speech,SpeechRecognizer
from shutil import copyfile
import random

CATEGORY = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']  # 10 categories

def splitTrainAndTest(trainDir,testDir,mainSpokenDir):
    rnd = random.sample(range(0,50), 12)
    i = 1
    cntTest = 0
    cntTrain = 0
    for f in os.listdir(mainSpokenDir):
        if os.path.splitext(f)[1] == '.wav':
            srcname = os.path.join(mainSpokenDir, f)
            traindstname = os.path.join(trainDir, f)
            testdstname = os.path.join(testDir, f)
            if (i in rnd):
                cntTest += 1
                copyfile(srcname, testdstname)
            else:
                cntTrain += 1
                copyfile(srcname, traindstname)
            if(i==50):
                i=1
                rnd = random.sample(range(0, 50), 12)
            else:
                i+=1
    print("rnd: ",rnd)
    print("number of test data : " ,cntTest)
    print("number of train data : ",cntTrain)

def loadData(dirName):
    fileList = [f for f in os.listdir(dirName) if os.path.splitext(f)[1] == '.wav']

    speechList = []

    for fileName in fileList:
        speech = Speech(dirName, fileName)
        speech.extractFeature()
        speechList.append(speech)

    # print(speechList)
    return speechList


def training(speechList):
    ''' HMM training '''
    speechRecognizerList = []

    # initialize speechRecognizer
    for categoryId in CATEGORY:
        speechRecognizer = SpeechRecognizer(categoryId)
        speechRecognizerList.append(speechRecognizer)

    # organize data into the same category
    for speechRecognizer in speechRecognizerList:
        for speech in speechList:
            # print(speech.categoryId)
            # print(speechRecognizer.categoryId)
            # print("+++++++++++++++++++++++")
            if speech.categoryId == speechRecognizer.categoryId:
                speechRecognizer.trainData.append(speech.features)

        # get hmm model
        speechRecognizer.initModelParam(nComp=5, nMix=2,covarianceType='diag', n_iter=10,bakisLevel=2)
        speechRecognizer.getHmmModel()
    return speechRecognizerList


def recognize(testSpeechList, speechRecognizerList):
    ''' recognition '''
    predictCategoryIdList = []

    for testSpeech in testSpeechList:
        scores = []

        for recognizer in speechRecognizerList:
            score = recognizer.hmmModel.score(testSpeech.features)
            scores.append(score)

        idx = scores.index(max(scores))
        predictCategoryId = speechRecognizerList[idx].categoryId
        predictCategoryIdList.append(predictCategoryId)

    return predictCategoryIdList


def calculateRecognitionRate(groundTruthCategoryIdList, predictCategoryIdList):
    ''' calculate recognition rate '''
    score = 0
    length = len(groundTruthCategoryIdList)

    for i in range(length):
        gt = groundTruthCategoryIdList[i]
        pr = predictCategoryIdList[i]

        if gt == pr:
            score += 1

    recognitionRate = float(score) / length
    return recognitionRate

def main():
    trainDir = 'train_spoken_digit/'
    testDir = 'test_spoken_digit/'
    mainSpokenDir = 'spoken_digit/'
    splitTrainAndTest(trainDir, testDir, mainSpokenDir)
    #
    # ### Step.1 Loading training data
    # print('Step.1 Training data loading...')
    # trainSpeechList = loadData(trainDir)
    # print('done!')
    #
    # ### Step.2 Training
    # print('Step.2 Training model...')
    # speechRecognizerList = training(trainSpeechList)
    # print('done!')
    #
    # ### Step.3 Loading test data
    # print('Step.3 Test data loading...')
    # testSpeechList = loadData(testDir)
    # print('done!')
    #
    # ### Step.4 Recognition
    # print('Step.4 Recognizing...')
    # predictCategoryIdList = recognize(testSpeechList, speechRecognizerList)
    #
    # ### Step.5 Print result
    # groundTruthCategoryIdList = [speech.categoryId for speech in testSpeechList]
    # recognitionRate = calculateRecognitionRate(groundTruthCategoryIdList, predictCategoryIdList)
    #
    # print('===== Final result =====')
    # print('Ground Truth:\t', groundTruthCategoryIdList)
    # print('Prediction:\t', predictCategoryIdList)
    # print('Accuracy:\t', recognitionRate)
    #

if __name__ == '__main__':
    main()
