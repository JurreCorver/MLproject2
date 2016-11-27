# This script produces the final submission as shown on
# the kaggle leaderboard.
#
# It runs correctly if placed next to the folders /src
# and /data. The folder /src contains whatever other
# scripts you need (provided by you). The folder /data
# can be assumed to contain two folders /set_train and
# /set_test which again contain the training and test
# samples respectively (provided by user, i.e. us).
#
# Its output is "final_sub.csv"

#We first create the features
import sys
sys.path.insert(0, './src/')
from adaBoostGrad.py import *
from fMRIToFeature.py import *


bins = 64

block = 8
testDir = '../data/set_test/'
trainDir = '../data/set_train/'

featureTest = mriToHistFeature(testDir, bins, block)
csvOutput('./src/testFeatureHist.csv', featureTest)
featureTrain = mriToHistFeature(trainDir, bins, block)
csvOutput('./src/trainFeatureHist.csv', featureTrain)

#Then we use adaBoost

# main begins
# load data
# read the training targets
trainTarget = '../data/targets.csv'
temp = []
temp.append(genfromtxt(trainTarget, delimiter=','))
trainingTemp = (listToMat(temp)).T
temp = []
for i in range(trainingTemp.shape[0]):
    # if trainingTemp[i] == 0:
    #     trainingTemp[i] = -1
    temp.append((int(trainingTemp[i])))
trainingTargets = listToMat(temp).T

# read the training features
trainFeature = './src/trainFeatureHist.csv'
temp = []
temp.append(genfromtxt(trainFeature, delimiter=','))
trainingFeatures = (listToMat(temp))

# read the testing features
testFeature = './src/testFeatureHist.csv'
temp = []
temp.append(genfromtxt(testFeature, delimiter=','))
testingFeatures = (listToMat(temp))

X = trainingFeatures
y = np.reshape(trainingTargets, [-1,])

# sgd + adaboost
numBoost = 600
gradBoostParaGrid = {'loss': ['deviance', 'exponential'],
                   'learning_rate': [0.0001,0.001,0.01,0.05,0.1,0.2,0.3,0.4,
                   0.5,0.6,0.7,0.8,0.9,1],
                   'n_estimators': [100,150,200],
                   'criterion': ['friedman_mse', 'mse','mae'],
                   'max_features':['sqrt']}

gradBoost = GradientBoostingClassifier(max_depth=1)
gradBoostBest = searchClassifier(gradBoost, gradBoostParaGrid,
                                trainingFeatures, trainingTargets)
boostEps, boostAlp, boostModel = adaBoost(gradBoostBest, numBoost, trainingFeatures, trainingTargets)
adaBoostSgdPred, adaBoostSgdPredProb = adaBoostPredict(boostModel,boostAlp, numBoost, testingFeatures)
csvFormatedOutput('./final_sub.csv', adaBoostSgdPredProb[:,1])


