import numpy as np
from numpy import genfromtxt
#from sklearn.datasets import make_friedman1
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import RandomizedPCA
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, normalize
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import make_scorer, mean_squared_error
import csv


# csv reader
def csvReader(fileName):
    readData = []
    with open(fileName) as f:
        for line in f:
            readData.append(line.strip().split(','))
    return readData


# csv write a formated csv file
def csvFormatedOutput(fileName, ans):
    with open(fileName, 'w', newline='') as f:
        c = csv.writer(f, delimiter=',',
                       quotechar='|', quoting=csv.QUOTE_MINIMAL)
        c.writerow(['ID', 'Prediction'])
        for i in range(0, len(ans)):
            c.writerow([i + 1, ans[i]])
    return 0


def forwardSubsetSelection(model, feature0, featureCandidate, targets, cv):
    feature = [];
    feature.append(feature0);
    feature.append(featureCandidate);

    #     get the cv score
    score = cross_val_score(model, feature.T, targets.T, cv)
    #     return np.median(score)
    return score

def cutoffAge(sTemp, minAge, maxAge):
    # minAge = np.min(targets);
    # maxAge = np.max(targets);

    # temp = []
    # temp.append(genfromtxt(fileName, delimiter=','))
    # temp = (np.array(temp).reshape(-1, np.array(temp).shape[-1]))
    # sTemp = temp[:, 1]
    tempOut = []
    for i in range(1, len(sTemp)):
        tempOut.append(int(sTemp[i] + 0.5))

    #
    for i in range(len(tempOut)):
        if tempOut[i] < minAge:
            tempOut[i] = minAge;
        if tempOut[i] > maxAge:
            tempOut[i] = maxAge;
    return tempOut

def listToMat(l):
    return np.array(l).reshape(-1, np.array(l).shape[-1])

def linearRegression(trainingFeatures, trainingTargets, trainingExamples):
    lr = linear_model.LinearRegression()
    cvList = []
    cvList.append(10)
    cvList.append(np.sqrt(trainingExamples))
    lr.fit(trainingFeatures.T, trainingTargets.T)
    lrPredicted = cross_val_predict(lr, trainingFeatures.T, trainingTargets.T, cv=int(np.min(cvList)))
    return lr, lrPredicted;

def ridgeRegression(trainingFeatures, trainingTargets, trainingExamples, alpha):
    cvList = []
    cvList.append(10)
    cvList.append(np.sqrt(trainingExamples))
    ridge = linear_model.Ridge(alpha)
    ridge.fit(trainingFeatures.T, trainingTargets.T)
    ridgePredicted = cross_val_predict(ridge, trainingFeatures.T, trainingTargets.T, cv=int(np.min(cvList)))
    return ridge, ridgePredicted;

def elasticNetRegression(trainingFeatures, trainingTargets, trainingExamples):

    return 0

def plotter(measured, predicted):
    # plot the result
    fig, ax = plt.subplots()
    y = measured
    ax.scatter(y, predicted)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()
    return 0

# load training and test
# training data
temp = []
temp.append(genfromtxt("trainFeature.csv", delimiter=','))
trainingFeatures = np.array(temp).reshape(-1, np.array(temp).shape[-1])
featureLength = np.array(trainingFeatures).shape[0]
trainingExamples = np.array(trainingFeatures).shape[1]

temp = []
temp.append(genfromtxt("trainTargets.csv", delimiter=','))
trainingTargets = np.array(temp).reshape(-1, np.array(temp).shape[-1])

# testing data
temp = []
temp.append(genfromtxt("testFeature.csv", delimiter=','))
testFeatures = (np.array(temp).reshape(-1, np.array(temp).shape[-1]))
testExamples = np.array(testFeatures).shape[1]


#lasso
cvList = []
cvList.append(10)
cvList.append(np.sqrt(trainingExamples))
lasso = linear_model.LassoLarsCV(cv=int(np.min(cvList)), max_iter=5000, max_n_alphas=1000)
lasso.fit(trainingFeatures.T, trainingTargets.T)
lassoEstimation = lasso.predict(testFeatures.T)
outputFileName = 'submissionCZLassoLarsOnlyOrg.csv'
csvFormatedOutput(outputFileName, lassoEstimation)

# rescaling the output
minAge = np.min(trainingTargets);
maxAge = np.max(trainingTargets);

temp = []
temp.append(genfromtxt(outputFileName, delimiter=','))
temp = (np.array(temp).reshape(-1, np.array(temp).shape[-1]))
sTemp = temp[:,1]
csvFormatedOutput('submissionCZLassoLarsFinal.csv', cutoffAge(sTemp, minAge, maxAge))
