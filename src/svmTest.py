import csv
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.svm import SVC, NuSVC, LinearSVC

# csv write a formated csv file
def csvFormatedOutput(fileName, ans):
    with open(fileName, 'w', newline='') as f:
        c = csv.writer(f, delimiter=',',
                       quotechar='|', quoting=csv.QUOTE_MINIMAL)
        c.writerow(['ID', 'Prediction'])
        for i in range(0, len(ans)):
#             temp = listToMat()
            c.writerow([i + 1, ans[i]])
    return 0

def listToMat(l):
    return np.array(l).reshape(-1, np.array(l).shape[-1])

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
def DimRedPlot(x,y):
    plt.figure()
    colors = ['navy', 'turquoise']
    labelName = ['bad', 'good']
    lw = 2
    # y= np.reshape(trainingTargets,[-1,])
    for color, i, label in zip(colors, [0, 1], labelName):
        plt.scatter(x[y == bin(i), 0],
                    x[(y) == bin(i), 1],
                    color=color, alpha=.8, lw=lw, label=label)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.show()
    return 0

def calibrationProb(classifier, X, y, test):
    caliClassifier = CalibratedClassifierCV(classifier, cv=10, method='isotonic')
    caliClassifier.fit(X, y)
    preTest = caliClassifier.predict(test)
    probTest = caliClassifier.predict_proba(test)
    return preTest,probTest[:,1]

def searchClassifier(baseClassifier, paraGrid, X, y):
    clf = GridSearchCV(baseClassifier, paraGrid, cv =10,
                   scoring=make_scorer(metrics.accuracy_score))
    clf = clf.fit(X, np.reshape(y,[-1,]))
    return clf.best_estimator_

# load data
# read the training targets
trainTarget = '../data/targets.csv'
temp = []
temp.append(genfromtxt(trainTarget, delimiter=','))
trainingTemp = (listToMat(temp)).T
temp = []
for i in range(trainingTemp.shape[0]):
    if trainingTemp[i] == 0:
        trainingTemp[i] = -1
    temp.append((int(trainingTemp[i])))
trainingTargets = listToMat(temp).T

# read the training features
trainFeature = '../trainFeature.csv'
temp = []
temp.append(genfromtxt(trainFeature, delimiter=','))
trainingFeatures = (listToMat(temp)).T

# read the testing features
testFeature = '../testFeature.csv'
temp = []
temp.append(genfromtxt(testFeature, delimiter=','))
testingFeatures = (listToMat(temp)).T

X = trainingFeatures
y = np.reshape(trainingTargets, [-1,])

print('import success')
# print(trainingTargets)
# lda
# ldaParaGrid = {'solver': ['eigen'],
#                'n_components': [1, 2],}
# lda = LinearDiscriminantAnalysis(shrinkage='auto')
# # ldaClf = GridSearchCV(lda, ldaParaGrid, scoring=make_scorer(metrics.accuracy_score), cv= 10)
# # ldaClf.fit(X, y)
# ldaBest = searchClassifier(lda, ldaParaGrid,
#                            trainingFeatures, trainingTargets)
# ldaPreTest, ldaProbTest = calibrationProb(ldaBest,
#                                           trainingFeatures, trainingTargets, testingFeatures)
# # print(ldaProbTest)
# csvFormatedOutput('../ldaBestPredictedProb.csv', ldaProbTest)

# sgd
# sgdParaGrid = {'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
              # 'penalty': ['l1','l2'],
#              'alpha': [0.00001,0.0001,0.001,0.01,0.1]}
# sgd = SGDClassifier(n_iter=1000)
# sgdBest = searchClassifier(sgd, sgdParaGrid,
#                            trainingFeatures, trainingTargets)
# sgdPreTest, sgdProbTest = calibrationProb(sgdBest,
#                                           trainingFeatures, trainingTargets, testingFeatures)
# csvFormatedOutput('../sgdBestHistRPredictedProb.csv', sgdProbTest)

# # adaBoost + sgdBest
# baseClassifier = sgdBest
# adaParaGrid = {'n_estimators': [10,50,100,200],
#               'learning_rate':[0.001,0.01,0.1,0.5,1],
#               'algorithm': ['SAMME'],}
# ada = AdaBoostClassifier(base_estimator=baseClassifier, random_state=0)
# adaBest = searchClassifier(ada, adaParaGrid, trainingFeatures, trainingTargets)
# adaPreTest,adaProbTest = calibrationProb(adaBest, trainingFeatures, trainingTargets,
#                                          testingFeatures)
# csvFormatedOutput('../adaSGDBestPredictedProb.csv', adaProbTest)

# bagging + sgdBest
# baseClassifier = sgdBest
# baggingParaGrid = {'n_estimators': [10, 50, 100, 200]}
# bagging = BaggingClassifier(base_estimator=baseClassifier, max_features=1.0, max_samples=1.0)
# baggingBest = searchClassifier(bagging, baggingParaGrid, trainingFeatures, trainingTargets)
# baggingPreTest, baggingProbTest = calibrationProb(baggingBest,
#                                                   trainingFeatures, trainingTargets, testingFeatures)
# csvFormatedOutput('../baggingSGDBestNew1PredictedProb.csv', baggingProbTest)

# svm
svcParaGrid = {'C':[1e-1,1e0,1e1,1e2,1e3,1e4,1e5],
              'kernel': ['rbf','linear','poly','sigmoid'],
              'degree': [3,4,5,6,7],
              'gamma': [1e-2,1e-1,1e0,1e1,1e2,1e3,1e4,1e5]}
svc = SVC(probability=True)
svcBest = searchClassifier(svc, svcParaGrid, trainingFeatures, trainingTargets)
svcProbTest = svcBest.predict_proba(testingFeatures)
csvFormatedOutput('../svcBestNew1PredictedProb.csv', svcProbTest[:,1])

