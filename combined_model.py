# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 22:37:33 2018

@author: harsh
"""

import os, pickle
import numpy as np, pandas as pd
from collections import defaultdict
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

questionnaire_data_path = os.path.join(os.path.dirname(__file__), 'questionnaire_data.pkl')

with open(questionnaire_data_path, 'rb') as f:
    questionnaire_data = pickle.load(f)

idsPath = os.path.join(os.path.dirname(__file__), 'imageIds.pkl')
with open(idsPath, 'rb') as f:
    imageIds = pickle.load(f)

imageDataPath = os.path.join(os.path.dirname(__file__), 'data.pkl')
imageLabelsPath = os.path.join(os.path.dirname(__file__), 'labels.pkl')
with open(imageDataPath, 'rb') as f1, open(imageLabelsPath, 'rb') as f2:
    image_data = pickle.load(f1)
    image_labels = pickle.load(f2)
image_data = pd.DataFrame(image_data)
image_data.index = imageIds
image_data['Labels'] = image_labels

missingQuestionnaire = set(image_data.index) - set(questionnaire_data.index)
missingImage = set(questionnaire_data.index) - set(image_data.index)
badIndices = missingQuestionnaire | missingImage
goodIndices = set([i for i in imageIds + list(questionnaire_data.index) if i not in badIndices])
questionnaire_data = questionnaire_data.loc[goodIndices]
image_data = image_data.loc[goodIndices]
image_data = image_data.reindex(questionnaire_data.index)

labels = pd.DataFrame(image_data.Labels)
labels['questionnaire'] = questionnaire_data.ssi
labels.rename(columns={'Labels': 'image'}, inplace=True)
goodIndices -= set((labels[labels.image != labels.questionnaire]).index)
questionnaire_data = questionnaire_data.loc[goodIndices]
image_data = image_data.loc[goodIndices]

normedQuestionnaireData = pd.DataFrame(scale(questionnaire_data.drop('ssi', axis='columns'), axis=0))
normedQuestionnaireData.columns = questionnaire_data.columns[:-1]
normedQuestionnaireData.index = questionnaire_data.index
combinedData = pd.concat([normedQuestionnaireData, image_data], axis=1)
X, Y = combinedData.drop('Labels', axis='columns'), combinedData.Labels

numSplits = 100
accs = []
relevances = []
rates = np.zeros((0, 4))
rocAucs = []
rocCurves = []
usedFeatures = defaultdict(lambda: [])
intercepts = []
for i in range(numSplits):
    numSamples = X.shape[0]
    numTrain = int(0.75 * numSamples)
    split = np.random.permutation(numSamples)
    trainIndices, testIndices = split[:numTrain], split[numTrain:]
    XTrain, YTrain = X.iloc[trainIndices,:], Y.iloc[trainIndices]
    XTest, YTest = X.iloc[testIndices,:], Y.iloc[testIndices]

    model = LogisticRegression(penalty='l1', solver='liblinear')
    model.fit(XTrain, YTrain)

    predictions = model.predict(XTest)
    #predictions = pd.Series(model.predict(test_data))
    #predictions.index = test_data.index
    #print('Features:', XTrain.shape)
    #print(np.sum(YTest), YTest[predictions != YTest])
    #print('Accuracy:', model.score(XTest, YTest))
    
    acc = model.score(XTest, YTest)
    accs.append(acc)
    relevantFeatures = (model.coef_ != 0).flatten()
    numRelevantFeatures = np.sum(relevantFeatures)
    relevances.append(numRelevantFeatures)
    used = np.arange(1, XTest.shape[1] + 1)[relevantFeatures]
    for feature in used:
        usedFeatures[feature].append(model.coef_[0,feature-1])
    confMat = confusion_matrix(YTest, predictions)
    totNo, totYes = np.sum(confMat, axis=1)
    #tn, fp, fn, tp
    rates = np.vstack((rates, confMat.ravel()/np.array([totNo, totNo, totYes, totYes])))
    rocAucs.append(roc_auc_score(YTest, predictions))
    rocCurves.append(roc_curve(YTest, predictions, drop_intermediate=False))
    intercepts.append(model.intercept_)

accs = np.array(accs)
minAcc, maxAcc, medAcc = np.amin(accs), np.amax(accs), np.median(accs)
print('Median Accuracy:', medAcc, '\nAccuracy Range: (', minAcc, '-', maxAcc, ')')
relevances = np.array(relevances)
minRel, maxRel, medRel = np.amin(relevances), np.amax(relevances), np.median(relevances)
print('Median Number of Relevant Features:', medRel, '\nNumber of Relevant Features Range: (', minRel, '-', maxRel, ')')
minTn, minFp, minFn, minTp = np.amin(rates, axis=0)
medTn, medFp, medFn, medTp = np.median(rates, axis=0)
maxTn, maxFp, maxFn, maxTp = np.amax(rates, axis=0)
print('True Negative Rate:', medTn, '(', minTn, '-', maxTn, ')')
print('False Positive Rate:', medFp, '(', minFp, '-', maxFp, ')')
print('False Negative Rate:', medFn, '(', minFn, '-', maxFn, ')')
print('True Positive Rate:', medTp, '(', minTp, '-', maxTp, ')')
rocAucs = np.array(rocAucs)
sortedAucIndices = np.argsort(rocAucs)
medRocIndex = sortedAucIndices[rocAucs.shape[0]//2]
minAuc, maxAuc, medAuc = np.amin(rocAucs), np.amax(rocAucs), rocAucs[medRocIndex]
print('Median ROC AUC:', medAuc, '(', minAuc, '-', maxAuc, ')')
fpr, tpr, _ = rocCurves[medRocIndex]
rocCurves = np.array(rocCurves)
plt.figure()
plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.show()

def getCoefStats(coefvals):
    coefvals = sorted(coefvals)
    minVal, maxVal, sumVals = coefvals[0], coefvals[-1], sum(coefvals)
    medVal = coefvals[len(coefvals)//2]
    return (minVal, maxVal, medVal, sumVals)
usedFeatures = dict([(k, getCoefStats(v)) for k, v in usedFeatures.items()])
sortedFeatures = sorted(usedFeatures, key=lambda k: abs(usedFeatures[k][-1]), reverse=True)


#print('Feature Importances:')
relevantCoefs = []
for feature in sortedFeatures[:int(maxRel)]:
    minVal, maxVal, medVal, sumVals = usedFeatures[feature]
    relevantCoefs.append(medVal)
    #print(feature, ':', medVal, '(', minVal, '-', maxVal, ')', sumVals)


relevantIndices = np.array([sortedFeatures[:int(maxRel)]]).flatten() - 1
relevantData = X.iloc[:,relevantIndices]
relevantCoefs = np.array(relevantCoefs)
pca = PCA(n_components=3)
pca.fit(relevantData)
print('Total explained variance:', np.sum(pca.explained_variance_ratio_))
a,b,c = np.dot(pca.components_, relevantCoefs)
intercepts = np.array(intercepts)
d = np.median(intercepts)
print('Median intercept:', d)
reducedData = pca.transform(relevantData)

x, y, z = reducedData[:,0], reducedData[:,1], reducedData[:,2]

delta = 0.1
planeX, planeY = np.meshgrid(np.arange(np.amin(x), np.amax(x), delta), np.arange(np.amin(y), np.amax(y), delta))
planeZ = np.zeros(planeX.shape)
for py in np.arange(planeX.shape[0]):
    for px in np.arange(planeX.shape[1]):
        xval, yval = planeX[py,px], planeY[py,px]
        zval = (0.5 - a*xval - b*yval - d)/c
        planeZ[py,px] = zval

colors = Y.replace({True: 'r', False:'g'})
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_wireframe(planeX, planeY, planeZ, alpha=.15)
ax.scatter(x, y, z, s=5, c=colors)
plt.show()