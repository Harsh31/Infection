import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import os, pickle

columnNames = {'study_id': 'id',\
			   'fever3': 'fever',\
			   'incision_pain3': 'pain',\
			   'incision_redness3': 'redness',\
			   'incision_swelling3': 'swelling',\
			   'incision_firmness3': 'firmness',\
			   'incision_draining_32': 'draining',\
			   'incision_draininge_thick_thin3': 'draining_thickness',\
			   'incision_draining_color3': 'draining_color',\
			   'incision_draining_smell3': 'draining_smell',\
			   'incision_gape3': 'gape',\
			   'ssi_present': 'ssi'}

data = pd.read_excel('YES_NO SSI LIST.xls', header=1)
data.rename(columns=columnNames, inplace=True)
# Get rid of empty rows
nonEmptyRows = data.apply(lambda r: not all(pd.isnull(r[1:])), axis=1)
data = data[nonEmptyRows]

# Get rid of patients with missing required values
requiredFeatures = {'redness', 'swelling', 'firmness', 'draining', 'gape'}
patientsToKeep = data.id.notnull()
for feature in requiredFeatures:
	patientsToKeep = patientsToKeep & data[feature].notnull()
data = data[patientsToKeep]

# draining_thickness, draining_color, and draining_smell are N/A if draining is 'No'
# draining_thickness takes on 2 values: 'Thick' and 'Thin'
# draining_color takes on 2 values: 'Brown, yellow, green, white' and 'Red, pink, clear'
# draining_smell is either 'Yes' or 'No'
# We will split these each up into two one hot columns
idx = 7
thick_draining = data.draining_thickness == 'Thick'
data.insert(loc=idx, column='thick_draining', value=thick_draining)
idx += 1
thin_draining = data.draining_thickness == 'Thin'
data.insert(loc=idx, column='thin_draining', value=thin_draining)
idx += 1
draining_color_1 = data.draining_color == 'Brown, yellow, green, white'
data.insert(loc=idx, column='draining_color_1', value=draining_color_1)
idx += 1
draining_color_2 = data.draining_color == 'Red, pink, clear'
data.insert(loc=idx, column='draining_color_2', value=draining_color_2)
idx += 1
draining_with_smell = data.draining_smell == 'Yes'
data.insert(loc=idx, column='draining_with_smell', value=draining_with_smell)
idx += 1
draining_without_smell = data.draining_smell == 'No'
data.insert(loc=idx, column='draining_without_smell', value=draining_without_smell)

# Get rid of the original columns for the categoricals
data.drop(['draining_thickness', 'draining_color', 'draining_smell'], axis=1, inplace=True)

# Change our data to contain boolean values and its index to be patient ids
data.replace({'Yes': True, 'No': False}, inplace=True)
data.set_index('id', inplace=True)

data_path = os.path.join(os.path.dirname(__file__), 'questionnaire_data.pkl')
with open(data_path, 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
labels = data.ssi
data.drop('ssi', axis=1, inplace=True)

numSplits = 100
accs = []
tpfns = np.zeros((0,4))
rates = np.zeros((0, 4))
rocAucs = []
rocCurves = []
relevances = []
coefs = np.zeros((0, data.shape[1]))
for i in range(numSplits):
    numSamples = data.shape[0]
    numTrain = int(0.75 * numSamples)
    split = np.random.permutation(numSamples)
    trainIndices, testIndices = split[:numTrain], split[numTrain:]
    train_data, train_labels = data.iloc[trainIndices,:], labels.iloc[trainIndices]
    test_data, test_labels = data.iloc[testIndices,:], labels.iloc[testIndices]
    
    # Train a logistic regression model and evaluate it on our test set
    model = SVC(gamma='auto')
    #model = LogisticRegression(penalty = 'l1', solver='liblinear')
    #model = LogisticRegression(solver='liblinear')
    model.fit(train_data, train_labels)

    # Make predictions on the test set
    predictions = pd.Series(model.predict(test_data))
    predictions.index = test_data.index
    accs.append(model.score(test_data, test_labels))
    confMat = confusion_matrix(test_labels, predictions)
    totNo, totYes = np.sum(confMat, axis=1)
    #tn, fp, fn, tp
    tpfn = confMat.ravel()
    tpfns = np.vstack((tpfns, tpfn))
    rates = np.vstack((rates, tpfn/np.array([totNo, totNo, totYes, totYes])))
    rocCurves.append(roc_curve(test_labels, predictions, drop_intermediate=False))
    rocAucs.append(roc_auc_score(test_labels, predictions))
    #coefs = np.vstack((coefs, model.coef_.flatten()))

# See the accuracy of the model on the test data
accs = np.array(accs)
minAcc, maxAcc, medAcc = np.amin(accs), np.amax(accs), np.median(accs)
print('Model accuracy:', medAcc, '(', minAcc, '-', maxAcc, ')')

# See the true positive and true negative rates of the model's classifications
minTN, minFP, minFN, minTP = np.amin(tpfns, axis=0)
maxTN, maxFP, maxFN, maxTP = np.amax(tpfns, axis=0)
medTN, medFP, medFN, medTP = np.median(tpfns, axis=0)
minSpecificity, _, _, minSensitivity = np.amin(rates, axis=0)
maxSpecificity, _, _, maxSensitivity = np.amax(rates, axis=0)
medSpecificity, _, _, medSensitivity = np.median(rates, axis=0)
print('Sensitivity:', medSensitivity, '(', minSensitivity, '-', maxSensitivity, ')')
print('Specificity:', medSpecificity, '(', minSpecificity, '-', maxSpecificity, ')')

# View the confusion matrix of the model's classifications
tnStr = ' '.join([str(medTN), '(', str(minTN), '-', str(maxTN), ')'])
fpStr = ' '.join([str(medFP), '(', str(minFP), '-', str(maxFP), ')'])
fnStr = ' '.join([str(medFN), '(', str(minFN), '-', str(maxFN), ')'])
tpStr = ' '.join([str(medTP), '(', str(minTP), '-', str(maxTP), ')'])
confMat = pd.DataFrame([[tnStr, fpStr], [fnStr, tpStr]])
confMat.index = ['Actual Not Infected', 'Actual Infected']
confMat.columns = ['Predicted Not Infected', 'Predicted Infected']
print('\nConfusion Matrix:', '\n', confMat, '\n')

rocAucs = np.array(rocAucs)
sortedAucIndices = np.argsort(rocAucs)
medRocIndex = sortedAucIndices[rocAucs.shape[0]//2]
minAuc, maxAuc, medAuc = rocAucs[sortedAucIndices[0]], rocAucs[sortedAucIndices[-1]], rocAucs[medRocIndex]
print('ROC AUC:', medAuc, '(', minAuc, '-', maxAuc, ')')
fpr, tpr, _ = rocCurves[medRocIndex]
plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.show()

minCoefs, maxCoefs, medCoefs = np.amin(coefs, axis=0), np.amax(coefs, axis=0), np.median(coefs,axis=0)
importances = pd.DataFrame([medCoefs, minCoefs, maxCoefs]).transpose()
importances.index = test_data.columns
importances.columns = ['Median', 'Min', 'Max']
importances = importances.reindex(importances.abs().sort_values(by='Median', ascending=False).index)
print(importances)