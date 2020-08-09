import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

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

# Partition our data into training and test sets.
# We have 572 total patients left in our dataset
# 61 patients are infected and 511 are healthy
# We'll use 80% of the dataset (458 patients) to train
# and the remaining 20% (114 patients) to test,
# while keeping the proportions of healthy to infected
# patients roughly the same as in the whole dataset.
# So we'll have:
#	Training: 409 healthy patients, 49 infected
#	Test: 102 healthy patients, 12 infected
num_healthy_training, num_infected_training = 409, 49

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
healthy_patient_ids = np.random.permutation(data.index[data.ssi == False])
infected_patient_ids = np.random.permutation(data.index[data.ssi == True])
train_healthy_ids, train_infected_ids = healthy_patient_ids[:num_healthy_training],\
										infected_patient_ids[:num_infected_training]
test_healthy_ids, test_infected_ids = healthy_patient_ids[num_healthy_training:],\
										infected_patient_ids[num_infected_training:]
train_ids = np.random.permutation(np.concatenate((train_healthy_ids, train_infected_ids)))
test_ids = np.random.permutation(np.concatenate((test_healthy_ids, test_infected_ids)))
train_data = data.loc[train_ids]
test_data = data.loc[test_ids]

train_labels = train_data.ssi
train_data.drop('ssi', axis=1, inplace=True)
test_labels = test_data.ssi
test_data.drop('ssi', axis=1, inplace=True)

# Train a logistic regression model and evaluate it on our test set
model = LogisticRegression()
model.fit(train_data, train_labels)

# Make predictions on the test set
predictions = pd.Series(model.predict(test_data))
predictions.index = test_data.index
# print(predictions)
# See the accuracy of the model on the test data
print('Model accuracy:', model.score(test_data, test_labels))

# See the true positive and true negative rates of the model's classifications
print('True Positive Rate:', np.mean(predictions.loc[test_infected_ids]))
print('True Negative Rate:', np.mean(~predictions.loc[test_healthy_ids]))

# View the confusion matrix of the model's classifications
confMat = pd.DataFrame(confusion_matrix(test_labels, predictions))
confMat.index = ['Actual Not Infected', 'Actual Infected']
confMat.columns = ['Predicted Not Infected', 'Predicted Infected']
print('\nConfusion Matrix:', '\n', confMat, '\n')

# Evaluate the model in terms of ROC AUC
fpr, tpr, _ = roc_curve(test_labels, predictions, drop_intermediate=False)

# Plot the ROC curve
plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.show()

# See ROC AUC
print('ROC AUC:', roc_auc_score(test_labels, predictions), '\n')

# See which features were most important for classification
importances = pd.Series(model.coef_[0])
importances.index = test_data.columns
importances = importances.reindex(importances.abs().sort_values(ascending=False).index)
print(importances)