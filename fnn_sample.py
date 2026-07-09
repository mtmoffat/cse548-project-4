# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 19:43:04 2019

Updated on Wed Jan 29 10:18:09 2020

@author: created by Sowmya Myneni and updated by Dijiang Huang

M. Tyler Moffat: Modified for CSE 548 Project 4.
M. Tyler Moffat: This version uses customized training/testing files for scenarios SA, SB, and SC.
"""

########################################
# Part 1 - Data Pre-Processing
#######################################

# To load a dataset file in Python, you can use Pandas.
import pandas as pd

# Import numpy to perform operations on the dataset.
import numpy as np

# M. Tyler Moffat: Added sys so the scenario can be selected from the command line.
import sys


# Variable Setup
# M. Tyler Moffat: Modified for Project 4 to use separate training and testing files.
# M. Tyler Moffat: Run examples:
# M. Tyler Moffat: python3 fnn_simple.py SA
# M. Tyler Moffat: python3 fnn_simple.py SB
# M. Tyler Moffat: python3 fnn_simple.py SC

TrainingDataPath = ''

# Batch Size
BatchSize = 10

# Epoch Size
NumEpoch = 10


# M. Tyler Moffat: Choose scenario from the command line. Default is SA.
if len(sys.argv) > 1:
    ScenarioName = sys.argv[1].upper()
else:
    ScenarioName = 'SA'


# M. Tyler Moffat: Project scenario setup.
# M. Tyler Moffat: A1 = DoS, A2 = Probe, A3 = U2R, A4 = R2L.
SCENARIOS = {
    'SA': {
        'train_file': 'Training-a1-a3.csv',
        'test_file': 'Testing-a2-a4.csv',
        'trained_classes': [1, 3],
        'test_classes': [2, 4]
    },
    'SB': {
        'train_file': 'Training-a1-a2.csv',
        'test_file': 'Testing-a1.csv',
        'trained_classes': [1, 2],
        'test_classes': [1]
    },
    'SC': {
        'train_file': 'Training-a1-a2.csv',
        'test_file': 'Testing-a1-a2-a3.csv',
        'trained_classes': [1, 2],
        'test_classes': [1, 2, 3]
    }
}


if ScenarioName not in SCENARIOS:
    print('Invalid scenario. Use SA, SB, or SC.')
    sys.exit(1)


Scenario = SCENARIOS[ScenarioName]
TrainingData = Scenario['train_file']
TestingData = Scenario['test_file']


print('Running Scenario:', ScenarioName)
print('Training file:', TrainingData)
print('Testing file:', TestingData)
print('Trained attack classes:', Scenario['trained_classes'])
print('Testing attack classes:', Scenario['test_classes'])


# M. Tyler Moffat: Attack class mapping for unknown attack analysis.
# M. Tyler Moffat: A1 = DoS, A2 = Probe, A3 = U2R, A4 = R2L.
ATTACKS_SUBCLASS = {
    1: [
        'apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod',
        'processtable', 'smurf', 'teardrop', 'udpstorm', 'worm'
    ],
    2: [
        'ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan'
    ],
    3: [
        'buffer_overflow', 'buffer.overflow', 'loadmodule', 'perl',
        'ps', 'rootkit', 'sqlattack', 'xterm'
    ],
    4: [
        'ftp_write', 'ftp.write', 'guess_passwd', 'guess.passwd',
        'httptunnel', 'httputunnel', 'imap', 'multihop', 'named',
        'phf', 'sendmail', 'snmpgetattack', 'spy', 'snmpguess',
        'snmmpguess', 'warezclient', 'warezmaster', 'warezserver',
        'xlock', 'xsnoop'
    ]
}


# M. Tyler Moffat: Load one NSL-KDD file and return features, binary labels, and original text labels.
def load_dataset(filename):
    dataset = pd.read_csv(filename, header=None, encoding='ISO-8859-1')

    X = dataset.iloc[:, 0:-2].values
    label_column = dataset.iloc[:, -2].astype(str).str.lower().values

    y = []

    for i in range(len(label_column)):
        if label_column[i] == 'normal':
            y.append(0)
        else:
            y.append(1)

    # Convert list to array.
    y = np.array(y)

    return X, y, label_column


# M. Tyler Moffat: Load training and testing files separately.
# M. Tyler Moffat: This replaces the original train_test_split approach.
X_train_raw, y_train, train_labels = load_dataset(TrainingDataPath + TrainingData)
X_test_raw, y_test, test_labels = load_dataset(TrainingDataPath + TestingData)


# Encoding categorical data.
# M. Tyler Moffat: Columns 1, 2, and 3 are categorical in NSL-KDD.
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# M. Tyler Moffat: handle_unknown='ignore' prevents errors if testing data has categories not seen in training.
# M. Tyler Moffat: sparse_output works in newer sklearn; sparse works in older sklearn.
try:
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
except TypeError:
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)


ct = ColumnTransformer(
    [('one_hot_encoder', encoder, [1, 2, 3])],
    remainder='passthrough'
)


# M. Tyler Moffat: Fit the encoder on training data, then transform testing data using the same encoder.
X_train = ct.fit_transform(X_train_raw)
X_test = ct.transform(X_test_raw)


# M. Tyler Moffat: Convert sparse matrices to arrays if needed.
if hasattr(X_train, 'toarray'):
    X_train = X_train.toarray()

if hasattr(X_test, 'toarray'):
    X_test = X_test.toarray()


# M. Tyler Moffat: np.float is deprecated, so use float instead.
X_train = np.array(X_train, dtype=float)
X_test = np.array(X_test, dtype=float)


# Perform feature scaling.
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

# M. Tyler Moffat: Fit scaler only on training data, then transform testing data.
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


########################################
# Part 2: Building FNN
#######################################

# Importing the Keras libraries and packages.
# M. Tyler Moffat: Import Keras in a way that works in different Python environments.
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
except Exception:
    from keras.models import Sequential
    from keras.layers import Dense


# Initialising the ANN.
classifier = Sequential()


# Adding the input layer and the first hidden layer.
classifier.add(Dense(
    units=6,
    kernel_initializer='uniform',
    activation='relu',
    input_dim=len(X_train[0])
))


# Adding the second hidden layer.
classifier.add(Dense(
    units=6,
    kernel_initializer='uniform',
    activation='relu'
))


# Adding the output layer.
classifier.add(Dense(
    units=1,
    kernel_initializer='uniform',
    activation='sigmoid'
))


# Compiling the ANN.
classifier.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# Fitting the ANN to the Training set.
# M. Tyler Moffat: Same batch size and epoch count are used for each scenario.
classifierHistory = classifier.fit(
    X_train,
    y_train,
    batch_size=BatchSize,
    epochs=NumEpoch
)


# Evaluate the model on the training dataset.
loss, accuracy = classifier.evaluate(X_train, y_train)

print('Print the loss and the accuracy of the model on the training dataset')
print('Loss [0,1]: %.4f' % loss, 'Accuracy [0,1]: %.4f' % accuracy)


########################################
# Part 3 - Making predictions and evaluating the model
#######################################

# Predicting the Test set results.
y_pred_prob = classifier.predict(X_test).ravel()

# M. Tyler Moffat: The original FNN lab uses 0.9 as the attack threshold.
y_pred = (y_pred_prob > 0.9).astype(int)


# Making the Confusion Matrix.
# [TN, FP]
# [FN, TP]
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
test_accuracy = accuracy_score(y_test, y_pred)


print('Print the Testing Accuracy:')
print('Testing Accuracy [0,1]: %.4f' % test_accuracy)

print('Print the Confusion Matrix:')
print('[ TN, FP ]')
print('[ FN, TP ]=')
print(cm)


# M. Tyler Moffat: Analyze unknown attack classes.
trained_classes = set(Scenario['trained_classes'])
test_classes = set(Scenario['test_classes'])
unknown_classes = sorted(list(test_classes - trained_classes))

print('Unknown attack class analysis:')

unknown_rows = []

if len(unknown_classes) == 0:
    print('No unknown attack classes in this scenario.')
else:
    for class_number in unknown_classes:
        subclass_names = ATTACKS_SUBCLASS[class_number]

        indexes = []

        for i, label in enumerate(test_labels):
            if label in subclass_names:
                indexes.append(i)

        if len(indexes) == 0:
            print('A' + str(class_number) + ': no rows found in testing file.')
            continue

        predicted_attack_count = int(np.sum(y_pred[indexes] == 1))
        predicted_normal_count = int(np.sum(y_pred[indexes] == 0))
        detection_accuracy = predicted_attack_count / len(indexes)
        average_attack_probability = float(np.mean(y_pred_prob[indexes]))

        print(
            'A' + str(class_number) +
            ': rows=' + str(len(indexes)) +
            ', predicted_attack=' + str(predicted_attack_count) +
            ', predicted_normal=' + str(predicted_normal_count) +
            ', attack_detection_accuracy=' + format(detection_accuracy, '.4f') +
            ', average_attack_probability=' + format(average_attack_probability, '.4f')
        )

        unknown_rows.append({
            'scenario': ScenarioName,
            'attack_class': 'A' + str(class_number),
            'rows': len(indexes),
            'predicted_attack': predicted_attack_count,
            'predicted_normal': predicted_normal_count,
            'attack_detection_accuracy': detection_accuracy,
            'average_attack_probability': average_attack_probability
        })


# M. Tyler Moffat: Save prediction and summary output files.
prediction_output = pd.DataFrame({
    'actual_label': test_labels,
    'actual_binary': y_test,
    'predicted_probability_attack': y_pred_prob,
    'predicted_binary': y_pred
})

prediction_output.to_csv('predictions_' + ScenarioName + '.csv', index=False)


scenario_summary = pd.DataFrame([{
    'scenario': ScenarioName,
    'train_file': TrainingData,
    'test_file': TestingData,
    'train_accuracy': accuracy,
    'train_loss': loss,
    'test_accuracy': test_accuracy,
    'tn': int(cm[0][0]),
    'fp': int(cm[0][1]),
    'fn': int(cm[1][0]),
    'tp': int(cm[1][1])
}])

scenario_summary.to_csv('summary_' + ScenarioName + '.csv', index=False)


if len(unknown_rows) > 0:
    unknown_summary = pd.DataFrame(unknown_rows)
    unknown_summary.to_csv('unknown_attack_' + ScenarioName + '.csv', index=False)


print('Generated files for Scenario ' + ScenarioName)
print('predictions_' + ScenarioName + '.csv')
print('summary_' + ScenarioName + '.csv')
print('unknown_attack_' + ScenarioName + '.csv, if unknown attacks existed')


########################################
# Part 4 - Visualizing
#######################################

# Import matplotlib libraries for plotting the figures.
import matplotlib.pyplot as plt


# You can plot the accuracy.
print('Plot the accuracy')

# M. Tyler Moffat: Some Keras versions use "acc" and others use "accuracy".
if 'accuracy' in classifierHistory.history:
    accuracy_key = 'accuracy'
else:
    accuracy_key = 'acc'

plt.plot(classifierHistory.history[accuracy_key])
plt.title('model accuracy - ' + ScenarioName)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.savefig('accuracy_' + ScenarioName + '.png')
plt.close()


# You can plot history for loss.
print('Plot the loss')

plt.plot(classifierHistory.history['loss'])
plt.title('model loss - ' + ScenarioName)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.savefig('loss_' + ScenarioName + '.png')
plt.close()
