from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import json
import os
import numpy as np

# Read in data
X_train = np.genfromtxt("../data/train_features.csv")
y_train = np.genfromtxt("../data/train_labels.csv")
X_test = np.genfromtxt("../data/test_features.csv")
y_test = np.genfromtxt("../data/test_labels.csv")

# Fit a model
depth = 10
num_estimators = 50
clf = RandomForestClassifier(max_depth=depth, n_estimators=num_estimators)
clf.fit(X_train, y_train)

y_scores = clf.predict(X_test)
acc = clf.score(X_test, y_test)

# Plot confusion matrix
cm_ = plot_confusion_matrix(clf, X_test, y_test, normalize='true', cmap=plt.cm.Blues)
plt.savefig('..\\output_metrics\\confusion_matrix.png')

roc_ = plot_roc_curve(clf, X_test, y_test)

plt.savefig('..\\output_metrics\\roc_curve.png')

auc_val = roc_.roc_auc
print('accuracy ', acc)
print('auc ', auc_val)
print('depth ', depth)
print('number of estimators ', num_estimators)
with open("..\\output_metrics\\metrics.txt", 'w') as outfile:
    outfile.writelines(["Accuracy: " + str(acc) + "\n",
                        "AUC: " + str(auc_val) + "\n"
                        "Depth: " + str(depth) + "\n",
                        "Number estimators: " + str(num_estimators) + "\n"])

