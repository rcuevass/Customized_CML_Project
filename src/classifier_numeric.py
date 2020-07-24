from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from ..utils.logger import get_log_object
import numpy as np

# instantiate log
log = get_log_object()

# Read in data
log.info('Reading training and test data ...')
X_train = np.genfromtxt("data/train_features.csv")
y_train = np.genfromtxt("data/train_labels.csv")
X_test = np.genfromtxt("data/test_features.csv")
y_test = np.genfromtxt("data/test_labels.csv")
log.info('Training and test data read...')

# Fit a model
depth = 10
num_estimators = 50
log.info('Training RF classifier with %i estimators and depth %i', num_estimators, depth)

clf = RandomForestClassifier(max_depth=depth, n_estimators=num_estimators)
clf.fit(X_train, y_train)

log.info('Classifier trained...')

y_scores = clf.predict(X_test)
acc = clf.score(X_test, y_test)

# Plot confusion matrix
log.info('Generating plots for ROC and CM')
cm_ = plot_confusion_matrix(clf, X_test, y_test, normalize='true', cmap=plt.cm.Blues)
plt.savefig('..\\output_metrics\\confusion_matrix.png')

roc_ = plot_roc_curve(clf, X_test, y_test)

plt.savefig('..\\output_metrics\\roc_curve.png')

log.info('CM and ROC plots generated...')

auc_val = round(roc_.roc_auc, 3)

log.info('Accuracy= %f', acc)
log.info('AUC= %f', auc_val)
log.info('Depth= %i', depth)
log.info('Number of estimators= %i', num_estimators)

with open("..\\output_metrics\\metrics.txt", 'w') as outfile:
    outfile.writelines(["Accuracy: " + str(acc) + "\n",
                        "AUC: " + str(auc_val) + "\n"
                        "Depth: " + str(depth) + "\n",
                        "Number estimators: " + str(num_estimators) + "\n"])

