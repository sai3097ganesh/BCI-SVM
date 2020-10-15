print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import bci_workshop_tools as BCIw  # Our own functions for the workshop


"""INITIALIZE PARAMETERS"""
n_channels = 4
index_channel = [0,1,2,3]
ch_names = ['TP9','AF7','AF8','TP10']
fs = 256

""" 2. SET EXPERIMENTAL PARAMETERS """

# Length of the EEG data buffer (in seconds)
# This buffer will hold last n seconds of data and be used for calculations
buffer_length = 15

# Length of the epochs used to compute the FFT (in seconds)
epoch_length = 1

# Amount of overlap between two consecutive epochs (in seconds)
overlap_length = 0.9

# Amount to 'shift' the start of each next consecutive epoch
shift_length = epoch_length - overlap_length


# Get names of features
# ex. ['delta - CH1', 'pwr-theta - CH1', 'pwr-alpha - CH1',...]
feature_names = BCIw.get_feature_names(ch_names)

# Number of seconds to collect training data for (one class)
training_length = 20
# Import some data to play with
data_c_t = np.array(np.load('concentrated.npy'))[:,index_channel]
data_r_t = np.array(np.load('relaxed.npy'))[:,index_channel]
print(data_c_t,data_r_t)
print(data_c_t.shape,data_r_t.shape)
eeg_epochs0 = BCIw.epoch(data_r_t, epoch_length * fs,
						 overlap_length * fs)
eeg_epochs1 = BCIw.epoch(data_c_t, epoch_length * fs,
						 overlap_length * fs)
feat_matrix0 = BCIw.compute_feature_matrix(eeg_epochs0, fs)
feat_matrix1 = BCIw.compute_feature_matrix(eeg_epochs1, fs)

class0 = np.zeros((feat_matrix0.shape[0], 1))
class1 = np.ones((feat_matrix1.shape[0], 1))

# Concatenate feature matrices and their respective labels

y = np.concatenate((class0, class1), axis=0)
features_all = np.concatenate((feat_matrix0, feat_matrix1),
							  axis=0)
mu_ft = np.mean(features_all, axis=0)
std_ft = np.std(features_all, axis=0)

X = (features_all - mu_ft) / std_ft
print(X.shape)

# Add noisy features to make the problem harder
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
#X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                    random_state=0)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=0))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)


# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, label='ROC curve for \'linear\' kernel' )
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

classifier = OneVsRestClassifier(svm.SVC(kernel='rbf', probability=True,
                                 random_state=0))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC curve for \'rbf\' kernel' )

classifier = OneVsRestClassifier(svm.SVC(kernel='poly', probability=True,
                                 random_state=0))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC curve for \'poly\' kernel ')


plt.xlabel('False Positive Rate ')
plt.ylabel('True Positive Rate ')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

# Plot ROC curve
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                   ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()