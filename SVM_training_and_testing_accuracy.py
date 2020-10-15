print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data
import msvcrt
import bci_workshop_tools as BCIw  # Our own functions for the workshop
from scipy.io import loadmat
import pickle
import time
from sklearn.model_selection import KFold, ShuffleSplit
import msvcrt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from funcsigs import signature
from sklearn.metrics import average_precision_score
start = time.time()

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

#TRAINING DATA
data_c_t = np.array(np.load('concentrated.npy'))[:,index_channel]
data_r_t = np.array(np.load('relaxed.npy'))[:,index_channel]
print(data_c_t)
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


#TESTING DATA
data_c_t = np.array(np.load('c_t10.npy'))[:,index_channel]
data_r_t = np.array(np.load('r_t10.npy'))[:,index_channel]
print(data_c_t)
eeg_epochs0 = BCIw.epoch(data_r_t, epoch_length * fs,
						 overlap_length * fs)
eeg_epochs1 = BCIw.epoch(data_c_t, epoch_length * fs,
						 overlap_length * fs)
feat_matrix0 = BCIw.compute_feature_matrix(eeg_epochs0, fs)
feat_matrix1 = BCIw.compute_feature_matrix(eeg_epochs1, fs)

class0 = np.zeros((feat_matrix0.shape[0], 1))
class1 = np.ones((feat_matrix1.shape[0], 1))

# Concatenate feature matrices and their respective labels
y_t = np.concatenate((class0, class1), axis=0)
features_all = np.concatenate((feat_matrix0, feat_matrix1),
							  axis=0)
mu_ft = np.mean(features_all, axis=0)
std_ft = np.std(features_all, axis=0)

X_t = (features_all - mu_ft) / std_ft

#TRAINING DIFFERENT ALGORITHMS
clf = SVC(kernel='rbf', C=0.1, gamma= 0.1)
clf.fit(X, y)
print('Training accuracy of RBF',clf.score(X,y.ravel()))
print('Testing accuracy of RBF', clf.score(X_t,y_t.ravel()))
y_score = clf.decision_function(X_t)
precision, recall, _ = precision_recall_curve(y_t, y_score)

plt.figure()
# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
average_precision = average_precision_score(y_t, y_score)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))
		  
print('tn, fp, fn, tp =',confusion_matrix(y_t,clf.predict(X_t)).ravel())

clf = SVC(kernel='linear')
clf.fit(X, y)
print('\n\nTraining accuracy of Linear',clf.score(X,y.ravel()))
print('Testing accuracy of Linear', clf.score(X_t,y_t.ravel()))
print('tn, fp, fn, tp =',confusion_matrix(y_t,clf.predict(X_t)).ravel())

clf = SVC(kernel='poly')
clf.fit(X, y)
print('\n\nTraining accuracy of Poly',clf.score(X,y.ravel()))
print('Testing accuracy of Poly', clf.score(X_t,y_t.ravel()))
print('tn, fp, fn, tp =',confusion_matrix(y_t,clf.predict(X_t)).ravel())

msvcrt.getch()

#TUNING FOR BEST PARAMETERS
C_range = [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
gamma_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = {'C': C_range, 'gamma' : gamma_range}

grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=StratifiedKFold(n_splits=5, random_state=None, shuffle=False))
grid.fit(X, y)
for params, mean_score, scores in grid.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r"
    % (mean_score, scores.std() / 2, params))
print()
print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

	  
#PLOTTING VALIDATION ACCURACY

# Utility function to move the midpoint of a colormap to be around
# the values of interest.

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
		


scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                     len(gamma_range))


													 
													 
# Draw heatmap of the validation accuracy as a function of gamma and C

# The score are encoded as colors with the hot colormap which varies from dark
# red to bright yellow. As the most interesting scores are all located in the
# 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
# as to make it easier to visualize the small variations of score values in the
# interesting range while not brutally collapsing all the low score values to
# the same color.
end = time.time()
print("Time elapsed:",end-start)
plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
           norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
plt.yticks(np.arange(len(C_range)), C_range)
plt.title('Validation accuracy')
plt.show()

