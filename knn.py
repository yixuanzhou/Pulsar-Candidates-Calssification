import pandas as pd
import os
from sklearn import model_selection, metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from pcfm import plot_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import preprocess
import evaluation

# Load data set
df = pd.read_csv("./HTRU_2.csv", hearer=None)
X = df.iloc[:,:-1]
y = df.iloc[:,-1:].values

# Normalization
normalizer = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_train = normalizer.fit_transform(X_train)
X_test = normalizer.transform(X_test)
X_train, y_train = preprocess.upsampling(X_train, y_train, ratio=1/5)

# Train model with fine-tuned parameters
knn = KNeighborsClassifier(n_neighbors=5)
clf_knn = knn.fit(X_train, y_train)

# Cross validation
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=95)
res = {}
for scoring in ('f1', 'roc_auc', 'precision', 'recall'):
    res[scoring] = cross_val_score(clf_knn, X_test, y_test, cv=cv, scoring=scoring, n_jobs=-1)
print(res['f1'].mean(), res['roc_auc'].mean(), res['precision'].mean(), res['recall'].mean())

# Final result for training set and test set
print(evaluation.test_score(clf_knn, X_train, y_train))
print(evaluation.test_score(clf_knn, X_test, y_test))

# Plot confusion
cmyy = metrics.confusion_matrix(y_test, y_pred)
metcm = [cmyy]
plot_confusion_matrix(np.mean(metcm,axis=0), ['RFI','PULSAR'], cmap=plt.cm.RdPu, title='KNN', colmap=True)
plt.show()