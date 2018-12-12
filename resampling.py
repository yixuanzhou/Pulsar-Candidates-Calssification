import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
import evaluation
import preprocess

# Load data set
df = pd.read_csv("./HTRU_2.csv", hearer=None)
X = df.iloc[:,:-1]
y = df.iloc[:,-1:].values

# Normalization
normalizer = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_train = normalizer.fit_transform(X_train)
X_test = normalizer.transform(X_test)

# Load models
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
lr = LogisticRegression(solver='liblinear')
svc = SVC(gamma='auto')
knn = KNeighborsClassifier(n_neighbors=2)
models = {'dt':dt, 'rf':rf, 'lr':lr, 'svc':svc, 'knn':knn}

# Original data set
for name, model in models.items():
    print(name, evaluation.cross_validation(model, X_train, y_train))

# Random over-sampling with ratio 1/5
X_train_up, y_train_up = preprocess.upsampling(X_train, y_train, ratio=1/5)
for name, model in models.items():
    print(name, evaluation.cross_validation(model, X_train_up, y_train_up))

# Random under-sampling with ratio 1/5
X_train_down, y_train_down = preprocess.downsampling(X_train, y_train, ratio=1/5)
for name, model in models.items():
    print(name, evaluation.cross_validation(model, X_train_down, y_train_down))

# Standard SMOTE with ratio 1/6
X_train_smote, y_train_smote = preprocess.smote(X_train, y_train, ratio=1/6)
for name, model in models.items():
    print(name, evaluation.cross_validation(model, X_train_smote, y_train_smote))