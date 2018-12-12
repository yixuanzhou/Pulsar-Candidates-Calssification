import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
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

models = {
    'DecisionTree': DecisionTreeClassifier(random_state=95),
    'RandomForest': RandomForestClassifier(random_state=95),
    'SVM': SVC(probability=True, random_state=95),
    'LogsticRegression': LogisticRegression(solver='liblinear', n_jobs=-1, random_state=95)
    'KNN': KNeighborsClassifier()
}

models_params = {
    'KNN': [
        {'n_neighbors': np.linspace(3, 51, 24, dtype=np.int16)}
    ],
    
    'SVM': [
        {'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100]},
        {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100]},
        {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100]}
    ],

    'DecisionTree': [
        {'max_depth': np.linspace(1, 32, 32, dtype=np.int16)},
        {'min_samples_split': np.linspace(2, 32, 31, dtype=np.int16)},
        {'min_samples_leaf': np.linspace(2, 16, 15, dtype=np.int16)}
    ],
    
    'RandomForest': [
        {'n_estimators': [2, 4, 8, 16, 32, 64, 100, 200]},
        {'max_depth': np.linspace(1, 32, 32, dtype=np.int16)},
        {'min_samples_split':  np.linspace(2, 32, 31, dtype=np.int16)},
        {'min_samples_leaf':  np.linspace(2, 16, 15, dtype=np.int16)}        
    ],

    'LogsticRegression': [
        {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'penalty': ['l2', 'l1']},
        {'tol': [1e-4, 1e-3, 1e-5]}
    ]
}

# Find the best parameters
dict_best_params = {}
dict_best_estimators = {}
dict_scores = {}

for model_name, md in models.items():
    print(model_name)
    param = models_params[model_name]
    best_param, df_score, best_estimator_ = evaluation.best_param_search(md, param, X_train, y_train)
    dict_best_params[model_name] = best_param
    dict_best_estimators[model_name] = best_estimator_
    dict_scores[model_name] = df_score
    print(best_param)