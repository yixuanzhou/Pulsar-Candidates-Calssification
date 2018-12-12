import numpy as np
import pandas as pd
from sklearn.base import clone
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, ShuffleSplit, GridSearchCV


def cross_validation(cls, X, y, n_jobs=-1, n_splits=5):

    cv = ShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=95)
    output = {}
    for scoring in ('f1', 'roc_auc', 'precision', 'recall'):
        output[scoring] = cross_val_score(cls, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs).mean()

    return output


def test_score(model, X, y):

    y_pred = clf.predict(X)
    recall = recall_score(y, y_pred)
    prec = precision_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    return recall, prec, roc_auc, f1


def best_param_search(estimator, params, X, y, verbose=True, n_jobs=-1):

    best_params = {}
    df_scores = pd.DataFrame(columns=['test_score', 'train_score', 'fit_time', 'score_time'])
    _estimator = estimator
    clf = None

    for ps in params:
        estimator = clone(_estimator)
        for name, value in best_params.items():
            if name not in ps:
                ps[name] = [value]

        cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
        clf = GridSearchCV(estimator, ps, scoring='f1', cv=cv, n_jobs=n_jobs, return_train_score=True)
        clf.fit(X, y)
        for name, value in clf.best_params_.items():
            best_params[name] = value

        for i, dikt in enumerate(clf.cv_results_['params']):
            index_name = ';'.join(['{}:{}'.format(a, b) for a, b in dikt.items()])
            df_scores.loc[index_name] = [
                clf.cv_results_['mean_test_score'][i],
                clf.cv_results_['mean_train_score'][i],
                clf.cv_results_['mean_fit_time'][i],
                clf.cv_results_['mean_score_time'][i],
            ]

    return best_params, df_scores, getattr(clf, 'best_estimator_', None)
