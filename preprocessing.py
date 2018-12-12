import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
import random

def upsampling(X, y, ratio=1.0, random_state=123):
    pos_num = sum(y)
    target_num = int(0.5 + ratio * (len(y) - pos_num))
    if not pos_num < target_num:
        return X, y
    X_pos = X[np.where(y == 1)[0]]
    X_pos = np.concatenate([X_pos for _ in range(target_num // pos_num[0])] + [X_pos[:target_num % pos_num[0]]])
    X_neg = X[np.where(y == 0)[0]]
    X, y = np.concatenate([X_pos, X_neg]), np.array([1] * len(X_pos) + [0] * len(X_neg))
    return shuffle(X, y, random_state=random_state)


def downsampling(X, y, ratio=1.0, random_state=123):
    random.seed(random_state)
    pos_num = sum(y)
    if not pos_num < ratio * (len(y) - pos_num):
        return X, y
    X_neg = X[np.where(y == 0)[0]]
    X_neg = X_neg[:int(pos_num[0] / ratio)]
    X_pos = X[np.where(y == 1)[0]]
    X = np.concatenate([X_neg, X_pos])
    y = np.array([0] * len(X_neg) + [1] * pos_num[0])
    return shuffle(X, y, random_state=random_state)


def smote(X, y, ratio=1.0):
    s = SMOTE(random_state=95, n_jobs=-1, ratio=ratio, kind='regular')
    return s.fit_sample(X, y)
