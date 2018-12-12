import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

# Bar chart
plt.rcdefaults()
fig, ax = plt.subplots()

people = ('DT Original', 'RF Original', 'LR Original', 'SVM Original','',
          'DT over-sampled', 'RF over-sampled', 'LR over-sampled', 'SVM over-sampled','',
          'DT under-sampled', 'RF under-sampled', 'LR under-sampled', 'SVM under-sampled','',
          'DT SMOTE', 'RF SMOTE', 'LR SMOTE', 'SVM SMOTE')

y_pos = np.arange(len(people))
performance = (0.814910711, 0.873970758, 0.83129434, 0.853985758,0,
          0.921202354, 0.943638331, 0.888036014, 0.89858451,0,
          0.846024248, 0.881349506, 0.871285347, 0.883163383,0,
          0.863377479, 0.896279846, 0.873292818, 0.892032413)

ax.barh(y_pos, performance, align='center',
        color='green', ecolor='black')
ax.tick_params(labelsize=8)
ax.set_yticks(y_pos)
ax.set_yticklabels(people)
ax.set_xlim([0.8, 0.95])
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('F1 score', x=0.35)
ax.set_title('Resampling Methods Comparison',x=0.35)
plt.show()
fig.savefig('resampling.png', dpi=150, bbox_inches='tight')

# PCA
pca = PCA(n_components=2)
sca = StandardScaler()
X_2 = pca.fit_transform(sca.fit_transform(X))
plt.figure(figsize=(15, 6))
plt.subplot(121)
plt.title('Scatter View After PCA (with positive samples in front of)')
plt.scatter(X_2[y==0][:, 0], X_2[y==0][:, 1], c='r', alpha=0.5)
plt.scatter(X_2[y==1][:, 0], X_2[y==1][:, 1], c='b', alpha=0.5)
plt.legend(['Negative', 'Positive'])

plt.subplot(122)
plt.title('Scatter View After PCA (with negative samples in front of)')
plt.scatter(X_2[y==1][:, 0], X_2[y==1][:, 1], c='b', alpha=0.5)
plt.scatter(X_2[y==0][:, 0], X_2[y==0][:, 1], c='r', alpha=0.5)
plt.legend(['Positive', 'Negative'])
plt.savefig('PCA.png', dpi=150)