from os import listdir
from os.path import join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa as lr
import librosa.display as lrd
#from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
import time

from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model

from sklearn.decomposition import PCA


#print('reading training data...')
#data = pd.read_feather('.kaggle/train.feather')
#print('done')

def read(**kwargs):
    return pd.read_csv('.kaggle/train.csv', **kwargs)

import librosa as lr
import librosa.display as lrd

from scipy.signal import firls, convolve, decimate
from scipy.spatial.distance import pdist

filt = firls(2001, bands=[0,240e3,245e3,250e3,255e3,2e6], desired=[0,0,1,1,0,0], fs=4e6)

def resample(xs):
    xs = convolve(xs.astype(float), filt, mode='valid')
    t = 2*np.pi*250e3/4e6*np.arange(len(xs))
    xs = xs*(np.cos(t) + 1j*np.sin(t))
    xs = decimate(xs, 150, ftype='fir')
    return xs

nrows = 629145481
chunksize = 150000

xs = []
ys = []
for df in tqdm(read(chunksize=chunksize), total=nrows//chunksize):
    xs += [resample(df.acoustic_data)]
    ys += [df.time_to_failure.iloc[-1]]

def icdf(x):
    qs = np.linspace(0,1,200)
    return 2*np.quantile(pdist(np.column_stack([x.real,x.imag])), qs)
print('computing distances')
dists = np.array([icdf(xi) for xi in tqdm(xs)])
print('done')
print('computing pca')
X_pre = dists
from sklearn.manifold import spectral_embedding, SpectralEmbedding
X_post =SpectralEmbedding(n_components=2).fit_transform(X_pre)
#X_post =SpectralEmbedding(n_components=2).fit_transform([np.column_stack([x.real,x.imag]) for x in xs])



"""
pca = PCA(n_components=20)
X = pca.fit_transform(X_pre)

print('done')
X_post = X[:,:2]

import matplotlib.pyplot as plt
print('plotting')
xlim = [-50,50]
ylim = [-50,50]
plt.xlim(xlim)
plt.ylim(ylim)
plt.savefig('pca.png')
print('done')
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
exp_var = pca.explained_variance_ratio_
cum = exp_var[0] + exp_var[1]
print(exp_var[0] + exp_var[1])
"""
plt.scatter(X_post[:,0],X_post[:,1],c=ys,s=1)
#plt.title('%var explained: ' + str(round(cum,4)))
plt.savefig('spec.png')
from sklearn.neighbors import KNeighborsRegressor

reg = KNeighborsRegressor(n_neighbors=100, weights='distance')
print(-cross_val_score(reg, X_post, ys, cv=5, scoring='neg_mean_absolute_error'))

