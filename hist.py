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

print('reading training data...')
data = pd.read_feather('.kaggle/train.feather')
print('done')

# Create a training file with simple derived features
rows = 150_000
segments = int(np.floor(data.shape[0] / rows))
y_train = pd.DataFrame(index=range(segments),dtype=np.float64)
for idx in tqdm(range(segments)):
    seg = data.iloc[idx*rows:idx*rows+rows]
    y = seg['time_to_failure'].values[-1]
    
    y_train.loc[idx, 'time_to_failure'] = y

plt.hist(y_train.values.flatten(), normed=False, bins=50)
plt.xlabel('time-to-failure (s)')
plt.title('Histogram of time-to-failure')
plt.savefig('hist.png')
