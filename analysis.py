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
print('generating features...')
rows = 150_000
segments = int(np.floor(data.shape[0] / rows))

#X_train = pd.DataFrame(index=range(segments), dtype=np.float64,
#                       columns=['ave', 'std', 'max', 'min'])
#y_train = pd.DataFrame(index=range(segments), dtype=np.float64,
#                       columns=['time_to_failure'])

X_train = pd.DataFrame(index=range(segments), dtype=np.float64)
y_train = pd.DataFrame(index=range(segments),dtype=np.float64)

for idx in tqdm(range(segments)):
    seg = data.iloc[idx*rows:idx*rows+rows]
    #x = seg['acoustic_data']
    y = seg['time_to_failure'].values[-1]
    
    y_train.loc[idx, 'time_to_failure'] = y
"""    
#    X_train.loc[segment, 'ave'] = x.mean()
#    X_train.loc[segment, 'std'] = x.std()
#    X_train.loc[segment, 'max'] = x.max()
#    X_train.loc[segment, 'min'] = x.min()
    X_train.loc[idx, "X_train"] = x.iloc[-1].item()
    X_train.loc[idx, "mean"] = x.mean().item()
    X_train.loc[idx, "std"] = x.std().item()
    X_train.loc[idx, "max"] = x.max().item()
    X_train.loc[idx, "min"] = x.min().item()
    X_train.loc[idx, "mad"] = x.mad()
    X_train.loc[idx, "kurt"] = x.kurtosis().item()
    X_train.loc[idx, "skew"] = x.skew().item()
    X_train.loc[idx, "median"] = x.median().item()
    X_train.loc[idx, "q01"] = np.quantile(x, 0.01)
    X_train.loc[idx, "q05"] = np.quantile(x, 0.05)
    X_train.loc[idx, "q95"] = np.quantile(x, 0.95)
    X_train.loc[idx, "q99"] = np.quantile(x, 0.99)
    X_train.loc[idx, "iqr"] = np.subtract(*np.percentile(x, [75, 25]))
    X_train.loc[idx, "abs_mean"] = x.abs().mean().item()
    X_train.loc[idx, "abs_std"] = x.abs().std().item()
    X_train.loc[idx, "abs_max"] = x.abs().max().item()
    X_train.loc[idx, "abs_min"] = x.abs().min().item()
    X_train.loc[idx, "abs_mad"] = x.abs().mad()
    X_train.loc[idx, "abs_kurt"] = x.abs().kurtosis().item()
    X_train.loc[idx, "abs_skew"] = x.abs().skew().item()
    X_train.loc[idx, "abs_median"] = x.abs().median().item()
    X_train.loc[idx, "abs_q01"] = np.quantile(x.abs(), 0.01)
    X_train.loc[idx, "abs_q05"] = np.quantile(x.abs(), 0.05)
    X_train.loc[idx, "abs_q95"] = np.quantile(x.abs(), 0.95)
    X_train.loc[idx, "abs_q99"] = np.quantile(x.abs(), 0.99)
    X_train.loc[idx, "abs_iqr"] = np.subtract(*np.percentile(x.abs(), [75, 25]))

    for window in [10, 100, 1000]:
        X_train_roll_mean = x.rolling(window).mean().dropna()
        X_train.loc[idx, f"mean_mean_{window}"] = X_train_roll_mean.mean()
        X_train.loc[idx, f"std_mean_{window}"] = X_train_roll_mean.std()
        X_train.loc[idx, f"max_mean_{window}"] = X_train_roll_mean.max()
        X_train.loc[idx, f"min_mean_{window}"] = X_train_roll_mean.min()
        X_train.loc[idx, f"mad_mean_{window}"] = X_train_roll_mean.mad()
        X_train.loc[idx, f"kurt_mean_{window}"] = X_train_roll_mean.kurtosis()
        X_train.loc[idx, f"skew_mean_{window}"] = X_train_roll_mean.skew()
        X_train.loc[idx, f"median_mean_{window}"] = X_train_roll_mean.median()
        X_train.loc[idx, f"q01_mean_{window}"] = np.quantile(X_train_roll_mean, 0.01)
        X_train.loc[idx, f"q05_mean_{window}"] = np.quantile(X_train_roll_mean, 0.05)
        X_train.loc[idx, f"q95_mean_{window}"] = np.quantile(X_train_roll_mean, 0.95)
        X_train.loc[idx, f"q99_mean_{window}"] = np.quantile(X_train_roll_mean, 0.99)
        X_train.loc[idx, f"iqr_mean_{window}"] = np.subtract(
            *np.percentile(X_train_roll_mean, [75, 25])
        )
        X_train.loc[idx, f"abs_mean_mean_{window}"] = X_train_roll_mean.abs().mean()
        X_train.loc[idx, f"abs_std_mean_{window}"] = X_train_roll_mean.abs().std()
        X_train.loc[idx, f"abs_max_mean_{window}"] = X_train_roll_mean.abs().max()
        X_train.loc[idx, f"abs_min_mean_{window}"] = X_train_roll_mean.abs().min()
        X_train.loc[idx, f"abs_mad_mean_{window}"] = X_train_roll_mean.abs().mad()
        X_train.loc[idx, f"abs_kurt_mean_{window}"] = (
            X_train_roll_mean.abs().kurtosis()
        )
        X_train.loc[idx, f"abs_skew_mean_{window}"] = X_train_roll_mean.abs().skew()
        X_train.loc[idx, f"abs_median_mean_{window}"] = (
            X_train_roll_mean.abs().median()
        )
        X_train.loc[idx, f"abs_q01_mean_{window}"] = np.quantile(
            X_train_roll_mean.abs(), 0.01
        )
        X_train.loc[idx, f"abs_q05_mean_{window}"] = np.quantile(
            X_train_roll_mean.abs(), 0.05
        )
        X_train.loc[idx, f"abs_q95_mean_{window}"] = np.quantile(
            X_train_roll_mean.abs(), 0.95
        )
        X_train.loc[idx, f"abs_q99_mean_{window}"] = np.quantile(
            X_train_roll_mean.abs(), 0.99
        )
        X_train.loc[idx, f"abs_iqr_mean_{window}"] = np.subtract(
            *np.percentile(X_train_roll_mean.abs(), [75, 25])
        )

        X_train_roll_std = x.rolling(window).std().dropna()
        X_train.loc[idx, f"mean_std_{window}"] = X_train_roll_std.mean()
        X_train.loc[idx, f"std_std_{window}"] = X_train_roll_std.std()
        X_train.loc[idx, f"max_std_{window}"] = X_train_roll_std.max()
        X_train.loc[idx, f"min_std_{window}"] = X_train_roll_std.min()
        X_train.loc[idx, f"mad_std_{window}"] = X_train_roll_std.mad()
        X_train.loc[idx, f"kurt_std_{window}"] = X_train_roll_std.kurtosis()
        X_train.loc[idx, f"skew_std_{window}"] = X_train_roll_std.skew()
        X_train.loc[idx, f"median_std_{window}"] = X_train_roll_std.median()
        X_train.loc[idx, f"q01_std_{window}"] = np.quantile(X_train_roll_mean, 0.01)
        X_train.loc[idx, f"q05_std_{window}"] = np.quantile(X_train_roll_mean, 0.05)
        X_train.loc[idx, f"q95_std_{window}"] = np.quantile(X_train_roll_mean, 0.95)
        X_train.loc[idx, f"q99_std_{window}"] = np.quantile(X_train_roll_mean, 0.99)
        X_train.loc[idx, f"iqr_std_{window}"] = np.subtract(
            *np.percentile(X_train_roll_std, [75, 25])
        )
        X_train.loc[idx, f"abs_mean_std_{window}"] = X_train_roll_std.abs().mean()
        X_train.loc[idx, f"abs_std_std_{window}"] = X_train_roll_std.abs().std()
        X_train.loc[idx, f"abs_max_std_{window}"] = X_train_roll_std.abs().max()
        X_train.loc[idx, f"abs_min_std_{window}"] = X_train_roll_std.abs().min()
        X_train.loc[idx, f"abs_mad_std_{window}"] = X_train_roll_std.abs().mad()
        X_train.loc[idx, f"abs_kurt_std_{window}"] = X_train_roll_std.abs().kurtosis()
        X_train.loc[idx, f"abs_skew_std_{window}"] = X_train_roll_std.abs().skew()
        X_train.loc[idx, f"abs_median_std_{window}"] = X_train_roll_std.abs().median()
        X_train.loc[idx, f"abs_q01_std_{window}"] = np.quantile(X_train_roll_std.abs(), 0.01)
        X_train.loc[idx, f"abs_q05_std_{window}"] = np.quantile(X_train_roll_std.abs(), 0.05)
        X_train.loc[idx, f"abs_q95_std_{window}"] = np.quantile(X_train_roll_std.abs(), 0.95)
        X_train.loc[idx, f"abs_q99_std_{window}"] = np.quantile(X_train_roll_std.abs(), 0.99)
        X_train.loc[idx, f"iqr_std_{window}"] = np.subtract(
            *np.percentile(X_train_roll_std, [75, 25])
        )

print(X_train.head())
X_train = X_train.fillna(method='ffill')
X_train.to_feather('scaled_features.feather')
"""
X_train = pd.read_feather('features.feather')

X_train = X_train.values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

print('done')
print(X_train_scaled.shape)

print('training model...')
#model = NuSVR(gamma='scale')
#model =ensemble.GradientBoostingRegressor()
#model = linear_model.Lasso()
#model = linear_model.LinearRegression()
model.fit(X_train_scaled, y_train.values.flatten())
t1 = time.time_ns()
#score = cross_val_score(model, X_train_scaled, y_train.values.flatten(), cv=5, scoring='neg_mean_absolute_error')
#y_pred = model.predict(X_train_scaled)
t2 = time.time_ns()
score = mean_absolute_error(y_train.values.flatten(), y_pred)
print('score', np.mean(score), np.var(score))
print('time', (t2-t1) / (10 ** 9))
#nonzer = model.coef_.shape[0] - np.isclose(model.coef_,np.zeros(model.coef_.shape[0])).sum()
#print('sparsity',nonzer, nonzer/model.coef_.shape[0], model.coef_.shape[0])

