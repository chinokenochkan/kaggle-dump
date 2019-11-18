import numpy as np
import pandas as pd
# import pydicom
# import os
# import matplotlib.pyplot as plt
# import collections
# from tqdm import tqdm_notebook as tqdm
# import tqdm
# from datetime import datetime
from utils import read_testset, read_trainset
from model import MyDeepModel
from keras_applications.inception_resnet_v2 import InceptionResNetV2
# from keras_applications.nasnet import NASNetLarge
# from keras_applications.resnext import ResNeXt101
from keras_applications.resnet import ResNet50
from sklearn.model_selection import ShuffleSplit, GroupKFold
from keras_applications.mobilenet_v2 import MobileNetV2



test_df = read_testset()
df = read_trainset()

#oversampling
# epidural_df = df[df.Label['epidural'] == 1]
# df = pd.concat([df, epidural_df])

splits = 5
k_fold = 2
# train set (90%) and validation set (10%)
ss = ShuffleSplit(n_splits=k_fold, test_size=0.1).split(df.index)
# ss = GroupKFold(n_splits=splits).split(df.index, groups=df.index)
#
# lets go for the first fold only
# train_idx, valid_idx = next(ss)
# valid_idx = valid_idx[0:600]

# obtain model
# model = MyDeepModel(engine=InceptionResNetV2, input_dims=(256, 256, 3), batch_size=55, learning_rate=1e-3,
#                     num_epochs=7, decay_rate=0.65, decay_steps=1, weights="imagenet", verbose=1)

# results = []
# for k in range(k_fold):
#     print('Fold '+ str(k+1))
#     train_idx, valid_idx = next(ss)
#     valid_idx = valid_idx[0:600]
#     history = model.fit_and_predict(df.iloc[train_idx], df.iloc[valid_idx], test_df)
#     results.append(history.test_predictions)

results = []
for k in range(k_fold):
    print('Fold '+ str(k+1))
    train_idx, valid_idx = next(ss)
    valid_idx = valid_idx[0:600]
    model = MyDeepModel(engine='resnet_34', input_dims=(256, 256, 3), batch_size=80, learning_rate=5e-4,
                num_epochs=7, decay_rate=0.65, decay_steps=1, weights="imagenet", verbose=1)
    history = model.fit_and_predict(df.iloc[train_idx], df.iloc[valid_idx], test_df)
    preds = np.average(history.test_predictions, axis=0, weights=[0, 1, 2, 4, 6, 8, 12])
    results.append(preds)

# history = model.fit_and_predict(df.iloc[train_idx], df.iloc[valid_idx], test_df)
# test_df.iloc[:, :] = np.average(history.test_predictions, axis=0, weights=[0, 0, 0, 1, 2, 4, 6])
test_df.iloc[:, :] = np.average(results, axis=0, weights=[1]*k_fold)
# test_df.iloc[:,:] = history.test_predictions[-1]

test_df = test_df.stack().reset_index()

test_df.insert(loc=0, column='ID', value=test_df['Image'].astype(str) + "_" + test_df['Diagnosis'])

test_df = test_df.drop(["Image", "Diagnosis"], axis=1)

test_df.to_csv('submission_incep_2folds.csv', index=False)