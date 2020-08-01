from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
import h5py

images_per_class = 80
fixed_size       = tuple((500, 500))
train_path       = "dataset/train"
h5_data          = 'output/data.h5'
h5_labels        = 'output/labels.h5'
bins             = 8

def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

train_labels = os.listdir(train_path)

train_labels.sort()

global_features = []
labels = []

for training_name in train_labels:
    dir = os.path.join(train_path, training_name)
    current_label = training_name
    for x in range(1, images_per_class+1):
        file = dir + "/" + str(x) + ".jpg"
        image = cv2.imread(file)
        image = cv2.resize(image, fixed_size)
        fv_hu_moments = fd_hu_moments(image)
        global_feature = np.hstack(fv_hu_moments)
        labels.append(current_label)
        global_features.append(global_feature)

targetNames = np.unique(labels)
le = LabelEncoder()
target = le.fit_transform(labels)

scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)

h5f_data = h5py.File(h5_data, 'w')
h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

h5f_label = h5py.File(h5_labels, 'w')
h5f_label.create_dataset('dataset_1', data=np.array(target))

h5f_data.close()
h5f_label.close()
