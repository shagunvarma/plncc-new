#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from joblib import dump
os.environ['KERAS_BACKEND'] = 'theano'
import cv2
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
from scipy import misc


# In[36]:


filelist = glob.glob("plankFrames/*.jpg")

# x = np.array([np.array(Image.open(fname)) for fname in filelist])

directoryFrames = r'./plankFrames'
directoryVids = r'./plankvids'
index = 0
frame_type = []
train_files = []
print("starting to read files")
for entry in filelist:
    input_source = entry
    # print(input_source)
    image = cv2.imread (input_source)
    # train_files.append (image)
    if input_source.startswith('plankFrames\plankvid0'):
        frame_type.append(1)
        train_files.append (image)
        # print("Appending: 1")
    if input_source.startswith('plankFrames\plankvid1'):
        frame_type.append(-1)
        train_files.append (image)
        # print("Appending: -1")
    if input_source.startswith('plankFrames\plankvid2'):
        frame_type.append(0)
        # print("Appending: 0")
        train_files.append (image)


# In[25]:


print("length of fourth dimension")
print(len(train_files))
print("length of third dimension")
for a in train_files:
    print(len(a))
    for b in a:
        print("length of second dimension")
        print(len(b))


# In[27]:


print(train_files)


# In[37]:


train_files = np.array(train_files, dtype=np.uint8)
print("list to np array complete")
print(train_files.shape)


# In[39]:


x = train_files.reshape(train_files.shape[1]*train_files.shape[2]*train_files.shape[3],train_files.shape[0]).T
print(x.shape)


# In[42]:


frame_type = np.array(frame_type)


# In[44]:


y = frame_type.reshape(frame_type.shape[0],)
print(y.shape)


# In[15]:


# data = train_files.reshape(921600,4140)
# train_files.resize(3,1380,720,1280)
# print(train_files.shape)
data = []
for x in train_files:
    for y in x:
        data.append(y)
data = np.array(data, dtype=np.uint8)
print(data.shape)


# In[8]:


n_samples = len(data)
data = data.reshape((n_samples, -1))
print(data.shape)
print("reshaped")


# In[48]:


# from sklearn.ensemble import RandomForestClassifier
classifier = svm.SVC(gamma=0.001)
print("created classifier")
# clf = RandomForestClassifier()


# In[53]:


X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.4, shuffle=False)
print("finished splitting")


# In[54]:


classifier.fit(X_train, y_train)
print("fitted")


# In[ ]:


from joblib import dump
dump(classifier, classifier.joblib)

