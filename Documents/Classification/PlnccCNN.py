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
    # if input_source.startswith('plankFrames\plankvid3'):
    #     frame_type.append(0)
    #     # print("Appending: 0")
    # if input_source.startswith('plankFrames\plankvid4'):
    #     frame_type.append(-1)
    #     # print("Appending: -1")
    # if input_source.startswith('plankFrames\plankvid5'):
    #     frame_type.append(1)
    #     # print("Appending: 1")
    # if input_source.startswith('plankFrames\plankvid6'):
    #     frame_type.append(1)
    #     # print("Appending: 1")
    # if input_source.startswith('plankFrames\plankvid7'):
    #     frame_type.append(-1)
    #     # print("Appending: -1")
    # if input_source.startswith('plankFrames\plankvid8'):
    #     frame_type.append(-1)
    #     # print("Appending: -1")
    # if input_source.startswith('plankFrames\plankvid9'):
    #     frame_type.append(-1)
        # print("Appending: -1")
print("exited loop")
# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
train_files = np.array(train_files, dtype=np.uint8)
print("list to np array complete")
n_samples = len(train_files)
data = train_files.reshape((n_samples, -1))
print("reshaped")

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)
print("created classifier")
# We learn the digits on the first half of the digits
X_train, X_test, y_train, y_test = train_test_split(
    data, frame_type, test_size=0.4, shuffle=False)
print("finished splitting")
# classifier.fit(train_files[:((6 * n_samples) // 10)], frame_type[:((6 * n_samples) // 10)])
classifier.fit(X_train, y_train)
print("fitted")
# Now predict the value of the digit on the second half:
expected = train_files[((6 * n_samples) // 10):]
predicted = classifier.predict(X_test)
print("predicting")
# expected = train_files[((6 * n_samples) // 10):]
# predicted = classifier.predict(data[((6 * n_samples) // 10):])

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

dump(classifier, classifier.joblib)
# y = frame_type

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))

# model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

# model.fit(X_train, y_train, validation_data=(X_test, y_test))