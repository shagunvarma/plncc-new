import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import cv2
import warnings
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
from scipy import misc
from PIL import Image

from resizeimage import resizeimage

train_image = []

# for i in tqdm(range(plankvids.shape[0])):
frame_type = []

directory = r'./plankvids'
index = 0
for entry in os.scandir(directory):
    input_source = entry.path
    print(input_source)
    # if (entry.path.endswith(".jpg")
    #         or entry.path.endswith(".png")) and entry.is_file():
    vidcap = cv2.VideoCapture(input_source)
    success,image = vidcap.read()
    print(image)
    count = 0
    success = True
    while success:
        # path = 'D:/OpenCV/Scripts/Images'
        # cv2.imwrite(os.path.join(path, 'plankvid%d_%d.jpg' % (index, count)), image)     # save frame as JPEG file
        # cv2.waitKey(0)
        cv2.imwrite("plankvid%d_%d.jpg" % (index, count), image)
        success,image = vidcap.read()
        print ('Read a new frame: '), success
        count += 1
        print(count)

        # train_image.append(image)
        if index == 0:
            frame_type.append(-1)
        if index == 1:
            frame_type.append(1)
        if index == 2:
            frame_type.append(1)
        if index == 3:
            frame_type.append(1)
        if index == 4:
            frame_type.append(1)
        if index == 5:
            frame_type.append(-1)
        if index == 6:
            frame_type.append(-1)
        if index == 7:
            frame_type.append(-1)
        if index == 8:
            frame_type.append(0)
        if index == 9:
            frame_type.append(0)
        if index == 10:
            frame_type.append(0)
    index += 1

for i in frame_type:
    print(i)
    # cap = cv2.VideoCapture(input_source)
    # hasFrame, frame = cap.read()

    # vid_writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
    #                             (frame.shape[1], frame.shape[0]))

    # while cv2.waitKey(1) < 0:
        # t = time.time()
        # hasFrame, frame = cap.read()
        # frameCopy = np.copy(frame)
        # if not hasFrame:
        #     cv2.waitKey()
        #     break
        # height, width, channels = frame.shape
        # use "frame" to access each individual frame

# print(train_image.length)

# X = np.array(train_image)
# # y=train['label'].values
# # y = to_categorical(y)
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