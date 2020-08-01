from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from imutils.object_detection import non_max_suppression
from imutils import paths
import cv2
import time
import argparse
import imutils


def img_skel():
    # Read the network into Memory
    protoFile = "./models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "./models/pose/mpi/pose_iter_160000.caffemodel"
    nPoints = 15
    POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14], [14, 8], [8, 9], [9, 10],
                  [14, 11], [11, 12], [12, 13]]

    frame = cv2.imread("./img/dudePlank.jpg")
    frameCopy = np.copy(frame)
    frameBlank = np.copy(frame)
    frameBlank.fill(255)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    threshold = 0.1

    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    t = time.time()
    # input image dimensions for the network
    inWidth = 368
    inHeight = 368
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                    (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)

    output = net.forward()
    print("time taken by network : {:.3f}".format(time.time() - t))

    H = output.shape[2]
    W = output.shape[3]

    # Empty list to store the detected keypoints
    points = []

    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > threshold:
            cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        lineType=cv2.LINE_AA)


            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else:
            points.append(None)

    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
            cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.line(frameBlank, points[partA], points[partB], (0, 255, 255), 2)
            cv2.circle(frameBlank, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)


    cv2.imshow('Output-Keypoints', frameCopy)
    cv2.imshow('Output-Skel-Blank', frameBlank)
    cv2.imshow('Output-Skeleton', frame)

    cv2.imwrite('Output-Keypoints.jpg', frameCopy)
    #THE ONE
    cv2.imwrite('Output-Skel-Blank.jpg', frameBlank)
    cv2.imwrite('Output-Skeleton.jpg', frame)

    print("Total time taken : {:.3f}".format(time.time() - t))

    cv2.waitKey(0)


def vid_skel():
    protoFile = "./models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "./models/pose/mpi/pose_iter_160000.caffemodel"
    nPoints = 15
    POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14], [14, 8], [8, 9], [9, 10],
                  [14, 11], [11, 12], [12, 13]]

    inWidth = 368 #368
    inHeight = 368
    threshold = 0.1
    frames = []

    input_source = "./img/goodPlank.mov"
    cap = cv2.VideoCapture(input_source)
    hasFrame, frame = cap.read()

    vid_writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                                 (frame.shape[1], frame.shape[0]))

    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    while cv2.waitKey(1) < 0:
        t = time.time()
        hasFrame, frame = cap.read()
        frameCopy = np.copy(frame)
        imgBlank = np.copy(frame)
        imgBlank.fill(255)
        if not hasFrame:
            cv2.waitKey()
            break

        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]

        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                        (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)
        output = net.forward()

        H = output.shape[2]
        W = output.shape[3]
        # Empty list to store the detected keypoints
        points = []

        for i in range(nPoints):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            # Scale the point to fit on the original image
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H

            if prob > threshold:
                cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                            lineType=cv2.LINE_AA)

                # Add the point to the list if the probability is greater than the threshold
                points.append((int(x), int(y)))
            else:
                points.append(None)

        # Draw Skeleton
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]

            if points[partA] and points[partB]:
                cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
                cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.line(imgBlank, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
                cv2.circle(imgBlank, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(imgBlank, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frame, str(partA), points[partA], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, str(partB), points[partB], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, str('HERE'), points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, str('AND HERE'), points[10], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)


        cv2.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8,
                    (255, 50, 0), 2, lineType=cv2.LINE_AA)
        # cv2.putText(frame, "OpenPose using OpenCV", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 50, 0), 2, lineType=cv2.LINE_AA)
        # cv2.imshow('Output-Keypoints', frameCopy)
        # cv2.imshow('Output-Skeleton', frame)
        height, width, channels = frame.shape
        xStart = 0
        xEnd = 0
        if points[10][0] < width/2:
            xStart = int(0 + 3*(points[10][0]/5))
            xEnd = int(width - 7*(width - points[0][0])/10)
        else:
            xStart = int(0 + 3*(points[0][0]/5))
            xEnd = int(width - 7*(width - points[10][0])/10)
        yStart = int(0 + 3 * (points[11][1] / 5))
        yEnd = int(height - 7*(height - points[7][1])/10)
        cropped = frame[yStart:yEnd, xStart:xEnd]
        croppedBlank = imgBlank[yStart:yEnd, xStart:xEnd]
        cv2.imshow("cropped", cropped)
        cv2.imshow("blank cropped", croppedBlank)
        # cv2.imshow("orig", frame)


        # plt.show()

    #     vid_writer.write(frame)
    #
    # vid_writer.release()

img_skel()
#---------------------ADAPTIVE THRESH STUFF----------------------
# gray_image = cv2.imread('./img/dudePlank.jpg', 0)
#
# ret, thresh_global = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
# # here 11 is the pixel neighbourhood that is used to calculate the threshold value
# thresh_mean = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
#
# thresh_gaussian = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
#
# names = ['Original Image', 'Global Thresholding', 'Adaptive Mean Threshold', 'Adaptive Gaussian Thresholding']
# images = [gray_image, thresh_global, thresh_mean, thresh_gaussian]

# for i in range(4):
#     plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
#     plt.title(names[i])
#     plt.xticks([]), plt.yticks([])
#
# plt.show()

#---------------------BITWISE AND WATERSHED SEGMENTATION-------------------------
# # reading the image
# image = cv2.imread('./img/dudePlank.jpg')
# # converting image to grayscale format
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# # apply thresholding
# ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# # get a kernel
# kernel = np.ones((3, 3), np.uint8)
# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
# # extract the background from image
# sure_bg = cv2.dilate(opening, kernel, iterations=3)
#
# dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
# ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
#
# sure_fg = np.uint8(sure_fg)
# unknown = cv2.subtract(sure_bg,sure_bg)
#
# ret, markers = cv2.connectedComponents(sure_fg)
#
# markers = markers+1
#
# markers[unknown == 255] = 0
#
# markers = cv2.watershed(image, markers)
# image[markers == -1] = [255, 0, 0]
#
# plt.imshow(sure_fg)
#
#
# # read the image
# # apply thresholdin
# ret, mask = cv2.threshold(sure_fg,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# # apply AND operation on image and mask generated by thrresholding
# final = cv2.bitwise_and(image, image, mask=mask)
# # plot the result
# plt.imshow(final)
#
#
# plt.show()

# hog = cv2.HOGDescriptor()
# hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
#
# imagePath = './img/plank.jpg'
# image = cv2.imread(imagePath)
# # rows,cols = image.shape[:2]
# # M = cv2.getRotationMatrix2D((cols/2,rows/2),270,1)
# # dst = cv2.warpAffine(image,M,(cols,rows))
# # image=dst
# image = imutils.resize(image, width=min(400, image.shape[1]))
# orig = image.copy()
#
# # detect people in the image
# (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
#                                         padding=(8, 8), scale=1.05)
#
# # draw the original bounding boxes
# for (x, y, w, h) in rects:
#     cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
#
# # apply non-maxima suppression to the bounding boxes using a
# # fairly large overlap threshold to try to maintain overlapping
# # boxes that are still people
# rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
# pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
#
# # draw the final bounding boxes
# for (xA, yA, xB, yB) in pick:
#     cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
#
# # show some information on the number of bounding boxes
# filename = imagePath[imagePath.rfind("/") + 1:]
# print("[INFO] {}: {} original boxes, {} after suppression".format(
#     filename, len(rects), len(pick)))
#
# # show the output images
# cv2.imshow("Before NMS", orig)
# cv2.imshow("After NMS", image)
# cv2.waitKey(0)
# plt.show()

