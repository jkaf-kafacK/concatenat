# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import glob
import cv2
import os

from pyspark.sql.catalog import Column

path1 = r'C:\Users\jacques kafack\PycharmProjects\300W\01_Indoor' # use your path
path2 = r'C:\Users\jacques kafack\PycharmProjects\300W\02_Outdoor'
# all_files = glob.glob(path + "/*.png")
#
# li = []
#
# for filename in all_files:
#     df = cv2.imread(path, 0)
#     #df = pd.read(filename, index_col=None, header=0)
#     li.append(df)
#
# frame = pd.concat(li, axis=0, ignore_index=True)
# print(frame.head())
#dirs = os.listdir(path)

# print all the files and directories
import cv2
import dlib

#cap = cv2.VideoCapture(0)
# img = "C:/Users/jacques kafack/PycharmProjects/300W/01_Indoor/indoor_008.png"
hog_face_detector = dlib.get_frontal_face_detector()
dataFile="C:/Users/jacques kafack/PycharmProjects/shape_predictor_68_face_landmarks.dat"
dlib_facelandmark = dlib.shape_predictor(dataFile)
#
# while True:
#     _, frame = cv2.imread(img,0)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     faces = hog_face_detector(gray)
#     for face in faces:
#
#         face_landmarks = dlib_facelandmark(gray, face)
#
#         for n in range(0, 68):
#             x = face_landmarks.part(n).x
#             y = face_landmarks.part(n).y
#             cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)
#
#
#     cv2.imshow("Face_Landmarks", frame)
#
#     key = cv2.waitKey(1)
#     if key == 27:
#         break
# img.release()
# cv2.destroyAllWindows(0)

path_img = "C:/Users/jacques kafack/PycharmProjects/10.jpg"
img = cv2.imread(path_img)
#resize images
img = cv2.resize(img,(512,512))
#displays image
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face = hog_face_detector(img)
for side in face:
    face_landmands_detection = dlib_facelandmark(img,side)
    for number in range(0,68):
        x = face_landmands_detection.part(number).x
        y = face_landmands_detection.part(number).y
        cv2.circle(img,(x,y),1, (2,0,255),1)

cv2.imshow("frame",img)
cv2.waitKey()


