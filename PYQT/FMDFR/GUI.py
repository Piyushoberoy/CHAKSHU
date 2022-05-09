import sys
from PyQt5.uic import loadUi
from pathlib import Path
from PyQt5.QtGui import QIcon, QImage, QPixmap, QCloseEvent
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QDialog, QApplication, QStackedWidget, QWidget
from PySide2.QtCore import QFile
# from PySide2.QtUiTools import QUiLoader
#Face Recognition
from importlib.resources import path
from pydoc import classname
from pyexpat import model
import cv2
import face_recognition
import os
import numpy as np

#Face Mask Detection
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils
import time

path = "K:\PROJECT\FACE_RECOGNIZE\IMAGES"

images = []
classname = []
mylist = os.listdir(path)

for cl in mylist:
    curimg = cv2.imread(f'{path}/{cl}')
    images.append(curimg)
    classname.append(os.path.splitext(cl)[0])

def findEncoding(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        # facescurframe = face_recognition.face_locations(img)
        # fl = face_recognition.face_landmarks(img, facescurframe)
        encodelist.append(encode)
    return encodelist

encodelistknown = findEncoding(images)

prototxtPath = "K:/PROJECT/FaceMaskDetection-main/face_detector/deploy.prototxt"
weightsPath = "K:/PROJECT/FaceMaskDetection-main/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

print("Lets go")
class MainScreen(QDialog):
    def __init__(self):
        super(MainScreen, self).__init__()
        
        path = os.fspath(Path(__file__).resolve().parent / "form.ui")
        # ui_file = QFile(path)
        loadUi(path,self)
        self.FR.clicked.connect(self.F_R)
        self.FMD.clicked.connect(self.F_M_D)
        # self.show()
        # print(app.aboutToQuit.connect(self.closeEvent))
        # self.back_from_FR.clicked.connect(self.__init__)
        # self.back_from_FMD.clicked.connect(self.__init__)
    
    def F_R(self):
        fr=Face_Recognization()
        widget.addWidget(fr)
        widget.setCurrentIndex(widget.currentIndex()+1)
        # fr.back_from_FR.clicked.connect(self.__init__)
    
    def F_M_D(self):
        fmd=Face_Mask_Detection()
        widget.addWidget(fmd)
        widget.setCurrentIndex(widget.currentIndex()+1)
        # fmd.back_from_FMD.clicked.connect(self.__init__)

class Face_Recognization(QDialog):
    def __init__(self):
        super(Face_Recognization,self).__init__()
        path = os.fspath(Path(__file__).resolve().parent / "FR.ui")
        loadUi(path,self)
        self.back_from_FR.clicked.connect(self.Main)
        self.Capture.clicked.connect(self.cap_img)
        self.cap = cv2.VideoCapture(0)
        app.aboutToQuit.connect(self.close_all)

    def close_all(self):
        print("done")
        self.cap.release()
        cv2.destroyAllWindows()

    def Main(self):
        self.cap.release()
        M=MainScreen()
        widget.addWidget(M)
        widget.setCurrentIndex(widget.currentIndex()+1)
    def cap_img(self):
        # cap = cv2.VideoCapture(0)
        default=""
        while (self.cap.isOpened()):
            success, img = self.cap.read()
            if success == True:
                self.display(img,1)
                cv2.waitKey(1)
                imgs = cv2.resize(img, (0, 0), None, 0.25, 0.25)
                imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
                # self.NAME.setText(default)
                facescurframe = face_recognition.face_locations(imgs)
                fl = face_recognition.face_landmarks(imgs, facescurframe, model="medium")
                # print(fl)
                encodecurframe = face_recognition.face_encodings(imgs, facescurframe)
                app.aboutToQuit.connect(self.close_all)
                for encodeface, faceloc in zip(encodecurframe, facescurframe):
                    matches = face_recognition.compare_faces(encodelistknown,encodeface)
                    facedis = face_recognition.face_distance(encodelistknown,encodeface)

                    matchindex = np.argmin(facedis)

                    if matches[matchindex]:
                        name = classname[matchindex].upper()
                        self.NAME.setText(name)
                        # cv2.waitKey()
                # app.aboutToQuit.connect(self.close_all)
                # print(app.aboutToQuit.connect(self.close_all))
    
    def display(self, img, window=1):
        qformat=QImage.Format_Indexed8

        if len(img.shape)==3:
            if (img.shape[2])==4:
                qformat=QImage.Format_RGBA888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(img, img.shape[1], img.shape[0], qformat)
        img = img.rgbSwapped()
        self.rec_img.setPixmap(QPixmap.fromImage(img))
        self.rec_img.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)


class Face_Mask_Detection(QDialog):
    def __init__(self):
        super(Face_Mask_Detection,self).__init__()
        path = os.fspath(Path(__file__).resolve().parent / "FMD.ui")
        loadUi(path,self)
        self.back_from_FMD.clicked.connect(self.Main)
        self.Capture.clicked.connect(self.cap_img)
        self.cap = cv2.VideoCapture(0)
        app.aboutToQuit.connect(self.close_all)

    def close_all(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def Main(self):
        self.cap.release()
        M=MainScreen()
        widget.addWidget(M)
        widget.setCurrentIndex(widget.currentIndex()+1)
    
    def detect_and_predict_mask(self, frame, faceNet, maskNet):
        # grab the dimensions of the frame and then construct a blob
        # from it
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
            (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the face detections
        faceNet.setInput(blob)
        detections = faceNet.forward()
        # print(detections.shape)

        # initialize our list of faces, their corresponding locations,
        # and the list of predictions from our face mask network
        faces = []
        locs = []
        preds = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the detection
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for
                # the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # ensure the bounding boxes fall within the dimensions of
                # the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                # add the face and bounding boxes to their respective
                # lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))
        # only make a predictions if at least one face was detected
        if len(faces) > 0:
            # for faster inference we'll make batch predictions on *all*
            # faces at the same time rather than one-by-one predictions
            # in the above `for` loop
            faces = np.array(faces, dtype="float32")
            preds = maskNet.predict(faces, batch_size=32)

        # return a 2-tuple of the face locations and their corresponding
        # locations
        return (locs, preds)
    
    def cap_img(self):
        while (self.cap.isOpened()):
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
            success,frame = self.cap.read()
            # frame = imutils.resize(frame, width=400)
            if success==True:
                self.display(frame,1)
                cv2.waitKey()
                # detect faces in the frame and determine if they are wearing a
                # face mask or not
                imgs = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
                imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
                (locs, preds) = self.detect_and_predict_mask(imgs, faceNet, maskNet)

                # loop over the detected face locations and their corresponding
                # locations
                app.aboutToQuit.connect(self.close_all)
                for (box, pred) in zip(locs, preds):
                    # unpack the bounding box and predictions
                    # (startX, startY, endX, endY) = box
                    (mask, withoutMask) = pred

                    # determine the class label and color we'll use to draw
                    # the bounding box and text
                    label = "Mask" if mask > withoutMask else "No Mask"
                    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                    # include the probability in the label
                    label = "{}".format(label)
                    print(label)
                    # display the label and bounding box rectangle on the output
                    # frame
                    # cv2.putText(frame, label, (startX, startY - 10),
                    #     cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    # cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                    self.REMARK.setText(label)
                    

            # show the output frame
            # cv2.imshow("Frame", frame)
            # key = cv2.waitKey(1) & 0xFF
    
    def display(self, img, window=1):
        qformat=QImage.Format_Indexed8

        if len(img.shape)==3:
            if (img.shape[2])==4:
                qformat=QImage.Format_RGBA888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(img, img.shape[1], img.shape[0], qformat)
        img = img.rgbSwapped()
        self.rec_img.setPixmap(QPixmap.fromImage(img))
        self.rec_img.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    

app = QApplication(sys.argv)
welcome=MainScreen()
widget=QStackedWidget()
widget.setWindowTitle("CHAKSHU")
widget.setWindowIcon(QIcon("K:\PROJECT\PYQT\FMDFR\LOGO.png"))
widget.addWidget(welcome)
widget.setFixedHeight(600)
widget.setFixedWidth(1200)
widget.show()

try:
    sys.exit(app.exec_())
except:
    # q.cap.release()
    # cv2.destroAllWindows()
    print("Exiting")