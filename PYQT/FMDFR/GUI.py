import sys
from PyQt5.uic import loadUi
from pathlib import Path
from PyQt5.QtGui import QIcon, QImage, QPixmap, QCloseEvent
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QDialog, QApplication, QStackedWidget, QWidget
from PySide2.QtCore import QFile

#Face Recognition
from importlib.resources import path
from pydoc import classname
import cv2
import face_recognition
import os
import numpy as np

#Face Mask Detection
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

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
maskNet = load_model("K:\PROJECT\PYQT\FMDFR\mask_detector.model")

print("Lets go")
class MainScreen(QDialog):
    def __init__(self):
        super(MainScreen, self).__init__()
        
        path = os.fspath(Path(__file__).resolve().parent / "form.ui")
        loadUi(path,self)
        self.FR.clicked.connect(self.F_R)
        self.FMD.clicked.connect(self.F_M_D)
    
    def F_R(self):
        fr=Face_Recognization()
        widget.addWidget(fr)
        widget.setCurrentIndex(widget.currentIndex()+1)
    
    def F_M_D(self):
        fmd=Face_Mask_Detection()
        widget.addWidget(fmd)
        widget.setCurrentIndex(widget.currentIndex()+1)

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
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
            (104.0, 177.0, 123.0))

        faceNet.setInput(blob)
        detections = faceNet.forward()

        faces = []
        locs = []
        preds = []

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                faces.append(face)
                locs.append((startX, startY, endX, endY))
        if len(faces) > 0:
            faces = np.array(faces, dtype="float32")
            preds = maskNet.predict(faces, batch_size=32)

        return (locs, preds)
    
    def cap_img(self):
        while (self.cap.isOpened()):
            success,frame = self.cap.read()
            if success==True:
                self.display(frame,1)
                cv2.waitKey()

                imgs = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
                imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
                (locs, preds) = self.detect_and_predict_mask(imgs, faceNet, maskNet)

                app.aboutToQuit.connect(self.close_all)
                for (box, pred) in zip(locs, preds):

                    (mask, withoutMask) = pred

                    label = "Mask" if mask > withoutMask else "No Mask"
                    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                    label = "{}".format(label)
                    print(label)

                    self.REMARK.setText(label)
    
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

    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    welcome=MainScreen()
    widget=QStackedWidget()
    widget.setWindowTitle("CHAKSHU")
    widget.setWindowIcon(QIcon("K:\PROJECT\PYQT\FMDFR\LOGO.png"))
    widget.addWidget(welcome)
    widget.setFixedHeight(600)
    widget.setFixedWidth(1200)
    widget.show()
    app.exec()
    # try:
    #     sys.exit(app.exec_())
    # except:
    #     # q.cap.release()
    #     # cv2.destroAllWindows()
    #     print("Exiting")