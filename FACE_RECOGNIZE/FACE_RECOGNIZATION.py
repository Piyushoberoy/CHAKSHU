from importlib.resources import path
from pydoc import classname
import cv2
import face_recognition
import os
import numpy as np

path = "K:\PROGRAMS\PROGRAMS_OF_PYTHON_BY_PIYUSH\INOVATION\FACE_RECOGNIZE\IMAGES"

images = []
classname = []
mylist = os.listdir(path)

for cl in mylist:
    curimg = cv2.imread(f'{path}/{cl}')
    images.append(curimg)
    classname.append(os.path.splitext(cl)[0])

if len(mylist)>5:
    print("Lots of images, but don't worry I can memorize it.")

def findEncoding(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

encodelistknown = findEncoding(images)
print("Almost done")

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgs = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    facescurframe = face_recognition.face_locations(imgs)
    encodecurframe = face_recognition.face_encodings(imgs, facescurframe)

    for encodeface, faceloc in zip(encodecurframe, facescurframe):
        matches = face_recognition.compare_faces(encodelistknown,encodeface)
        facedis = face_recognition.face_distance(encodelistknown,encodeface)

        matchindex = np.argmin(facedis)

        if matches[matchindex]:
            name = classname[matchindex].upper()
            print(name)
            y1, x2, y2, x1 = faceloc
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)
        
    cv2.imshow('webcam', img)
    
    if cv2.waitKey(1) == 27:
        break
