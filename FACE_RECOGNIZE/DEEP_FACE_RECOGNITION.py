import cv2
import mediapipe as mp

# Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

cap = cv2.VideoCapture(0)
LEFT_EYE = (466, 388)
RIGHT_EYE = [(33, 7), (7, 163), (163, 144), (144, 145),
                                (145, 153), (153, 154), (154, 155), (155, 133),
                                (33, 246), (246, 161), (161, 160), (160, 159),
                                (159, 158), (158, 157), (157, 173), (173, 133)]
draw=True
facial_areas = {
    'Contours': mp_face_mesh.FACEMESH_CONTOURS
    , 'Lips': mp_face_mesh.FACEMESH_LIPS
    , 'Face_oval': mp_face_mesh.FACEMESH_FACE_OVAL
    , 'Left_eye': mp_face_mesh.FACEMESH_LEFT_EYE
    , 'Left_eye_brow': mp_face_mesh.FACEMESH_LEFT_EYEBROW
    , 'Right_eye': mp_face_mesh.FACEMESH_RIGHT_EYE
    , 'Right_eye_brow': mp_face_mesh.FACEMESH_RIGHT_EYEBROW
    , 'Tesselation': mp_face_mesh.FACEMESH_TESSELATION
}
while True:
    # Image
    ret, image = cap.read()
    if ret is not True:
        break
    height, width, _ = image.shape
    # print("Height, width", height, width)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # print(mp_face_mesh.FACEMESH_LEFT_EYE)
    # Facial landmarks
    result = face_mesh.process(rgb_image)


    
    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0]
        for source_idx, target_idx in mp_face_mesh.FACEMESH_LEFT_EYE:
            source = landmarks.landmark[source_idx]
            target = landmarks.landmark[target_idx]
        
            relative_source = (int(image.shape[1] * source.x), int(image.shape[0] * source.y))
            relative_target = (int(image.shape[1] * target.x), int(image.shape[0] * target.y))
            print(relative_source, relative_target)
        
            cv2.line(image, relative_source, relative_target, (255, 255, 255), thickness = 2)

        # for source_idx, target_idx in mp_face_mesh.FACEMESH_RIGHT_EYE:
        #     source = landmarks.landmark[source_idx]
        #     target = landmarks.landmark[target_idx]
        
        #     relative_source = (int(image.shape[1] * source.x), int(image.shape[0] * source.y))
        #     relative_target = (int(image.shape[1] * target.x), int(image.shape[0] * target.y))
        
        #     cv2.line(image, relative_source, relative_target, (255, 255, 255), thickness = 2)

    # if result.multi_face_landmarks:
    #     lmList=[]
    #     for facial_landmarks in result.multi_face_landmarks:
    #         myHand=result.multi_face_landmarks[0]
    #         for id, lm in enumerate(myHand.landmark):
    #             # print(id, lm)
    #             h, w, c = image.shape
    #             cx, cy = int(lm.x * w), int(lm.y * h)
    #             # print(id, cx, cy)
    #             lmList.append([id, cx, cy])
        
    #             if draw:
    #                 cv2.circle(image, (cx, cy), 1, (255, 0, 255), cv2.FILLED)
        # print(lmList)
        
            # for i in range(0, 468):
            #     pt1 = facial_landmarks.landmark[i]
            #     x = int(pt1.x * width)
            #     y = int(pt1.y * height)
            #     # print(i)
            #     # if (x,y) in LEFT_EYE:
            #     #     print("x, y", x, y)
            #     cv2.circle(image, (x, y), 2, (100, 100, 0), -1)
                # cv2.putText(image, str(i), (263, 249), 0, 1, (0, 0, 0))
    cv2.imshow("Image", image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()