import cv2
def detect_faces(image, faceDetModel):
    biggestFace = 0
    if faceDetModel == dlibStr:
        detector = dlib.get_frontal_face_detector()
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)
        for result in faces:
            x = result.left()
            y = result.top()
            x1 = result.right()
            y1 = result.bottom()
            size = y1-y
            if biggestFace < size:
                biggestFace = size

    elif faceDetModel == haarStr:
        face_cascade = cv2.CascadeClassifier(haarXml)
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            if biggestFace < h:
                biggestFace = h

    elif faceDetModel == mtcnnStr:
        detector = MTCNN()
        img = cv2.imread(image)
        faces = detector.detect_faces(img)

        for result in faces:
            x, y, w, h = result['box']
            size = h
            if biggestFace < h:
                biggestFace = h

    elif faceDetModel == dnnStr:
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        img = cv2.imread(image)
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
        net.setInput(blob)
        faces = net.forward()
        for i in range(faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence > 0.5:
                box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                height = y1 - y
                if biggestFace < height:
                    biggestFace = height

    else:
        print("No face detection model in use")
    return biggestFace