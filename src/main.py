import cv2
import numpy as np
import pathlib

videoCapture = cv2.VideoCapture(0)

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyesCascade = cv2.CascadeClassifier('haarcascade_eye.xml')

faces = []
labels = []
subjects = [
    'com mascara', 
    'com mascara', 
    'com mascara', 
    'com mascara',
    'com mascara',
    'com mascara',
    'com mascara',
    'com mascara',
    'com mascara',
    'sem mascara',
    'sem mascara',
    'sem mascara',
    'sem mascara',
    'sem mascara',
    'sem mascara',
    'sem mascara',
    'sem mascara',
    'sem mascara',
    'sem mascara',
    'sem mascara',
    'sem mascara',
    'sem mascara',
    'sem mascara'
]

items = 0

for path in pathlib.Path("Images").iterdir():
    if path.is_file():
        items += 1

print(items)
for x in range(1, items):
    faces.append(cv2.cvtColor(cv2.imread(f'./Images/{x}.jpeg'), cv2.COLOR_BGR2GRAY))
    labels.append(x-1)

faceRecognizer = cv2.face.LBPHFaceRecognizer_create()

faceRecognizer.train(faces, np.array(labels))

if videoCapture.isOpened():
    rval, frame = videoCapture.read()
else:
    rval = False

while rval:
    rval, frame = videoCapture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    facesz = faceCascade.detectMultiScale(gray, 1.3, 5)
    faces2 = eyesCascade.detectMultiScale(gray, 1.3, 1)

    for (x, y, w, h) in facesz:
        for (x2, y2, w2, h2) in faces2:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
            cv2.rectangle(frame, (x2, y2), (x2+w2, y2+h2), (255, 0, 255), 2)

            r_gr = gray[y:y+h, x:x+w]
            r_gr2 = gray[y2:y2+h2, x2:x2+w2]

            lb, indice = faceRecognizer.predict(r_gr)
            lb2, indice2 = faceRecognizer.predict(r_gr2)

            if indice < 80 and indice2 < 120:
                label_text = subjects[lb]
            elif indice > 80 and indice2 < 120:
                label_text = subjects[lb2]
            else:
                label_text = 'Desconhecido'

            cv2.putText(frame, label_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(20)

    if key == 27:
        break

videoCapture.release()
cv2.destroyWindow("Camera")