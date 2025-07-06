import cv2 as cv

capture = cv.VideoCapture(0)
haar_cascade = cv.CascadeClassifier('haar_face.xml') 
clear = 0

people = ['Ben Afleck', 'natasha romanoff', 'nicolas cage']

faces_recognizer = cv.face.LBPHFaceRecognizer_create()
faces_recognizer.read('face_trained.yml')

while True:
    isTrue, frame = capture.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)

    conf_val = 0

    for (x, y, w, h) in faces_rect:
        face_roi = gray[y:y+h, x:x+w]  
        label, confidence = faces_recognizer.predict(face_roi)
        print(f'{people[label]} with a confidence of {confidence}')

        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if confidence < 97:
            text = "Processing..."
        elif confidence >= 97: 
            clear = 1
            text = f"{people[label]} has been verified"

        text_position = (x, y + h + 25)
        cv.putText(frame, text, text_position, cv.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
        conf_val = confidence

    cv.imshow('Detected Faces', frame)
    if conf_val > 97:
        break
    if cv.waitKey(20) & 0xff==ord('d'): 
        break

cv.imshow('Detected Faces', frame)

capture.release()
cv.waitKey(0)




