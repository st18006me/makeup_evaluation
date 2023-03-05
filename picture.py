import cv2

face_cascade = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #顔の学習データ
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)

    cv2.imshow('image', img)
    key = cv2.waitKey(1)
    if key == 32:   #スペース
        cv2.imwrite('makeup/makeup.jpg', img)
    if key == 27:   #escキー
        break
 
cap.release()
cv2.destroyAllWindows()
