import cv2
import datetime

capture = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

while True:
    _, frame = capture.read()
    original_frame = frame.copy()
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    
    for x, y, w, h in face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        face_roi = frame[y:y+h, x:x+w]
        gray_roi = gray_frame[y:y+h, x:x+w]
        smile = smile_cascade.detectMultiScale(gray_roi, 1.7, 5)
        
        for x1, y1, w1, h1 in smile:
            cv2.rectangle(face_roi, (x1, y1), (x1+w1, y1+h1), (0, 0, 255), 2)
            time_stamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            file_name = f'selfie-{time_stamp}.png'
            cv2.imwrite(file_name, original_frame)
            
    cv2.imshow('Photo', frame)
    if cv2.waitKey(10) == ord('q'):
        break
  
capture.release()
cv2.destroyAllWindows()
    
