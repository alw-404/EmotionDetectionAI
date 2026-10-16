import pickle
import cv2
from util import get_face_landmarks

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#usato per conoscere versione di OpenCV
#print(cv2.__version__)

emotions = ['ANGRY' ,'HAPPY', 'SAD']

code_path = os.path.join(BASE_DIR, 'code.pkl')
cap = cv2.VideoCapture(0, cv2.CAP_V4L2) #se webcam integrata == 0, se esterna o secondaria == 1 / 2
with open(code_path, 'rb') as f:
    model = pickle.load(f)

#usato per controllo webcam
if not cap.isOpened():
    print("Errore: Webcam non disponibile!")
    exit()
else:
    print("Webcam aperta con successo")

ret, frame = cap.read()

while ret:
    ret, frame = cap.read()
    
    if not ret:
        print("Impossibile ottenere frame dalla webcam...")
        break

    height, width = frame.shape[:2]
    face_landmarks = get_face_landmarks(frame, draw=True, static_image_mode=False, image_dimensions=(width, height))

    if face_landmarks:
        output = model.predict([face_landmarks])

    cv2.putText(frame,
                emotions[int(output[0])],
               (10, frame.shape[0] - 1),
               cv2.FONT_HERSHEY_SIMPLEX,
               3,
               (0, 255, 0),
               5)

    cv2.imshow('frame', frame)

    print(len(face_landmarks))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()