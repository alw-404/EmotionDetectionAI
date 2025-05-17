import pickle
import cv2
from util import get_face_landmarks

#usato per conoscere versione di OpenCV
#print(cv2.__version__)

emotions = ['ANGRY' ,'HAPPY', 'SAD']

code_path = r"C:\Users\drj4k\OneDrive\Desktop\AIProjectsAlw\EmotionDetectionAI\code\code.pkl"
with open(code_path, 'rb') as f:
    model = pickle.load(f)
    
cap = cv2.VideoCapture(0) #se webcam integrata == 0, se esterna o secondaria == 1 / 2

#usato per controllo webcam
#if not cap.isOpened():
#    print("Errore: Webcam non disponibile!")
#    exit()
#else:
#    print("Webcam aperta con successo")

#if not cap.isOpened():
#    print("Errore: Webcam non disponibile!")
#    exit()

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

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()