from util import get_face_landmarks
import os
import cv2
import numpy as np

data_dir = r'C:\Users\drj4k\OneDrive\Desktop\AIProjectsAlw\EmotionDetectionAI\dataset\faces'

output = []

for emotion_indx, emotion in enumerate(os.listdir(data_dir)):
    for image_path_ in os.listdir(os.path.join(data_dir, emotion)):
        image_path = os.path.join(data_dir, emotion, image_path_)
        
        image = cv2.imread(image_path)
        
        face_landmarks = get_face_landmarks(image)
        
        if len(face_landmarks) == 1404:
            face_landmarks.append(int(emotion_indx))
            output.append(face_landmarks)
            
output_path = r'C:\Users\drj4k\OneDrive\Desktop\AIProjectsAlw\EmotionDetectionAI\code\data.txt'
np.savetxt(output_path, np.asarray(output))