import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data_file = os.path.join(BASE_DIR, 'data.txt')
code_path = os.path.join(BASE_DIR, 'code.pkl')
data = np.loadtxt(data_file)

X = data[:, :-1]
y = data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

code_path = r"C:\Users\drj4k\OneDrive\Desktop\AIProjectsAlw\EmotionDetectionAI\code\code.pkl"
with open(code_path, 'wb') as f:
    pickle.dump(rf_classifier, f)