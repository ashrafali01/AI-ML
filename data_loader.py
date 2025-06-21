"""import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(data_dir, img_size=(128, 128)):
    X = []
    y = []
    class_names = sorted(os.listdir(data_dir))
    class_map = {name: idx for idx, name in enumerate(class_names)}

    for label in class_names:
        folder = os.path.join(data_dir, label)
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            X.append(img)
            y.append(class_map[label])
    
    X = np.array(X) / 255.0
    y = np.array(y)

    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42), class_names
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def detect_and_crop_face(img, target_size=(128, 128)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        return None  # No face found

    (x, y, w, h) = faces[0]  # Take first detected face
    face_img = img[y:y + h, x:x + w]
    face_img = cv2.resize(face_img, target_size)
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    return face_img / 255.0

def load_and_process_data(data_dir, target_size=(128, 128)):
    X = []
    y = []
    class_names = sorted(os.listdir(data_dir))
    class_map = {name: idx for idx, name in enumerate(class_names)}

    for label in class_names:
        folder = os.path.join(data_dir, label)
        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            cropped_face = detect_and_crop_face(img, target_size)
            if cropped_face is not None:
                X.append(cropped_face)
                y.append(class_map[label])

    X = np.array(X)
    y = np.array(y)
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42), class_names
