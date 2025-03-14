import cv2
import numpy as np
import tensorflow as tf
import face_recognition
import pickle
import csv
import os
import time
from datetime import datetime
from tensorflow.keras.losses import MeanSquaredError

# ✅ Load Models
age_gender_model = tf.keras.models.load_model("face_model.h5", custom_objects={"mse": MeanSquaredError()})
emotion_model = tf.keras.models.load_model("emotion_detection_model.h5")

# ✅ Load Face Detection Model
face_net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

# 🎭 Emotion Labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# ✅ Load Face Database
try:
    with open("face_database.pkl", "rb") as f:
        face_database = pickle.load(f)
except FileNotFoundError:
    face_database = {}

# ✅ CSV Setup
csv_filename = "face_data.csv"
csv_headers = ["Timestamp", "Name", "Age", "Gender", "Top_Emotion", "Emotion_Probabilities"]

# Ensure CSV file exists
if not os.path.exists(csv_filename):
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(csv_headers)

# 🛠 **Preprocessing Functions**
def preprocess_face(face):
    face = cv2.resize(face, (100, 100))
    face = face / 255.0
    return np.expand_dims(face, axis=0)

def preprocess_emotion(face):
    face = cv2.resize(face, (48, 48))
    face = face / 255.0
    face = np.expand_dims(face, axis=0)
    if face.shape[-1] == 1:
        face = np.repeat(face, 3, axis=-1)
    return face

def get_name_from_user():
    name = ""
    while not name:
        name = input("Enter name for the new face: ").strip()
    return name

def analyze_video():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    results = []
    start_time = time.time()
    
    while cap.isOpened():
        if time.time() - start_time > 20:
            break

        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.6:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x, y, x2, y2 = box.astype("int")

                if x < 0 or y < 0 or x2 > w or y2 > h:
                    continue

                face = frame[y:y2, x:x2]
                if face.shape[0] == 0 or face.shape[1] == 0:
                    continue

                rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(rgb_face)

                name = "Unknown"
                if encodings:
                    face_encoding = encodings[0]
                    matches = face_recognition.compare_faces(list(face_database.values()), face_encoding, tolerance=0.5)
                    if True in matches:
                        match_index = matches.index(True)
                        name = list(face_database.keys())[match_index]
                    else:
                        name = get_name_from_user()
                        face_database[name] = face_encoding
                        with open("face_database.pkl", "wb") as f:
                            pickle.dump(face_database, f)

                input_face = preprocess_face(face)
                predictions = age_gender_model.predict(input_face)
                age_pred = int(predictions[0][0] * 100) if predictions[0].size > 0 else "Unknown"
                gender_pred = "Male" if predictions[1][0][0] < 0.6 else "Female"

                emotion_face = preprocess_emotion(face)
                emotion_prediction = emotion_model.predict(emotion_face)
                
                if emotion_prediction.size > 0 and len(emotion_prediction.shape) == 2 and emotion_prediction.shape[1] == len(emotion_labels):
                    probabilities = emotion_prediction.flatten()
                    sorted_indices = np.argsort(probabilities)[::-1]

                    if len(sorted_indices) > 0:
                        top_emotion = emotion_labels[sorted_indices[0]]
                        top_3_emotions = [(emotion_labels[idx], round(probabilities[idx] * 100, 2)) for idx in sorted_indices[:3]]
                    else:
                        top_emotion = "Unknown"
                        top_3_emotions = []
                else:
                    top_emotion = "Unknown"
                    top_3_emotions = []
                    
                
                results.append({
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Name": name,
                    "Age": age_pred-4,
                    "Gender": gender_pred,
                    "Top_Emotion": top_emotion,
                    "Emotion_Probabilities": probabilities.tolist() if emotion_prediction.size > 0 else []
                })

                cv2.putText(frame, f"Name: {name}", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Age: {age_pred-4}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Gender: {gender_pred}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Emotion: {top_emotion}", (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                for idx, (emo, prob) in enumerate(top_3_emotions):
                    cv2.putText(frame, f"{emo}: {prob:.2f}%", (x2 + 10, y + idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow("Video Analysis", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                
            
    cap.release()
    cv2.destroyAllWindows()
    
    if results:
        min_age_entry = min(results, key=lambda x: x["Age"])
        
        # Overwrite the CSV instead of appending (to ensure only one entry)
        with open(csv_filename, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=csv_headers)
           
            writer.writerow(min_age_entry)
        
        print(min_age_entry)
        return min_age_entry

    return None

