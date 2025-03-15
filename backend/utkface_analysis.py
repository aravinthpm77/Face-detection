from flask import Flask, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import face_recognition
import pickle
import csv
import os
import time
from fpdf import FPDF
from datetime import datetime
from tensorflow.keras.losses import MeanSquaredError


app = Flask(__name__)
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
csv_headers = ["Timestamp", "Name", "Age", "Gender", "Top_Emotion", "Emotion_Probabilities","Snapshot","Report"]
face_database = {}

# Load existing face database if available
if os.path.exists("face_database.pkl"):
    with open("face_database.pkl", "rb") as f:
        face_database = pickle.load(f)

# Ensure CSV file exists
if not os.path.exists(csv_filename):
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(csv_headers)

if not os.path.exists( "images"):
    os.makedirs( "images")


def generate_report(data):
    if not data:
        print("No data to generate report.")
        return None
    
    report_filename = f"reports/{data['Name']}_{datetime.now().strftime('%Y%m%d')}.pdf"
    if not os.path.exists("reports"):
        os.makedirs("reports")
    
    pdf = FPDF()
    
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Face Analysis Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(10)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "User Information", ln=True, align="L", border=1)
    pdf.ln(5)
    
    pdf.set_font("Arial", "", 12)
    details = [
        ("Timestamp:", data["Timestamp"]),
        ("Name:", data["Name"]),
        ("Age:", str(data["Age"])),
        ("Gender:", data["Gender"]),
        ("Top Emotion:", data["Top_Emotion"]),
    ]

    for label, value in details:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(40, 10, label, ln=False)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, value, ln=True)
    
    pdf.ln(5)

    # Emotion Probabilities Table
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Emotion Probabilities", ln=True, align="L", border=1)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(90, 10, "Emotion", border=1, align="C")
    pdf.cell(90, 10, "Probability (%)", border=1, align="C", ln=True)
    
    pdf.set_font("Arial", "", 12)
    for emo, prob in zip(emotion_labels, data["Emotion_Probabilities"]):
        pdf.cell(90, 10, emo, border=1, align="C")
        pdf.cell(90, 10, f"{prob:.2f}%", border=1, align="C", ln=True)
    
    pdf.ln(10)
    
    
    
    
    
    if os.path.exists(data["Snapshot"]):
        pdf.ln(10)
        pdf.image(data["Snapshot"], x=10, y=pdf.get_y(), w=100)
    
    pdf.set_y(-15)
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 10, f"Page {pdf.page_no()}", align="C")
    
    pdf.output(report_filename)
    print(f"Report generated: {report_filename}")
    return report_filename


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

def get_name_from_opencv():
    name = ""
    while True:
        frame = np.zeros((200, 500, 3), dtype=np.uint8)
        cv2.putText(frame, "Enter Name: " + name, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Name Input", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13:  # Enter key
            cv2.destroyWindow("Name Input")
            return name
        elif key == 8:  # Backspace key
            name = name[:-1]
        elif 32 <= key <= 126:  # Printable characters
            name += chr(key)

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
                        name = get_name_from_opencv()
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
                    
                snapshot_filename = f"{"images"}/{name}_{datetime.now().strftime('%Y%m%d')}.jpg"
                cv2.imwrite(snapshot_filename, face)
                
                
        
                data = {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Name": name,
                    "Age": age_pred-4,
                    "Gender": gender_pred,
                    "Top_Emotion": top_emotion,
                    "Emotion_Probabilities": probabilities.tolist() if emotion_prediction.size > 0 else [],
                    "Snapshot": snapshot_filename
                    
                } 
                report_path = generate_report(data)
                results.append(data)
                results[-1]["Report"] = report_path

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


    