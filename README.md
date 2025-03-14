# Face Detection with Age, Gender, and Emotion Recognition

## Overview
This repository contains a face detection system that not only detects faces but also predicts their age, gender, and emotions using deep learning models. The implementation leverages OpenCV and pre-trained deep learning models for accurate predictions.

## Features
- **Face Detection:** Identifies faces in images and videos.
- **Age Prediction:** Estimates the age range of detected faces.
- **Gender Classification:** Determines the gender (Male/Female) of the detected faces.
- **Emotion Recognition:** Recognizes different emotions such as happy, sad, angry, surprised, neutral, etc.

## Technologies Used
- Python
- OpenCV
- Deep Learning Models (Pre-trained on datasets for face detection, age, gender, and emotion recognition)
- NumPy
- TensorFlow/Keras (for deep learning models)

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/aravinthpm77/Face-detection.git
   cd Face-detection
   ```

2. Install required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3.Install react js :
```sh
npx create-react-app ./
```

## Usage
Run the face detection script to detect faces and predict age, gender, and emotions in the backend and for the frontend run the application using the following commands

```sh
python app.py
```

```sh
npm start
```


Ensure that the required pre-trained models (face detection, age, gender, emotion recognition) are downloaded and placed in the correct directory.

## Model Details
- **Face Detection Model:** Uses OpenCV's Haar cascades or DNN-based models for real-time detection.
- **Age & Gender Prediction Model:** Trained deep learning models trained on labeled datasets and CNN Models for Face Detections.
- **Emotion Recognition Model:** Uses a convolutional neural network (CNN) trained on facial expression datasets.

## Output
For each detected face, the system will:
- Draw a bounding box around the face.
- Display the estimated age.
- Show the predicted gender.
- Display the recognized emotion.
- Graphical Display Center.
  
## Contributing
Feel free to contribute by improving the model accuracy, adding new features, or optimizing performance. Create a pull request with your modifications.

## License
This project is open-source and available under the MIT License.

## Contact
For any issues or feature requests, please open an issue in the repository or contact via GitHub or Linkedin.

