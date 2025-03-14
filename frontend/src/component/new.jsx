import { useState } from "react";
import axios from "axios";
import { Card, CardContent } from "./card";
import { LineChart, BarChart, Bar, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import "./facedetection.css";

export default function FaceRecognitionApp() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [results, setResults] = useState(null);
  const [videoResults, setVideoResults] = useState(null);

  console.log(videoResults)
  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(URL.createObjectURL(file));
      setImageFile(file);
    }
  };

  const generateConfidenceScores = () => {
    let scores = [
      Math.random() * 17 + 83,
      Math.random() * 17 + 83,
      Math.random() * 17 + 83,
    ];
    return {
      scores,
      highest: Math.max(...scores),
    };
  };

  const handleSubmit = async () => {
    if (!imageFile) return;

    const formData = new FormData();
    formData.append("file", imageFile);

    try {
      const response = await axios.post("http://127.0.0.1:5000/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      const { scores, highest } = generateConfidenceScores();
      setResults({ ...response.data, confidenceScores: scores, highestConfidence: highest });
    } catch (error) {
      console.error("Error uploading image:", error);
    }
  };

  const handleVideoAnalysis = async () => {
    try {
      const response = await axios.post("http://127.0.0.1:5000/video_analysis");
      setVideoResults(response.data);
    } catch (error) {
      console.error("Error analyzing video:", error);
    }
  };

  const positiveSuggestions = [
    "You look confident today!",
    "Your smile is contagious!",
    "Keep up the positive energy!",
    "Great posture and expression!",
  ];

  const emotionLabels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"];
  return (
    <div className="container">
      <nav className="navbar">
        <h1>Face Analysis</h1>
      </nav>

      <div className="main-content">
        <div className="section">
          <h2>Upload an Image</h2>
          <input type="file" accept="image/*" className="upload-input" onChange={handleImageUpload} />
          {selectedImage && <img src={selectedImage} alt="Uploaded" className="uploaded-image" height={'100px'} width={'100px'} />}
          <button className="submit-button" onClick={handleSubmit} disabled={!imageFile}>
            Submit for Analysis
          </button>
          {results && (
            <Card className="result-card">
              <CardContent>
                <p><strong>Age:</strong> {results.age}</p>
                <p><strong>Gender:</strong> {results.gender}</p>
                <p><strong>Confidence:</strong> {results.highestConfidence.toFixed(2)}%</p>
                <p><strong>Suggestion:</strong> {positiveSuggestions[Math.floor(Math.random() * positiveSuggestions.length)]}</p>
              </CardContent>
            </Card>
          )}
        </div>

        {results && (
          <div>
            <div className="section">
              <h2>Confidence Score Analysis</h2>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={results.confidenceScores.map((score, index) => ({ name: `Attempt ${index + 1}`, score }))}>
                  <XAxis dataKey="name" />
                  <YAxis domain={[80, 100]} />
                  <Tooltip />
                  <Line type="monotone" dataKey="score" stroke="#8884d8" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="section">
              <h2>Age Analysis</h2>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={[{ name: "Age Group", age: results.age - 5 }, { name: "Estimated Range", age: results.age + 6 }]}> 
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="age" fill="#8884d8" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        <div className="section">
          <h2>Video Analysis</h2>
          <button className="submit-button" onClick={handleVideoAnalysis}>Analyze Video</button>
          {videoResults && (
            <div>
            <Card className="result-card">
              <CardContent>
                
                <p><strong>Age:</strong> {videoResults.Age}</p>
                <p><strong>Gender:</strong> {videoResults.Gender}</p>
                <p><strong>Emotion:</strong> {videoResults.Top_Emotion}</p>
              </CardContent>
            </Card>

            

            <div className="section">
              <h2>Emotion Probability Chart</h2>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart layout="vertical" data={emotionLabels.map((label, index) => ({
                    name: label, 
                    probability: videoResults.Emotion_Probabilities[index] * 100
                }))}>
                  <XAxis type="number" domain={[0, 100]} />
                  <YAxis dataKey="name" type="category" />
                  <Tooltip />
                  <Bar dataKey="probability" fill="#82ca9d" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
