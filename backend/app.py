from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from testing import analyze_image  # Import the function from testing.py
from utkface_analysis import analyze_video  # Import video analysis function
from flask import Flask, request, send_from_directory, jsonify

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
REPORTS_FOLDER = os.path.join(os.getcwd(), "reports")

@app.route('/download', methods=['POST'])
def download_file():
    data = request.json
    pdf_url = data.get("pdfUrl")  # Get pdfUrl from request body

    if not pdf_url:
        return jsonify({"error": "pdfUrl is required"}), 400
    
    filename = os.path.basename(pdf_url)  # Extract filename from pdfUrl
    file_path = os.path.join(REPORTS_FOLDER, filename)

    if os.path.exists(file_path):
        return send_from_directory(REPORTS_FOLDER, filename, as_attachment=True)
    else:
        return jsonify({"error": "File not found"}), 404
    
    
@app.route("/upload", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file:
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)
        
        # Call the function from testing.py
        results = analyze_image(filepath)
        return jsonify(results)

@app.route("/video_analysis", methods=["POST"])
def video_analysis():
    try:
        # Call utkface_analysis.py function to process video
        results = analyze_video()
        print(results,"00")
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
