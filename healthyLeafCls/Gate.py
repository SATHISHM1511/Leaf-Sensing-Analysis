from flask import Flask, render_template, Response, jsonify
import cv2
import json
from leafCls_Identify import HealthyLeafClassification

app = Flask(__name__)

# Load configuration
with open("config.json", 'r') as file:
    configJson = json.load(file)

# Camera Configs
camType = configJson["cameraDetails"]["cameraType"]
captureFPS = configJson["cameraDetails"]["fps"]

# Initialize Model
leafClsModelRes = HealthyLeafClassification(model=r"leafOD_Yolo11_150n.pt")

# Global variable to store classification result
latest_result = {"Result": "No Data"}

def generate_frames():
    """Captures video frames, processes them, and streams them to the frontend"""
    global latest_result

    if camType == 2:
        ipCam = configJson["cameraDetails"]["ipCamUrl"]
        vid = cv2.VideoCapture(ipCam + "/video")
        vid.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    else:
        vid = cv2.VideoCapture(camType)

    if not vid.isOpened():
        return "Error: Could not open video capture."

    fps = int(vid.get(cv2.CAP_PROP_FPS))
    flag = 0
    ExtractionRate = fps // captureFPS

    while True:
        success, frame = vid.read()
        if not success:
            break

        if flag % ExtractionRate == 0:
            latest_result = leafClsModelRes.Identiify_HealthyLeaf(frame)  # Store result globally
            flag = 0

        flag += 1

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Loads the index.html page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Streams video feed to the HTML page"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_result')
def get_result():
    """Returns the latest classification result as JSON"""
    return jsonify(latest_result)

if __name__ == "__main__":
    app.run(debug=True)
