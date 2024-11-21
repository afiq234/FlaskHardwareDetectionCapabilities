from flask import Flask, redirect, url_for, render_template, request, Response
import os
import cv2
import torch
from matplotlib import pyplot as plt
import numpy as np

model = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path="exp3/weights/last.pt",
    force_reload=True,
)
app = Flask(__name__)
camera = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            results = model(frame)
            detected_frame = np.squeeze(
                results.render()
            )  # Render detections on the frame

            ret, buffer = cv2.imencode(".jpg", detected_frame)
            frame = buffer.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(debug=True)
    # app.run(port=int(os.environ.get("PORT", 8080)), host="0.0.0.0", debug=True)
