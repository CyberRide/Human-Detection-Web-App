import cv2
import tempfile
import os
import imutils
import numpy as np
from flask import Flask, render_template, request


# initialize the flask app
app = Flask(__name__)


# initialize the hog descriptor and set SVM to pre-trained pedestrian detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


# route the app to the home page
@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")


# route the app to the image detection page
@app.route("/image", methods=["GET", "POST"])
def image():
    if request.method == "POST":
        file = request.files["image"]
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        image = imutils.resize(image, width=min(500, image.shape[1]))
        regions, _ = hog.detectMultiScale(image, winStride=(4, 4), padding=(4, 4), scale=1.05)
        for (x, y, w, h) in regions:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image, f"Total Persons: {len(regions)}", (20, 315), cv2.FONT_ITALIC, 0.5, (0, 0, 255), 1)
        cv2.imshow("Human Detection from Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return render_template("image.html")
    else:
        return render_template("home.html")

# route the app to the video detection page
@app.route("/video", methods=["GET", "POST"])
def video():
    if request.method == "POST":
        file = request.files["video"]
        video_bytes = file.read()
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(video_bytes)
            video_path = f.name
        
        cap = cv2.VideoCapture(video_path)
        
        while cap.isOpened():
            ret, image = cap.read()
            if ret:
                image = imutils.resize(image, width=min(500, image.shape[1]))
                regions, _ = hog.detectMultiScale(image, winStride=(4, 4), padding=(4, 4), scale=1.05)
                for (x, y, w, h) in regions:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(image, f"Total Persons: {len(regions)}", (20, 250), cv2.FONT_ITALIC, 0.5, (0, 0, 255), 1)
                cv2.imshow("Human Detection from Video", image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
            else:
                break
        os.unlink(video_path)
        
        return render_template("video.html")
    else:
        return render_template("home.html")




hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

@app.route('/webcam')
def webcam():
    # OpenCV code to capture video from webcam and perform human detection
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = imutils.resize(frame, width=min(800, frame.shape[1]))
        regions, _ = hog.detectMultiScale(frame, winStride=(4, 4), padding=(4, 4), scale=1.05)
        for (x, y, w, h) in regions:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, f"Total Persons: {len(regions)}", (20, 50), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)
        cv2.imshow('Human Detection from Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return render_template('webcam.html')


# run the app
# if __name__ == '__main__':
#     app.run(debug=True, port=5000)
if __name__ == '__main__':
    app.run(debug=False)
