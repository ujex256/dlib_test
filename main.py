import cv2
import dlib
import base64
import numpy as np
from fastapi import FastAPI

from test import detect_eye

app = FastAPI()

def b64_to_cv2_img(data: str):
    data = data.replace("-", "+").replace("_", "/")
    if "," in data:
        data = data.split(",")[1]
    decoded_bytes = base64.b64decode(data)
    ndarray = np.frombuffer(decoded_bytes, np.uint8)
    img = cv2.imdecode(ndarray, cv2.IMREAD_COLOR)
    return img

def img_to_b64(data):
    ret, dst = cv2.imencode(".jpg", data)
    return base64.b64encode(dst).decode("utf-8", "strict")

@app.post("/detect")
def detect(img):
    img_ = b64_to_cv2_img(img)
    detector = dlib.get_frontal_face_detector()
    for i in range(5):
        faces = detector(img_, i)
        if faces:
            break
    predictor = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")
    for face in faces:
        top_left, bottom_right = detect_eye(predictor(img_, face), "left")
        cv2.rectangle(img_, top_left, bottom_right, (0, 255, 0), 0)

        top_left, bottom_right = detect_eye(predictor(img_, face), "right")
        cv2.rectangle(img_, top_left, bottom_right, (0, 255, 0), 0)
    return img_to_b64(img_)