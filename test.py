import time

import cv2
import dlib
from imutils import face_utils


DEFAULT_LEVEL = 0

img = cv2.imread("main.png")
if img is None:
    raise FileNotFoundError

# イラストは不可
def detect_eye(landmarks, side):
    landmarks = face_utils.shape_to_np(landmarks)
    if side == "left":
        eye_left = 36
        eye_right = 39
        eye_top = 38
        eye_bottom = 41
    elif side == "right":
        eye_left = 42
        eye_right = 45
        eye_top = 43
        eye_bottom = 46
    else:
        raise ValueError("left or right")

    left_x = landmarks[eye_left][0]
    right_x = landmarks[eye_right][0]
    eye_width_half = (right_x - left_x) // 2

    top_y = landmarks[eye_top][1]
    bottom_y = landmarks[eye_bottom][1]

    top_left = (left_x - eye_width_half, top_y - eye_width_half)
    bottom_right = (right_x + eye_width_half, bottom_y + eye_width_half)
    return top_left, bottom_right


# 顔を検出
start = time.time()
detector = dlib.get_frontal_face_detector()

faces = []
level = int(DEFAULT_LEVEL)
while len(faces) == 0:
    if level > 5:
        break
    faces = detector(img, level)
    if len(faces) == 0:
        print("顔が見つからなかったのでレベルを一つ上げます")
        print("(レベルが高いほど時間がかかります)\n")
    level += 1

print("顔検出にかかった時間:", time.time()-start)


# 顔のランドマーク検出
start = time.time()

predictor = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")
for face in faces:
    top_left, bottom_right = detect_eye(predictor(img, face), "left")
    cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 0)

    top_left, bottom_right = detect_eye(predictor(img, face), "right")
    cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 0)

print("顔のランドマーク検出にかかった時間:", time.time()-start)


cv2.imshow("eye", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
