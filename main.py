import cv2
import cvzone
import mediapipe as mp

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
video = cv2.VideoCapture(0)

video.set(3, 1000)
video.set(4, 780)

while(True):
    ret, image = video.read()
    image = cv2.flip(image,1)
    rgbimg=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result=hands.process(rgbimg)
    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(image, hand, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Image",image)
    k= cv2.waitKey(1)
    if k==ord('q'):
        break