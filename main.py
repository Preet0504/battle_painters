import cv2
import cvzone
import mediapipe as mp

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
video = cv2.VideoCapture(0)

video.set(3, 1000)
video.set(4, 780)

def position_data(lmlist):
    global wrist, thumb_tip, index_mcp, index_tip, midle_mcp, midle_tip, ring_tip, pinky_tip
    wrist = (lmlist[0][0], lmlist[0][1])
    thumb_tip = (lmlist[4][0], lmlist[4][1])
    index_mcp = (lmlist[5][0], lmlist[5][1])
    index_tip = (lmlist[8][0], lmlist[8][1])
    midle_mcp = (lmlist[9][0], lmlist[9][1])
    midle_tip = (lmlist[12][0], lmlist[12][1])
    ring_tip  = (lmlist[16][0], lmlist[16][1])
    pinky_tip = (lmlist[20][0], lmlist[20][1])

while(True):
    ret, image = video.read()
    image = cv2.flip(image,1)
    rgbimg=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result=hands.process(rgbimg)
    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(image, hand, mpHands.HAND_CONNECTIONS)
            lmList=[]
            for id, lm in enumerate(hand.landmark):
                h,w,c = image.shape
                coorx, coory=int(lm.x*w), int(lm.y*h)
                lmList.append([coorx, coory])
            position_data(lmList)
            print(thumb_tip)
    cv2.imshow("Image",image)
    k= cv2.waitKey(1)
    if k==ord('q'):
        break