import cv2
import numpy as np
import mediapipe as mp

# Initialize mediapipe hands
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
video = cv2.VideoCapture(0)

# Set video dimensions
video.set(3, 1000)
video.set(4, 780)

# Load and resize brush images
img_1 = cv2.imread('images/brush_1.png', -1)
img_2 = cv2.imread('images/brush_2.png', -1)
brush_1 = cv2.resize(img_1, (100, 100), interpolation=cv2.INTER_AREA)
brush_2 = cv2.resize(img_2, (100, 100), interpolation=cv2.INTER_AREA)

# Function to position the data
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

# Function to overlay transparent image
def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    x, y = pos
    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Overlay ranges
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, None]

    img_crop[:] = alpha * img_overlay_crop + (1 - alpha) * img_crop

# Create a canvas for drawing
canvas = np.zeros((780, 1000, 3), dtype=np.uint8)

while True:
    ret, image = video.read()
    image = cv2.flip(image, 1)
    rgbimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgbimg)
    flag = False
    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(image, hand, mpHands.HAND_CONNECTIONS)
            lmList = []
            for id, lm in enumerate(hand.landmark):
                h, w, c = image.shape
                coorx, coory = int(lm.x * w), int(lm.y * h)
                lmList.append([coorx, coory])
            position_data(lmList)

            # Determine hand and assign brush
            hand_label = result.multi_handedness[0].classification[0].label
            if flag == False:
                if hand_label == 'Left':
                    overlay_image_alpha(image, brush_1[:, :, :3], thumb_tip, brush_1[:, :, 3] / 255.0)
                    cv2.circle(canvas, thumb_tip, 10, (0, 0, 255), -1)  # Red color for left hand
                else:
                    overlay_image_alpha(image, brush_2[:, :, :3], thumb_tip, brush_2[:, :, 3] / 255.0)
                    cv2.circle(canvas, thumb_tip, 10, (255, 0, 0), -1)  # Blue color for right hand
                flag = True
            elif flag == True:
                if hand_label == 'Right':
                    overlay_image_alpha(image, brush_1[:, :, :3], thumb_tip, brush_1[:, :, 3] / 255.0)
                    cv2.circle(canvas, thumb_tip, 10, (0, 255, 0), -1)  # Green color for right hand
                else:
                    overlay_image_alpha(image, brush_2[:, :, :3], thumb_tip, brush_2[:, :, 3] / 255.0)
                    cv2.circle(canvas, thumb_tip, 10, (255, 255, 0), -1)  # Cyan color for left hand

    # Reduce the opacity of the background
    canvas_resized = cv2.resize(canvas, (image.shape[1], image.shape[0]))
    image = cv2.addWeighted(image, 0.5, canvas_resized, 0.5, 0)

    cv2.imshow("Image", image)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
