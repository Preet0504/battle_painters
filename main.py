import time
import cv2
import numpy as np
import mediapipe as mp

# Initialize mediapipe hands
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
video = cv2.VideoCapture(0)

# Set video dimensions
video.set(3, 3000)
video.set(4, 1500)

# Load and resize brush images
img_1 = cv2.imread('images/brush_1.png', -1)
img_2 = cv2.imread('images/brush_2.png', -1)
brush_1 = cv2.resize(img_1, (100, 100), interpolation=cv2.INTER_AREA)
brush_2 = cv2.resize(img_2, (100, 100), interpolation=cv2.INTER_AREA)

# Define the region of interest (ROI) for coloring
roi_x1, roi_y1 = 100, 100  # Top-left corner of the ROI
roi_x2, roi_y2 = 900, 680  # Bottom-right corner of the ROI

game_started = False
start_time = 0

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
canvas[roi_y1+5:roi_y2+50, roi_x1-5:roi_x2-200] = (255, 255, 255)  # Set ROI to white

def start_game():
    global game_started, start_time
    game_started = True
    start_time = time.time()

while True:
    ret, image = video.read()
    image = cv2.flip(image, 1)
    # image[:,:] = (0,0,0)
    rgbimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgbimg)
    if game_started:
        elapsed_time = time.time() - start_time
        if elapsed_time < 3:
            countdown = int(3 - elapsed_time)
            cv2.putText(image, str(countdown), (500, 400), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 255, 0), 15)
        elif elapsed_time < 18:
            result_text = f"Time: {19-int(elapsed_time)}"
            cv2.putText(image, result_text, (500,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 5) 
            if result.multi_hand_landmarks:
                for idx, hand in enumerate(result.multi_hand_landmarks):
                    mpDraw.draw_landmarks(image, hand, mpHands.HAND_CONNECTIONS)
                    lmList = []
                    for id, lm in enumerate(hand.landmark):
                        h, w, c = image.shape
                        coorx, coory = int(lm.x * w), int(lm.y * h)
                        lmList.append([coorx, coory])
                    position_data(lmList)

                    # Determine hand and assign brush
                    hand_label = result.multi_handedness[idx].classification[0].label
                    if roi_x1-5 <= thumb_tip[0] <= roi_x2-200 and roi_y1+5<= thumb_tip[1] <= roi_y2+50:
                        if hand_label == 'Left':
                            overlay_image_alpha(image, brush_1[:, :, :3], thumb_tip, brush_1[:, :, 3] / 255.0)
                            cv2.circle(canvas, thumb_tip, 20, (0, 0, 255), -1)  # Red color for left hand
                        else:
                            overlay_image_alpha(image, brush_2[:, :, :3], thumb_tip, brush_2[:, :, 3] / 255.0)
                            cv2.circle(canvas, thumb_tip, 20, (255, 0, 0), -1)  # Blue color for right hand
        else: 
            roi_canvas = canvas[roi_y1:roi_y2, roi_x1:roi_x2]
            red_pixels = np.sum(np.all(roi_canvas == [0, 0, 255], axis=-1))
            blue_pixels = np.sum(np.all(roi_canvas == [255, 0, 0], axis=-1))
            winner = "Red" if red_pixels > blue_pixels else "Blue"
            result_text = f"Winner: {winner}" 
            cv2.putText(image, result_text, (start_button_x1 + 11, start_button_y1 + 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5) 
            if(elapsed_time>25):
                canvas = np.zeros((780, 1000, 3), dtype=np.uint8)  # Reset canvas for the next game
                canvas[roi_y1+5:roi_y2+50, roi_x1-5:roi_x2-200] = (255, 255, 255)  # Set ROI to white
                game_started = False  
        
        roi_canvas = canvas[roi_y1:roi_y2, roi_x1:roi_x2]
        red_pixels = np.sum(np.all(roi_canvas == [0, 0, 255], axis=-1))/1000
        blue_pixels = np.sum(np.all(roi_canvas == [255, 0, 0], axis=-1))/1000
        result_text = f"Red: {red_pixels} "
        cv2.putText(image, result_text, (start_button_x1 + 15, start_button_y1 + 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
        result_text = f"Blue: {blue_pixels}"
        cv2.putText(image, result_text, (start_button_x1 + 13, start_button_y1 + 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
        
    else:
        # Draw the start button
        start_button_x1, start_button_y1 = image.shape[1] - 300, 50
        start_button_x2, start_button_y2 = image.shape[1] - 50, 150
        cv2.rectangle(image, (start_button_x1, start_button_y1), (start_button_x2, start_button_y2), (255, 0, 0), -1)
        cv2.putText(image, "Start", (start_button_x1 + 30, start_button_y1 + 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
        
        
        if cv2.waitKey(1) & 0xFF == ord('s'):
            start_game()

    # Reduce the opacity of the background
    canvas_resized = cv2.resize(canvas, (image.shape[1], image.shape[0]))
    image = cv2.addWeighted(image, 0.5, canvas_resized, 0.8, 0)

    # Draw the ROI on the image
    cv2.rectangle(image, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 50)
    
    cv2.imshow("Image", image)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Calculate the total red and blue pixels within the ROI
roi_canvas = canvas[roi_y1:roi_y2, roi_x1:roi_x2]
red_pixels = np.sum(np.all(roi_canvas == [0, 0, 255], axis=-1))
blue_pixels = np.sum(np.all(roi_canvas == [255, 0, 0], axis=-1))

# Determine the winner
# winner = "Red" if red_pixels > blue_pixels else "Blue"
# print(f"Red pixels: {red_pixels}, Blue pixels: {blue_pixels}")
# print(f"The winner is: {winner}")

video.release()
cv2.destroyAllWindows()
