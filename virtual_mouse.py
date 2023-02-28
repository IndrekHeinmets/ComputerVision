import hand_tracking_module as htm
import pyautogui as ap
import numpy as np
import time
import cv2

# Disable AutoPy Failsafe:
ap.FAILSAFE = False
ap.PAUSE = 0

w_cam, h_cam = 640, 480
frame_red, bot_frame_red, smoothen = 120, 190, 5
draw = True

p_time = 0
p_loc_x, p_loc_y = 0, 0
c_loc_x, c_loc_y = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, w_cam)
cap.set(4, h_cam)
w_screen, h_screen = ap.size()
detector = htm.Hand_detector(max_hands=1, detection_conf=0.8)
clicked = False

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.find_hands(img, draw=draw)
    lm_list, b_box = detector.find_pos(img, draw=draw)

    # Get Fingertip Info:
    if len(lm_list) != 0:
        x1, y1 = lm_list[8][1:]
        x2, y2 = lm_list[12][1:]

        # Check Fingers Up:
        fingers = detector.fingers_up()
        cv2.rectangle(img, (frame_red, frame_red), (w_cam - frame_red, h_cam - bot_frame_red), (255, 0, 255), 2)

        # Index and Middle Finger Up (Move Mouse):
        if fingers[1] and fingers[2]:
            clicked = False

            # Convert Coordinates:
            x3 = np.interp(x1, (frame_red, w_cam - frame_red), (0, w_screen))
            y3 = np.interp(y1, (frame_red, h_cam - bot_frame_red), (0, h_screen))

            # Smoothen Response:
            c_loc_x = p_loc_x + (x3 - p_loc_x) / smoothen
            c_loc_y = p_loc_y + (y3 - p_loc_y) / smoothen

            # Move Mouse:
            ap.moveTo(c_loc_x, c_loc_y)
            cv2.circle(img, (x1, y1), 8, (0, 255, 0), cv2.FILLED)
            p_loc_x, p_loc_y = c_loc_x, c_loc_y

        # Index Up, Middle Down (left Click):
        if fingers[1] and not fingers[2] and not fingers [4] and not clicked:
            cv2.circle(img, (x1, y1), 8, (255, 0, 0), cv2.FILLED)
            ap.click(button='left')
            clicked = True

        # Index Up, Pinky Up, Middle Down (Right Click):
        if fingers[1] and fingers [4] and not fingers[2] and not clicked:
            cv2.circle(img, (x1, y1), 8, (255, 0, 255), cv2.FILLED)
            ap.click(button='right')
            clicked = True

        # Only Thumb Up (Scroll Down):
        if fingers[0] and not fingers[1] and not fingers[2] and not fingers[4]:
            ap.scroll(20)

        # Only Pinky Up (Scroll Up):
        if not fingers[0] and not fingers[1] and not fingers[2] and fingers[4]:
            ap.scroll(-20)

    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    cv2.putText(img, f'Fps: {int(fps)}', (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (234, 255, 0), 2)
    cv2.imshow('Virtual Mouse', img)
    
    # Exit if ESC Pressed or Window Closed:
    if cv2.waitKey(1) == 27 or cv2.getWindowProperty('Virtual Mouse', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
    
