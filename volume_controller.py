from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import hand_tracking_module as htm
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
import numpy as np
import time
import cv2

w_cam, h_cam = 640, 480
draw = True

cap = cv2.VideoCapture(0)
cap.set(3, w_cam)
cap.set(4, h_cam)
p_time = 0
detector = htm.Hand_detector(max_hands=1, detection_conf=0.8)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

vol_range = volume.GetVolumeRange()
min_vol, max_vol = vol_range[0], vol_range[1]
vol_per = 0
col_vol = (234, 255, 0)
area = 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Find Hand:
    img = detector.find_hands(img, draw=draw)
    lm_list, b_box = detector.find_pos(img, draw=draw)
    if len(lm_list) != 0:
        # Filter Based on Size:
        area = ((b_box[2] - b_box[0]) * (b_box[3] - b_box[1])) // 100
        if 200 < area < 1600:

            # Find Distance Between Fingers:
            length, img, line_info = detector.find_distance(4, 8, img, draw=draw)
            if draw:
                if length < 20 or length > 140:
                    cv2.circle(img, (line_info[4], line_info[5]), 10, (0, 255, 0), cv2.FILLED)

            # Convert Length -> Volume:
            vol_per = np.interp(length, [20, 140], [0, 100])

            # Reduce Resolution to Increase Smoothness:
            increments = 2
            vol_per = increments * round(vol_per / increments)

            # Check Which Fingers Up (If Pinky Up -> Set Volume):
            fingers = detector.fingers_up()

            if not fingers[3] and fingers[4]:
                volume.SetMasterVolumeLevelScalar(vol_per / 100, None)
                col_vol = (0, 255, 0)
            else:
                col_vol = (234, 255, 0)

    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    cv2.putText(img, f'Fps: {int(fps)}', (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (234, 255, 0), 2)

    current_vol = int(volume.GetMasterVolumeLevelScalar() * 100)
    cv2.putText(img, f'Vol: {int(vol_per)}', (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (234, 255, 0), 2)
    cv2.putText(img, f'Vol set: {int(current_vol)}', (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col_vol, 2)
    cv2.imshow('Volume Controller', img)
    
    # Exit if ESC Pressed or Window Closed:
    if cv2.waitKey(1) == 27 or cv2.getWindowProperty('Volume Controller', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
    

