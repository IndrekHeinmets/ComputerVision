import mediapipe as mp
import time
import math
import cv2


class Hand_detector():
    def __init__(self, mode=False, max_hands=2, model_complexity=1, detection_conf=0.5, track_conf=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.detection_conf = detection_conf
        self.track_conf = track_conf

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.model_complexity, self.detection_conf, self.track_conf)
        self.mp_draw = mp.solutions.drawing_utils

        self.tip_ids = [4, 8, 12, 16, 20]

    def find_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        return img

    def find_pos(self, img, hand_no=0, draw=True):
        x_list = []
        y_list = []
        self.lm_list = []
        b_box = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]

            for id, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                x_list.append(cx)
                y_list.append(cy)
                self.lm_list.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 5, (47, 0, 255), cv2.FILLED)

            x_min, x_max = min(x_list) - 20, max(x_list) + 20
            y_min, y_max = min(y_list) - 20, max(y_list) + 20
            b_box = x_min, y_min, x_max, y_max

            if draw:
                cv2.rectangle(img, (b_box[0], b_box[1]), (b_box[2], b_box[3]), (0, 255, 0), 2)

        return self.lm_list, b_box

    def find_distance(self, p1, p2, img, draw=True):
        x1, y1 = self.lm_list[p1][1], self.lm_list[p1][2]
        x2, y2 = self.lm_list[p2][1], self.lm_list[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.circle(img, (x1, y1), 8, (195, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 8, (195, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), 8, (195, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (195, 0, 255), 2)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]

    def fingers_up(self):
        fingers= []
        # Thumb:
        if self.lm_list[self.tip_ids[0]][1] < self.lm_list[self.tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 Fingers:
        for id in range(1, 5):
            if self.lm_list[self.tip_ids[id]][2] < self.lm_list[self.tip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

def main():
    w_cam, h_cam = 640, 480
    draw = True

    cap = cv2.VideoCapture(0)
    cap.set(3, w_cam)
    cap.set(4, h_cam)
    p_time = 0
    detector = Hand_detector()

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = detector.find_hands(img, draw=draw)
        lm_list, b_box = detector.find_pos(img, draw=draw)
        if len(lm_list) != 0:
            pass

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        cv2.putText(img, f'Fps: {int(fps)}', (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (234, 255, 0), 2)
        cv2.imshow('CV', img)
        
        # Exit if ESC Pressed or Window Closed:
        if cv2.waitKey(1) == 27 or cv2.getWindowProperty('CV', cv2.WND_PROP_VISIBLE) < 1:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()