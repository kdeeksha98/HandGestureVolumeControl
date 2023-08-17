import cv2
import mediapipe as mp
import time
import subprocess
import math

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0 #previous time
cTime = 0 #current time

# Calculate the maximum distance between thumb and index finger
max_distance = 0.2  # Adjust this value based on your hand's maximum span

# Initialize the position and length of the volume bar
bar_x = 50
bar_y = 400
bar_width = 400
bar_height = 20

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                '''
                h = height, w = width, c = centre, lm = landmark
                '''
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            thumb_tip = handLms.landmark[4]
            index_tip = handLms.landmark[8]

            distance = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
            
            # Calculate the volume and clamp it between 0 and 87
            new_volume = max(0, min(87, int(distance * 87 / max_distance)))
            subprocess.run(['amixer', '-D', 'pulse', 'sset', 'Master', f'{new_volume}%'])
            
            # Calculate the length of the volume bar based on the clamped volume
            bar_length = int(new_volume * bar_width / 87)
            
            # Draw the volume bar
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_length, bar_y + bar_height), (0, 255, 0), cv2.FILLED)
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 0, 0), 2)
            
            # Show the current volume value
            cv2.putText(img, str(new_volume), (bar_x + bar_width + 10, bar_y + bar_height // 2), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (18, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
