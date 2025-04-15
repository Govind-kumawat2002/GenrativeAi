import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
print(cvzone.__version__)
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
detectore =HandDetector(detectionCon=0.8)

while True:
    success, img = cap.read()
    img =detectore.findHands(img)
    lmtList, _=detectore.findPosition(img)
    cv2.imshow("Image", img)
    cv2.waitKey(1)





# import cv2
# import mediapipe as mp
# import numpy as np

# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(max_num_hands=1)
# mp_draw = mp.solutions.drawing_utils

# # Dummy object position
# obj_pos = [300, 300]
# dragging = False

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     frame = cv2.flip(frame, 1)
#     h, w, _ = frame.shape

#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     result = hands.process(rgb)

#     if result.multi_hand_landmarks:
#         hand_landmarks = result.multi_hand_landmarks[0]
#         mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#         lm = hand_landmarks.landmark

#         # Get thumb tip and index tip
#         x1, y1 = int(lm[4].x * w), int(lm[4].y * h)
#         x2, y2 = int(lm[8].x * w), int(lm[8].y * h)

#         center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
#         distance = np.hypot(x2 - x1, y2 - y1)

#         # Draw pointer
#         cv2.circle(frame, (center_x, center_y), 8, (255, 0, 255), -1)

#         # If pinch (drag)
#         if distance < 40:
#             dragging = True
#         else:
#             dragging = False

#         # Move object if dragging
#         if dragging:
#             obj_pos[0] = center_x
#             obj_pos[1] = center_y

#     # Draw draggable object
#     cv2.rectangle(frame, (obj_pos[0]-40, obj_pos[1]-40), (obj_pos[0]+40, obj_pos[1]+40), (0, 255, 0), -1)

#     cv2.imshow("Virtual Drag & Drop", frame)
#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()
