import cv2
import numpy as np
import HandTrackingModule as htm
import time

# === 1. Initialize webcam ===
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# === 2. Initialize drawing settings ===
detector = htm.HandDetector(detectionCon=0.85, maxHands=1)
drawColor = (255, 0, 255)
brushThickness = 8
eraserThickness = 130
xp, yp = 0, 0

imgCanvas = np.zeros((720, 1280, 3), np.uint8)
strokes = []
currentStroke = []

# === 3. Define colors and UI styles ===
colors = [
    {"name": "Purple", "color": (255, 0, 255)},
    {"name": "Green", "color": (0, 255, 0)},
    {"name": "Red", "color": (0, 0, 255)},
    {"name": "Eraser", "color": (0, 0, 0)}
]
font = cv2.FONT_HERSHEY_SIMPLEX

undo_cooldown = 0
clear_cooldown = 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    fingers = [0, 0, 0, 0, 0]

    if lmList:
        x1, y1 = lmList[8][1:]
        tipIds = [4, 8, 12, 16, 20]

        fingers[0] = 1 if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1] else 0
        for i in range(1, 5):
            fingers[i] = 1 if lmList[tipIds[i]][2] < lmList[tipIds[i] - 2][2] else 0

        totalFingers = sum(fingers)

        # Clear canvas (with cooldown)
        if totalFingers == 5 and time.time() - clear_cooldown > 1:
            imgCanvas = np.zeros((720, 1280, 3), np.uint8)
            strokes.clear()
            clear_cooldown = time.time()
            cv2.putText(img, "Canvas Cleared", (950, 680), font, 1, (0, 255, 255), 3)

        # Undo (with cooldown)
        elif fingers[1] and fingers[2] and fingers[3] and not fingers[4] and time.time() - undo_cooldown > 1:
            if strokes:
                strokes.pop()
                imgCanvas = np.zeros((720, 1280, 3), np.uint8)
                for stroke in strokes:
                    for p1, p2, color, thick in stroke:
                        cv2.line(imgCanvas, p1, p2, color, thick)
                undo_cooldown = time.time()
                cv2.putText(img, "Undo", (1100, 680), font, 1, (0, 255, 255), 3)

        # Color selection
        elif fingers[1] and fingers[2]:
            xp, yp = 0, 0
            currentStroke = []
            for i, col in enumerate(colors):
                xStart = i * 320
                if xStart < x1 < xStart + 320 and y1 < 70:
                    drawColor = col["color"]
                    break
            cv2.rectangle(img, (x1 - 25, y1 - 25), (x1 + 25, y1 + 25), drawColor, cv2.FILLED)

        # Drawing
        elif fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 8, drawColor, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            thickness = eraserThickness if drawColor == (0, 0, 0) else brushThickness
            p1, p2 = (xp, yp), (x1, y1)
            cv2.line(img, p1, p2, drawColor, thickness)
            cv2.line(imgCanvas, p1, p2, drawColor, thickness)
            currentStroke.append((p1, p2, drawColor, thickness))
            xp, yp = x1, y1

        else:
            if currentStroke:
                strokes.append(currentStroke)
                currentStroke = []

    # Merge canvas with webcam feed
    gray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, inv)
    img = cv2.bitwise_or(img, imgCanvas)

    # Draw color palette UI
    for i, col in enumerate(colors):
        xStart = i * 320
        color = col["color"]
        name = col["name"]
        overlay = img.copy()
        cv2.rectangle(overlay, (xStart + 20, 10), (xStart + 300, 60), color, cv2.FILLED)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
        cv2.rectangle(img, (xStart + 20, 10), (xStart + 300, 60), (255, 255, 255), 2)
        cv2.putText(img, name, (xStart + 40, 45), font, 1.2, (255, 255, 255), 2)

    # Show instructions
    cv2.putText(img, "Draw: Index Up | Select: Index+Middle | Undo: 3 Fingers | Clear: 5 Fingers", 
                (20, 690), font, 0.7, (0, 255, 255), 2)

    # Display output
    cv2.imshow("Virtual Painter", img)

    # Save canvas
    key = cv2.waitKey(1)
    if key == ord('s'):
        filename = f"art_{int(time.time())}.png"
        cv2.imwrite(filename, imgCanvas)
        print(f"[INFO] Saved as {filename}")
    if key == ord('q'):
        break