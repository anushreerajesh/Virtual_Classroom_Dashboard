from gaze_tracking import GazeTracking
from deepface import DeepFace
import cv2
import time

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

distraction_start_time = None
distraction_threshold = 2  # seconds

while True:
    _, frame = webcam.read()
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    text = ""

    # Detect gaze direction
    if gaze.is_blinking():
        text = "Blinking"
        distraction_start_time = None

    elif gaze.is_center():
        text = "Looking center - FOCUSED"
        distraction_start_time = None

    elif gaze.is_left() or gaze.is_right():
        if gaze.is_left():
            text = "Looking left"
        else:
            text = "Looking right"

        if distraction_start_time is None:
            distraction_start_time = time.time()
        elif time.time() - distraction_start_time > distraction_threshold:
            text += " - NOT CONCENTRATING"
            cv2.putText(frame, "⚠️ You are not concentrating!", (90, 130), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 2)
    else:
        text = "Undetected"
        distraction_start_time = None

    # Display gaze direction
    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (255, 0, 0), 2)

    # Emotion detection (optional)
    try:
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = analysis[0]['dominant_emotion']
        cv2.putText(frame, f"Emotion: {emotion}", (90, 100), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2)
    except:
        cv2.putText(frame, "Emotion: Unknown", (90, 100), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 2)

    # Show frame
    cv2.imshow("Engagement Tracker", frame)

    # Exit on ESC
    if cv2.waitKey(1) == 27:
        break

webcam.release()
cv2.destroyAllWindows()