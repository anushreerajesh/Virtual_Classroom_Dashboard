import cv2
from deepface import DeepFace

# Load your video file here
video = cv2.VideoCapture("student_feed.mp4")  # Replace with your actual file name

frame_count = 0
while True:
    ret, frame = video.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 10 != 0:
        continue  # Process every 10th frame to improve speed

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        cv2.putText(frame, f"Emotion: {emotion}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    except:
        cv2.putText(frame, "Emotion: Unknown", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Emotion Detection", frame)
    if cv2.waitKey(1) == 27:  # ESC key to exit
        break

video.release()
cv2.destroyAllWindows()
