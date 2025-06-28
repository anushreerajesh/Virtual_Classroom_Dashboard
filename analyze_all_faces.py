import cv2
import pandas as pd
from deepface import DeepFace

def get_student_data():
    video = cv2.VideoCapture("student_feed.mp4")

    frame_count = 0
    data = []

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 10 != 0:
            continue  # Skip frames for performance

        try:
            results = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
            if not isinstance(results, list):
                results = [results]

            for face in results:
                region = face["region"]
                x, y, w, h = region["x"], region["y"], region["w"], region["h"]
                emotion = face["dominant_emotion"]

                # Face center point
                face_center_x = x + w // 2
                frame_center_x = frame.shape[1] // 2

                # Concentration logic based on horizontal offset
                offset = abs(face_center_x - frame_center_x)
                concentration = "Attentive" if offset < 100 else "Distracted"

                # Append to data list
                data.append({
                    "Frame": frame_count,
                    "Dominant Emotion": emotion,
                    "Concentration": concentration
                })

        except Exception as e:
            print("Error:", e)
            continue

    video.release()
    # Optionally, group by frame and get average if multiple faces
    df = pd.DataFrame(data)

    if df.empty:
        return pd.DataFrame(columns=["Frame", "Dominant Emotion", "Concentration"])

    # Just take 1st face per frame for now (you can extend this for multiple students)
    df_summary = df.groupby("Frame").first().reset_index()
    df_summary["Student"] = ["Student " + str(i+1) for i in range(len(df_summary))]

    return df_summary[["Student", "Dominant Emotion", "Concentration"]]
