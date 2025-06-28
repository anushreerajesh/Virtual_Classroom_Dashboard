# from deepface import DeepFace
import csv
from datetime import datetime
import pandas as pd
import os

def evaluate_teacher_from_csv():
    try:
        df = pd.read_csv("structured_log.csv")
        latest_frame = df["Frame"].max()
        latest_data = df[df["Frame"] == latest_frame]

        total_students = 3  # fixed total
        focused_students = latest_data[latest_data["Focus"].str.lower() == "focused"]
        focused_count = len(focused_students)

        focus_percent = round((focused_count / total_students) * 100, 2)
        status = "Effective" if focus_percent >= 50 else "Not Effective"

        return {
            "focus_percent": focus_percent,
            "status": status,
            "total_students": total_students
        }
    except Exception as e:
        return {"error": str(e)}

def evaluate_teacher_effectiveness(video_path="student_feed.mp4", output_csv="structured_log.csv"):
    video = cv2.VideoCapture(video_path)
    frame_count = 0

    csv_file = open(output_csv, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "Frame", "Timestamp", "Student ID", "Emotion", "Focus", "Focus % (overall)", "Total Students"
    ])

    total_focus_percent = 0
    focus_samples = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 10 != 0:
            continue

        height, width = frame.shape[:2]
        total_faces = 0
        concentrating_count = 0
        face_data = []

        try:
            results = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
            if not isinstance(results, list):
                results = [results]

            for i, face in enumerate(results, start=1):
                region = face["region"]
                x, y, w, h = region["x"], region["y"], region["w"], region["h"]
                emotion = face["dominant_emotion"]
                total_faces += 1

                face_center_x = x + w // 2
                frame_center_x = width // 2
                offset = abs(face_center_x - frame_center_x)

                if offset < 100:
                    focus = "Focused"
                    concentrating_count += 1
                else:
                    focus = "Distracted"

                face_data.append((i, emotion, focus))

        except Exception as e:
            print("DeepFace error:", e)

        focus_percent = round((concentrating_count / total_faces) * 100, 3) if total_faces > 0 else 0
        timestamp = datetime.now().strftime("%H:%M:%S")

        if total_faces > 0:
            total_focus_percent += focus_percent
            focus_samples += 1

        for student_id, emotion, focus in face_data:
            csv_writer.writerow([
                frame_count, timestamp, student_id, emotion, focus, focus_percent, total_faces
            ])

    video.release()
    csv_file.close()

    return round(total_focus_percent / focus_samples, 3) if focus_samples > 0 else 0.0
