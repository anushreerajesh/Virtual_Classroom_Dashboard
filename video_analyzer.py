import cv2
import os
import pandas as pd
from gaze_tracking import GazeTracking
from deepface import DeepFace

print("✅ video_analyzer.py was loaded")

gaze = GazeTracking()

def analyze_video(video_path, output_csv="data/engagement_log.csv"):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Failed to open video file: {video_path}")
        return pd.DataFrame()

    frame_count = 0
    student_data = []

    print("▶️ Starting video analysis...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ End of video or failed to read frame at frame {frame_count}")
            break

        frame_count += 1
        print(f"🔍 Processing frame {frame_count}")

        gaze.refresh(frame)
        attention_status = gaze.is_center()

        try:
            print("🧠 Running DeepFace...")
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            dominant_emotion = analysis[0]['dominant_emotion']
            print(f"✅ DeepFace Result: {dominant_emotion}")
        except Exception as e:
            print(f"⚠️ DeepFace failed: {str(e)}")
            dominant_emotion = "unknown"

        if frame_count % 10 == 0:
            student_data.append({
                "frame": frame_count,
                "attention": "Yes" if attention_status else "No",
                "emotion": dominant_emotion
            })

    cap.release()

    if not student_data:
        print("❌ No student data collected. Exiting.")
        return pd.DataFrame()

    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame(student_data)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved analysis to: {output_csv}")
    print(f"📝 Total frames processed: {frame_count}")
    return df

