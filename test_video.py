from video_analyzer import analyze_video

print("▶️ test_video.py is running")
df = analyze_video("sample_class.mp4")
print("✅ test complete")
print(df.head())