import streamlit as st
import pandas as pd
import re
import os
from fuzzywuzzy import fuzz
from teacher_effectiveness import evaluate_teacher_from_csv, evaluate_teacher_effectiveness

# -------------------- Page Setup -------------------- #
st.set_page_config(page_title="Virtual Classroom Dashboard", layout="wide")
st.title("📚 Virtual Classroom Dashboard")

# -------------------- School Info -------------------- #
school_info = {
    "name": "Sunrise Public School",
    "principal": "Mrs. Anjali Sharma",
    "timings": "8:00 AM to 2:00 PM (Mon–Fri)",
    "location": "14th Cross, Indiranagar, Bengaluru",
    "students": "around 450",
    "reviews": "We have great feedback from parents and students!"
}

# -------------------- CSV Check -------------------- #
if not os.path.exists("structured_log.csv"):
    st.warning("Analyzing video data. Please wait...")
    avg_focus = evaluate_teacher_effectiveness()

# -------------------- View Selector -------------------- #
st.markdown("---")
view = st.selectbox("📋 How was your child today?", [
    "Student Performance",
    "Teacher Engagement",
    "Attendance Tracker",
    "Marks Overview"
])

if view == "Marks Overview":
    st.header("📚 Student Marks Analysis")
    
    marks_data = pd.DataFrame({
        "Student ID": [1, 2],
        "Name": ["Aditi", "Karan"],
        "Math": [85, 70],
        "Science": [78, 65],
        "English": [88, 72],
    })

    marks_data["Average"] = marks_data[["Math", "Science", "English"]].mean(axis=1)

    st.subheader("📊 Student Marks Overview")
    st.dataframe(marks_data)

    st.subheader("📈 Average Marks per Student")
    st.bar_chart(marks_data.set_index("Name")["Average"])

# -------------------- Student Performance -------------------- #
elif view == "Student Performance":
    st.subheader("🎓 Student Performance")
    try:
        df = pd.read_csv("structured_log.csv")

        # Fetch last 30 frames to increase chance of seeing more students
        recent_frames = df["Frame"].drop_duplicates().sort_values(ascending=False).head(30)
        student_data = df[df["Frame"].isin(recent_frames)]

        # Optional: remove duplicates if you want only 1 row per student
        student_data = student_data.sort_values(by="Frame", ascending=False).drop_duplicates(subset=["Student ID"])

        # Rename and display only the top 3 students seen most recently
        student_data["Focus"] = student_data.apply(
        lambda row: "Focused" if row["Emotion"].lower() == "happy" else row["Focus"], axis=1
    )

        display_data = student_data[["Student ID", "Emotion", "Focus"]].rename(
        columns={"Emotion": "Dominant Emotion"}
        ).head(3)
        st.dataframe(display_data, use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ Could not load student data: {e}")

# -------------------- Teacher Engagement -------------------- #
elif view == "Teacher Engagement":
    st.subheader("👩‍🏫 Teacher Effectiveness")

    try:
        df = pd.read_csv("structured_log.csv")
        latest_frame = df["Frame"].max()
        latest_data = df[df["Frame"] == latest_frame]

        total_students = 2
        focused_students = latest_data[latest_data["Focus"].str.lower() == "focused"]
        focused_count = len(focused_students)
        print(focused_students)

        if total_students > 0:
            focus_percent = round((focused_count / total_students) * 100, 2)
            if focus_percent >= 50:
                st.success(f"✅ Teacher is effective ({focus_percent}% students focused)")
            else:
                st.warning(f"⚠️ Teacher engagement low ({focus_percent}% students focused out of {total_students})")
        else:
            st.info("ℹ️ No student data available yet.")

    except Exception as e:
        st.error(f"⚠️ Could not evaluate teacher effectiveness: {e}")

# -------------------- Attendance Tracker -------------------- #
elif view == "Attendance Tracker":
    st.subheader("📌 Attendance Tracker")

    # Try to use the latest student data from the structured_log
    try:
        df = pd.read_csv("structured_log.csv")
        latest_frame = df["Frame"].max()
        attendance_df = df[df["Frame"] == latest_frame]
        attendance_df = attendance_df[["Student ID"]].drop_duplicates()
        attendance_df["Attendance"] = "Present"

        # Add mock absent students (optional)
        all_students = list(range(1, 3))  # Assume 2 students total
        for sid in all_students:
            if sid not in attendance_df["Student ID"].values:
                attendance_df = pd.concat([
                    attendance_df,
                    pd.DataFrame({"Student ID": [sid], "Attendance": ["Absent"]})
                ], ignore_index=True)

        st.dataframe(attendance_df.sort_values("Student ID").reset_index(drop=True), use_container_width=True)
    except Exception as e:
        st.error(f"⚠️ Could not load attendance data: {e}")



# -------------------- Chatbot Section -------------------- #
st.markdown("---")
st.markdown("### 💬 Ask the School Assistant")

faq = {
    "hi": "👋 Hi there! Welcome to the school assistant.",
    "hello": "👋 Hello! How can I help you today?",
    "how are you": "🤖 I'm just code, but ready to help! 😊",
    "where is the school located": f"📍 Location: {school_info['location']}",
    "what is the school name": f"🏫 The school is called {school_info['name']}",
    "who is the principal": f"👩‍🏫 Principal: {school_info['principal']}",
    "what are the timings": f"⏰ Timings: {school_info['timings']}",
    "how many students": f"🎓 {school_info['students']} students are currently enrolled.",
    "tell me about the school": f"📘 {school_info['name']} is a vibrant learning community located at {school_info['location']}.",
    "does the school have good reviews": f"⭐ Yes! {school_info['reviews']}",
}

def clean_text(text):
    return re.sub(r"[^\w\s]", "", text.lower().strip())

def find_best_match(question):
    question = clean_text(question)
    best_score = 0
    best_match = None
    for key in faq:
        score = fuzz.partial_ratio(clean_text(key), question)
        if score > best_score:
            best_score = score
            best_match = key
    return best_match if best_score > 60 else None

user_input = st.chat_input("I am here to help you. Feel free to ask me anything!")

if user_input:
    match = find_best_match(user_input)
    if match:
        st.write(f"🤖 {faq[match]}")
    else:
        st.write("🤔 I'm not sure how to answer that. Try asking about the school, principal, or timings.")

# -------------------- Footer -------------------- #
st.markdown("---")
st.caption("📍 Sunrise Public School ")
