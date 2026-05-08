import streamlit as st

# 🔥 MUST be first Streamlit command
st.set_page_config(
    page_title="AutoCaptionAI",
    page_icon="🎬"
)

import os
import cv2
from faster_whisper import WhisperModel

# Load model (optimized for CPU)
model = WhisperModel("tiny", compute_type="int8")


# 🎬 Function to generate subtitled video
def generate_subtitled_video(video_path):
    segments, _ = model.transcribe(video_path)

    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0 or fps is None:
    fps = 24  # fallback fix

    output_path = "output.mp4"

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        time_sec = frame_count / fps

        text = ""
        for seg in segments:
            if seg.start <= time_sec <= seg.end:
                text = seg.text
                break

        if text:
            # Black background box
            cv2.rectangle(
                frame,
                (20, height - 120),
                (width - 20, height - 40),
                (0, 0, 0),
                -1
            )

            # Subtitle text
            cv2.putText(
                frame,
                text,
                (40, height - 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    return output_path


# 🎬 UI
st.title("🎬 AutoCaptionAI")
st.write("Generate subtitles for your videos instantly using AI ⚡")

uploaded_file = st.file_uploader("Upload video", type=["mp4", "mov"])

if uploaded_file:
    with open("input.mp4", "wb") as f:
        f.write(uploaded_file.read())

    st.video("input.mp4")

    if st.button("Generate Subtitles"):
        output_path = generate_subtitled_video("input.mp4")

        if os.path.exists(output_path):
            st.success("Video created!")
            st.video(output_path)
        else:
            st.error("Video not generated ❌")

if not cap.isOpened():
    st.error("Failed to open video ❌")
    return None

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    st.error("Failed to open video ❌")
    return None

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0 or fps is None:
    fps = 24
