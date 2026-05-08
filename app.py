import streamlit as st

# 🔥 MUST be first Streamlit command
st.set_page_config(
    page_title="AutoCaptionAI",
    page_icon="🎬"
)

import os
from faster_whisper import WhisperModel

# Load model (use tiny for speed)
model = WhisperModel("tiny", compute_type="int8")

# Function to generate subtitles
def generate_subtitled_video(video_path):
    segments, _ = model.transcribe(video_path)

    video = VideoFileClip(video_path)
    subtitles = []

    for segment in segments:
        txt = segment.text
        start = segment.start
        end = segment.end

import cv2

def generate_subtitled_video(video_path):
    segments, _ = model.transcribe(video_path)

    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(
        "output.mp4",
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
                break   # 🔥 IMPORTANT

        if text:
            # 🔥 BLACK BACKGROUND BOX
            cv2.rectangle(frame, (20, height - 120), (width - 20, height - 40), (0, 0, 0), -1)

            # 🔥 BIG CLEAR TEXT
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

    return "output.mp4"


    final = CompositeVideoClip([video] + subtitles)
    output_path = "output.mp4"
    final.write_videofile(output_path)

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
    output_path = generate_subtitled_video(video_path)

    import os

    if os.path.exists(output_path):
        st.success("Video created!")
        st.video(output_path)
    else:
        st.error("Video not generated ❌")
