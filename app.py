import streamlit as st

st.set_page_config(page_title="AutoCaptionAI", page_icon="🎬")

import os
import cv2
from faster_whisper import WhisperModel

# Load model
model = WhisperModel("tiny", compute_type="int8")


def generate_subtitled_video(video_path):
    segments, _ = model.transcribe(video_path)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 🔥 FIX 1: fallback fps
    if fps == 0 or fps is None:
        fps = 24

    output_path = "output.mp4"

    # 🔥 FIX 2: better codec
    fourcc = cv2.VideoWriter_fourcc(*"avc1")

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

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
            cv2.rectangle(frame, (20, height - 120), (width - 20, height - 40), (0, 0, 0), -1)

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

    # 🔥 FIX 3: ensure file exists + delay write
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_path
    else:
        return None


# UI
st.title("🎬 AutoCaptionAI")
st.write("Generate subtitles for your videos instantly using AI ⚡")

uploaded_file = st.file_uploader("Upload video", type=["mp4", "mov"])

if uploaded_file:
    with open("input.mp4", "wb") as f:
        f.write(uploaded_file.read())

    st.video("input.mp4")

    if st.button("Generate Subtitles"):
        with st.spinner("Processing... ⏳"):
            output_path = generate_subtitled_video("input.mp4")

        if output_path:
            st.success("Video created!")
            st.video(output_path)
        else:
            st.error("Video failed ❌")
