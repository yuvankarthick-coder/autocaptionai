import streamlit as st

st.set_page_config(page_title="AutoCaptionAI", page_icon="🎬")

import os
from faster_whisper import WhisperModel

# Load model
model = WhisperModel("tiny", compute_type="int8")


import imageio

def generate_subtitled_video(video_path):
    try:
        segments, _ = model.transcribe(video_path)

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps is None:
            fps = 24

        output_path = "output.mp4"

        writer = imageio.get_writer(output_path, fps=fps)

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
                h, w, _ = frame.shape

                cv2.rectangle(frame, (20, h - 120), (w - 20, h - 40), (0, 0, 0), -1)

                cv2.putText(
                    frame,
                    text,
                    (40, h - 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA
                )

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            writer.append_data(frame)   # ✅ STREAM write (no RAM issue)

            frame_count += 1

        cap.release()
        writer.close()

        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            return output_path

        return None

    except Exception as e:
        print("ERROR:", e)
        return None 

def generate_subtitled_video(video_path):
    try:
        segments, _ = model.transcribe(video_path)

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps is None:
            fps = 24

        output_path = "output.mp4"

        writer = imageio.get_writer(output_path, fps=fps)

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
                h, w, _ = frame.shape

                cv2.rectangle(frame, (20, h - 120), (w - 20, h - 40), (0, 0, 0), -1)

                cv2.putText(
                    frame,
                    text,
                    (40, h - 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA
                )

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            writer.append_data(frame)   # ✅ STREAM write (no RAM issue)

            frame_count += 1

        cap.release()
        writer.close()

        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            return output_path

        return None

    except Exception as e:
        print("ERROR:", e)
        return Noneko

def generate_subtitled_video(video_path):
    try:
        segments, _ = model.transcribe(video_path)

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps is None:
            fps = 24

        output_path = "output.mp4"

        writer = imageio.get_writer(output_path, fps=fps)

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
                h, w, _ = frame.shape

                cv2.rectangle(frame, (20, h - 120), (w - 20, h - 40), (0, 0, 0), -1)

                cv2.putText(
                    frame,
                    text,
                    (40, h - 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA
                )

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            writer.append_data(frame)   # ✅ STREAM write (no RAM issue)

            frame_count += 1

        cap.release()
        writer.close()

        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            return output_path

        return None

    except Exception as e:
        print("ERROR:", e)
        return Non

def generate_subtitled_video(video_path):
    try:
        segments, _ = model.transcribe(video_path)

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps is None:
            fps = 24

        output_path = "output.mp4"

        writer = imageio.get_writer(output_path, fps=fps)

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
                h, w, _ = frame.shape

                cv2.rectangle(frame, (20, h - 120), (w - 20, h - 40), (0, 0, 0), -1)

                cv2.putText(
                    frame,
                    text,
                    (40, h - 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA
                )

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            writer.append_data(frame)   # ✅ STREAM write (no RAM issue)

            frame_count += 1

        cap.release()
        writer.close()

        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            return output_path

        return None

    except Exception as e:
        print("ERROR:", e)
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
