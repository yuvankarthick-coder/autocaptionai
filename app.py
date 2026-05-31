import streamlit as st
import os
import cv2
import imageio
import textwrap
import subprocess
from faster_whisper import WhisperModel

# Page config
st.set_page_config(
    page_title="AutoCaptionAI",
    page_icon="🎬"
)

# Load Whisper model
@st.cache_resource
def load_model():
    return WhisperModel("tiny", compute_type="int8")

model = load_model()


def generate_subtitled_video(video_path):
    try:
        # Generate subtitles
        segments, info = model.transcribe(video_path)
        segments = list(segments)

        # Open video
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            st.error("Could not open video.")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 24

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = "output.mp4"

        writer = imageio.get_writer(
            output_path,
            fps=fps
        )

        frame_count = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            current_time = frame_count / fps

            subtitle_text = ""

            for seg in segments:
                if seg.start <= current_time <= seg.end:
                    subtitle_text = seg.text.strip()
                    break

            if subtitle_text:

                wrapped_text = textwrap.wrap(
                    subtitle_text,
                    width=30
                )

                line_count = max(1, len(wrapped_text))
                box_height = 50 + (line_count * 35)

                cv2.rectangle(
                    frame,
                    (20, height - box_height - 20),
                    (width - 20, height - 20),
                    (0, 0, 0),
                    -1
                )

                y = height - box_height + 35

                for line in wrapped_text:
                    cv2.putText(
                        frame,
                        line,
                        (40, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA
                    )
                    y += 35

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            writer.append_data(frame_rgb)

            frame_count += 1

        cap.release()
        writer.close()

        # Merge original audio using FFmpeg
        final_output = "final_output.mp4"

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", output_path,
                "-i", video_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0",
                final_output
            ],
            check=True
        )

        if os.path.exists(final_output):
            return final_output

        return None

    except Exception as e:
        st.error(f"Error: {e}")
        return None


# UI
st.title("🎬 AutoCaptionAI")
st.write("Generate subtitles for your videos using AI ⚡")

uploaded_file = st.file_uploader(
    "Upload Video",
    type=["mp4", "mov", "avi", "mkv"]
)

if uploaded_file is not None:

    with open("input.mp4", "wb") as f:
        f.write(uploaded_file.read())

    st.subheader("Original Video")
    st.video("input.mp4")

    if st.button("🚀 Generate Subtitles"):

        with st.spinner("Generating subtitles..."):

            output_file = generate_subtitled_video("input.mp4")
            srt_file = generate_srt("input.mp4")
        if output_file:
            st.success("✅ Video created successfully!")

            st.subheader("Subtitled Video")
            st.video(output_file)

            with open(output_file, "rb") as f:
                st.download_button(
                    "⬇ Download Video",
                    data=f,
                    file_name="subtitled_video.mp4",
                    mime="video/mp4"
                )

        else:
            st.error("❌ Failed to generate video.")
