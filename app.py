import streamlit as st
import os
import cv2
import imageio
import textwrap
import subprocess
from faster_whisper import WhisperModel

# Page Config
st.set_page_config(
    page_title="AutoCaptionAI - Free AI subtitle generator",
    page_icon="🎬"
)

# Load Whisper Model
@st.cache_resource
def load_model():
    return WhisperModel("tiny", compute_type="int8")

model = load_model()

# ----------------------------
# CUSTOM UI
# ----------------------------

with st.sidebar:
    
    st.image("https://cdn-icons-png.flaticon.com/512/2991/2991148.png",
        width=100
    )

    st.title("AutoCaptionAI")

    st.info("Generate AI subtitles for your videos.")

hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""

st.markdown(
    hide_st_style,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="
    padding:30px;
    border-radius:20px;
    background:
    linear-gradient(135deg,#0ea5e9,#8b5cf6);
    text-align:center;
    color:white;
    margin-bottom:20px;
    ">
    <h1>🎬 AutoCaptionAI</h1>
    <p>Create AI Powered subtitles for
    Youtube shorts,TikTok and Instagram 
    Reels</p>
    </div>
    """, unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.metric("🌍 Languages", "4")

with col2:
    st.metric("🎨 Styles", "3")

col1, col2, col3 = st.columns(3)

with col1:
    st.success("⚡ Fast AI Captions")

with col2:
    st.success("🌍 Multi-Language")

with col3:
    st.success("📄 SRT Download")
             
# ----------------------------
# Timestamp Formatter
# ----------------------------
def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)

    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


# ----------------------------
# Get Transcription
# ----------------------------
def get_segments(video_path, language):

    lang_map = {
        "English": "en",
        "Tamil": "ta",
        "Hindi": "hi",
        "Telugu": "te",
        "Malayalam": "ml",
        "Kannada": "kn",
    }
    
    if language == "Auto Detect":
        segments, _ = model.transcribe(video_path)
    else:
        segments, _ = model.transcribe(
            video_path,
            language=lang_map[language]
        )

    return list(segments)


# ----------------------------
# Generate SRT
# ----------------------------
def generate_srt(video_path, language):

    segments = get_segments(video_path, language)

    srt_path = "subtitles.srt"

    with open(srt_path, "w", encoding="utf-8") as f:

        for i, seg in enumerate(segments, start=1):

            start = format_timestamp(seg.start)
            end = format_timestamp(seg.end)

            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{seg.text.strip()}\n\n")

    return srt_path


# ----------------------------
# Generate Video
# ----------------------------
def generate_subtitled_video(
    video_path,
    subtitle_style,
    language,
    font_size,
    subtitle_position,
    subtitle_color,
    background_color
):

    try:

        # Convert background color
        bg_hex = background_color.lstrip("#")

        bg_color = (bg_rgb[2],
                    bg_rgb[1],
                    bg_rgb[0]
                   )

        text_hex = subtitle_color.lstrip("#")

        text_bgr = tuple(
            int(text_hex[i:i+2], 16)
            for i in (0,2,4)
        )

        text_color = (
            text_bgr[2],
            text_bgr[1],
            text_bgr[0]
        )   

        segments = get_segments(
            video_path,
            language
        )

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

                line_count = max(
                    1,
                    len(wrapped_text)
                )

                box_height = 50 + (
                    line_count * 35
                )

                cv2.rectangle(
                    frame,
                    (20, height - box_height - 20),
                    (width - 20, height - 20),
                    bg_color,
                    -1
                )

            if  subtitle_position == "Top":
                y = 80

            elif subtitle_position == "Center":
                y = height // 2

            else:
                y = height - box_height + 35

                for line in wrapped_text:

                    if subtitle_style == "YouTube Shorts":

                        cv2.putText(
                            frame,
                            line,
                            (40, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_size,
                            text_color,
                            2,
                            cv2.LINE_AA
                        )

                    elif subtitle_style == "TikTok":

                        cv2.putText(
                            frame,
                            line,
                            (40, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_size,
                            (255, 255, 255),
                            3,
                            cv2.LINE_AA
                        )

                    else:

                        (text_width, _), _ = cv2.getTextSize(
                            line,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_size,
                            2
                        )

                        x = (width - text_width) // 2

                        cv2.putText(
                            frame,
                            line,
                            (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA
                        )

                    y += 35

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            writer.append_data(frame_rgb)

            frame_count += 1

        cap.release()
        writer.close()

        final_output = "final_output.mp4"

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                output_path,
                "-i",
                video_path,
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
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


# ----------------------------
# UI
# ----------------------------

with st.sidebar:

    st.title("⚙️ Settings")

    subtitle_style = st.selectbox(
        "🎨 Subtitle style",
        ["Youtube Shorts", "TikTok","Instagram Reels"]
        )

    language = st.selectbox(
        "🌍 Language",
        ["Auto Detect", "English", "Tamil", "Hindi"]
    )

    font_size = st.slider(
        "🔤 Font Size",
        0.5,
        3.0,
        1.0,
    )

    subtitle_position = st.selectbox(
        "📍 Subtitle Position",
        ["Bottom","Center","Top"]
        
        )

    subtitle_color = st.color_picker(
        "🎨 Subtitle Text Color",
        "#FFFFFF"
        )

    background_color = st.color_picker(
        "⬛ Background Color",
        "#000000"
        )

st.markdown("---")

st.info("""AutoCaptionAI creates AI-powered subtitles for Youtube shorts, tiktok and Instagram Reels.""")

st.subheader("🎥 Upload your video")
st.write("Supports MP4, MOV, AVI and MKV files")

uploaded_file = st.file_uploader(
    "Upload Video",
    type=[
        "mp4",
        "mov",
        "avi",
        "mkv"
    ]
)

if uploaded_file is not None:

    with open("input.mp4", "wb") as f:
        f.write(uploaded_file.read())

    st.subheader("Original Video")
    st.video("input.mp4")

    if st.button("🚀 Generate Subtitles"):

        with st.spinner(
            "Generating subtitles..."
        ):

            progress = st.progress(0)

            progress.progress(20)

            output_file = generate_subtitled_video(
                "input.mp4",
                subtitle_style,
                language,
                font_size,
                subtitle_position,
                subtitle_color,
                background_color
            )

            progress.progress(80)

            srt_file = generate_srt(
                "input.mp4",
                language
            )

        progress.progress(100)

        if output_file:

            st.success(
                "✅ Video created successfully!"
            )

            st.subheader(
                "Subtitled Video"
            )

            st.video(output_file)

            with open(
                output_file,
                "rb"
            ) as f:

                st.download_button(
                    "⬇ Download Video",
                    data=f,
                    file_name="subtitled_video.mp4",
                    mime="video/mp4"
                )

            with open(
                srt_file,
                "rb"
            ) as f:

                st.download_button(
                    "📄 Download SRT",
                    data=f,
                    file_name="subtitles.srt",
                    mime="text/plain"
                )

        else:

            st.error(
                "❌ Failed to generate video."
            )

            st.markdown("---")
            
            st.caption("© 2026 AutoCaptionAI • Powered by Whisper AI")
