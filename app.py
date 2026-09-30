import os
import cv2
import imageio
import tempfile
import subprocess
import textwrap
import shutil
from pathlib import Path

import streamlit as st
from faster_whisper import WhisperModel
from openai import OpenAI

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="AutoCaptionAI - AI Subtitle Generator",
    page_icon="logo.png",
    layout="wide",
)

# ----------------------------
# Constants
# ----------------------------
LANG_MAP = {
    "English": "en",
    "Tamil": "ta",
    "Hindi": "hi",
    "Telugu": "te",
    "Malayalam": "ml",
    "Kannada": "kn",
}

SUPPORTED_LANGUAGES = ["Auto Detect", *LANG_MAP.keys()]
TRANSLATION_LANGUAGES = ["None", *LANG_MAP.keys()]

# ----------------------------
# Model
# ----------------------------
@st.cache_resource
def load_model():
    # PyAV is pinned below 19 in requirements.txt because faster-whisper
    # currently calls av.open(..., metadata_errors="ignore").
    return WhisperModel("tiny", device="cpu", compute_type="int8")


# ----------------------------
# Helpers
# ----------------------------
def get_openai_client():
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def format_timestamp(seconds):
    total_ms = max(0, int(round(float(seconds) * 1000)))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1_000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def get_segments(video_path, language):
    model = load_model()

    kwargs = {}
    if language != "Auto Detect":
        kwargs["language"] = LANG_MAP[language]

    segments, info = model.transcribe(
        video_path,
        beam_size=5,
        vad_filter=True,
        **kwargs,
    )

    segments = list(segments)
    full_transcript = " ".join(
        segment.text.strip()
        for segment in segments
        if segment.text.strip()
    ).strip()

    detected_language = getattr(info, "language", None)
    return segments, full_transcript, detected_language


def write_srt(segments, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            text = seg.text.strip()
            if not text:
                continue
            f.write(
                f"{i}\n"
                f"{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}\n"
                f"{text}\n\n"
            )


def hex_to_bgr(hex_color):
    value = hex_color.lstrip("#")
    if len(value) != 6:
        return (0, 0, 0)
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    return (b, g, r)


def choose_font(font_style):
    if font_style == "Bold":
        return cv2.FONT_HERSHEY_DUPLEX
    if font_style == "Classic":
        return cv2.FONT_HERSHEY_TRIPLEX
    return cv2.FONT_HERSHEY_SIMPLEX


def wrap_text_to_width(text, font, font_size, thickness, max_width):
    words = text.split()
    if not words:
        return []

    lines = []
    current = words[0]

    for word in words[1:]:
        candidate = f"{current} {word}"
        (width, _), _ = cv2.getTextSize(
            candidate, font, font_size, thickness
        )
        if width <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word

    lines.append(current)
    return lines


def draw_subtitle(
    frame,
    text,
    position,
    font,
    font_size,
    text_color,
    background_color,
    style,
):
    if not text:
        return

    height, width = frame.shape[:2]
    thickness = 3 if style == "TikTok" else 2
    max_text_width = max(200, width - 80)

    lines = wrap_text_to_width(
        text,
        font,
        font_size,
        thickness,
        max_text_width,
    )
    if not lines:
        return

    line_height = max(30, int(45 * font_size))
    padding_x = 24
    padding_y = 18
    box_height = len(lines) * line_height + padding_y * 2

    text_sizes = [
        cv2.getTextSize(line, font, font_size, thickness)[0]
        for line in lines
    ]
    box_width = min(
        width - 30,
        max(max(size[0] for size in text_sizes) + padding_x * 2, 180),
    )

    if position == "Top":
        box_y1 = 20
    elif position == "Center":
        box_y1 = max(10, (height - box_height) // 2)
    else:
        box_y1 = max(10, height - box_height - 20)

    box_y2 = min(height - 1, box_y1 + box_height)
    box_x1 = max(10, (width - box_width) // 2)
    box_x2 = min(width - 10, box_x1 + box_width)

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (box_x1, box_y1),
        (box_x2, box_y2),
        background_color,
        -1,
    )
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    for index, line in enumerate(lines):
        (text_width, text_height), _ = cv2.getTextSize(
            line, font, font_size, thickness
        )
        x = max(10, (width - text_width) // 2)
        y = box_y1 + padding_y + text_height + index * line_height

        if style == "TikTok":
            color = (255, 255, 255)
        elif style == "Instagram Reels":
            color = text_color
        else:
            color = text_color

        # Black outline improves readability over bright video.
        cv2.putText(
            frame,
            line,
            (x, y),
            font,
            font_size,
            (0, 0, 0),
            thickness + 3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (x, y),
            font,
            font_size,
            color,
            thickness,
            cv2.LINE_AA,
        )


def add_watermark(frame):
    height, width = frame.shape[:2]
    text = "AutoCaptionAI"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)

    x = max(10, width - tw - 20)
    y = max(th + 10, height - 20)

    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        scale,
        (255, 255, 255),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        scale,
        (40, 40, 40),
        thickness,
        cv2.LINE_AA,
    )


def render_subtitled_video(
    video_path,
    output_path,
    segments,
    subtitle_style,
    font_size,
    subtitle_position,
    subtitle_color,
    background_color,
    font_style,
    watermark,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 24.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Could not read the video dimensions.")

    font = choose_font(font_style)
    text_color = hex_to_bgr(subtitle_color)
    bg_color = hex_to_bgr(background_color)

    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        codec="libx264",
        quality=7,
    )

    segment_index = 0
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = frame_count / fps

            while (
                segment_index < len(segments)
                and current_time > segments[segment_index].end
            ):
                segment_index += 1

            subtitle_text = ""
            if segment_index < len(segments):
                seg = segments[segment_index]
                if seg.start <= current_time <= seg.end:
                    subtitle_text = seg.text.strip()

            if subtitle_text:
                draw_subtitle(
                    frame,
                    subtitle_text,
                    subtitle_position,
                    font,
                    font_size,
                    text_color,
                    bg_color,
                    subtitle_style,
                )

            if watermark:
                add_watermark(frame)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            writer.append_data(frame_rgb)
            frame_count += 1

            if total_frames > 0 and frame_count % 10 == 0:
                yield frame_count / total_frames

    finally:
        cap.release()
        writer.close()


def merge_audio(video_without_audio, original_video, final_output):
    # Try to preserve original audio. If the source has no audio stream,
    # simply keep the rendered video.
    probe = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=index",
            "-of", "csv=p=0",
            str(original_video),
        ],
        capture_output=True,
        text=True,
    )

    has_audio = bool(probe.stdout.strip())

    if not has_audio:
        shutil.copyfile(video_without_audio, final_output)
        return

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", str(video_without_audio),
            "-i", str(original_video),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            str(final_output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def ai_request(prompt):
    client = get_openai_client()
    if client is None:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured in Streamlit Secrets."
        )

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


def generate_titles(transcript):
    return ai_request(
        "Generate 5 catchy YouTube Shorts titles based on this transcript. "
        "Return one title per line and do not number them.\n\n"
        f"Transcript:\n{transcript}"
    )


def generate_description(transcript):
    return ai_request(
        "Write a concise, engaging YouTube Shorts description based on this "
        "transcript. Include a natural call to action but do not invent facts.\n\n"
        f"Transcript:\n{transcript}"
    )


def generate_hashtags(transcript):
    return ai_request(
        "Generate 10 relevant social-media hashtags for this video based only "
        "on the transcript. Return hashtags separated by spaces.\n\n"
        f"Transcript:\n{transcript}"
    )


def translate_transcript(text, target_language):
    return ai_request(
        f"Translate the following subtitle text into {target_language}. "
        "Preserve meaning and return only the translated text.\n\n{text}"
    )


# ----------------------------
# Session State
# ----------------------------
defaults = {
    "segments": None,
    "transcript": "",
    "detected_language": None,
    "translated_segments": None,
    "output_file": None,
    "srt_file": None,
    "workdir": None,
    "titles": "",
    "description": "",
    "hashtags": "",
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)

# ----------------------------
# UI
# ----------------------------
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=120)

    st.title("⚙️ Settings")

    subtitle_style = st.selectbox(
        "🎨 Subtitle style",
        ["YouTube Shorts", "TikTok", "Instagram Reels"],
    )

    language = st.selectbox(
        "🌍 Language",
        SUPPORTED_LANGUAGES,
    )

    font_style = st.selectbox(
        "🔤 Font Style",
        ["Simple", "Bold", "Classic"],
    )

    font_size = st.slider(
        "🔤 Font Size",
        0.5,
        2.5,
        1.0,
        0.1,
    )

    subtitle_position = st.selectbox(
        "📍 Subtitle Position",
        ["Bottom", "Center", "Top"],
    )

    subtitle_color = st.color_picker(
        "🎨 Subtitle Text Color",
        "#FFFFFF",
    )

    background_color = st.color_picker(
        "⬛ Background Color",
        "#000000",
    )

    add_watermark = st.checkbox(
        "🏷️ Add AutoCaptionAI Watermark",
        value=True,
    )

    target_language = st.selectbox(
        "🌍 Translate To",
        TRANSLATION_LANGUAGES,
    )

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="
        padding:30px;
        border-radius:20px;
        background:linear-gradient(135deg,#0ea5e9,#8b5cf6);
        text-align:center;
        color:white;
        margin-bottom:20px;">
        <h1>🎬 AutoCaptionAI</h1>
        <p>Create AI-powered subtitles for YouTube Shorts, TikTok and Instagram Reels</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if os.path.exists("logo.png"):
    st.image("logo.png", width=180)

col1, col2 = st.columns(2)
with col1:
    st.metric("🌍 Languages", str(len(LANG_MAP)))
with col2:
    st.metric("🎨 Styles", "3")

col1, col2, col3 = st.columns(3)
with col1:
    st.success("⚡ Fast AI Captions")
with col2:
    st.success("🌍 Multi-Language")
with col3:
    st.success("📄 SRT Download")

st.markdown("---")
st.info(
    "AutoCaptionAI creates AI-powered subtitles for YouTube Shorts, "
    "TikTok and Instagram Reels."
)

st.subheader("🎥 Upload your video")
st.write("Supports MP4, MOV, AVI and MKV files")

uploaded_file = st.file_uploader(
    "Upload Video",
    type=["mp4", "mov", "avi", "mkv"],
)

if uploaded_file is not None:
    # Create a fresh per-upload working directory.
    if st.session_state.workdir is None:
        st.session_state.workdir = tempfile.mkdtemp(prefix="autocaptionai_")

    workdir = Path(st.session_state.workdir)
    input_path = workdir / "input_video"

    if not input_path.exists():
        input_path.write_bytes(uploaded_file.getbuffer())

    st.subheader("Original Video")
    st.video(str(input_path))

    if st.button("🚀 Generate Subtitles", type="primary"):
        try:
            # Reset outputs for this run.
            st.session_state.output_file = None
            st.session_state.srt_file = None
            st.session_state.titles = ""
            st.session_state.description = ""
            st.session_state.hashtags = ""

            progress = st.progress(0)
            status = st.empty()

            status.info("Loading Whisper and transcribing audio...")
            progress.progress(10)

            segments, transcript, detected = get_segments(
                str(input_path),
                language,
            )

            if not segments or not transcript:
                raise RuntimeError(
                    "No speech was detected in the uploaded video."
                )

            st.session_state.segments = segments
            st.session_state.transcript = transcript
            st.session_state.detected_language = detected

            progress.progress(35)
            status.info("Rendering subtitles onto the video...")

            rendered_path = workdir / "rendered_no_audio.mp4"
            final_path = workdir / "subtitled_video.mp4"
            srt_path = workdir / "subtitles.srt"

            render_generator = render_subtitled_video(
                str(input_path),
                str(rendered_path),
                segments,
                subtitle_style,
                font_size,
                subtitle_position,
                subtitle_color,
                background_color,
                font_style,
                add_watermark,
            )

            last_progress = 35
            for fraction in render_generator:
                value = 35 + int(max(0.0, min(1.0, fraction)) * 45)
                if value > last_progress:
                    progress.progress(value)
                    last_progress = value

            status.info("Restoring original audio...")
            merge_audio(
                rendered_path,
                input_path,
                final_path,
            )

            progress.progress(90)

            write_srt(segments, srt_path)

            st.session_state.output_file = str(final_path)
            st.session_state.srt_file = str(srt_path)

            progress.progress(100)
            status.success("✅ Subtitles generated successfully!")

        except Exception as exc:
            st.error(f"Could not generate subtitles: {exc}")

# ----------------------------
# Transcript + AI Assistant
# ----------------------------
if st.session_state.transcript:
    st.markdown("---")
    st.subheader("📝 Transcript")

    detected = st.session_state.detected_language
    if detected:
        st.caption(f"Detected language: {detected}")

    st.text_area(
        "Transcript",
        value=st.session_state.transcript,
        height=180,
        disabled=True,
    )

    st.subheader("📝 AI Content Assistant")

    ai_col1, ai_col2, ai_col3 = st.columns(3)

    with ai_col1:
        if st.button("✨ Generate Titles"):
            with st.spinner("Generating titles..."):
                try:
                    st.session_state.titles = generate_titles(
                        st.session_state.transcript
                    )
                except Exception as exc:
                    st.error(str(exc))

    with ai_col2:
        if st.button("📄 Generate Description"):
            with st.spinner("Generating description..."):
                try:
                    st.session_state.description = generate_description(
                        st.session_state.transcript
                    )
                except Exception as exc:
                    st.error(str(exc))

    with ai_col3:
        if st.button("🏷️ Generate Hashtags"):
            with st.spinner("Generating hashtags..."):
                try:
                    st.session_state.hashtags = generate_hashtags(
                        st.session_state.transcript
                    )
                except Exception as exc:
                    st.error(str(exc))

    if st.session_state.titles:
        st.subheader("Suggested Titles")
        st.code(st.session_state.titles)

    if st.session_state.description:
        st.subheader("Suggested Description")
        st.text_area(
            "Description",
            value=st.session_state.description,
            height=180,
        )

    if st.session_state.hashtags:
        st.subheader("Suggested Hashtags")
        st.code(st.session_state.hashtags)

    if target_language != "None":
        st.subheader("🌍 Translation")

        if st.button(f"Translate Subtitles to {target_language}"):
            with st.spinner("Translating subtitles..."):
                try:
                    translated = []
                    for seg in st.session_state.segments:
                        translated_text = translate_transcript(
                            seg.text.strip(),
                            target_language,
                        )
                        translated.append(
                            {
                                "start": seg.start,
                                "end": seg.end,
                                "text": translated_text,
                            }
                        )
                    st.session_state.translated_segments = translated
                    st.success("Translation generated.")
                except Exception as exc:
                    st.error(str(exc))

        if st.session_state.translated_segments:
            st.text_area(
                "Translated subtitles",
                value="\n".join(
                    item["text"]
                    for item in st.session_state.translated_segments
                ),
                height=180,
                disabled=True,
            )

# ----------------------------
# Downloads
# ----------------------------
if st.session_state.output_file and os.path.exists(
    st.session_state.output_file
):
    st.markdown("---")
    st.subheader("🎬 Subtitled Video")
    st.video(st.session_state.output_file)

    with open(st.session_state.output_file, "rb") as f:
        st.download_button(
            "⬇️ Download Video",
            data=f.read(),
            file_name="subtitled_video.mp4",
            mime="video/mp4",
        )

    if st.session_state.srt_file and os.path.exists(
        st.session_state.srt_file
    ):
        with open(st.session_state.srt_file, "rb") as f:
            st.download_button(
                "📄 Download SRT",
                data=f.read(),
                file_name="subtitles.srt",
                mime="text/plain",
            )
