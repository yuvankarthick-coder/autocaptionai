import os
import re
import subprocess
import tempfile
import textwrap
from pathlib import Path

import cv2
import imageio
import streamlit as st
from faster_whisper import WhisperModel
from openai import OpenAI


# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AutoCaptionAI - AI Subtitle Generator",
    page_icon="logo.png",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
LANGUAGES = {
    "English": "en",
    "Tamil": "ta",
    "Hindi": "hi",
    "Telugu": "te",
    "Malayalam": "ml",
    "Kannada": "kn",
}

TRANSLATION_LANGUAGES = ["None", *LANGUAGES.keys()]

FONT_STYLES = {
    "Simple": cv2.FONT_HERSHEY_SIMPLEX,
    "Bold": cv2.FONT_HERSHEY_DUPLEX,
    "Classic": cv2.FONT_HERSHEY_TRIPLEX,
}

SUPPORTED_VIDEO_TYPES = ["mp4", "mov", "avi", "mkv"]

# Keep the maximum number of characters per line conservative. The actual
# wrapping below is pixel-aware, so this is only a first-pass safeguard.
MAX_CHARS_PER_LINE = 42


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def get_openai_client():
    """Create the OpenAI client only when an AI feature is requested."""
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def safe_filename(name: str) -> str:
    """Return a filesystem-safe filename."""
    name = Path(name).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name or "uploaded_video.mp4"


def hex_to_bgr(hex_color: str):
    """Convert #RRGGBB to an OpenCV BGR tuple."""
    value = hex_color.lstrip("#")
    if len(value) != 6:
        return (0, 0, 0)
    rgb = tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])


def format_timestamp(seconds: float) -> str:
    """Format seconds as an SRT timestamp without floating-point overflow."""
    total_ms = max(0, int(round(float(seconds) * 1000)))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def get_video_metadata(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps <= 0:
        fps = 24.0
    if width <= 0 or height <= 0:
        raise ValueError("Could not read the video's dimensions.")

    duration = frame_count / fps if frame_count > 0 else 0
    return fps, width, height, frame_count, duration


def has_audio_stream(video_path: str) -> bool:
    """Check whether the input contains an audio stream using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                video_path,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return bool(result.stdout.strip())
    except FileNotFoundError:
        # If ffprobe is unavailable, let the later ffmpeg step report the real
        # issue rather than crashing the UI here.
        return True


# -----------------------------------------------------------------------------
# Whisper
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Whisper model...")
def load_model():
    # tiny/int8 is a good default for Streamlit Cloud. This can be upgraded
    # later to a model selector once the app is stable.
    return WhisperModel("tiny", compute_type="int8")


def transcribe_video(video_path: str, language: str):
    """Transcribe once and return serializable segment dictionaries."""
    model = load_model()

    whisper_language = LANGUAGES.get(language)
    kwargs = {}
    if whisper_language:
        kwargs["language"] = whisper_language

    segments, info = model.transcribe(video_path, **kwargs)
    segment_list = []

    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        segment_list.append(
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "text": text,
            }
        )

    transcript = " ".join(item["text"] for item in segment_list).strip()
    detected_language = getattr(info, "language", None)
    return segment_list, transcript, detected_language


# -----------------------------------------------------------------------------
# Subtitle generation
# -----------------------------------------------------------------------------
def generate_srt_from_segments(segments, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        for index, segment in enumerate(segments, start=1):
            f.write(f"{index}\n")
            f.write(
                f"{format_timestamp(segment['start'])} --> "
                f"{format_timestamp(segment['end'])}\n"
            )
            f.write(f"{segment['text'].strip()}\n\n")

    return output_path


def wrap_text_to_width(text: str, font, font_scale: float, thickness: int, max_width: int):
    """Wrap subtitle text using actual rendered pixel width."""
    words = textwrap.wrap(text, width=MAX_CHARS_PER_LINE) or [""]
    lines = []

    for paragraph in words:
        current = ""
        for word in paragraph.split():
            candidate = word if not current else f"{current} {word}"
            (text_width, _), _ = cv2.getTextSize(
                candidate, font, font_scale, thickness
            )
            if text_width <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)

    return lines or [""]


def draw_subtitle(
    frame,
    text,
    position,
    style,
    font,
    font_scale,
    text_color,
    background_color,
):
    """Draw one subtitle block safely at top, center, or bottom."""
    height, width = frame.shape[:2]
    thickness = 2 if style != "TikTok" else 3
    max_text_width = max(200, width - 80)
    lines = wrap_text_to_width(
        text, font, font_scale, thickness, max_text_width
    )

    line_height = max(30, int(45 * font_scale))
    padding_x = 24
    padding_y = 18
    block_height = len(lines) * line_height + padding_y * 2

    text_widths = [
        cv2.getTextSize(line, font, font_scale, thickness)[0][0]
        for line in lines
    ]
    content_width = max(text_widths) if text_widths else 0
    box_width = min(width - 40, content_width + padding_x * 2)

    if position == "Top":
        box_y = 20
    elif position == "Center":
        box_y = max(20, (height - block_height) // 2)
    else:
        box_y = max(20, height - block_height - 20)

    box_x = max(20, (width - box_width) // 2)
    box_right = min(width - 20, box_x + box_width)
    box_bottom = min(height - 20, box_y + block_height)

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (box_x, box_y),
        (box_right, box_bottom),
        background_color,
        -1,
    )
    # Slight transparency makes subtitles easier to read without hiding the
    # video completely.
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    total_text_height = len(lines) * line_height
    first_baseline = box_y + padding_y + (line_height + total_text_height) // 2 - line_height // 3

    for line_index, line in enumerate(lines):
        (line_width, line_height_px), _ = cv2.getTextSize(
            line, font, font_scale, thickness
        )
        x = (width - line_width) // 2
        y = first_baseline + line_index * line_height

        if style == "TikTok":
            # Dark outline + white text for readability.
            cv2.putText(
                frame,
                line,
                (x, y),
                font,
                font_scale,
                (0, 0, 0),
                thickness + 3,
                cv2.LINE_AA,
            )
            draw_color = (255, 255, 255)
        elif style == "Instagram Reels":
            draw_color = (255, 255, 255)
        else:
            draw_color = text_color

        cv2.putText(
            frame,
            line,
            (x, y),
            font,
            font_scale,
            draw_color,
            thickness,
            cv2.LINE_AA,
        )


def add_watermark_to_frame(frame):
    height, width = frame.shape[:2]
    text = "AutoCaptionAI"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2
    (text_width, text_height), _ = cv2.getTextSize(
        text, font, scale, thickness
    )
    x = max(10, width - text_width - 20)
    y = max(text_height + 10, height - 20)

    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def render_subtitled_video(
    video_path,
    segments,
    output_video_path,
    subtitle_style,
    font_size,
    subtitle_position,
    subtitle_color,
    background_color,
    font_style,
    add_watermark,
    progress_callback=None,
):
    """Render subtitles onto video frames. Does not transcribe again."""
    fps, width, height, frame_count, _ = get_video_metadata(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video for subtitle rendering.")

    font = FONT_STYLES.get(font_style, cv2.FONT_HERSHEY_SIMPLEX)
    text_color = hex_to_bgr(subtitle_color)
    background_bgr = hex_to_bgr(background_color)

    writer = imageio.get_writer(
        output_video_path,
        fps=fps,
        codec="libx264",
        macro_block_size=1,
    )

    segment_index = 0
    frame_number = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = frame_number / fps

            # Advance through finished subtitle segments instead of scanning
            # the entire transcript for every frame.
            while (
                segment_index < len(segments)
                and segments[segment_index]["end"] < current_time
            ):
                segment_index += 1

            subtitle_text = ""
            if segment_index < len(segments):
                current_segment = segments[segment_index]
                if current_segment["start"] <= current_time <= current_segment["end"]:
                    subtitle_text = current_segment["text"]

            if subtitle_text:
                draw_subtitle(
                    frame,
                    subtitle_text,
                    subtitle_position,
                    subtitle_style,
                    font,
                    font_size,
                    text_color,
                    background_bgr,
                )

            if add_watermark:
                add_watermark_to_frame(frame)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            writer.append_data(frame_rgb)
            frame_number += 1

            if progress_callback and frame_count > 0:
                progress_callback(min(frame_number / frame_count, 1.0))
    finally:
        cap.release()
        writer.close()


def mux_audio(video_without_audio: str, original_video: str, final_output: str):
    """Attach original audio when available; otherwise keep silent output."""
    if not has_audio_stream(original_video):
        os.replace(video_without_audio, final_output)
        return final_output

    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_without_audio,
        "-i",
        original_video,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        final_output,
    ]

    subprocess.run(command, check=True, capture_output=True, text=True)
    return final_output


# -----------------------------------------------------------------------------
# OpenAI content assistant
# -----------------------------------------------------------------------------
def call_ai(prompt: str):
    client = get_openai_client()
    if client is None:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. Add it to Streamlit secrets to use AI Content Assistant."
        )

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def generate_titles(transcript: str):
    return call_ai(
        """
Generate 5 catchy, natural YouTube Shorts title options based on the transcript below.
Keep them concise, avoid clickbait that is unsupported by the transcript, and return
one title per line without numbering or quotation marks.

Transcript:
"""
        + transcript
    )


def generate_description(transcript: str):
    return call_ai(
        """
Write a concise YouTube Shorts description based only on the transcript below.
Make it engaging and suitable for a creator. Do not invent facts that are not present.

Transcript:
"""
        + transcript
    )


def generate_hashtags(transcript: str):
    return call_ai(
        """
Generate 10 relevant social-media hashtags for this video transcript.
Return only hashtags separated by spaces. Prefer specific topics over generic tags.

Transcript:
"""
        + transcript
    )


def translate_segments(segments, target_language: str):
    """Translate subtitle text while preserving timings."""
    if target_language == "None" or not segments:
        return segments

    translated_text = call_ai(
        f"""
Translate the following subtitle lines into {target_language}.
Return exactly the same number of lines, one translation per line, in the same order.
Do not add numbering, explanations, or quotation marks.

Subtitle lines:
"""
        + "\n".join(item["text"] for item in segments)
    )

    lines = [line.strip() for line in translated_text.splitlines() if line.strip()]
    if len(lines) != len(segments):
        raise RuntimeError(
            "The translation response did not preserve the subtitle line count."
        )

    translated = []
    for segment, text in zip(segments, lines):
        translated.append(
            {
                "start": segment["start"],
                "end": segment["end"],
                "text": text,
            }
        )
    return translated


# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
if "job_dir" not in st.session_state:
    st.session_state.job_dir = None
if "uploaded_video" not in st.session_state:
    st.session_state.uploaded_video = None
if "segments" not in st.session_state:
    st.session_state.segments = []
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "detected_language" not in st.session_state:
    st.session_state.detected_language = None
if "subtitle_video" not in st.session_state:
    st.session_state.subtitle_video = None
if "srt_file" not in st.session_state:
    st.session_state.srt_file = None
if "translated" not in st.session_state:
    st.session_state.translated = False
if "titles" not in st.session_state:
    st.session_state.titles = ""
if "description" not in st.session_state:
    st.session_state.description = ""
if "hashtags" not in st.session_state:
    st.session_state.hashtags = ""


# -----------------------------------------------------------------------------
# CSS / header
# -----------------------------------------------------------------------------
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
    margin-bottom:20px;
">
    <h1>🎬 AutoCaptionAI</h1>
    <p>Create AI-powered subtitles for YouTube Shorts, TikTok and Instagram Reels.</p>
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=120)

    st.title("⚙️ Settings")

    subtitle_style = st.selectbox(
        "🎨 Subtitle Style",
        ["YouTube Shorts", "TikTok", "Instagram Reels"],
    )

    language = st.selectbox(
        "🌍 Spoken Language",
        ["Auto Detect", *LANGUAGES.keys()],
    )

    font_style = st.selectbox(
        "🔤 Font Style",
        list(FONT_STYLES.keys()),
    )

    font_size = st.slider(
        "🔤 Font Size",
        0.5,
        3.0,
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
        "⬛ Subtitle Background",
        "#000000",
    )

    target_language = st.selectbox(
        "🌍 Translate Subtitles To",
        TRANSLATION_LANGUAGES,
    )

    watermark_enabled = st.checkbox(
        "🏷️ Add AutoCaptionAI Watermark",
        value=True,
    )

    st.info(
        f"Style: {subtitle_style}\n\n"
        f"Language: {language}\n\n"
        f"Position: {subtitle_position}"
    )


# -----------------------------------------------------------------------------
# Main UI
# -----------------------------------------------------------------------------
st.image("logo.png", width=180)

col1, col2 = st.columns(2)
with col1:
    st.metric("🌍 Languages", str(len(LANGUAGES)))
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
st.subheader("🎥 Upload your video")
st.write("Supports MP4, MOV, AVI and MKV files.")

uploaded_file = st.file_uploader(
    "Upload Video",
    type=SUPPORTED_VIDEO_TYPES,
)


# -----------------------------------------------------------------------------
# New upload / job directory
# -----------------------------------------------------------------------------
if uploaded_file is not None:
    upload_key = f"{uploaded_file.name}:{uploaded_file.size}"

    if st.session_state.uploaded_video != upload_key:
        # A unique directory prevents concurrent users from overwriting files.
        job_dir = Path(tempfile.mkdtemp(prefix="autocaption_"))
        input_path = job_dir / safe_filename(uploaded_file.name)
        input_path.write_bytes(uploaded_file.getvalue())

        st.session_state.job_dir = str(job_dir)
        st.session_state.uploaded_video = upload_key
        st.session_state.input_path = str(input_path)
        st.session_state.segments = []
        st.session_state.transcript = ""
        st.session_state.detected_language = None
        st.session_state.subtitle_video = None
        st.session_state.srt_file = None
        st.session_state.translated = False
        st.session_state.titles = ""
        st.session_state.description = ""
        st.session_state.hashtags = ""

    input_path = st.session_state.input_path

    st.subheader("Original Video")
    st.video(input_path)

    if st.button("🚀 Generate Subtitles", type="primary", use_container_width=True):
        try:
            with st.status("Generating subtitles...", expanded=True) as status:
                st.write("🎙️ Transcribing audio with Whisper...")
                segments, transcript, detected_language = transcribe_video(
                    input_path,
                    language,
                )

                if not segments:
                    raise RuntimeError(
                        "No speech was detected. Try a video with clearer speech or a supported language."
                    )

                st.session_state.segments = segments
                st.session_state.transcript = transcript
                st.session_state.detected_language = detected_language

                if detected_language:
                    st.write(f"Detected language: `{detected_language}`")

                render_segments = segments
                if target_language != "None":
                    st.write(f"🌍 Translating subtitles to {target_language}...")
                    render_segments = translate_segments(segments, target_language)
                    st.session_state.translated = True
                else:
                    st.session_state.translated = False

                job_dir = Path(st.session_state.job_dir)
                rendered_video = job_dir / "subtitled_video.mp4"
                silent_video = job_dir / "subtitled_video_silent.mp4"
                srt_path = job_dir / "subtitles.srt"
                final_video = job_dir / "final_output.mp4"

                st.write("🎬 Rendering subtitles onto the video...")
                progress = st.progress(0)

                render_subtitled_video(
                    input_path,
                    render_segments,
                    str(silent_video),
                    subtitle_style,
                    font_size,
                    subtitle_position,
                    subtitle_color,
                    background_color,
                    font_style,
                    watermark_enabled,
                    progress_callback=lambda value: progress.progress(int(value * 100)),
                )

                st.write("🔊 Restoring original audio...")
                mux_audio(str(silent_video), input_path, str(final_video))

                generate_srt_from_segments(render_segments, str(srt_path))

                # Keep translated captions in memory for the SRT/video generated
                # in this run, while preserving the original transcript for AI content.
                st.session_state.subtitle_video = str(final_video)
                st.session_state.srt_file = str(srt_path)

                status.update(label="Subtitles generated successfully!", state="complete")

        except subprocess.CalledProcessError as exc:
            error_text = exc.stderr or str(exc)
            st.error(f"FFmpeg failed while processing the video:\n\n{error_text[-2000:]}")
        except Exception as exc:
            st.error(f"Could not generate subtitles: {exc}")


# -----------------------------------------------------------------------------
# Results + AI assistant
# -----------------------------------------------------------------------------
if st.session_state.transcript:
    st.markdown("---")
    st.subheader("📝 Transcript")
    st.text_area(
        "Transcript",
        value=st.session_state.transcript,
        height=180,
        label_visibility="collapsed",
    )

    if st.session_state.detected_language:
        st.caption(
            f"Detected language: {st.session_state.detected_language}"
        )

    st.subheader("📝 AI Content Assistant")
    ai_col1, ai_col2, ai_col3 = st.columns(3)

    with ai_col1:
        if st.button("✨ Generate Titles", use_container_width=True):
            try:
                with st.spinner("Generating titles..."):
                    st.session_state.titles = generate_titles(
                        st.session_state.transcript
                    )
            except Exception as exc:
                st.error(str(exc))

    with ai_col2:
        if st.button("📄 Generate Description", use_container_width=True):
            try:
                with st.spinner("Generating description..."):
                    st.session_state.description = generate_description(
                        st.session_state.transcript
                    )
            except Exception as exc:
                st.error(str(exc))

    with ai_col3:
        if st.button("🏷️ Generate Hashtags", use_container_width=True):
            try:
                with st.spinner("Generating hashtags..."):
                    st.session_state.hashtags = generate_hashtags(
                        st.session_state.transcript
                    )
            except Exception as exc:
                st.error(str(exc))

    if st.session_state.titles:
        st.subheader("✨ Suggested Titles")
        st.text_area(
            "Titles",
            value=st.session_state.titles,
            height=160,
            label_visibility="collapsed",
        )

    if st.session_state.description:
        st.subheader("📄 Suggested Description")
        st.text_area(
            "Description",
            value=st.session_state.description,
            height=200,
            label_visibility="collapsed",
        )

    if st.session_state.hashtags:
        st.subheader("🏷️ Suggested Hashtags")
        st.code(st.session_state.hashtags)


# -----------------------------------------------------------------------------
# Downloads / final video
# -----------------------------------------------------------------------------
if st.session_state.subtitle_video and os.path.exists(st.session_state.subtitle_video):
    st.markdown("---")
    st.subheader("🎬 Final Video")
    st.video(st.session_state.subtitle_video)

    download_col1, download_col2 = st.columns(2)

    with download_col1:
        with open(st.session_state.subtitle_video, "rb") as video_file:
            st.download_button(
                "⬇️ Download MP4",
                data=video_file,
                file_name="autocaption_subtitled_video.mp4",
                mime="video/mp4",
                use_container_width=True,
            )

    with download_col2:
        if st.session_state.srt_file and os.path.exists(st.session_state.srt_file):
            with open(st.session_state.srt_file, "rb") as srt_file:
                st.download_button(
                    "⬇️ Download SRT",
                    data=srt_file,
                    file_name="autocaption_subtitles.srt",
                    mime="application/x-subrip",
                    use_container_width=True,
                )
