import streamlit as st

st.set_page_config(page_title="AutoCaptionAI", page_icon="🎬")

import os
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip


def generate_subtitled_video(video_path):
    segments, _ = model.transcribe(video_path)

    video = VideoFileClip(video_path)

    subtitle_clips = []

    for seg in segments:
        txt = seg.text

        txt_clip = TextClip(
            txt,
            fontsize=40,
            color='yellow',
            size=(video.w - 100, None),
            method='caption'
        ).set_position(('center', 'bottom')).set_start(seg.start).set_end(seg.end)

        subtitle_clips.append(txt_clip)

    final = CompositeVideoClip([video] + subtitle_clips)

    output_path = "output.mp4"
    final.write_videofile(output_path, codec="libx264", audio_codec="aac")

    return output_path


# UI
st.title("🎬 AutoCaptionAI")
st.write("Generate subtitles for your videos instantly using AI ⚡")

uploaded_file = st.file_uploader("Upload video", type=["mp4", "mov"])

if uploaded_file:
    with open("input.mp4", "wb") as f:
        f.write(uploaded_file.read())

    st.video("input.mp4")

    if st.button("Generate Subtitles"):
        output_path = generate_subtitled_video("input.mp4")

        if output_path and os.path.exists(output_path):
            st.success("Video created!")
            st.video(output_path)
        else:
            st.error("Video not generated ❌")
