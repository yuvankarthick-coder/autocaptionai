import base64
import io
import os
import re

import streamlit as st
from openai import OpenAI

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from PIL import Image
except ImportError:
    Image = None

st.set_page_config(
    page_title="Tamil Study AI",
    page_icon="🇮🇳",
    layout="wide",
)

APP_NAME = "Tamil Study AI"
MODEL = "gpt-4.1-mini"
MAX_UPLOAD_MB = 20
MAX_PDF_PAGES = 30
MAX_TEXT_CHARS = 45000

SYSTEM_PROMPT = """
You are Tamil Study AI, a careful study assistant for Tamil-speaking students.
Explain the user's supplied study material in clear, natural, student-friendly Tamil.
Keep important English technical terms in parentheses when useful.
Base answers primarily on the supplied material. Do not invent unsupported facts.
If the material is incomplete or unclear, say so. Do not claim answers are guaranteed
for an exam. Preserve equations, units, scientific names, and important terminology.
"""


def get_openai_client():
    key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    return OpenAI(api_key=key) if key else None


def clean_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def truncate_text(text, limit=MAX_TEXT_CHARS):
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n\n[Material truncated for processing.]"


def extract_pdf_text(uploaded_file):
    if fitz is None:
        raise RuntimeError("PDF support requires PyMuPDF. Add PyMuPDF to requirements.txt.")
    data = uploaded_file.getvalue()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"Please upload a file smaller than {MAX_UPLOAD_MB} MB.")
    doc = fitz.open(stream=data, filetype="pdf")
    page_count = len(doc)
    if page_count == 0:
        doc.close()
        raise ValueError("The PDF does not contain any pages.")
    pages = min(page_count, MAX_PDF_PAGES)
    text = clean_text("\n\n".join(doc.load_page(i).get_text("text") for i in range(pages)))
    doc.close()
    if not text:
        raise ValueError("No readable text was found. For scanned PDFs, upload a page as an image.")
    if page_count > MAX_PDF_PAGES:
        text += f"\n\n[Only the first {MAX_PDF_PAGES} pages were processed from this {page_count}-page PDF.]"
    return truncate_text(text), page_count


def image_to_data_url(uploaded_file):
    if Image is None:
        raise RuntimeError("Image support requires Pillow. Add Pillow to requirements.txt.")
    data = uploaded_file.getvalue()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"Please upload an image smaller than {MAX_UPLOAD_MB} MB.")
    image = Image.open(io.BytesIO(data))
    image.verify()
    mime = uploaded_file.type or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode('utf-8')}"


def run_text_ai(client, material, task):
    prompt = f"""
Study material supplied by the student:
--- BEGIN MATERIAL ---
{truncate_text(material)}
--- END MATERIAL ---

Task:
{task}

Respond in clear Tamil. Keep important English subject terms in parentheses when useful.
"""
    response = client.responses.create(model=MODEL, instructions=SYSTEM_PROMPT, input=prompt)
    return response.output_text.strip()


def run_image_ai(client, image_url, task):
    response = client.responses.create(
        model=MODEL,
        instructions=SYSTEM_PROMPT,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Read the study material in this image carefully. " + task},
                {"type": "input_image", "image_url": image_url},
            ],
        }],
    )
    return response.output_text.strip()


def run_ai(material, image_url, task):
    client = get_openai_client()
    if client is None:
        raise RuntimeError("OPENAI_API_KEY is not configured in Streamlit secrets.")
    if image_url:
        return run_image_ai(client, image_url, task)
    if not material:
        raise ValueError("Please provide study material first.")
    return run_text_ai(client, material, task)


st.markdown("""
<style>
#MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;}
.hero {padding:42px 24px;border-radius:24px;background:linear-gradient(135deg,#0ea5e9,#7c3aed);color:white;text-align:center;margin-bottom:24px;}
.hero h1 {font-size:46px;margin-bottom:10px;}
.hero p {font-size:20px;margin:0 auto;max-width:720px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>🇮🇳 Tamil Study AI</h1>
<p>Understand your lessons in simple Tamil.</p>
<p>Upload your study material and turn difficult lessons into clear explanations.</p>
</div>
""", unsafe_allow_html=True)

for key, default in {
    "material_text": "", "material_name": "", "material_source": "", "image_data_url": None,
    "explanation": "", "key_points": "", "practice": "", "last_file_key": ""
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.subheader("📚 Add your study material")
upload_tab, text_tab = st.tabs(["📄 Upload file", "✍️ Paste text"])

with upload_tab:
    uploaded = st.file_uploader(
        "Upload a PDF or image of your lesson/question",
        type=["pdf", "png", "jpg", "jpeg", "webp"],
        help=f"Maximum file size: {MAX_UPLOAD_MB} MB.",
    )
    if uploaded is not None:
        file_key = f"{uploaded.name}:{uploaded.size}"
        if st.session_state.last_file_key != file_key:
            st.session_state.last_file_key = file_key
            st.session_state.material_name = uploaded.name
            st.session_state.explanation = ""
            st.session_state.key_points = ""
            st.session_state.practice = ""
            try:
                if uploaded.name.lower().endswith(".pdf"):
                    text, pages = extract_pdf_text(uploaded)
                    st.session_state.material_text = text
                    st.session_state.image_data_url = None
                    st.session_state.material_source = f"PDF • {pages} page(s)"
                else:
                    st.session_state.material_text = ""
                    st.session_state.image_data_url = image_to_data_url(uploaded)
                    st.session_state.material_source = "Image"
                st.success(f"Loaded: {uploaded.name}")
            except Exception as exc:
                st.error(str(exc))

with text_tab:
    pasted = st.text_area("Paste your lesson, notes, or question", height=220, placeholder="Paste English study material here...")
    if pasted.strip():
        st.session_state.material_text = truncate_text(clean_text(pasted))
        st.session_state.image_data_url = None
        st.session_state.material_name = "Pasted text"
        st.session_state.material_source = "Pasted text"

has_material = bool(st.session_state.material_text.strip() or st.session_state.image_data_url)

if has_material:
    st.divider()
    left, right = st.columns([2, 1])
    with left:
        st.subheader("📖 Material ready")
        if st.session_state.material_source == "Image":
            st.info(f"🖼️ {st.session_state.material_name} is ready for AI vision analysis.")
        else:
            preview = st.session_state.material_text[:1500]
            if len(st.session_state.material_text) > 1500:
                preview += "\n..."
            st.text_area("Extracted text preview", value=preview, height=220, disabled=True)
    with right:
        st.metric("Input", st.session_state.material_source or "Material")
        if st.session_state.material_text:
            st.metric("Characters", f"{len(st.session_state.material_text):,}")
        st.caption("AI output can contain mistakes. Verify important academic information with your textbook or teacher.")

if has_material:
    st.divider()
    st.subheader("✨ Learn from your material")
    a, b, c = st.columns(3)
    with a:
        explain = st.button("📖 Explain in Tamil", use_container_width=True, type="primary")
    with b:
        points = st.button("🧠 Key Points", use_container_width=True)
    with c:
        practice = st.button("❓ Practice Questions", use_container_width=True)

    if explain:
        with st.spinner("தமிழில் எளிய விளக்கம் உருவாக்கப்படுகிறது..."):
            try:
                st.session_state.explanation = run_ai(
                    st.session_state.material_text or None, st.session_state.image_data_url,
                    """
Explain the lesson in simple Tamil for a student.
Structure it with a short heading, main idea, small concept sections, and simple examples.
Keep important English technical terms in parentheses. Preserve formulas, equations,
steps, and examples from the material. Do not add unsupported facts.
"""
                )
            except Exception as exc:
                st.error(f"Could not generate the explanation: {exc}")

    if points:
        with st.spinner("முக்கிய குறிப்புகள் உருவாக்கப்படுகின்றன..."):
            try:
                st.session_state.key_points = run_ai(
                    st.session_state.material_text or None, st.session_state.image_data_url,
                    """
Create a concise study sheet based only on the supplied material.
Return a section called '🧠 முக்கிய குறிப்புகள்' with 5-10 points in simple Tamil,
and a section called '📌 முக்கிய சொற்கள்' with important English terms and short Tamil meanings.
"""
                )
            except Exception as exc:
                st.error(f"Could not generate key points: {exc}")

    if practice:
        with st.spinner("பயிற்சி கேள்விகள் உருவாக்கப்படுகின்றன..."):
            try:
                st.session_state.practice = run_ai(
                    st.session_state.material_text or None, st.session_state.image_data_url,
                    """
Create a practice set based only on the supplied material.
Include 3 MCQs with four options each and 2 short-answer questions.
Then provide an answer key with brief Tamil explanations.
Do not ask about information absent from the supplied material.
"""
                )
            except Exception as exc:
                st.error(f"Could not generate practice questions: {exc}")

if st.session_state.explanation or st.session_state.key_points or st.session_state.practice:
    st.divider()
    st.subheader("📚 Your Study Pack")
    if st.session_state.explanation:
        with st.container(border=True):
            st.markdown("## 📖 Simple Tamil Explanation")
            st.markdown(st.session_state.explanation)
    if st.session_state.key_points:
        with st.container(border=True):
            st.markdown("## 🧠 Key Points")
            st.markdown(st.session_state.key_points)
    if st.session_state.practice:
        with st.container(border=True):
            st.markdown("## ❓ Practice")
            st.markdown(st.session_state.practice)

st.divider()
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("### 📄 Upload")
    st.caption("PDF, image, or pasted text.")
with c2:
    st.markdown("### 🇮🇳 Understand")
    st.caption("Get explanations in simple Tamil.")
with c3:
    st.markdown("### 🧠 Practice")
    st.caption("Turn your material into revision questions.")
st.caption("Tamil Study AI is an educational assistant. Verify important information with your textbook or teacher.")
