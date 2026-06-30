import io
import os
import tempfile
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image


MODEL_ID = os.getenv("MODEL_ID", "umm-maybe/AI-image-detector")
FAKE_THRESHOLD = float(os.getenv("FAKE_THRESHOLD", "0.5"))
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_VIDEO_SECONDS = 60


st.set_page_config(
    page_title="DeepGuard | AI Media Authenticity Demo",
    layout="centered",
)


st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #171923 0%, #0f172a 100%);
    }
    .stApp {
        color: #f8fafc;
    }
    .scan-panel {
        background: rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        padding: 1.25rem;
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.28);
        margin-bottom: 1.5rem;
    }
    .result-real,
    .result-fake,
    .result-unknown {
        font-weight: 700;
        font-size: 1.15rem;
        text-align: center;
        padding: 0.75rem;
        border-radius: 8px;
        margin-bottom: 0.75rem;
    }
    .result-real {
        color: #34d399;
        background: rgba(52, 211, 153, 0.12);
        border: 1px solid rgba(52, 211, 153, 0.25);
    }
    .result-fake {
        color: #fb7185;
        background: rgba(251, 113, 133, 0.12);
        border: 1px solid rgba(251, 113, 133, 0.25);
    }
    .result-unknown {
        color: #fbbf24;
        background: rgba(251, 191, 36, 0.12);
        border: 1px solid rgba(251, 191, 36, 0.25);
    }
    h1 {
        text-align: center;
        font-weight: 800 !important;
    }
    .subtitle,
    .disclaimer {
        text-align: center;
        color: #cbd5e1;
    }
    .disclaimer {
        font-size: 0.85rem;
        line-height: 1.45;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 16px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the Hugging Face image-classification pipeline once per session."""
    try:
        from transformers import pipeline

        return pipeline("image-classification", model=MODEL_ID), None
    except Exception as exc:
        return None, str(exc)


def class_to_fake_score(label: str, score: float) -> float | None:
    """Map common model labels to a probability-like fake score."""
    normalized = label.lower()
    fake_tokens = ("fake", "deepfake", "synthetic", "generated", "ai", "artificial")
    real_tokens = ("real", "human", "natural", "authentic", "photograph")

    if any(token in normalized for token in fake_tokens):
        return float(score)
    if any(token in normalized for token in real_tokens):
        return float(1.0 - score)
    return None


def predict_pil(model, pil_image: Image.Image) -> float:
    """Return a fake-probability score for a PIL image."""
    if model is None:
        raise RuntimeError("The model is not available. Check deployment logs for the load error.")

    results = model(pil_image.convert("RGB"), top_k=None)
    if not isinstance(results, list):
        raise ValueError("The model returned an unexpected response.")

    for result in results:
        label = str(result.get("label", ""))
        score = float(result.get("score", 0.0))
        fake_score = class_to_fake_score(label, score)
        if fake_score is not None:
            return max(0.0, min(1.0, fake_score))

    raise ValueError("The model response did not include a recognizable real/fake label.")


def fetch_image_from_url(url: str) -> Image.Image:
    """Fetch and decode a direct image URL with basic safety checks."""
    clean_url = url.strip()
    if not clean_url:
        raise ValueError("Please enter an image URL.")

    if clean_url.startswith("data:image"):
        import base64

        _, encoded = clean_url.split(",", 1)
        data = base64.b64decode(encoded)
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError("Image is too large. Please use an image under 10 MB.")
        return Image.open(io.BytesIO(data)).convert("RGB")

    parsed = urlparse(clean_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http and https image URLs are supported.")

    if "google.com/imgres" in clean_url:
        query = parse_qs(parsed.query)
        if "imgurl" in query:
            clean_url = query["imgurl"][0]

    headers = {
        "User-Agent": "Mozilla/5.0 DeepGuard/1.0 (+https://streamlit.io)"
    }
    response = requests.get(clean_url, headers=headers, timeout=15, stream=True)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()
    if content_type and "image" not in content_type:
        raise ValueError("URL does not point to a direct image file.")

    content = response.content
    if len(content) > MAX_IMAGE_BYTES:
        raise ValueError("Image is too large. Please use an image under 10 MB.")

    return Image.open(io.BytesIO(content)).convert("RGB")


def extract_frames(video_path: str, n_frames: int = 30) -> list[np.ndarray]:
    """Extract evenly spaced RGB frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, min(n_frames, total), dtype=int)
    frames = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return frames


def get_video_duration(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return float(frame_count / fps) if fps and fps > 0 else 0.0


def analyse_frames(model, frames: list[np.ndarray]) -> list[float]:
    return [predict_pil(model, Image.fromarray(frame)) for frame in frames]


def render_verdict(fake_score: float, fake_ratio: float | None = None) -> None:
    is_fake = fake_score >= FAKE_THRESHOLD
    confidence = fake_score if is_fake else 1.0 - fake_score

    if confidence < 0.6:
        st.markdown(
            '<div class="result-unknown">Uncertain result</div>',
            unsafe_allow_html=True,
        )
    elif is_fake:
        st.markdown(
            '<div class="result-fake">Likely AI-generated or manipulated</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="result-real">Likely real image</div>',
            unsafe_allow_html=True,
        )

    if fake_ratio is None:
        st.write(f"Model confidence: **{confidence * 100:.1f}%**")
    else:
        st.write(
            f"Model confidence: **{confidence * 100:.1f}%** | "
            f"Frames above fake threshold: **{fake_ratio * 100:.1f}%**"
        )
    st.progress(float(confidence))


def render_model_status(model_error: str | None) -> bool:
    if model_error is None:
        return True

    st.error("The AI model could not be loaded in this environment.")
    with st.expander("Model load details"):
        st.code(model_error)
    st.info("Check internet access, package installation, and available memory on the deployment host.")
    return False


def render_header() -> None:
    st.markdown("<h1>DeepGuard</h1>", unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">AI media authenticity demo for images and short videos.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="disclaimer">Research/demo tool only. Results are model probabilities, '
        'not forensic proof. Always verify important claims with multiple sources.</p>',
        unsafe_allow_html=True,
    )


def render_image_tab() -> None:
    st.markdown('<div class="scan-panel">', unsafe_allow_html=True)
    input_mode = st.radio(
        "Input method",
        ["Upload file", "Enter URL"],
        horizontal=True,
        key="image_input_mode",
    )
    image = None

    if input_mode == "Upload file":
        uploaded = st.file_uploader(
            "Upload an image to scan",
            type=["jpg", "jpeg", "png", "webp"],
            key="image_upload",
        )
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
    else:
        url = st.text_input(
            "Paste a direct image URL",
            placeholder="https://example.com/photo.jpg",
            key="image_url",
        )
        if url:
            with st.spinner("Fetching image..."):
                try:
                    image = fetch_image_from_url(url)
                except Exception as exc:
                    st.error(f"Could not load image: {exc}")

    st.markdown("</div>", unsafe_allow_html=True)

    if image is None:
        return

    st.image(image, caption="Target image", use_container_width=True)
    model, model_error = load_model()
    if not render_model_status(model_error):
        return

    with st.spinner("Analyzing image..."):
        try:
            score = predict_pil(model, image)
            render_verdict(score)
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")


def render_video_tab() -> None:
    st.markdown('<div class="scan-panel">', unsafe_allow_html=True)
    n_frames = st.slider(
        "Frames to sample",
        min_value=10,
        max_value=60,
        value=30,
        step=5,
        help="More frames can improve coverage but will run slower.",
    )
    uploaded = st.file_uploader(
        "Upload a short video to scan",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        key="video_upload",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded is None:
        return

    temp_path = None
    try:
        suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(uploaded.read())
            temp_path = handle.name

        duration = get_video_duration(temp_path)
        if duration > MAX_VIDEO_SECONDS:
            st.error(
                f"Video is {duration:.1f} seconds long. Please upload a video "
                f"under {MAX_VIDEO_SECONDS} seconds."
            )
            return

        model, model_error = load_model()
        if not render_model_status(model_error):
            return

        with st.spinner(f"Extracting and analyzing {n_frames} frames..."):
            frames = extract_frames(temp_path, n_frames=n_frames)
            if not frames:
                st.error("Could not extract frames from this video.")
                return

            scores = analyse_frames(model, frames)
            average_score = float(np.mean(scores))
            fake_ratio = sum(score >= FAKE_THRESHOLD for score in scores) / len(scores)
            render_verdict(average_score, fake_ratio)

            chart_data = pd.DataFrame(
                {
                    "Frame": range(1, len(scores) + 1),
                    "Fake probability": scores,
                }
            ).set_index("Frame")
            st.markdown("#### Per-frame fake probability")
            st.line_chart(chart_data)

            st.markdown("#### Sampled frames")
            sample_indices = np.linspace(0, len(frames) - 1, min(6, len(frames)), dtype=int)
            columns = st.columns(min(3, len(sample_indices)))
            for position, frame_index in enumerate(sample_indices):
                score = scores[frame_index]
                verdict = "Fake" if score >= FAKE_THRESHOLD else "Real"
                confidence = (score if score >= FAKE_THRESHOLD else 1.0 - score) * 100
                columns[position % len(columns)].image(
                    frames[frame_index],
                    caption=f"Frame {frame_index + 1}: {verdict} ({confidence:.1f}%)",
                    use_container_width=True,
                )
    except Exception as exc:
        st.error(f"Video analysis failed: {exc}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def main() -> None:
    render_header()
    left, center, right = st.columns([1, 6, 1])
    with center:
        image_tab, video_tab = st.tabs(["Image", "Video"])
        with image_tab:
            render_image_tab()
        with video_tab:
            render_video_tab()

    st.markdown("---")
    st.markdown(
        '<p class="disclaimer">Built for research, education, and security-awareness demos.</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
