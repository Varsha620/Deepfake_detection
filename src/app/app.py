import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import sys
import io
import cv2
import requests
import tempfile

# Add src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepGuard | Deepfake Detection",
    page_icon="🛡️",
    layout="centered"
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #1e1e2f 0%, #121212 100%);
    }
    .stApp {
        color: #ffffff;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 2rem;
    }
    .result-real {
        color: #00ff88;
        font-weight: bold;
        font-size: 24px;
        text-align: center;
        padding: 10px;
        border-radius: 10px;
        background: rgba(0, 255, 136, 0.1);
    }
    .result-fake {
        color: #ff4b2b;
        font-weight: bold;
        font-size: 24px;
        text-align: center;
        padding: 10px;
        border-radius: 10px;
        background: rgba(255, 75, 43, 0.1);
    }
    h1 {
        text-align: center;
        background: linear-gradient(to right, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }
    p {
        text-align: center;
        color: #cccccc;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 16px;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("<h1>DeepGuard</h1>", unsafe_allow_html=True)
st.markdown("<p>Next-Generation AI-Powered Deepfake Detection</p>", unsafe_allow_html=True)

# ── Model Loader (cached) ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    from transformers import pipeline
    try:
        # Load Hugging Face pipeline for universal AI image classification
        model = pipeline("image-classification", model="umm-maybe/AI-image-detector")
        loaded = True
    except Exception as e:
        print(f"Failed loading HF model: {e}")
        model = None
        loaded = False

    return model, loaded

# ── Helpers ────────────────────────────────────────────────────────────────────
def predict_pil(model, pil_image: Image.Image) -> float:
    """Return raw score for a PIL image using Hugging Face (>=0.5 means Fake)."""
    # The HF pipeline intrinsically handles preprocessing and RGB alignment natively.
    results = model(pil_image, top_k=None)
    
    # Extract the confidence ratio for the 'Deepfake' class
    # Default to 0.5 (uncertain) if no mapping is found
    fake_score = 0.5
    for res in results:
        label = res['label'].lower()
        if 'fake' in label or 'artificial' in label or 'ai' in label or 'synthetic' in label:
            fake_score = res['score']
            break
        elif 'real' in label or 'human' in label or 'natural' in label:
            fake_score = 1.0 - res['score']
            break
            
    return float(fake_score)


def fetch_image_from_url(url: str) -> Image.Image:
    # Handle embedded base64 Data URIs natively (bypassing requests)
    if url.startswith("data:image"):
        import base64
        header, encoded = url.split(",", 1)
        data = base64.b64decode(encoded)
        return Image.open(io.BytesIO(data)).convert("RGB")
        
    # Handle Google Image Search indirect URLs (extracts the actual imgurl)
    from urllib.parse import urlparse, parse_qs
    parsed = urlparse(url)
    if "google.com/imgres" in url:
        qs = parse_qs(parsed.query)
        if "imgurl" in qs:
            url = qs["imgurl"][0]

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    
    # Fail gracefully if it's a webpage instead of an image
    content_type = resp.headers.get("Content-Type", "")
    if "text/html" in content_type:
        raise ValueError("URL points to a webpage, not a direct image file. Please provide a direct link to the image (e.g., ending in .jpg or .png).")

    return Image.open(io.BytesIO(resp.content)).convert("RGB")




def extract_frames(video_path: str, n_frames: int = 30) -> list:
    """Extract n evenly-spaced frames from a video; return list of RGB ndarrays."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = np.linspace(0, total - 1, min(n_frames, total), dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def analyse_frames(model, frames: list) -> list:
    """Run model on each frame; return list of sigmoid scores."""
    scores = []
    for frame in frames:
        pil = Image.fromarray(frame)
        scores.append(predict_pil(model, pil))
    return scores


def render_verdict(avg_score: float, fake_ratio: float):
    """Render final verdict banner."""
    st.markdown("---")
    # In CIFAKE, if REAL=0 and FAKE=1 alphabetically:
    is_fake = avg_score >= 0.5
    confidence = avg_score if is_fake else 1 - avg_score
    if is_fake:
        st.markdown('<div class="result-fake">⚠️ Likely AI-Generated / Deepfake</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-real">✅ Likely Real</div>', unsafe_allow_html=True)
    st.write(f"Overall confidence: **{confidence*100:.1f}%** | Fake frames: **{fake_ratio*100:.1f}%**")
    st.progress(float(confidence))


# ══════════════════════════════════════════════════════════════════════════════
# Main App
# ══════════════════════════════════════════════════════════════════════════════
def main():
    col1, col2, col3 = st.columns([1, 6, 1])

    with col2:
        tab_img, tab_vid = st.tabs(["🖼️ Image", "🎬 Video"])

        # ── IMAGE TAB ──────────────────────────────────────────────────────────
        with tab_img:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            img_mode = st.radio("Input method", ["📁 Upload File", "🔗 Enter URL"], horizontal=True, key="img_mode")
            image = None

            if img_mode == "📁 Upload File":
                uploaded = st.file_uploader("Upload an image to scan…", type=["jpg", "jpeg", "png"], key="img_upload")
                if uploaded:
                    image = Image.open(uploaded).convert("RGB")

            else:
                url = st.text_input("Paste a direct image URL", placeholder="https://example.com/photo.jpg", key="img_url")
                if url:
                    with st.spinner("Fetching image…"):
                        try:
                            image = fetch_image_from_url(url)
                        except Exception as e:
                            st.error(f"Could not load image from URL: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

            if image is not None:
                st.image(image, caption="Target Image", width="stretch")

                with st.spinner("Analyzing pixels for deepfake artifacts…"):
                    try:
                        model, loaded = load_model()
                        if not loaded:
                            st.info("💡 Running in demo mode — weights not found in /models.")

                        score = predict_pil(model, image)
                        confidence = score if score >= 0.5 else 1 - score

                        st.markdown("---")
                        if score >= 0.5:
                            st.markdown('<div class="result-fake">⚠️ Potential Deepfake Detected</div>', unsafe_allow_html=True)
                            st.write(f"Confidence score: **{confidence*100:.2f}%** (Likely AI-Generated)")
                        else:
                            st.markdown('<div class="result-real">✅ Real Image Verified</div>', unsafe_allow_html=True)
                            st.write(f"Confidence score: **{confidence*100:.2f}%** (Likely Real)")
                        st.progress(float(confidence))

                    except Exception as e:
                        st.error(f"Error during analysis: {e}")

        # ── VIDEO TAB ──────────────────────────────────────────────────────────
        with tab_vid:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            n_frames = st.slider("Frames to sample", min_value=10, max_value=60, value=30, step=5,
                                 help="More frames = more accurate but slower")
            video_path = None
            cleanup_video = False

            uploaded_vid = st.file_uploader("Upload a video to scan (Max 60 seconds)…",
                                            type=["mp4", "mov", "avi", "mkv", "webm"],
                                            key="vid_upload")
            if uploaded_vid:
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                    f.write(uploaded_vid.read())
                    temp_path = f.name
                
                # Check video duration using OpenCV
                cap = cv2.VideoCapture(temp_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                cap.release()
                
                duration = 0
                if fps > 0:
                    duration = frame_count / fps
                
                if duration > 60:
                    st.error(f"⚠️ Video is too long ({duration:.1f}s). To ensure the app performs efficiently, please upload a video under 60 seconds.")
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                else:
                    video_path = temp_path
                    cleanup_video = True
            st.markdown('</div>', unsafe_allow_html=True)

            if video_path is not None:
                with st.spinner(f"Extracting & analyzing {n_frames} frames…"):
                    try:
                        model, loaded = load_model()
                        if not loaded:
                            st.info("💡 Running in demo mode — weights not found in /models.")

                        frames = extract_frames(video_path, n_frames=n_frames)
                        if not frames:
                            st.error("Could not extract frames. Please check the video file.")
                        else:
                            scores = analyse_frames(model, frames)
                            fake_ratio = sum(1 for s in scores if s >= 0.5) / len(scores)
                            avg_score  = float(np.mean(scores))

                            render_verdict(avg_score, fake_ratio)

                            # Per-frame chart
                            import pandas as pd
                            chart_data = pd.DataFrame({
                                "Frame": range(1, len(scores) + 1),
                                "Fake Probability": [s for s in scores]
                            }).set_index("Frame")
                            st.markdown("#### 📊 Per-Frame Fake Probability")
                            st.line_chart(chart_data)

                            # Frame grid (up to 6 thumbnails)
                            st.markdown("#### 🎞️ Sampled Frames")
                            thumb_indices = np.linspace(0, len(frames) - 1, min(6, len(frames)), dtype=int)
                            cols = st.columns(min(3, len(thumb_indices)))
                            for i, idx in enumerate(thumb_indices):
                                s = scores[idx]
                                conf = (s if s >= 0.5 else 1 - s) * 100
                                label = f"Frame {idx+1} — {'🔴 Fake' if s >= 0.5 else '🟢 Real'} ({conf:.1f}%)"
                                cols[i % 3].image(frames[idx], caption=label, width="stretch")

                    except Exception as e:
                        st.error(f"Error during video analysis: {e}")
                    finally:
                        if cleanup_video and os.path.exists(video_path):
                            os.remove(video_path)

    # Footer
    st.markdown("---")
    st.markdown("<p style='font-size: 0.8rem;'>Built for Research and Security Purposes • 2025</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
