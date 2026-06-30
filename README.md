# DeepGuard

DeepGuard is a Streamlit demo for AI media authenticity screening. It supports
image uploads, direct image URLs, and short-video analysis by sampling frames and
running each frame through a Hugging Face image-classification model.

This is a research and education project, not a forensic decision system. The
confidence shown in the app is a model probability, not proof that media is real
or fake.

## Live Demo

Deployment target: Streamlit Community Cloud, Render, or another Python web host.

Recommended Streamlit command:

```bash
streamlit run src/app/app.py
```

## Features

- Image scan from upload or direct URL.
- Short-video scan with configurable frame sampling.
- Per-frame fake-probability chart for videos.
- Model-load failure handling for deployment debugging.
- Clear limitations and responsible-use messaging in the interface.
- Lean Hugging Face inference path without local TensorFlow model loading.

## Model Story

The deployed app uses the Hugging Face model configured by the `MODEL_ID`
environment variable. By default:

```text
umm-maybe/AI-image-detector
```

Older local Keras experiments are preserved under `experiments/` for reference,
but they are not used by the deployed app. The included legacy evaluation
artifacts show limited performance, including an ROC AUC of about 0.68, so the
public app is intentionally framed as a demo rather than a production detector.

## Project Structure

```text
.
|-- .streamlit/
|   `-- config.toml
|-- figures/
|   |-- architecture.png
|   `-- workflow.png
|-- reports/
|   |-- confusion_matrix.png
|   |-- roc_curve.png
|   `-- training_history.png
|-- src/
|   `-- app/
|       `-- app.py
|-- tests/
|   `-- test_app_smoke.py
|-- packages.txt
|-- Procfile
|-- runtime.txt
`-- requirements.txt
```

## Local Setup

Use Python 3.11 for the smoothest deployment match.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run src/app/app.py
```

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run src/app/app.py
```

## Deployment

### Streamlit Community Cloud

1. Push this repository to GitHub.
2. Create a new Streamlit app.
3. Set the main file path to:

```text
src/app/app.py
```

4. Keep the default `MODEL_ID`, or set a different Hugging Face image
   classifier in app settings.

### Render

Use the included `Procfile`.

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
streamlit run src/app/app.py --server.port $PORT --server.address 0.0.0.0
```

## Testing

The smoke tests avoid downloading a real model. They use a fake classifier to
verify the scoring logic and model-load failure behavior.

```bash
python -m pytest
```

## Limitations

- This app is not forensic proof.
- Model confidence is not the same thing as truth.
- Results can be wrong on compressed images, screenshots, cartoons, AI-edited
  but mostly real images, and out-of-distribution media.
- Video support samples frames; it does not analyze audio, temporal consistency,
  face landmarks, or compression traces.
- Hosted deployments may need more memory or a paid tier depending on the model.

## Resume Summary

Built an AI media-authenticity demo with Streamlit, Hugging Face Transformers,
OpenCV frame sampling, image URL ingestion, deployment config, and responsible
AI limitation messaging.

## License

MIT License. See `LICENSE`.
