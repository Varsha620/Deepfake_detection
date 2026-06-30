import importlib.util
import sys
import types
import unittest
from pathlib import Path

from PIL import Image


APP_PATH = Path(__file__).resolve().parents[1] / "src" / "app" / "app.py"


class FakeStreamlit(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.cache_resource = lambda *args, **kwargs: self._cache_decorator

    @staticmethod
    def _cache_decorator(func):
        return func

    def __getattr__(self, name):
        if name == "columns":
            return lambda *args, **kwargs: [self, self, self]
        if name == "tabs":
            return lambda labels: [self for _ in labels]
        if name == "expander":
            return lambda *args, **kwargs: self
        return lambda *args, **kwargs: None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def load_app_with_fake_streamlit():
    sys.modules["streamlit"] = FakeStreamlit()
    sys.modules.setdefault("cv2", types.ModuleType("cv2"))
    sys.modules.setdefault("requests", types.ModuleType("requests"))
    spec = importlib.util.spec_from_file_location("deepguard_app", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeClassifier:
    def __init__(self, results):
        self.results = results

    def __call__(self, image, top_k=None):
        return self.results


class DeepGuardSmokeTests(unittest.TestCase):
    def setUp(self):
        self.app = load_app_with_fake_streamlit()
        self.image = Image.new("RGB", (16, 16), color="white")

    def test_predict_maps_fake_label(self):
        model = FakeClassifier([{"label": "fake", "score": 0.91}])
        self.assertAlmostEqual(self.app.predict_pil(model, self.image), 0.91)

    def test_predict_maps_real_label_to_inverse_score(self):
        model = FakeClassifier([{"label": "real", "score": 0.82}])
        self.assertAlmostEqual(self.app.predict_pil(model, self.image), 0.18)

    def test_predict_fails_when_model_is_missing(self):
        with self.assertRaises(RuntimeError):
            self.app.predict_pil(None, self.image)

    def test_unrecognized_model_label_is_explicit_error(self):
        model = FakeClassifier([{"label": "landscape", "score": 0.99}])
        with self.assertRaises(ValueError):
            self.app.predict_pil(model, self.image)


if __name__ == "__main__":
    unittest.main()
