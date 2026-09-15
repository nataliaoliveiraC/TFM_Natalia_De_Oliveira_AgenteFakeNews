from pathlib import Path
import joblib

from src.preprocessing import (build_model_text)


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "Artefactos"
    / "fake_news_pipeline.joblib"
)

pipeline = joblib.load(
    MODEL_PATH
)


LABEL_MAP = {
    0: "Fake",
    1: "Actual"
}


def predict_fake_news(
    title: str,
    text: str
):
    content = build_model_text(
        title,
        text
    )

    prediction = int(
        pipeline.predict(
            [content]
        )[0]
    )

    decision_score = float(
        pipeline.decision_function(
            [content]
        )[0]
    )

    return {
        "prediction": LABEL_MAP[
            prediction
        ],
        "label": prediction,
        "decision_score": (
            decision_score
        )
    }