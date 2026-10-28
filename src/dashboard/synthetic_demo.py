import io

import numpy as np
import pandas as pd

from src.data.preprocess import EXPECTED_FEATURES

SYNTHETIC_DEMO_ROWS = 20
SYNTHETIC_DEMO_SEED = 42
SYNTHETIC_DEMO_WARNING = (
    "Synthetic demo data and predictions are illustrative only; they are not validated "
    "real-world medical-device detections."
)


def generate_synthetic_demo_csv(scaler) -> bytes:
    rng = np.random.default_rng(SYNTHETIC_DEMO_SEED)
    standardized = rng.standard_normal((SYNTHETIC_DEMO_ROWS, len(EXPECTED_FEATURES)))
    raw_features = scaler.inverse_transform(standardized)
    demo_df = pd.DataFrame(raw_features, columns=EXPECTED_FEATURES)
    return demo_df.to_csv(index=False).encode("utf-8")
