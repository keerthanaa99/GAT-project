import io

import numpy as np
import pandas as pd

from src.data.preprocess import EXPECTED_FEATURES
from src.dashboard.synthetic_demo import (
    SYNTHETIC_DEMO_ROWS,
    generate_synthetic_demo_csv,
)


class DemoScaler:
    def __init__(self):
        self.input_shape = None

    def inverse_transform(self, values):
        self.input_shape = values.shape
        return values * 2 + 3


def test_synthetic_demo_csv_is_deterministic_and_uses_saved_scaler():
    first_scaler = DemoScaler()
    second_scaler = DemoScaler()

    first = generate_synthetic_demo_csv(first_scaler)
    second = generate_synthetic_demo_csv(second_scaler)

    assert first == second
    assert first_scaler.input_shape == (SYNTHETIC_DEMO_ROWS, len(EXPECTED_FEATURES))
    assert second_scaler.input_shape == first_scaler.input_shape


def test_synthetic_demo_csv_has_expected_schema_and_enough_rows():
    scaler = DemoScaler()
    demo = pd.read_csv(io.BytesIO(generate_synthetic_demo_csv(scaler)))

    assert demo.columns.tolist() == EXPECTED_FEATURES
    assert len(demo) == SYNTHETIC_DEMO_ROWS
    assert len(demo) >= 6
    assert not any(
        name.lower() in {"label_category", "label", "device_id", "source_ip", "destination_ip"}
        for name in demo.columns
    )
    assert np.isfinite(demo.to_numpy()).all()
