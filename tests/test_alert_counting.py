import pandas as pd
import pytest

from src.dashboard.alert_utils import count_alerts, severity_of


@pytest.mark.parametrize(
    ("category", "confidence", "expected"),
    [
        ("Benign", 0.90, "Normal"),
        ("ARP Spoofing", 0.90, "High"),
        ("ARP Spoofing", 0.49, "Review"),
        ("Benign", 0.49, "Review"),
        ("MQTT attacks", 0.60, "Medium"),
    ],
)
def test_severity_of(category, confidence, expected):
    assert severity_of(category, confidence) == expected


def test_count_alerts_keeps_review_rows_separate_from_attack_alerts():
    rows = [
        ("Benign", 0.90),
        ("ARP Spoofing", 0.90),
        ("TCP/IP DoS/DDoS", 0.49),
        ("Benign", 0.49),
        ("MQTT attacks", 0.60),
    ]
    table = pd.DataFrame(
        [
            {"Prediction": category, "_sev": severity_of(category, confidence)}
            for category, confidence in rows
        ]
    )

    attack_alerts, reviews = count_alerts(table)

    assert attack_alerts == 2
    assert reviews == 2
    assert attack_alerts + reviews == 4
