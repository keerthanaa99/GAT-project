REVIEW_BELOW = 0.50
HIGH_FROM = 0.80


def severity_of(category: str, confidence: float) -> str:
    if confidence < REVIEW_BELOW:
        return "Review"
    if category == "Benign":
        return "Normal"
    return "High" if confidence >= HIGH_FROM else "Medium"


def count_alerts(table) -> tuple[int, int]:
    attack_alerts = int(
        ((table["Prediction"] != "Benign") & table["_sev"].isin(("High", "Medium"))).sum()
    )
    reviews = int((table["_sev"] == "Review").sum())
    return attack_alerts, reviews
