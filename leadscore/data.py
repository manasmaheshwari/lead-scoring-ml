"""Synthetic lead data generator.

Conversion probability is a known logistic function of a few features plus
noise, so the model has real signal to find (and we can sanity-check SHAP).
"""
import numpy as np
import pandas as pd

SOURCES = ["organic", "paid_search", "referral", "webinar", "cold_outbound"]
SOURCE_EFFECT = {"organic": 0.2, "paid_search": -0.3, "referral": 1.0, "webinar": 0.5, "cold_outbound": -0.8}
FEATURES = ["source", "company_size", "pages_viewed", "demo_requested", "days_since_first_touch", "emails_opened"]


def generate_leads(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "source": rng.choice(SOURCES, n, p=[0.25, 0.30, 0.10, 0.10, 0.25]),
        "company_size": rng.lognormal(3.5, 1.2, n).round().astype(int).clip(1, 10000),
        "pages_viewed": rng.poisson(5, n),
        "demo_requested": rng.binomial(1, 0.15, n),
        "days_since_first_touch": rng.integers(0, 90, n),
        "emails_opened": rng.poisson(2, n),
    })
    logit = (
        -2.6
        + df["source"].map(SOURCE_EFFECT)
        + 0.25 * np.log1p(df["company_size"])
        + 0.12 * df["pages_viewed"]
        + 1.8 * df["demo_requested"]
        + 0.15 * df["emails_opened"]
        - 0.02 * df["days_since_first_touch"]
        + rng.normal(0, 0.5, n)
    )
    df["converted"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return df
