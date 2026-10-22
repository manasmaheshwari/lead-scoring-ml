"""Train, evaluate, persist, score and explain."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import FEATURES

CATEGORICAL = ["source"]
NUMERIC = [f for f in FEATURES if f not in CATEGORICAL]


def build_pipeline(kind: str = "gbm") -> Pipeline:
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if kind == "gbm":
        pre = ColumnTransformer([("cat", onehot, CATEGORICAL)],
                                remainder="passthrough", verbose_feature_names_out=False)
        clf = GradientBoostingClassifier(random_state=0)
    elif kind == "logreg":
        pre = ColumnTransformer([("cat", onehot, CATEGORICAL), ("num", StandardScaler(), NUMERIC)],
                                verbose_feature_names_out=False)
        clf = LogisticRegression(max_iter=1000)
    else:
        raise ValueError(f"unknown model kind: {kind}")
    return Pipeline([("pre", pre), ("clf", clf)])


def train(df: pd.DataFrame, kind: str = "gbm", seed: int = 0):
    """Fit on a stratified 75% split. Returns (pipeline, test_auc, test_df)."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[FEATURES], df["converted"], test_size=0.25, stratify=df["converted"], random_state=seed)
    pipe = build_pipeline(kind).fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
    return pipe, auc, X_te.assign(converted=y_te)


def score(pipe: Pipeline, leads: pd.DataFrame) -> pd.Series:
    """Conversion probability scaled to a 0-100 lead score."""
    return pd.Series(pipe.predict_proba(leads[FEATURES])[:, 1] * 100, index=leads.index).round(1)


def save(pipe: Pipeline, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)


def load(path: str) -> Pipeline:
    return joblib.load(path)


def _explainer_and_matrix(pipe: Pipeline, background: pd.DataFrame, rows: pd.DataFrame):
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    names = list(pre.get_feature_names_out())
    X = pre.transform(rows[FEATURES])
    if isinstance(clf, GradientBoostingClassifier):
        expl = shap.TreeExplainer(clf)
    else:
        expl = shap.LinearExplainer(clf, pre.transform(background[FEATURES]))
    return expl, names, X


def global_drivers(pipe: Pipeline, df: pd.DataFrame, top: int = 6, sample: int = 500) -> pd.Series:
    """Mean |SHAP| per feature (log-odds scale), largest first."""
    rows = df.sample(min(sample, len(df)), random_state=0)
    expl, names, X = _explainer_and_matrix(pipe, rows, rows)
    vals = np.abs(expl.shap_values(X)).mean(axis=0)
    return pd.Series(vals, index=names).sort_values(ascending=False).head(top)


def explain_lead(pipe: Pipeline, background: pd.DataFrame, lead: pd.DataFrame, top: int = 3) -> list[tuple[str, float]]:
    """Top per-lead SHAP contributions (signed, log-odds) for a single-row frame."""
    expl, names, X = _explainer_and_matrix(pipe, background, lead)
    vals = np.asarray(expl.shap_values(X))[0]
    order = np.argsort(-np.abs(vals))[:top]
    return [(names[i], float(vals[i])) for i in order]
