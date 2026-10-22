import pandas as pd
import pytest

from leadscore import model
from leadscore.cli import main
from leadscore.data import FEATURES, generate_leads


@pytest.fixture(scope="module")
def trained():
    df = generate_leads(3000, seed=1)
    pipe, auc, test = model.train(df)
    return df, pipe, auc, test


def test_generator_shape_and_determinism():
    a, b = generate_leads(200, seed=7), generate_leads(200, seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert set(FEATURES) | {"converted"} == set(a.columns)
    assert 0.05 < a["converted"].mean() < 0.6


def test_auc_beats_chance(trained):
    assert trained[2] > 0.70


def test_scores_in_range_and_demo_matters(trained):
    df, pipe, _, _ = trained
    s = model.score(pipe, df)
    assert s.between(0, 100).all()
    assert s[df["demo_requested"] == 1].mean() > s[df["demo_requested"] == 0].mean()


def test_shap_top_drivers_are_plausible(trained):
    _, pipe, _, test = trained
    drivers = model.global_drivers(pipe, test, top=4)
    assert {"demo_requested", "pages_viewed"} & set(drivers.index)


def test_explain_lead_returns_signed_contributions(trained):
    _, pipe, _, test = trained
    out = model.explain_lead(pipe, test, test.iloc[[0]], top=3)
    assert len(out) == 3 and all(isinstance(v, float) for _, v in out)


def test_logreg_variant():
    _, auc, _ = model.train(generate_leads(2000, seed=3), kind="logreg")
    assert auc > 0.7


def test_cli_end_to_end(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    assert main(["train", "--n", "1500"]) == 0
    assert main(["generate", "--n", "50", "--out", "data/new.csv"]) == 0
    assert main(["score", "data/new.csv", "--top", "3"]) == 0
    assert main(["explain", "data/new.csv", "--row", "0"]) == 0
    assert "test AUC" in capsys.readouterr().out
