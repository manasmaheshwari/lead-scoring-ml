"""Command line: generate | train | score | explain."""
import argparse
import sys

import pandas as pd

from . import model
from .data import generate_leads

MODEL_PATH = "artifacts/model.joblib"
BACKGROUND_PATH = "artifacts/background.csv"


def cmd_generate(a):
    generate_leads(a.n, a.seed).to_csv(a.out, index=False)
    print(f"wrote {a.n} synthetic leads to {a.out}")


def cmd_train(a):
    df = generate_leads(a.n, a.seed)
    pipe, auc, test = model.train(df, a.model)
    model.save(pipe, MODEL_PATH)
    df.sample(300, random_state=0).to_csv(BACKGROUND_PATH, index=False)
    print(f"model={a.model}  test AUC={auc:.3f}  (base rate {df['converted'].mean():.1%})")
    print("\nTop global drivers (mean |SHAP|, log-odds):")
    print(model.global_drivers(pipe, test).round(3).to_string())


def cmd_score(a):
    pipe = model.load(MODEL_PATH)
    leads = pd.read_csv(a.input)
    leads["score"] = model.score(pipe, leads)
    print(leads.sort_values("score", ascending=False).head(a.top).to_string())


def cmd_explain(a):
    pipe = model.load(MODEL_PATH)
    background = pd.read_csv(BACKGROUND_PATH)
    lead = pd.read_csv(a.input).iloc[[a.row]]
    print(f"score: {model.score(pipe, lead).iloc[0]}")
    for name, v in model.explain_lead(pipe, background, lead):
        print(f"  {name:28s} {v:+.3f} {'raises' if v > 0 else 'lowers'} score")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="leadscore")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="write synthetic leads to CSV")
    g.add_argument("--n", type=int, default=5000)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--out", default="data/leads.csv")
    g.set_defaults(fn=cmd_generate)

    t = sub.add_parser("train", help="train, report AUC and SHAP drivers, save model")
    t.add_argument("--model", choices=["gbm", "logreg"], default="gbm")
    t.add_argument("--n", type=int, default=5000)
    t.add_argument("--seed", type=int, default=42)
    t.set_defaults(fn=cmd_train)

    s = sub.add_parser("score", help="score a CSV of leads")
    s.add_argument("input")
    s.add_argument("--top", type=int, default=10)
    s.set_defaults(fn=cmd_score)

    e = sub.add_parser("explain", help="SHAP explanation for one lead")
    e.add_argument("input")
    e.add_argument("--row", type=int, default=0)
    e.set_defaults(fn=cmd_explain)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
