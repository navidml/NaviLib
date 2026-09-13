"""Executable workflow using synthetic data only: reports, modeling and themes.

Run from the project root: ``python examples/walkthrough.py``.
Outputs are written under examples/output (ignored by git).
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import NaviLib as nv


def main():
    rng = np.random.default_rng(42)
    n = 450
    age = rng.integers(18, 76, n).astype(float)
    spend = rng.lognormal(4.5, .65, n)
    city = rng.choice(["Tehran", "Shiraz", "Tabriz", "Isfahan"], n)
    probability = 1 / (1 + np.exp(-((age - 45) / 18 + (spend - 100) / 90)))
    target = rng.binomial(1, probability)
    age[rng.choice(n, 30, replace=False)] = np.nan
    df = pd.DataFrame({"age": age, "spend": spend, "city": city, "responded": target})
    output = ROOT / "examples" / "output"
    output.mkdir(exist_ok=True)
    train, test = nv.split_data(df, "responded")
    X, y = train.drop(columns="responded"), train.responded
    prep = nv.ChainTransformer([
        (nv.impute_missing, {"method": "median"}),
        (nv.encode_categorical, {"columns": ["city"]}),
        (nv.scale_features, {"columns": ["age", "spend"]}),
    ])
    pipeline = nv.make_pipeline(prep, LogisticRegression(max_iter=1000))
    cv = nv.cross_validate_model(pipeline, X, y, cv=3, n_jobs=1, verbose=False)
    artifact = nv.train_model(pipeline, X, y, verbose=False)
    prediction = nv.predict_model(artifact, test.drop(columns="responded"))
    metrics = nv.score_classification(test.responded, prediction.prediction, prediction.probability)
    cv.to_csv(output / "cross_validation.csv")
    metrics.to_csv(output / "metrics.csv")
    for theme in nv.available_themes():
        report = nv.create_report(df, target="responded", title="Customer response · Synthetic demo", theme=theme)
        report.to_html(output / f"report_{theme}.html")
        report.figures["Distributions"].savefig(output / f"distributions_{theme}.png", bbox_inches="tight")
        with nv.theme_context(theme):
            figure = nv.evaluation.plot_classification(test.responded, prediction.prediction,
                                                       prediction.probability, show=False)
            figure.savefig(output / f"classification_{theme}.png", bbox_inches="tight")
    print(f"Reports and figures: {output}")
    print(metrics.loc[["roc_auc", "average_precision", "brier"], ["value"]].to_string())


if __name__ == "__main__":
    main()
