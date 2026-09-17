<div align="center">

# NaviLib

### Know your data. Build with confidence.

Readable, reproducible workflows for tabular analysis and machine learning.
<br>
From the first quality check to a fitted model and a report worth sharing.

<p>
  <a href="https://pypi.org/project/NaviLib/"><img src="https://img.shields.io/pypi/v/navilib?style=flat-square&color=006dad&label=PyPI" alt="PyPI version"></a>
  <a href="https://github.com/navidml/NaviLib/actions/workflows/tests.yml"><img src="https://github.com/navidml/NaviLib/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10 or newer">
  <img src="https://img.shields.io/badge/pandas-DataFrame%20native-150458?style=flat-square&logo=pandas&logoColor=white" alt="pandas DataFrame native">
  <img src="https://img.shields.io/badge/scikit--learn-Compatible-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" alt="scikit-learn compatible">
  <img src="https://img.shields.io/badge/Reports-Offline%20HTML-0F766E?style=flat-square" alt="Offline HTML reports">
</p>

<p>
  <a href="#installation">Install</a> &middot;
  <a href="#quick-start">Quick start</a> &middot;
  <a href="#from-data-to-model">Modeling workflow</a> &middot;
  <a href="docs/API.md">API reference</a> &middot;
  <a href="examples/walkthrough.py">Full example</a>
</p>

</div>

---

NaviLib brings data cleaning, exploration, feature engineering, statistical testing, modeling, and evaluation into one Python library. Work with familiar pandas DataFrames, inspect the results of each step, and reuse fitted preprocessing when new data arrives.

Built on NumPy, pandas, SciPy, scikit-learn, Matplotlib, Seaborn, and statsmodels, it fits naturally into notebooks and Python scripts.

## Why NaviLib?

- **Understand the data first.** Inspect missing values, duplicates, suspicious columns, and label quality with suggested next steps.
- **Keep preprocessing reproducible.** Capture fitted transformations, inspect their state, and replay them on validation or production data.
- **Bring preprocessing into validation.** Use `ChainTransformer` with scikit-learn pipelines to fit learned transformations within each fold.
- **Keep results accessible.** Work directly with DataFrames, dictionaries, and Matplotlib figures for further analysis or export.
- **Make analysis look consistent.** Apply `light`, `dark`, or `paper` themes across NaviLib charts and styled tables.
- **Share a complete report.** Export standalone HTML with embedded figures and styles that opens offline.

## Installation

Requires **Python 3.10 or newer**.

```bash
pip install navilib
```

The import name is **`NaviLib`**:

```python
import NaviLib as nv
```

Add optional capabilities as needed:

| Extra | Install | Adds |
| :--- | :--- | :--- |
| Notebook | `pip install "navilib[notebook]"` | Jinja2 for styled pandas tables |
| Data I/O | `pip install "navilib[io]"` | Excel and Parquet dependencies |
| Class balancing | `pip install "navilib[balance]"` | imbalanced-learn |
| Explainability | `pip install "navilib[explain]"` | SHAP |

Extras can be combined, for example: `pip install "navilib[notebook,io]"`.

To work on NaviLib itself, clone the repository and install it in editable mode:

```bash
git clone https://github.com/navidml/NaviLib.git
cd NaviLib
python -m pip install -e ".[dev,notebook]"
```

## Quick start

Turn a DataFrame into a quality review and a shareable report:

```python
import pandas as pd
import NaviLib as nv

customers = pd.DataFrame({
    "age": [24, 31, None, 42, 29, 35],
    "city": ["Tehran", "Shiraz", "Tehran", "Tabriz", "Shiraz", "Tabriz"],
    "spend": [120.0, 85.5, 210.0, 160.0, None, 95.0],
})

nv.set_theme("light")

issues = nv.audit_data(customers)
print(issues)

report = nv.create_report(customers, title="Customer data overview")
report.to_html("outputs/customer_report.html")
```

Open `outputs/customer_report.html` in your browser. The report includes dataset summary cards, quality findings, suggested actions, descriptive tables, and plots, without changing the source DataFrame.

**The report stays useful beyond HTML.** Full analysis tables are available in `report.tables`, and reusable figure handles are in `report.figures`. Tables use all rows; plots use a reproducible sample of up to 5,000 rows by default. Profiles cover up to 12 columns per type by default, while quality checks cover every column.

## Explore the toolkit

| Module | What you can do |
| :--- | :--- |
| `cleaning` | Profile columns, handle missing values and outliers, remove duplicates, split datasets, and balance classes. |
| `eda` | Explore distributions, correlations, feature–target relationships, and differences between datasets. |
| `feature_engineering` | Transform and scale numeric features, encode categories, bin values, and create date, text, and interaction features. |
| `statistical_tests` | Run parametric and nonparametric tests, examine effect sizes and correlations, and adjust multiple p-values. |
| `modeling` | Build pipelines, cross-validate, tune parameters, compare algorithms, train, explain, and persist models. |
| `evaluation` | Evaluate classification, regression, clustering, ranking, and recommender outputs. |
| `quality` | Audit data with suggested actions, infer a reference schema, and validate incoming datasets. |
| `timeseries` | Split chronologically and create grouped lag and rolling features. |
| `theme` | Coordinate chart colors, typography, and notebook table styling. |
| `reporting` | Create reusable report objects and export standalone HTML. |

Common operations are available directly as `nv.function_name(...)`. Specialist tools remain organized under their modules, such as `nv.eda.plot_distribution(...)`.

## From data to model

This self-contained example creates synthetic classification data, introduces missing values, and trains a pipeline with preprocessing inside cross-validation.

```python
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import NaviLib as nv

# Create a reproducible dataset; no download required.
features, labels = make_classification(
    n_samples=400,
    n_features=4,
    n_informative=3,
    n_redundant=0,
    random_state=42,
)
data = pd.DataFrame(features, columns=["a", "b", "c", "d"])
data.loc[::17, "a"] = float("nan")
data["label"] = labels

# Reserve the test set before fitting any transformations.
train, test = nv.split_data(data, target="label", random_state=42)
X_train, y_train = train.drop(columns="label"), train["label"]
X_test, y_test = test.drop(columns="label"), test["label"]

steps = [
    (nv.impute_missing, {"method": "median"}),
    (nv.scale_features, {"method": "standard"}),
]
pipeline = nv.make_pipeline(
    nv.ChainTransformer(steps),
    LogisticRegression(max_iter=1000, random_state=42),
)

validation = nv.cross_validate_model(
    pipeline, X_train, y_train, cv=5, n_jobs=1, verbose=False,
)

artifact = nv.train_model(pipeline, X_train, y_train, verbose=False)
predictions = nv.predict_model(artifact, X_test)
metrics = nv.score_classification(
    y_test, predictions["prediction"], predictions["probability"],
)
print(metrics)

# Save the fitted pipeline and its evaluation together.
nv.save_model(artifact, "outputs/model.joblib", metrics=metrics)
restored = nv.load_model("outputs/model.joblib")
restored_predictions = nv.predict_model(restored, X_test)
```

Keep labels in `y`, separate from predictor columns. For grouped observations, pass `groups=` to cross-validation; for temporal data, use a time-aware splitter. Choose a fold count supported by your smallest class, and set `task="regression"` explicitly when an integer-valued target represents a regression problem.

### Fit once, replay on new data

For preprocessing outside a model pipeline, use `chain` to collect fitted states and `apply_state` to reuse them. Continuing with `X_train`, `X_test`, and `steps` from the example above:

```python
prepared_train, states = nv.chain(X_train, steps)
prepared_test = nv.apply_state(X_test, states)

print(nv.describe_states(states))

nv.save_state(states, "outputs/preprocessing.joblib")
restored_states = nv.load_state("outputs/preprocessing.joblib")
prepared_again = nv.apply_state(X_test, restored_states)
```

Use this pattern for a fixed training/test split or incoming batches. For cross-validation, keep learned preprocessing inside `ChainTransformer` so each fold fits its own transformations.

## Give your analysis a consistent look

Choose a theme for subsequent NaviLib figures, or apply one temporarily to a single part of your analysis:

```python
# Uses the customers DataFrame from the quick start.
nv.set_theme("dark", font_scale=1.1)

figure = nv.eda.plot_distribution(
    customers, columns="spend", kind="hist", show=False,
)
figure.savefig("spend_distribution.png", dpi=200, bbox_inches="tight")

with nv.theme_context("paper", palette=["#0072B2", "#E69F00", "#009E73"]):
    figure = nv.eda.plot_categorical(customers, "city", show=False)

# Requires the notebook extra; display this object in a notebook.
styled = nv.style_table(nv.audit_data(customers), caption="Data quality review")
```

| Theme | Intended use |
| :--- | :--- |
| `light` | Everyday exploration and reports |
| `dark` | Analysis on dark backgrounds |
| `paper` | Print-friendly charts and figures |

Customize the palette, font scale, and Matplotlib settings through `palette`, `font_scale`, and `rc`. Themes apply to subsequent NaviLib output, with Matplotlib settings scoped to plotting calls. Figures returned with `show=False` remain available for saving.

## More workflows

<details>
<summary><strong>Check incoming data against a reference schema</strong></summary>

Infer a schema from your reference dataset, then inspect changes in an incoming batch:

```python
schema = nv.infer_schema(customers)
incoming = customers.copy()
incoming["channel"] = "web"

changes = nv.validate_schema(incoming, schema)
print(changes)
```

Schema inference captures properties of the reference sample. You can use the resulting checks alongside your domain-specific data requirements.

</details>

<details>
<summary><strong>Build features from ordered observations</strong></summary>

```python
events = pd.DataFrame({
    "timestamp": pd.date_range("2025-01-01", periods=30, freq="D"),
    "store": ["A"] * 30,
    "sales": [20 + (day % 7) * 3 for day in range(30)],
})

train, test = nv.temporal_split(events, "timestamp", test_size=0.2, gap=2)
lagged = nv.add_lag_features(
    events, "sales", time="timestamp", group_by="store", lags=[1, 7],
)
rolling = nv.add_rolling_features(
    events, "sales", time="timestamp", group_by="store", windows=[7],
)
```

`gap` counts distinct timestamps; lags and windows count observations. Rolling features exclude the current row. Use numeric or datetime ordering columns, with unique timestamps within each group, and include only history available at prediction time.

</details>

<details>
<summary><strong>Find the right function from Python</strong></summary>

```python
nv.help_map()              # Browse the API catalog.
nv.help_map("missing")     # Search functions by purpose.
help(nv.impute_missing)    # Inspect parameters and return values.
```

The catalog includes module names, signatures, summaries, and compatibility aliases.

</details>

## Documentation and examples

| Resource | What you will find |
| :--- | :--- |
| [API reference](docs/API.md) | Function signatures, parameters, return values, and usage notes |
| [Migration guide](docs/MIGRATION.md) | Preferred names, compatibility aliases, and behavior changes in 0.5 |
| [Executable walkthrough](examples/walkthrough.py) | Synthetic data, model evaluation, and HTML reports in all three themes |

The walkthrough lives in the repository. Clone it (or download `examples/walkthrough.py`), then run from the repository root:

```bash
python examples/walkthrough.py
```

It writes reports, charts, cross-validation results, and metrics to `examples/output/`.

### Practical notes

- Quality findings suggest what to investigate; review them in the context of your dataset.
- Target and WOE encoding use internal randomized folds. Grouped and temporal problems need additional care when choosing these transformations.
- HTML reports contain dataset-derived values and embedded figures. Review their contents before sharing.
- Model and preprocessing files use joblib. Load only trusted artifacts and preserve a compatible dependency environment for reuse.

## Development

Install the development extras, then run the checks or regenerate the API reference:

```bash
python -m pip install -e ".[dev,notebook]"
python -m pytest -q
python -m build
python tools/build_docs.py
```

For a bug report, include a minimal reproducible example, your Python and NaviLib versions, and the expected behavior. Small synthetic datasets make examples easier to reproduce and share.

## License

Released under the [MIT License](LICENSE).

---

<div align="center">
  <strong>NaviLib</strong><br>
  Understand the data. Reproduce the workflow. Share the results.
</div>
