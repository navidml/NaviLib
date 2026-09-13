import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import NaviLib as nv


@pytest.fixture
def frame():
    rng = np.random.default_rng(1)
    return pd.DataFrame({"x": rng.lognormal(size=60), "z": rng.uniform(.5, 4, 60),
                         "city": np.tile(["a", "b", "c"], 20), "target": np.tile([0, 1], 30)})


@pytest.mark.parametrize("method", nv.feature_engineering.TRANSFORMS)
def test_numeric_transforms_replay(frame, method):
    out, state = nv.transform_numeric(frame, ["x"], method=method, return_state=True)
    assert_frame_equal(out, nv.apply_state(frame, state), atol=1e-8)


@pytest.mark.parametrize("method", nv.feature_engineering.SCALERS)
def test_scalers_replay(frame, method):
    out, state = nv.scale_features(frame, ["x", "z"], method=method, return_state=True)
    assert_frame_equal(out, nv.apply_state(frame, state), atol=1e-8)


@pytest.mark.parametrize("method", nv.feature_engineering.ENCODERS)
def test_encoders_replay_and_unseen(frame, method):
    options = {"target": "target"} if method in ("target", "woe") else {}
    if method == "ordinal":
        options["order"] = {"city": ["a", "b", "c"]}
    out, state = nv.encode_categorical(frame, ["city"], method=method, return_state=True, **options)
    replayed = nv.apply_state(frame, state)
    assert list(out) == list(replayed)
    if method not in ("target", "woe"):
        assert_frame_equal(out, replayed)
    incoming = frame.iloc[:2].copy()
    incoming["city"] = ["unseen", None]
    result = nv.apply_state(incoming.drop(columns="target"), state)
    assert len(result) == 2
    assert list(result) == [c for c in out if c != "target"]


@pytest.mark.parametrize("method", ["mean", "median", "mode", "constant", "knn", "mice", "ffill", "bfill"])
def test_imputers_replay(frame, method):
    frame.loc[1:3, "x"] = np.nan
    frame.loc[2:4, "city"] = None
    out, state = nv.impute_missing(frame, method=method, fill_value=0, return_state=True)
    assert_frame_equal(out, nv.apply_state(frame, state))


@pytest.mark.parametrize("method", ["iqr", "zscore", "mad", "quantile"])
def test_outliers_replay(frame, method):
    out, state = nv.handle_outliers(frame, ["x"], method=method, return_state=True)
    assert_frame_equal(out, nv.apply_state(frame, state))


@pytest.mark.parametrize("method", ["quantile", "uniform", "kmeans", "custom", "tree"])
def test_binning_replay(frame, method):
    options = {"bins": [0, 1, 2, 10]} if method == "custom" else {"bins": 3}
    out, state = nv.bin_numeric(frame, ["x"], method=method, target="target", return_state=True, **options)
    assert_frame_equal(out, nv.apply_state(frame, state))


def test_derived_features_replay(frame):
    frame["time"] = pd.date_range("2024-01-01", periods=len(frame), freq="h")
    for fn, args, kwargs in [
        (nv.add_datetime_features, (["time"],), {}),
        (nv.add_cyclical_features, ("x",), {"period": 24}),
        (nv.feature_engineering.add_interactions, (["x", "z"],), {}),
        (nv.feature_engineering.add_aggregates, (), {"group": "city", "values": ["x", "z"]}),
        (nv.feature_engineering.add_text_features, (["city"],), {}),
    ]:
        out, state = fn(frame, *args, return_state=True, **kwargs)
        assert_frame_equal(out, nv.apply_state(frame, state))


def test_new_pandas_string_targets_and_dates():
    frame = pd.DataFrame({"time": pd.Series(["2024-01-01", "2024-01-02"], dtype="string")})
    assert "time_year" in nv.add_datetime_features(frame, ["time"])
    target = pd.Series([f"c{i}" for i in range(25)] * 2, dtype="string")
    assert nv.modeling._infer_task(target) == "multiclass"


def test_imputation_nullable_and_all_missing_categories():
    df = pd.DataFrame({"x": pd.Series([1, 2, pd.NA], dtype="Int64"),
                       "c": pd.Series([None] * 3, dtype="category")})
    out, state = nv.impute_missing(df, method="median", fill_value="missing", return_state=True)
    assert out.x.iloc[-1] == 1.5
    assert out.c.eq("missing").all()
    assert_frame_equal(out, nv.apply_state(df, state))


def test_statistical_tests_and_multiple_correction(frame):
    stats = nv.statistical_tests
    for method in ["shapiro", "dagostino", "anderson", "ks"]:
        assert "statistic" in stats.test_normality(frame, "x", method=method, show_plot=False)
    for method in ["pearson", "spearman", "kendall", "partial"]:
        result = stats.test_correlation(frame, method, "x", "z", control="target" if method == "partial" else None,
                                         show_plot=False)
        assert 0 <= result["p_value"] <= 1
    for method in ["chi_square_independence", "chi_square_gof"]:
        result = stats.test_categorical(frame, method, col1="city", col2="target",
                                         expected=[1/3] * 3, show_plot=False)
        assert 0 <= result["p_value"] <= 1
    corrected = stats.adjust_pvalues(pd.Series([.01, .04, None], index=["a", "b", "c"]), method="holm")
    np.testing.assert_allclose(corrected.p_adjusted[:2], [.02, .04])
    assert not corrected.reject.iloc[2]


def test_regression_zero_smape():
    result = nv.score_regression([0., 1.], [0., 0.])
    assert result.loc["smape_pct", "value"] == 100.


def test_custom_palette_applies_after_import(frame):
    from matplotlib.colors import to_hex
    with nv.theme_context("dark", palette=["#ABCDEF", "#FEDCBA"]):
        fig = nv.eda.plot_distribution(frame, "x", kind="hist", show=False)
        assert to_hex(fig.axes[0].patches[0].get_facecolor()) == "#abcdef"


def test_string_class_scores_and_one_fit_per_fold(frame):
    from sklearn.linear_model import LogisticRegression
    from sklearn.base import BaseEstimator, ClassifierMixin

    class CountingClassifier(ClassifierMixin, BaseEstimator):
        fits = 0

        def fit(self, X, y):
            type(self).fits += 1
            self.model_ = LogisticRegression().fit(X, y)
            self.classes_ = self.model_.classes_
            return self

        def predict(self, X):
            return self.model_.predict(X)

        def predict_proba(self, X):
            return self.model_.predict_proba(X)

    result = nv.cross_validate_model(CountingClassifier(), frame[["x", "z"]],
        frame.target.map({0: "no", 1: "yes"}), cv=3, n_jobs=1, verbose=False)
    assert CountingClassifier.fits == 3
    assert result.attrs["classes"] == ["no", "yes"]
    assert result.loc["brier", "mean"] >= 0


def test_grouped_nested_validation(frame):
    from sklearn.linear_model import LogisticRegression
    result = nv.modeling.nested_validate(LogisticRegression(), {"C": [.1, 1]},
        frame[["x", "z"]], frame.target, groups=np.repeat(np.arange(12), 5),
        inner_cv=2, outer_cv=3, n_iter=2, scoring="accuracy", n_jobs=1, verbose=False)
    assert len(result) == 3


def test_pipeline_coefficient_feature_names(frame):
    from sklearn.linear_model import LogisticRegression
    prep = nv.modeling.make_preprocessor(numeric=["x", "z"], categorical=["city"])
    model = nv.make_pipeline(prep, LogisticRegression()).fit(frame.drop(columns="target"), frame.target)
    result = nv.explain(model, frame.drop(columns="target"), method="coef")
    assert len(result) == 5
    assert set(result.feature) == {"x", "z", "city_a", "city_b", "city_c"}


def test_resampling_is_training_only_and_pipeline_predict_preserves_rows(frame):
    pytest.importorskip("imblearn")
    from sklearn.linear_model import LogisticRegression
    df = frame[["x", "z"]].copy()
    df["target"] = np.r_[np.zeros(45, dtype=int), np.ones(15, dtype=int)]
    with pytest.warns(UserWarning, match="TRAINING"):
        out, report = nv.resample_data(df, "target", method="smote")
    assert report["counts_after"] == {0: 45, 1: 45}
    assert len(out) == 90
    pipe = nv.balance_pipeline("smote", LogisticRegression())
    pipe.fit(df.drop(columns="target"), df.target)
    assert len(pipe.predict(df.drop(columns="target"))) == len(df)


def test_positive_label_matrix_matches_vector_in_thresholds_and_plots():
    y = np.array([0, 1, 0, 1, 0, 1])
    p = np.array([.9, .2, .8, .1, .7, .3])
    matrix = np.column_stack([p, 1 - p])
    assert_frame_equal(nv.evaluation.threshold_sweep(y, matrix, pos_label=0),
                       nv.evaluation.threshold_sweep(y, p, pos_label=0))
    fig = nv.evaluation.plot_classification(y, y_prob=matrix, pos_label=0, show=False)
    assert len(fig.axes) >= 6


def test_weighted_regression_metrics_and_intervals():
    y = np.array([1., 2., 3., 4., 5.])
    p = y + np.array([0., 1., 0., 1., 2.])
    w = np.array([10., 1., 10., 1., 1.])
    result = nv.score_regression(y, p, sample_weight=w, ci=True, n_boot=30)
    assert result.loc["bias", "value"] == pytest.approx(np.average(y - p, weights=w), abs=1e-6)
    assert result.loc["mae", "value"] == pytest.approx(np.average(abs(y - p), weights=w), abs=1e-6)
    assert result.loc["mae", "ci_high"] >= result.loc["mae", "ci_low"]
