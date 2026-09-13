import importlib

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.linear_model import LogisticRegression, Ridge

import NaviLib as nv


def test_public_exports_and_old_import():
    for name in nv.__all__:
        assert hasattr(nv, name), name
    assert nv.impute_missing is nv.fix_missing
    assert importlib.import_module("NaviLib.Statistical_Tests") is nv.statistical_tests
    assert not nv.help_map("missing").empty


def test_mixed_state_round_trip(tmp_path):
    train = pd.DataFrame({"x": [1., 2., None, 4.], "city": ["a", "b", "a", "b"]})
    source = train.copy(deep=True)
    out, s1 = nv.impute_missing(train, return_state=True)
    out, s2 = nv.encode_categorical(out, ["city"], return_state=True)
    out, s3 = nv.scale_features(out, ["x"], return_state=True)
    states = [s1, s2, s3]
    assert_frame_equal(nv.apply_state(train, states), out)
    assert_frame_equal(train, source)
    file = tmp_path / "prep.joblib"
    nv.save_state(states, file)
    assert_frame_equal(nv.apply_state(train, nv.load_state(file)), out)
    test = nv.apply_state(pd.DataFrame({"x": [None], "city": ["new"]}), states)
    assert list(test) == list(out)
    assert test.filter(like="city_").to_numpy().sum() == 0
    with pytest.raises(KeyError, match="Missing source"):
        nv.apply_state(train.drop(columns="x"), states)
    with pytest.warns(UserWarning, match="skipped"):
        nv.apply_state(train.drop(columns="x"), s1, strict=False)


def test_pipeline_excludes_target_and_preserves_oof():
    X = pd.DataFrame({"x": [1., 2., None, 4., 5., 6., 7., 8.], "city": list("aabbaabb")})
    y = np.array([0, 1] * 4)
    prep = nv.ChainTransformer([
        (nv.impute_missing, {}),
        (nv.encode_categorical, {"columns": ["city"], "method": "target", "cv": 2}),
        (nv.scale_features, {}),
    ], target="label")
    out = prep.fit_transform(X, y)
    assert "label" not in out
    assert "label" not in prep.states_[0]["columns"]
    assert "label" not in prep.states_[2]["columns"]
    assert list(prep.transform(X)) == list(out)
    pipe = nv.make_pipeline(prep, LogisticRegression())
    result = nv.cross_validate_model(pipe, X, y, cv=2, scoring="accuracy", n_jobs=1, verbose=False)
    assert result.attrs["oof"].shape == (8,)
    with pytest.raises(ValueError, match="target"):
        prep.fit(X.assign(label=y), y)


def test_pipeline_rejects_row_dropping():
    prep = nv.ChainTransformer([(nv.impute_missing, {"method": "drop_rows"})])
    with pytest.raises(ValueError, match="row count"):
        prep.fit(pd.DataFrame({"x": [1, None]}), [0, 1])


def test_zero_threshold_positive_prevalence_and_saved_model(tmp_path):
    X = pd.DataFrame({"x": np.arange(20)})
    y = np.array([0] * 5 + [1] * 15)
    art = nv.train_model(LogisticRegression(), X, y, threshold=0, verbose=False)
    assert art["prevalence"] == .75
    assert nv.predict_model(art, X)["prediction"].eq(1).all()
    path = tmp_path / "model.joblib"
    nv.save_model(art, path)
    assert_frame_equal(nv.predict_model(art, X), nv.predict_model(nv.load_model(path), X))


def test_custom_negative_scorer_ranks_lower_error_first():
    from sklearn.dummy import DummyRegressor
    X = np.arange(30).reshape(-1, 1)
    y = X.ravel() * 2.
    scores = nv.compare_algorithms(X, y, models={"ridge": Ridge(), "dummy": DummyRegressor()},
        include_baseline=False, scoring="neg_mean_absolute_error", cv=3, task="regression", n_jobs=1, verbose=False)
    assert scores.index[0] == "ridge"


def test_column_names_are_unique_unicode_and_empty_safe():
    df = pd.DataFrame([[1, 2, 3, 4, 5]], columns=["A", "A", "A_2", "نام مشتری", "نام مشتری"])
    result = nv.clean_names(df)
    assert result.columns.is_unique
    assert "نام_مشتری" in result
    assert list(nv.clean_names(df.iloc[:0])) == list(result)


def test_woe_does_not_fall_back_to_in_sample():
    with pytest.raises(ValueError, match="two rows per class"):
        nv.encode_categorical(pd.DataFrame({"c": list("abcc"), "y": [0, 0, 0, 1]}),
                              ["c"], method="woe", target="y")


def test_probabilities_and_bootstrap_validation():
    y = np.array([0, 1, 0, 1])
    p = np.array([[.9, .1], [.2, .8], [.8, .2], [.1, .9]])
    result = nv.score_classification(y, y_prob=p, pos_label=0)
    assert result.loc["roc_auc", "value"] == 1
    with pytest.raises(ValueError, match="Probabilities"):
        nv.score_classification(y, y_prob=[-.1, .8, .2, .9])
    with pytest.raises(ValueError, match="equal lengths"):
        nv.bootstrap_ci(lambda a, b: np.mean(a - b), [1, 2], [1], n_boot=20)


def test_drift_string_column_selection():
    a = pd.DataFrame({"amount": np.arange(100), "city": ["a"] * 100})
    result = nv.compare_distributions(a, a, columns="amount")
    assert list(result.index) == ["amount"]
    assert result.loc["amount", "psi"] == 0


def test_quality_and_schema():
    reference = pd.DataFrame({"x": [1., 2., 3.], "city": ["a", "b", "a"]})
    current = pd.DataFrame({"x": [np.inf, np.nan], "city": ["z", " a "]})
    issues = nv.audit_data(current)
    assert {"infinite", "missing", "whitespace"}.issubset(set(issues.issue))
    schema = nv.infer_schema(reference)
    problems = nv.validate_schema(current, schema)
    assert {"unexpected_null", "unseen_category"}.issubset(set(problems.issue))
    assert nv.validate_schema(reference, schema).empty


def test_historical_features_do_not_cross_groups_or_use_current_values():
    df = pd.DataFrame({"time": [3, 1, 2, 1, 2], "id": ["a", "a", "a", "b", "b"],
                       "x": [30., 10., 20., 100., 200.]}, index=[0, 0, 2, 3, 4])
    lagged = nv.add_lag_features(df, "x", time="time", group_by="id")
    np.testing.assert_allclose(lagged.x_lag_1, [20, np.nan, 10, np.nan, 100], equal_nan=True)
    rolled = nv.add_rolling_features(df, "x", time="time", group_by="id", windows=[2], statistics=["mean"])
    np.testing.assert_allclose(rolled.x_rolling_mean_2, [15, np.nan, 10, np.nan, 100], equal_nan=True)
    assert lagged.index.equals(df.index)
    tr, te = nv.temporal_split(pd.DataFrame({"time": np.repeat(np.arange(10), 2)}), "time", gap=2)
    assert tr.time.max() == 5 and te.time.min() == 8
    with pytest.raises(ValueError, match="unique"):
        nv.add_lag_features(pd.concat([df, df]), "x", time="time", group_by="id")


@pytest.mark.parametrize("theme", ["light", "dark", "paper"])
def test_theme_report_and_no_global_plot_leaks(theme, tmp_path):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex
    before = mpl.rcParams.copy()
    df = pd.DataFrame({"value": [1., 2., 3., 4., 5., np.nan], "group": ["a", "a", "b", "b", "c", "c"]})
    with nv.theme_context(theme):
        fig = nv.eda.plot_distribution(df, "value", kind="hist", show=False)
        assert to_hex(fig.get_facecolor()) == nv.get_theme()["background"].lower()
        result = nv.create_report(df, title='<script>alert("x")</script>')
        html = result.to_html(tmp_path / f"{theme}.html")
        assert '<script>' not in html and '&lt;script&gt;' in html
        assert 'data:image/png;base64,' in html
        assert 'Quality and next steps' in html
    assert mpl.rcParams == before
    assert not plt.get_fignums()


def test_lilliefors_matches_statsmodels():
    from statsmodels.stats.diagnostic import lilliefors
    df = pd.DataFrame({"x": np.random.default_rng(42).normal(size=100)})
    result = nv.statistical_tests.test_normality(df, "x", method="ks", show_plot=False)
    assert result["p_value"] == pytest.approx(lilliefors(df.x)[1])
    assert result["figures"] == []
