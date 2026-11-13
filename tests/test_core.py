import numpy as np
import pandas as pd

from rfmseg.core import (
    clean_transactions,
    cluster_metrics,
    fit_kmeans,
    rfm_features,
    scale_features,
)


def _raw(rows):
    df = pd.DataFrame(
        rows,
        columns=["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price", "Customer ID"],
    )
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df.assign(Description="x", Country="x")


def test_cancelled_missing_and_nonpositive_rows_are_removed():
    raw = _raw(
        [
            ["1", "A", 2, "2011-01-01", 5.0, 10.0],
            ["C2", "A", 1, "2011-01-02", 5.0, 10.0],
            ["3", "A", 1, "2011-01-03", 5.0, None],
            ["4", "A", -1, "2011-01-04", 5.0, 10.0],
            ["5", "A", 1, "2011-01-05", 0.0, 10.0],
        ]
    )
    tx = clean_transactions(raw)
    assert list(tx["Invoice"]) == ["1"]
    assert tx["revenue"].iloc[0] == 10.0


def test_rfm_uses_only_transactions_before_snapshot():
    tx = clean_transactions(
        _raw(
            [
                ["1", "A", 1, "2011-05-01", 10.0, 1.0],
                ["2", "A", 2, "2011-05-20", 10.0, 1.0],
                ["3", "A", 1, "2011-06-05", 100.0, 1.0],  # after snapshot, ignored
            ]
        )
    )
    feats = rfm_features(tx, snapshot=pd.Timestamp("2011-06-01"))
    row = feats.loc[1]
    assert row["recency_days"] == 12
    assert row["frequency"] == 2
    assert row["monetary"] == 30.0
    assert row["tenure_days"] == 31
    assert row["avg_order_value"] == 15.0


def test_monetary_is_capped_at_quantile():
    tx = clean_transactions(
        _raw(
            [[str(i), "A", 1, "2011-05-01", float(v), float(i)] for i, v in enumerate([1, 2, 3, 4, 1000], start=1)]
        )
    )
    feats = rfm_features(tx, snapshot=pd.Timestamp("2011-06-01"), monetary_cap_quantile=0.8)
    assert feats["monetary"].max() < 1000


def test_scale_features_is_finite_and_centered_on_median():
    feats = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0, 100.0], "b": [1, 1, 1, 1, 1]})
    X = scale_features(feats)
    assert np.isfinite(X).all()
    assert abs(np.median(X[:, 0])) < 1e-12


def test_kmeans_is_reproducible_and_metrics_are_reported():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.1, (30, 2)), rng.normal(5, 0.1, (30, 2))])
    _, labels_a = fit_kmeans(X, 2)
    _, labels_b = fit_kmeans(X, 2)
    assert (labels_a == labels_b).all()
    metrics = cluster_metrics(X, labels_a)
    assert metrics["silhouette"] > 0.9
    assert set(metrics) == {"silhouette", "davies_bouldin", "calinski_harabasz"}
