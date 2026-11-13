"""RFM feature engineering, scaling and clustering metrics for Online Retail II."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import RobustScaler

RANDOM_STATE = 42
SNAPSHOT_DATE = pd.Timestamp("2011-06-01")
DEFAULT_DATA_PATH = Path.home() / ".local/share/clv-data/online-retail-ii/online_retail_II.csv"


def data_path() -> Path:
    return Path(os.environ.get("RETAIL_DATA_PATH", DEFAULT_DATA_PATH))


def load_retail(path: Path | None = None) -> pd.DataFrame:
    """Load transactions and apply the cleaning rules documented in the README."""
    raw = pd.read_csv(
        path or data_path(),
        encoding="latin1",
        parse_dates=["InvoiceDate"],
        dtype={"Invoice": str, "StockCode": str},
    )
    return clean_transactions(raw)


def clean_transactions(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns={"Customer ID": "customer_id"})
    df = df.dropna(subset=["customer_id"]).copy()
    df["customer_id"] = df["customer_id"].astype("int64")
    df = df[~df["Invoice"].str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)].copy()
    df["revenue"] = df["Quantity"] * df["Price"]
    return df[["Invoice", "customer_id", "InvoiceDate", "revenue"]]


def rfm_features(
    tx: pd.DataFrame,
    snapshot: pd.Timestamp = SNAPSHOT_DATE,
    monetary_cap_quantile: float = 0.99,
) -> pd.DataFrame:
    """Customer-level RFM features using only transactions strictly before the snapshot."""
    past = tx[tx["InvoiceDate"] < snapshot]
    grouped = past.groupby("customer_id")
    feats = pd.DataFrame(
        {
            "recency_days": (snapshot - grouped["InvoiceDate"].max()).dt.days,
            "frequency": grouped["Invoice"].nunique(),
            "monetary": grouped["revenue"].sum(),
            "tenure_days": (snapshot - grouped["InvoiceDate"].min()).dt.days,
        }
    )
    recent = past[past["InvoiceDate"] >= snapshot - pd.Timedelta(days=90)]
    feats["monetary_90d"] = recent.groupby("customer_id")["revenue"].sum().reindex(feats.index).fillna(0.0)
    feats["avg_order_value"] = feats["monetary"] / feats["frequency"]
    cap = feats["monetary"].quantile(monetary_cap_quantile)
    for col in ["monetary", "monetary_90d", "avg_order_value"]:
        feats[col] = feats[col].clip(upper=cap)
    return feats


def scale_features(feats: pd.DataFrame) -> np.ndarray:
    """log1p for skewed positive features, then a robust scaler."""
    logged = np.log1p(feats.clip(lower=0))
    return RobustScaler().fit_transform(logged)


def fit_kmeans(X: np.ndarray, k: int) -> tuple[KMeans, np.ndarray]:
    model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    labels = model.fit_predict(X)
    return model, labels


def cluster_metrics(X: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "silhouette": float(silhouette_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
    }


def segment_sizes(labels: np.ndarray) -> pd.Series:
    return pd.Series(labels).value_counts(normalize=True).sort_index()


def segment_profile(feats: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Median RFM values and size per segment, to make clusters readable."""
    prof = feats.assign(segment=labels).groupby("segment").agg(
        customers=("frequency", "size"),
        median_recency_days=("recency_days", "median"),
        median_frequency=("frequency", "median"),
        median_monetary=("monetary", "median"),
    )
    prof["share"] = prof["customers"] / prof["customers"].sum()
    return prof
