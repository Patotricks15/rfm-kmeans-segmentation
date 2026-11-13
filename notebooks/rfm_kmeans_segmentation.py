# %% [markdown]
# # CS01: RFM scoring and KMeans segmentation (Online Retail II)
#
# Snapshot date: 2011-06-01. Only transactions before the snapshot are used.
# Cancelled invoices, missing customer IDs and non-positive quantity/price rows are removed.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples

from rfmseg.core import (
    RANDOM_STATE,
    SNAPSHOT_DATE,
    cluster_metrics,
    data_path,
    fit_kmeans,
    load_retail,
    rfm_features,
    scale_features,
    segment_profile,
    segment_sizes,
)

FIG = Path.cwd().parent / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
MIN_SHARE = 0.02

# %%
tx = load_retail()
print("data:", data_path())
print("transactions after cleaning:", len(tx), "| range:", tx.InvoiceDate.min(), "to", tx.InvoiceDate.max())

feats = rfm_features(tx, SNAPSHOT_DATE)
X = scale_features(feats)
print("customers with history at snapshot:", len(feats))
feats.describe().round(2)

# %%
rows = []
models = {}
for k in range(3, 9):
    model, labels = fit_kmeans(X, k)
    models[k] = labels
    m = cluster_metrics(X, labels)
    m["k"] = k
    m["min_segment_share"] = segment_sizes(labels).min()
    rows.append(m)
sweep = pd.DataFrame(rows).set_index("k").round(4)
sweep

# %%
best_k = int(sweep["silhouette"].idxmax())
sweep_ok = sweep[sweep["min_segment_share"] >= MIN_SHARE]
print("best k by silhouette:", best_k)
print("k values where all segments reach the 2% minimum:", list(sweep_ok.index))
labels = models[best_k]
print("metrics at chosen k:", {k: round(v, 3) for k, v in cluster_metrics(X, labels).items()})

# %%
profile = segment_profile(feats, labels)
profile.round(2)

# %%
fig, ax = plt.subplots(figsize=(7, 4))
sweep[["silhouette"]].plot(ax=ax, marker="o", legend=False)
ax.set_title(f"Silhouette by k (snapshot {SNAPSHOT_DATE.date()}, n={len(X)})")
ax.set_ylabel("silhouette")
fig.tight_layout()
fig.savefig(FIG / "cs01_silhouette_by_k.png", dpi=120)
plt.show()

# %%
pca = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(x=pca[:, 0], y=pca[:, 1], hue=labels, palette="tab10", s=8, linewidth=0, ax=ax)
ax.set_title(f"PCA projection, k={best_k} (snapshot {SNAPSHOT_DATE.date()})")
fig.tight_layout()
fig.savefig(FIG / "cs01_pca.png", dpi=120)
plt.show()

# %%
heat = profile[["median_recency_days", "median_frequency", "median_monetary"]]
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap((heat - heat.mean()) / heat.std(), annot=heat.round(1), fmt="", cmap="RdBu_r", center=0, ax=ax)
ax.set_title(f"RFM medians by segment, k={best_k} (snapshot {SNAPSHOT_DATE.date()})")
fig.tight_layout()
fig.savefig(FIG / "cs01_rfm_heatmap.png", dpi=120)
plt.show()

# %%
sil = silhouette_samples(X, labels)
fig, ax = plt.subplots(figsize=(7, 5))
y_lower = 10
for c in range(best_k):
    vals = np.sort(sil[labels == c])
    y_upper = y_lower + len(vals)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, alpha=0.7)
    y_lower = y_upper + 10
ax.axvline(sil.mean(), color="red", linestyle="--")
ax.set_title(f"Silhouette plot, k={best_k}, mean={sil.mean():.3f}")
fig.tight_layout()
fig.savefig(FIG / "cs01_silhouette_plot.png", dpi=120)
plt.show()

# %%
fig, ax = plt.subplots(figsize=(6, 4))
segment_sizes(labels).plot.bar(ax=ax, color="steelblue")
ax.axhline(MIN_SHARE, color="red", linestyle="--", label="2% minimum")
ax.set_ylabel("share of customers")
ax.set_title(f"Segment sizes, k={best_k} (snapshot {SNAPSHOT_DATE.date()})")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "cs01_segment_sizes.png", dpi=120)
plt.show()
