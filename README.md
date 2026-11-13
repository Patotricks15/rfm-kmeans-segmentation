# RFM KMeans Segmentation (CS01)

Customer groups from recency, frequency and monetary value, built on real UK retailer transactions (Online Retail II).

**Core question:** what are the main customer groups by recency, frequency and monetary value?

## Setup

- Snapshot date: **2011-06-01**. Only transactions strictly before the snapshot are used.
- Cleaning: missing customer IDs, cancelled invoices (`C` prefix) and rows with non-positive quantity or price are removed. Monetary value is capped at the 99th percentile.
- Features: recency (days), frequency (distinct invoices), monetary, monetary in the last 90 days, tenure and average order value. `log1p` then `RobustScaler`.
- Model: `KMeans(n_init=10, random_state=42)`, with k from 3 to 8 chosen by silhouette.

## Results

Customers with purchase history at the snapshot: **4,933**.

| k | Silhouette | Davies-Bouldin | Calinski-Harabasz | Smallest segment |
|---|---|---|---|---|
| 3 | **0.359** | 1.030 | 2388.7 | 10.9% |
| 4 | 0.299 | 1.072 | 2382.3 | 10.0% |
| 5 | 0.282 | 1.156 | 2152.1 | 9.8% |
| 6 | 0.250 | 1.233 | 2002.5 | 9.6% |
| 7 | 0.260 | 1.135 | 1924.1 | 8.4% |
| 8 | 0.256 | 1.166 | 1854.1 | 2.6% |

Chosen k = 3 (highest silhouette, all segments above the 2% minimum):

| Segment | Customers | Share | Median recency (days) | Median frequency | Median monetary |
|---|---|---|---|---|---|
| 0 | 1,594 | 32% | 28 | 8 | 3,056.6 |
| 1 | 2,801 | 57% | 228 | 2 | 454.7 |
| 2 | 538 | 11% | 46 | 1 | 331.5 |

Segment 0 is the loyal, recent, high-value group. Segment 1 is the lapsed group that buys rarely. Segment 2 is recent but one-time buyers.

Figures: [silhouette by k](reports/figures/cs01_silhouette_by_k.png), [PCA](reports/figures/cs01_pca.png), [RFM heatmap](reports/figures/cs01_rfm_heatmap.png), [silhouette plot](reports/figures/cs01_silhouette_plot.png), [segment sizes](reports/figures/cs01_segment_sizes.png).

## Limitation

The silhouette score is modest (0.36 at the chosen k), so the segments are a useful summary, not sharply separated groups. Only 4,933 customers have history at this snapshot, because the data starts in December 2009.

## Reproduce

```bash
pip install -e ".[dev]"
RETAIL_DATA_PATH=/path/to/online_retail_II.csv pytest
cd notebooks && jupyter nbconvert --to notebook --execute --inplace rfm_kmeans_segmentation.ipynb
```

Code is in [src/rfmseg/core.py](src/rfmseg/core.py); tests are in [tests/test_core.py](tests/test_core.py). Data notes are in [data/README.md](data/README.md).
