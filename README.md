# Unsupervised Learning Final Project - Customer Segmentation

Final Project (M.Sc. Data Science, HIT). Customer segmentation on Online Retail II — 1,067,371 transactions — asking whether a segment is a stable property of the customer or a snapshot of the window it was measured in. A transfer test of the extended RFM framework in Ozcan (2026).

## Key Features

* **Temporal Split First**: 60/20/20 into three consecutive windows — 464 / 181 / 94 days — before any cleaning. Every fitted map is learned from the training window alone.
* **Feature Engineering**: 1,067,371 rows reduce to 470,345 clean training rows and 3,057 customers. 15 candidate features, volume expressed as a monthly rate so windows of different length stay comparable. Yeo-Johnson then min-max brings mean skew from 5.26 to 0.20.
* **Five Algorithms**: K-Means, K-Medoids (FasterPAM), Fuzzy C-Means, Agglomerative (Ward) and a Gaussian Mixture, each scanning its own grid of feature count against k. HDBSCAN as a density check.
* **Stability as the Deciding Metric**: SSE has no elbow and silhouette peaks at k=2, so neither can choose k. Selection was made on transfer instead — silhouette retention ≥ 0.9, zero vanished clusters, and ARI on customers active in both windows. That gives 9 features and k=3.
* **Supervised Judge**: LightGBM on the cluster labels, scored against the majority baseline in every window — the one measurement here that is not distance-based.

## Results

Five algorithms agree with each other at **ARI 0.41** inside one window. Each agrees with **itself** across two windows at only **0.08**. The boundary belongs to the window, not to the customer.

K-Means was selected: silhouette 0.208 / 0.188 / 0.157 across the three windows, and 0.928 supervised accuracy on validation against a 0.467 baseline. Three segments emerged — wholesale-deal buyers (684 customers, 58.4% of revenue), regular retail (1,251, 32.3%) and occasional buyers (1,122, 9.3%). Individual labels do not survive the next window (ARI 0.124 and 0.165), but the economics do: 23.5% of test-window customers hold 48.9% of its revenue. Unusable for per-customer targeting, usable for group-level budgeting.

SHAP puts `campaign_pct` and `interval_std` first and classic RFM last — the framework's extensions do the separating, not its core. And product categories correlate at most 0.021 with any segment while quantity and bulk buying reach 0.33: the segments differ in *how* customers buy, not *what*.

## Repository Structure

* `final_project_unsupervised_learning.ipynb`: Full end-to-end notebook, 18 stages
* `data.py`: Loading, temporal splitting and cleaning
* `eda.py`: Exploratory analysis and shared plotting
* `features.py`: Row features, customer aggregation, and scaling
* `clustering.py`: The five algorithms behind one interface, plus the configuration grid
* `validation.py`: Stability against the following window, and the test-window report
* `reproduce.py`: Rebuilds every published figure from the raw CSV. `python reproduce.py online_retail_II.csv`
* `online_retail_II.csv`: Raw dataset, 95 MB, via Git LFS
* `customer_segmentation_presentation.pdf`: Project presentation, 25 slides
* `final_project_instructions.pdf`: Course requirements
