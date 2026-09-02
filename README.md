# Unsupervised Learning Final Project - Customer Segmentation (Grade: XX)

Final Project (Grade XX, M.Sc. Data Science, HIT). Customer segmentation on the Online Retail II dataset — 1,067,371 transactions over two years — asking one question the standard pipeline never asks: is a segment a stable property of the customer, or a snapshot of the window it was measured in? A transfer test of the extended RFM framework in Ozcan (2026), whose own data and code are not public.

## Key Features

* **Temporal Split First**: 60/20/20 by row mass into three consecutive windows — 464 / 181 / 94 days — before any cleaning or feature work. Cut points round up to the following midnight so no invoice is divided, and every fitted map (descriptions, reference prices, scaler, cluster centres) is learned from the training window alone.
* **Feature Engineering**: 1,067,371 raw rows reduce to 470,345 clean training rows and 3,057 customers with at least two invoices. 15 candidate features are built per customer, with volume features expressed as a monthly rate so the three windows of different length stay comparable. Yeo-Johnson then min-max brings mean absolute skew from 5.26 down to 0.20.
* **Five Algorithms, One Route**: K-Means, K-Medoids (FasterPAM), Fuzzy C-Means, Agglomerative (Ward) and a Gaussian Mixture each rank features from their own centres and scan feature count against cluster count, so the comparison is between best configurations rather than forced parameters. HDBSCAN runs as a density check and finds zero clusters at `min_cluster_size` 100 — the data is a continuous cloud, not natural groups.
* **Stability as the Deciding Metric**: The elbow and silhouette curves give no answer here — SSE falls smoothly with no elbow, and silhouette peaks at k=2, the smallest value possible. k was therefore chosen by temporal transfer: silhouette retention into the next window ≥ 0.9, zero vanished clusters, and the adjusted Rand index on customers active in both windows. That selects 9 features and k=3.
* **Supervised Judge**: A LightGBM classifier on the cluster labels, scored against the majority baseline in every window — the one measurement in the project that is not distance-based, and therefore independent of everything above it.
* **Row-Level Control**: The labels are pushed back onto 436,298 transaction rows with every product category as its own feature, to test whether the segments could have been reached from products alone.

## Results

Five algorithms with different assumptions agree with each other at **ARI 0.41** inside one window. Each algorithm agrees with **itself** across two windows at only **0.08** — a factor of five. The boundary that gets measured belongs to the window, not to the customer.

K-Means was selected: silhouette **0.208 / 0.188 / 0.157** across train, validation and test, retention 0.904 and 0.835, and the best cross-window ARI of the five. The supervised judge reaches **0.928** accuracy on the validation window against a 0.467 baseline, and still **0.866** on the test window against 0.425 — the boundaries are a learnable rule, not a description of customers already seen.

Three segments emerged: **wholesale-deal buyers** (684 customers, 22.4%, holding 58.4% of revenue), **regular retail** (1,251, 40.9%, 32.3%) and **occasional buyers** (1,122, 36.7%, 9.3%). Individual labels do not survive the move to the next window — ARI 0.124 and 0.165 — but the economics do: in the test window one segment holds 23.5% of the customers and 48.9% of the revenue, a concentration ratio of 2.08, and it is the only segment above twice chance in both transitions. The segmentation is therefore unusable for per-customer targeting and usable for group-level budgeting.

Two supporting findings. SHAP puts `campaign_pct` and `interval_std` first, together about half of all contribution, while the classic RFM measures sit at the bottom — the framework's own extensions do the separating, not its core. And the twelve product categories correlate at most **0.021** with any segment while quantity and bulk buying reach **0.33**: the segments differ in *how* customers buy, not in *what* they buy.

## Repository Structure

* `final_project_unsupervised_learning.ipynb`: Full end-to-end notebook — 18 stages from raw rows to business profiling, with every decision documented in place
* `data.py`: Loading, temporal splitting and cleaning
* `eda.py`: Exploratory analysis, structure checks and shared plotting
* `features.py`: Row features, customer-level aggregation, and Yeo-Johnson → min-max scaling
* `clustering.py`: The five algorithms behind one interface, feature ranking, and the configuration grid
* `validation.py`: Stability against the following window — retention ratio, adjusted Rand index, migration matrices, the supervised judge, and the single test-window report
* `reproduce.py`: End-to-end reproducibility check. Run `python reproduce.py online_retail_II.csv` to rebuild every published figure from the raw CSV using only the modules
* `online_retail_II.csv`: Raw dataset, 95 MB, via Git LFS
* `customer_segmentation_presentation.pdf`: Project presentation, 25 slides
* `Final_Project_Instructions.pdf`: Original course requirements
