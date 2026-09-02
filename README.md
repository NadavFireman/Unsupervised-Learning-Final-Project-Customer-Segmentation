# Unsupervised Learning Final Project - Customer Segmentation

Final Project (M.Sc. Data Science, HIT). Customer segmentation on Online Retail II - 1,067,371 transactions - asking whether a segment is a stable property of the customer or a snapshot of the window it was measured in. A transfer test of the extended RFM framework in Ozcan (2026), whose own data and code are not public.

## Key Features

* **Temporal Split First**: 60/20/20 into three consecutive windows - 464 / 181 / 94 days - before any cleaning. Everything is fitted on the training window alone.
* **Feature Engineering**: 1,067,371 rows down to 3,057 customers. Volume as monthly rates so unequal windows compare; Yeo-Johnson then min-max cuts mean skew from 5.26 to 0.20.
* **Five Algorithms**: K-Means, K-Medoids, Fuzzy C-Means, Agglomerative and a Gaussian Mixture, each scanning its own grid. HDBSCAN as a density check.
* **Stability Decides k**: SSE has no elbow, silhouette peaks at k=2. The next window chose - 9 features, k=3.
* **Supervised Judge**: LightGBM on the cluster labels against the majority baseline - the one check that is not distance-based.

## Results

Five algorithms agree with each other at **ARI 0.41** inside one window. Each agrees with **itself** across two windows at only **0.08**. The boundary belongs to the window, not to the customer.

K-Means won: silhouette 0.208 / 0.188 / 0.157, and 0.928 accuracy against a 0.467 baseline. Three segments - wholesale-deal buyers (684 customers, 58.4% of revenue), regular retail (1,251, 32.3%), occasional (1,122, 9.3%). The labels don't survive the next window (ARI 0.124 and 0.165), but the economics do: 23.5% of test-window customers hold 48.9% of its revenue. Unusable for targeting, usable for budgeting.

SHAP puts `campaign_pct` and `interval_std` first and RFM last: the extensions separate, not the core. And product categories correlate at most 0.021 with any segment - customers differ in *how* they buy, not *what*.

## Repository Structure

* `final_project_unsupervised_learning.ipynb`: Full end-to-end notebook, 18 stages
* `data.py`: Loading, temporal splitting and cleaning
* `eda.py`: Exploratory analysis and shared plotting
* `features.py`: Row features, customer aggregation, and scaling
* `clustering.py`: The five algorithms behind one interface, plus the configuration grid
* `validation.py`: Stability against the following window, and the test-window report
* `validate.py`: Rebuilds every published figure from the raw CSV. `python validate.py online_retail_II.csv`
* `online_retail_II.csv`: Raw dataset, 95 MB - download from the Kaggle link below
* * `customer_segmentation_presentation.pdf`: Project presentation, 25 slides
* `final_project_instructions.pdf`: Course requirements

## References

* Ozcan, T. (2026). [Customer Segmentation Using an Extended RFM Model and Clustering Algorithms in E-Commerce](https://www.mdpi.com/0718-1876/21/5/142). *Journal of Theoretical and Applied Electronic Commerce Research*, 21(5), 142. The framework under test - it adds campaign share, basket depth and inter-order interval regularity to classic RFM.
* [Online Retail II (UCI, via Kaggle)](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci) - a UK online gift retailer, 1,067,371 transactions, December 2009 to December 2011.
* [Supratim0406/Customer-Segmentation-RFM-Analysis](https://github.com/Supratim0406/Customer-Segmentation-RFM-Analysis) - a public RFM clustering implementation, used as a reference point for the comparison in stage 17.
