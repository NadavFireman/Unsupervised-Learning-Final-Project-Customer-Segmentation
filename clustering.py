"""Clustering algorithms, feature ranking and configuration search.

Five algorithms behind one interface. Each ranks the features itself, from the
centres it produces, and scans its own grid — so the comparison between them is
between best configurations rather than between forced parameters.

One rule is enforced here and matters: unseen customers are always assigned to
the nearest centre, for every algorithm. The comparison is geometric, since all
validation metrics are distance-based, so the assignment must be geometric too.
"""

import math

try:
    import kmedoids as kmed
except ImportError:
    kmed = None
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import skfuzzy as fuzz
except ImportError:
    fuzz = None
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from tqdm.auto import tqdm

from features import apply_transformer, fit_transformer


def fit_clustering(algorithm, values, k, random_state, distances=None):
    if algorithm == 'kmeans':
        model = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(values)
        return model.labels_, model.cluster_centers_

    if algorithm == 'kmedoids':
        if kmed is None:
            raise ImportError('the kmedoids package is required for this algorithm')
        model = kmed.fasterpam(distances, k, random_state=random_state)
        return model.labels.astype(int), values[model.medoids]

    if algorithm == 'fcm':
        centers, membership, *_ = fuzz.cluster.cmeans(values.T, c=k, m=2.0,
                                                      error=1e-6, maxiter=400, seed=random_state)
        return membership.argmax(axis=0), centers

    if algorithm == 'agglomerative':
        labels = AgglomerativeClustering(n_clusters=k).fit_predict(values)
        return labels, np.vstack([values[labels == i].mean(axis=0) for i in range(k)])

    if algorithm == 'gmm':
        model = GaussianMixture(n_components=k, n_init=3, random_state=random_state).fit(values)
        return model.predict(values), model.means_

    raise ValueError(algorithm)


def assign_unseen(values, centers):
    return cdist(values, centers).argmin(axis=1)


def separation_scores(values, random_state, algorithm='kmeans', k_values=(3, 4, 5)):
    scores = {}
    distances = squareform(pdist(values.values)) if algorithm == 'kmedoids' else None

    for k in k_values:
        labels, _ = fit_clustering(algorithm, values.values, k, random_state, distances)

        for column in values.columns:
            grand_mean = values[column].mean()
            total = ((values[column] - grand_mean) ** 2).sum()
            between = sum(len(values[column][labels == group]) *
                          (values[column][labels == group].mean() - grand_mean) ** 2
                          for group in np.unique(labels))
            scores.setdefault(column, []).append(between / total if total > 0 else 0.0)

    return pd.Series({c: np.mean(v) for c, v in scores.items()}).sort_values(ascending=False)


def elbow_point(x_values, y_values):
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)

    numerator = np.abs((y[-1] - y[0]) * x - (x[-1] - x[0]) * y + x[-1] * y[0] - y[-1] * x[0])
    denominator = np.hypot(y[-1] - y[0], x[-1] - x[0])

    return int(x[np.argmax(numerator / denominator)])


def selection_grid(algorithm, customer_train, customer_val, feature_order,
                   feature_range, k_range, random_state):
    rows = []

    for size in tqdm(feature_range, desc=algorithm, leave=False):
        features = feature_order[:size]
        transformer = fit_transformer(customer_train, features)
        train_values = apply_transformer(customer_train, transformer)
        val_values = apply_transformer(customer_val, transformer)
        shared = train_values.index.intersection(val_values.index)
        distances = squareform(pdist(train_values.values)) if algorithm == 'kmedoids' else None

        for k in k_range:
            labels, centers = fit_clustering(algorithm, train_values.values, k, random_state, distances)
            val_labels = assign_unseen(val_values.values, centers)

            before = pd.Series(labels, index=train_values.index).loc[shared]
            after = pd.Series(val_labels, index=val_values.index).loc[shared]
            before_share = before.value_counts(normalize=True)
            after_share = after.value_counts(normalize=True)
            chance = sum(before_share.get(c, 0) * after_share.get(c, 0) for c in range(k))

            silhouette_train = silhouette_score(train_values.values, labels)
            silhouette_val = silhouette_score(val_values.values, val_labels)
            inertia = sum(((train_values.values[labels == c] - centers[c]) ** 2).sum()
                          for c in range(len(centers)))

            rows.append({'n_features': size, 'k': k, 'sse': inertia,
                         'silhouette_train': silhouette_train, 'silhouette_val': silhouette_val,
                         'retention_ratio': silhouette_val / silhouette_train if silhouette_train > 0 else 0.0,
                         'ari_returning': adjusted_rand_score(before, after),
                         'stay_rate': (before == after).mean(),
                         'lift': (before == after).mean() / chance if chance > 0 else 0.0,
                         'vanished_clusters': k - len(np.unique(val_labels))})

    return pd.DataFrame(rows)


def plot_selection_curves(grid, feature_order, n_cols=4):
    sizes = sorted(grid['n_features'].unique())
    n_rows = math.ceil(len(sizes) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 3.4 * n_rows))

    for ax, size in zip(axes.ravel(), sizes):
        subset = grid[grid['n_features'] == size].sort_values('k')
        sse_elbow = elbow_point(subset['k'], subset['sse'])
        silhouette_elbow = elbow_point(subset['k'], subset['silhouette_train'])

        ax.plot(subset['k'], subset['sse'], marker='o', markersize=3, color='#4C72B0')
        ax.set_ylabel('SSE', color='#4C72B0')
        ax.tick_params(axis='y', labelcolor='#4C72B0')
        ax.axvline(sse_elbow, color='#4C72B0', linestyle='--', linewidth=1.3)

        twin = ax.twinx()
        twin.plot(subset['k'], subset['silhouette_train'], marker='s', markersize=3, color='#DD8452')
        twin.set_ylabel('silhouette', color='#DD8452')
        twin.tick_params(axis='y', labelcolor='#DD8452')
        twin.axvline(silhouette_elbow, color='#DD8452', linestyle=':', linewidth=1.3)

        ax.set_title(f"{size} features (+{feature_order[size - 1]})" + chr(10) +
                     f"SSE elbow k={sse_elbow} | silhouette elbow k={silhouette_elbow}", fontsize=8)
        ax.set_xlabel('k')

    for ax in axes.ravel()[len(sizes):]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.show()


def qualified_configurations(grid, min_retention=0.9, top=12):
    qualified = grid[(grid['vanished_clusters'] == 0) &
                     (grid['retention_ratio'] >= min_retention)]

    return (qualified.sort_values('ari_returning', ascending=False)
            [['n_features', 'k', 'silhouette_train', 'silhouette_val', 'retention_ratio',
              'ari_returning', 'stay_rate', 'lift']].head(top).round(3))


def modified_dunn(values, labels, centers, random_state, sample_cap=2000):
    """Minimum centre separation over maximum cluster diameter; higher is better.

    This is the Dunn variant used by the reference framework: the numerator is
    the shortest distance between two centres, the denominator the largest
    distance between two points inside one cluster. Large clusters are sampled
    for the diameter, since an exact computation is quadratic.
    """
    generator = np.random.default_rng(random_state)

    separation = cdist(centers, centers)
    np.fill_diagonal(separation, np.inf)

    diameter = 0.0
    for label in np.unique(labels):
        points = values[labels == label]
        if len(points) > sample_cap:
            points = points[generator.choice(len(points), sample_cap, replace=False)]
        if len(points) > 1:
            diameter = max(diameter, pdist(points).max())

    return separation.min() / diameter if diameter > 0 else np.nan


def combined_index(grid, silhouette_column='silhouette_train',
                   dunn_column='dunn', davies_bouldin_column='davies_bouldin'):
    """Sum of silhouette, Dunn and inverse Davies-Bouldin, each scaled to [0, 1].

    This is the quality index the reference framework proposes, and the reason it
    gives for using three measures rather than one. It is reported here for
    comparison. On this dataset the three components turn out to be close to
    collinear, and all three track the number of features rather than cluster
    quality; the notebook quantifies that.
    """
    scored = grid.copy()
    scored['inverse_davies_bouldin'] = 1 / scored[davies_bouldin_column]

    columns = [silhouette_column, dunn_column, 'inverse_davies_bouldin']
    for column in columns:
        low, high = scored[column].min(), scored[column].max()
        scored['scaled_' + column] = ((scored[column] - low) / (high - low)
                                      if high > low else 0.0)

    scored['combined_index'] = scored[['scaled_' + c for c in columns]].sum(axis=1)
    return scored


def control_overlap(customer, control_columns, feature_columns, threshold=0.8):
    """Rank control variables by their strongest overlap with any clustering feature.

    A control variable only controls for anything if the clustering never saw it.
    Dividing a feature by a window constant preserves rank order, so frequency and
    n_invoices are the same variable and correlate at exactly 1.0; a control test
    built on them asks whether a segment can be recovered from the feature it was
    built on. Spearman is used because the relation only has to be monotone to be
    circular. Anything at or above the threshold is disqualified.
    """
    columns = list(control_columns) + list(feature_columns)
    overlap = customer[columns].corr(method='spearman')
    strongest = overlap.loc[list(control_columns), list(feature_columns)].abs()

    table = pd.DataFrame({
        'max_correlation': strongest.max(axis=1).round(3),
        'closest_feature': strongest.idxmax(axis=1),
    }).sort_values('max_correlation', ascending=False)
    table['kept'] = table['max_correlation'] < threshold
    return table
