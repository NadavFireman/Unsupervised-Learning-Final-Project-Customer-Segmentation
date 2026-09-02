"""Validation of a segmentation against the following time window.

The core of the project. A segmentation is judged not by how tidy it looks
inside the window it was learned from, but by whether the same customer receives
the same segment in the next period.

Four measures carry that: the retention ratio, the adjusted Rand index on
returning customers, the stay rate against a chance baseline, and the number of
clusters that receive no customer at all in the new window.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import skfuzzy as fuzz
except ImportError:
    fuzz = None
try:
    import lightgbm as lgb
except ImportError:
    lgb = None
try:
    import shap
except ImportError:
    shap = None
try:
    import squarify
except ImportError:
    squarify = None
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (accuracy_score, adjusted_rand_score, f1_score,
                             roc_auc_score, silhouette_score)
from sklearn.mixture import GaussianMixture

from clustering import assign_unseen
from features import apply_transformer, attach_segments, fit_transformer


def segment_profile(customer, segments, features):
    subset = customer.loc[segments.index]

    profile = subset.groupby(segments)[features].median().round(2)
    profile['customers'] = segments.value_counts().sort_index()
    profile['customers_pct'] = (100 * segments.value_counts(normalize=True).sort_index()).round(1)

    revenue = subset.groupby(segments)['spend_net'].sum()
    profile['revenue_pct'] = (100 * revenue / revenue.sum()).round(1)

    return profile


def migration_matrix(segment_train, segment_val, returning, k):
    before = segment_train.loc[returning]
    after = segment_val.loc[returning]

    matrix = pd.crosstab(before, after, normalize='index').round(2)
    matrix.index.name = 'train segment'
    matrix.columns.name = 'validation segment'

    after_share = after.value_counts(normalize=True)
    stability = pd.DataFrame({
        'customers': before.value_counts().sort_index(),
        'stayed_pct': (100 * (before == after).groupby(before).mean()).round(1),
        'chance_pct': (100 * after_share.reindex(sorted(before.unique())).fillna(0)).round(1),
    })
    stability['lift'] = (stability['stayed_pct'] / stability['chance_pct']).round(2)

    return matrix, stability


def representative_separation(values, labels, centers):
    distances = cdist(centers, centers)
    radii = {c: np.linalg.norm(values[labels == c] - centers[c], axis=1).mean()
             for c in range(len(centers))}

    rows = [{'pair': f'{a}-{b}', 'distance': round(distances[a, b], 3),
             'radii_sum': round(radii[a] + radii[b], 3),
             'separation_ratio': round(distances[a, b] / (radii[a] + radii[b]), 2)}
            for a in range(len(centers)) for b in range(a + 1, len(centers))]

    return pd.DataFrame(rows).sort_values('separation_ratio')


def merge_distances(values, k_range, method='ward'):
    """Merge height per number of clusters, and the gap down to the next merge.

    diff(-1) already returns the drop from this row to the next, so the gap is
    reported as a positive magnitude: a large value means the merge that would
    reduce k by one joins two distant groups, and k is a natural stopping point.
    """
    merge_matrix = linkage(values, method=method)
    distances = merge_matrix[-(max(k_range) - 1):, 2][::-1]

    table = pd.DataFrame({'k': list(k_range), 'merge_distance': distances[:len(k_range)]})
    table['gap_to_next'] = table['merge_distance'].diff(-1)
    return table.round(2)


def linkage_comparison(customer_train, customer_val, features, k_values, random_state,
                       methods=('ward', 'average', 'complete')):
    transformer = fit_transformer(customer_train, features)
    train_values = apply_transformer(customer_train, transformer)
    val_values = apply_transformer(customer_val, transformer)
    shared = train_values.index.intersection(val_values.index)

    rows = []
    for method in methods:
        for k in k_values:
            labels = AgglomerativeClustering(n_clusters=k, linkage=method).fit_predict(train_values.values)
            centers = np.vstack([train_values.values[labels == i].mean(axis=0) for i in range(k)])
            val_labels = assign_unseen(val_values.values, centers)

            before = pd.Series(labels, index=train_values.index).loc[shared]
            after = pd.Series(val_labels, index=val_values.index).loc[shared]
            before_share = before.value_counts(normalize=True)
            after_share = after.value_counts(normalize=True)
            chance = sum(before_share.get(c, 0) * after_share.get(c, 0) for c in range(k))

            silhouette_train = silhouette_score(train_values.values, labels)
            silhouette_val = silhouette_score(val_values.values, val_labels)

            rows.append({'linkage': method, 'k': k,
                         'silhouette_train': round(silhouette_train, 3),
                         'silhouette_val': round(silhouette_val, 3),
                         'retention_ratio': round(silhouette_val / silhouette_train, 3),
                         'ari_returning': round(adjusted_rand_score(before, after), 3),
                         'stay_rate': round((before == after).mean(), 3),
                         'smallest_cluster': int(np.bincount(labels).min())})

    return pd.DataFrame(rows)


def fuzzy_diagnostics(values, k_range, random_state, fuzziness=2.0):
    rows = []
    for k in k_range:
        centers, membership, *_ , partition_coefficient = fuzz.cluster.cmeans(
            values.T, c=k, m=fuzziness, error=1e-6, maxiter=400, seed=random_state)

        rows.append({'k': k,
                     'partition_coefficient': round(partition_coefficient, 3),
                     'random_baseline': round(1 / k, 3),
                     'mean_max_membership': round(membership.max(axis=0).mean(), 3),
                     'borderline_pct': round(100 * (membership.max(axis=0) < 0.5).mean(), 1)})

    return pd.DataFrame(rows)


def information_criteria(customer_train, features, k_range, random_state):
    transformer = fit_transformer(customer_train, features)
    values = apply_transformer(customer_train, transformer)

    rows = []
    for k in k_range:
        model = GaussianMixture(n_components=k, n_init=3, random_state=random_state).fit(values.values)
        rows.append({'k': k,
                     'bic': round(model.bic(values.values)),
                     'aic': round(model.aic(values.values)),
                     'mean_probability': round(model.predict_proba(values.values).max(axis=1).mean(), 3)})

    table = pd.DataFrame(rows)
    table['bic_reached_minimum'] = table['bic'].idxmin() != len(table) - 1
    return table


def _classification_scores(model, values, labels):
    predicted = model.predict(values)
    present = np.unique(labels)
    probabilities = model.predict_proba(values)
    try:
        area = roc_auc_score(labels, probabilities, multi_class='ovr',
                             average='macro', labels=present)
    except ValueError:
        area = np.nan
    return (accuracy_score(labels, predicted),
            f1_score(labels, predicted, average='macro'),
            area)


def _fit_classifier(values, labels, random_state):
    if lgb is None:
        raise ImportError('the lightgbm package is required for supervised validation')
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                              random_state=random_state, verbose=-1).fit(values, labels)


def supervised_validation(customer_train, customer_val, train_values, val_values,
                          segment_train, segment_val, control_columns, random_state,
                          window_name='validation'):
    """Three supervised checks on one segmentation, and what each of them answers.

    Sanity trains and scores on the clustering features of the training window. It
    should be near perfect, since the labels are a deterministic function of those
    features; anything lower means the segmentation is not even self-consistent.

    Generalisation scores that same model on the next window against the segments
    its customers were assigned. It asks whether the boundaries still describe
    customers the model never saw.

    Control trains only on variables held out of the clustering by design. If the
    segments can be recovered from information the algorithm never touched, they
    are a property of the customers rather than of the chosen feature space. Pass
    only genuinely independent columns: see control_overlap in clustering.

    Every accuracy is reported beside the majority-class share, without which a
    three-cluster accuracy of 0.62 cannot be read at all.

    window_name labels the second window in the report. Pass 'test' when the
    second window is the test one, so the table cannot be misread later.
    """
    baseline_train = segment_train.value_counts(normalize=True).max()
    baseline_val = segment_val.value_counts(normalize=True).max()

    model = _fit_classifier(train_values.values, segment_train.values, random_state)
    sanity = _classification_scores(model, train_values.values, segment_train.values)
    generalisation = _classification_scores(model, val_values.values, segment_val.values)

    control_train = customer_train.loc[train_values.index, list(control_columns)]
    control_val = customer_val.loc[val_values.index, list(control_columns)]
    control_model = _fit_classifier(control_train.values, segment_train.values, random_state)
    control = _classification_scores(control_model, control_val.values, segment_val.values)

    table = pd.DataFrame([
        {'test': 'sanity', 'window': 'train', 'accuracy': sanity[0],
         'macro_f1': sanity[1], 'roc_auc': sanity[2], 'majority_baseline': baseline_train},
        {'test': 'generalisation', 'window': window_name, 'accuracy': generalisation[0],
         'macro_f1': generalisation[1], 'roc_auc': generalisation[2],
         'majority_baseline': baseline_val},
        {'test': 'control', 'window': window_name, 'accuracy': control[0],
         'macro_f1': control[1], 'roc_auc': control[2], 'majority_baseline': baseline_val},
    ])
    table['above_baseline'] = table['accuracy'] - table['majority_baseline']
    return table.round(3), model


def shap_importance(model, values, max_display=10):
    """Mean absolute SHAP value per feature, averaged over classes.

    SHAP explains a segmentation that already exists; it does not score one. An
    importance ranking taken from a model trained on cluster labels can only
    rediscover what the clustering already used. The value is in the ranking
    inside a segment, which is what turns a cluster into a description.
    """
    if shap is None:
        raise ImportError('the shap package is required for this report')

    explainer = shap.TreeExplainer(model)
    contributions = explainer.shap_values(values)
    if isinstance(contributions, list):
        magnitude = np.mean([np.abs(part).mean(axis=0) for part in contributions], axis=0)
    else:
        magnitude = np.abs(contributions).mean(axis=0)
        if magnitude.ndim > 1:
            magnitude = magnitude.mean(axis=1)

    return (pd.Series(magnitude, index=values.columns)
            .sort_values(ascending=False).head(max_display).round(4))


def test_window_report(values, segments, k):
    """The single touch of the test window, for the chosen configuration only.

    Validation was consulted for every decision — feature count, k, algorithm — so
    it is no longer unseen. The test window is the only measurement untouched by
    those choices. That is why it is read once, and why no decision may be revised
    after reading it.

    values and segments are dicts keyed 'train', 'val', 'test'.
    """
    silhouettes = {name: silhouette_score(values[name].values, segments[name].values)
                   for name in ('train', 'val', 'test')}

    chain = pd.DataFrame([
        {'window': 'train', 'silhouette': silhouettes['train'],
         'retention_from_previous': np.nan},
        {'window': 'validation', 'silhouette': silhouettes['val'],
         'retention_from_previous': silhouettes['val'] / silhouettes['train']},
        {'window': 'test', 'silhouette': silhouettes['test'],
         'retention_from_previous': silhouettes['test'] / silhouettes['val']},
    ]).round(3)

    sizes = pd.DataFrame({name: segments[name].value_counts().sort_index()
                          for name in ('train', 'val', 'test')}).fillna(0).astype(int)

    transitions = {}
    for earlier, later in (('train', 'val'), ('val', 'test')):
        shared = segments[earlier].index.intersection(segments[later].index)
        matrix, stability = migration_matrix(segments[earlier], segments[later], shared, k)
        stability['ari'] = round(adjusted_rand_score(segments[earlier].loc[shared],
                                                     segments[later].loc[shared]), 3)
        stability['returning'] = len(shared)
        transitions[earlier + ' -> ' + later] = (matrix, stability)

    return chain, sizes, transitions


def revenue_concentration(customer, segments):
    """Customer share against revenue share per segment, and the ratio between them.

    The ratio is what makes a segment actionable. A fifth of the customers holding
    half the revenue is a retention budget; a fifth holding a fifth is not.
    """
    subset = customer.loc[segments.index]
    revenue = subset.groupby(segments)['spend_net'].sum()
    customers = segments.value_counts().sort_index()

    table = pd.DataFrame({
        'customers': customers,
        'customers_pct': (100 * customers / customers.sum()).round(1),
        'revenue': revenue.round(0),
        'revenue_pct': (100 * revenue / revenue.sum()).round(1),
    })
    table['concentration'] = (table['revenue_pct'] / table['customers_pct']).round(2)
    return table.sort_values('revenue_pct', ascending=False)


def segment_treemap(concentration, title, figsize=(7, 4)):
    """Treemap of the segments, area proportional to revenue share."""
    if squarify is None:
        raise ImportError('the squarify package is required for the treemap')

    labels = [f"segment {index}\n{row.revenue_pct:.1f}% revenue\n{row.customers_pct:.1f}% customers"
              for index, row in concentration.iterrows()]

    plt.figure(figsize=figsize)
    squarify.plot(sizes=concentration['revenue_pct'].values, label=labels,
                  alpha=0.8, text_kwargs={'fontsize': 10})
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def category_mix(transactions, segments, category_column='product_category'):
    """Share of each product category in every segment's spend."""
    labelled = attach_segments(transactions, segments)
    labelled = labelled[labelled['segment'].notna() & ~labelled['is_return']]

    spend = pd.crosstab(labelled['segment'], labelled[category_column],
                        values=labelled['total_price'], aggfunc='sum').fillna(0.0)
    return (100 * spend.div(spend.sum(axis=1), axis=0)).round(1)
