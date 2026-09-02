"""Exploratory analysis, structure checks and shared plotting.

Everything here is display-only and learns nothing from the data. The same
battery is run at every level of the pipeline — raw rows, cleaned rows, customer
table, scaled space — so that each transition is inspected before the next.
"""

import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def reset_seed(seed):
    """Reset every random source so a run is reproducible."""
    random.seed(seed)
    np.random.seed(seed)


def check_df(df):
    print(f"Dataset Shape: {df.shape}")

    rows_with_null = df.isna().any(axis=1).sum()
    print(f"Total rows with at least one NULL: {rows_with_null}")

    summary = pd.DataFrame({'Dtype': df.dtypes, 'Non-Null Count': df.count(), 'Null Count': df.isna().sum(),
        'Null Percent': (df.isna().sum() / len(df) * 100).round(2), 'Unique': df.nunique()})

    return summary


def window_state(train, val, test):
    frames = {'train': train, 'val': val, 'test': test}

    rows = {k: len(v) for k, v in frames.items()}
    customers = {k: (v['customer_id'].nunique() if 'customer_id' in v.columns
                     else v.index.nunique()) for k, v in frames.items()}

    total_rows = sum(rows.values())
    total_customers = sum(customers.values())

    state = pd.DataFrame({
        'rows': rows,
        'rows_pct': {k: round(100 * n / total_rows, 1) for k, n in rows.items()},
        'customers': customers,
        'customers_pct': {k: round(100 * n / total_customers, 1) for k, n in customers.items()},
        'columns': {k: v.shape[1] for k, v in frames.items()},
    })

    return state


def before_after(before, after):
    combined = before.join(after, lsuffix=' (before)', rsuffix=' (after)')
    return combined[sorted(combined.columns, key=lambda c: c.split(' ')[0])]


def graph_stats(df, exclude_cols=None, n_cols=2, cat_threshold=20, top_n=50, sample_frac=1.0):

    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)
        print(f"Visualization based on a {sample_frac*100:.1f}% sample ({len(df)} rows)")

    cols = [c for c in df.columns
            if (exclude_cols is None or c not in exclude_cols)]

    n_rows = math.ceil(len(cols) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12 * n_cols, 5.5 * n_rows))

    if n_rows * n_cols == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    for i, col in enumerate(cols):
        ax = axes[i]
        unique_count = df[col].nunique()
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        is_id_col = 'id' in col.lower()

        if is_id_col or not is_numeric or unique_count <= cat_threshold:
            data = df[col].value_counts().head(top_n)

            sns.barplot(x=data.index.astype(str), y=data.values, ax=ax, palette="Blues_r")

            title_suffix = f" (Top {min(top_n, unique_count)})" if unique_count > cat_threshold else ""
            ax.set_title(f"{col}{title_suffix}", fontsize=13, fontweight='bold')
            ax.set_ylabel('Count')
            ax.set_xlabel('')
            ax.grid(axis='y', alpha=0.3, linestyle='--')

            ax.tick_params(axis='x', rotation=90, labelsize=8)

        else:
            sns.histplot(df[col].dropna(), kde=True, ax=ax, color='#82B1C6', bins=30)
            ax.set_title(f"{col}", fontsize=13, fontweight='bold')
            ax.set_ylabel('Count')
            ax.set_xlabel('')
            ax.grid(axis='y', alpha=0.3, linestyle='--')

    for j in range(len(cols), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def correlation_matrix(df, exclude_cols=None, method='pearson', figsize=(9, 7), cmap=None):

    df_work = df.copy()

    if exclude_cols:
        df_work = df_work.drop(columns=exclude_cols, errors='ignore')

    df_numeric = df_work.select_dtypes(include='number')

    methods = ['pearson', 'spearman'] if method == 'both' else [method]

    results = {}
    for m in methods:
        corr_matrix = df_numeric.corr(method=m).round(2)

        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix.abs(), annot=corr_matrix, vmin=0, vmax=1, fmt='.2f', cmap=cmap)
        plt.title(m)
        plt.xticks(rotation=90, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

        display(corr_matrix)

        results[m] = corr_matrix

    return results if len(results) > 1 else results[methods[0]]


def plot_segment_projection(values, segments, title, random_state):
    model = PCA(n_components=2, random_state=random_state).fit(values)
    projection = model.transform(values)

    plt.figure(figsize=(6.5, 5.5))
    for segment in sorted(segments.unique()):
        mask = segments.values == segment
        plt.scatter(projection[mask, 0], projection[mask, 1], s=8, alpha=0.4,
                    label=f'segment {segment}')

    plt.xlabel(f'PC1 ({model.explained_variance_ratio_[0]:.0%})')
    plt.ylabel(f'PC2 ({model.explained_variance_ratio_[1]:.0%})')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def before_after_distributions(customer, scaled, features, n_pairs=8, pairs_per_row=1):
    baseline = pd.DataFrame(MinMaxScaler().fit_transform(customer[features]),
                            columns=features, index=customer.index)

    order = customer[features].skew().abs().sort_values(ascending=False).index[:n_pairs]

    n_rows = math.ceil(len(order) / pairs_per_row)
    fig, axes = plt.subplots(n_rows, 2 * pairs_per_row,
                             figsize=(6 * pairs_per_row, 2.4 * n_rows))
    axes = np.atleast_2d(axes)

    for index, column in enumerate(order):
        row, left = index // pairs_per_row, 2 * (index % pairs_per_row)

        sns.histplot(baseline[column], bins=50, ax=axes[row, left], color='#C68282')
        axes[row, left].set_title(
            f"{column} - minmax only (skew {baseline[column].skew():.1f})", fontsize=10)

        sns.histplot(scaled[column], bins=50, ax=axes[row, left + 1], color='#82B1C6')
        axes[row, left + 1].set_title(
            f"{column} - transformed (skew {scaled[column].skew():.1f})", fontsize=10)

    for ax in axes.ravel():
        ax.set_xlabel('')
        ax.set_ylabel('')

    for index in range(len(order), n_rows * pairs_per_row):
        row, left = index // pairs_per_row, 2 * (index % pairs_per_row)
        fig.delaxes(axes[row, left])
        fig.delaxes(axes[row, left + 1])

    plt.tight_layout()
    plt.show()


def before_after_boxplots(customer, scaled, features):
    baseline = pd.DataFrame(MinMaxScaler().fit_transform(customer[features]),
                            columns=features, index=customer.index)

    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    sns.boxplot(data=baseline, ax=axes[0], color='#C68282', fliersize=1)
    axes[0].set_title('minmax only')
    axes[0].tick_params(axis='x', rotation=90)

    sns.boxplot(data=scaled, ax=axes[1], color='#82B1C6', fliersize=1)
    axes[1].set_title('yeo-johnson then minmax')
    axes[1].tick_params(axis='x', rotation=90)

    plt.tight_layout()
    plt.show()


def before_after_projection(customer, scaled, features, random_state):
    baseline = MinMaxScaler().fit_transform(customer[features])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, values, title in [(axes[0], baseline, 'minmax only'),
                              (axes[1], scaled.values, 'yeo-johnson then minmax')]:
        pca = PCA(n_components=2, random_state=random_state).fit(values)
        projection = pca.transform(values)
        ax.scatter(projection[:, 0], projection[:, 1], s=7, alpha=0.35)
        ax.set_title(f"{title}  (PC1 {pca.explained_variance_ratio_[0]:.0%})")
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')

    plt.tight_layout()
    plt.show()
