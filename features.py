"""Feature engineering: row level, customer level, and scaling.

Row features are each derived from one original column. The customer table
aggregates them into one row per customer per window; volume features are
expressed as a monthly rate because the three windows differ in length.

Scaling lives here too, since transforming a feature is part of building it.
Yeo-Johnson is used rather than a log because a log is close to linear on
bounded ratios and leaves them untouched.

Constants are passed in as required arguments; none are defined here.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, PowerTransformer


# ---------------------------------------------------------------------------
# row level
# ---------------------------------------------------------------------------

def build_holiday_calendar(holiday_ranges, days_before, days_after):
    """Expand holiday ranges into a set of holiday days and a set of season days.

    A holiday is a range rather than a point, so that Christmas Eve and Boxing
    Day are covered along with the day itself.
    """
    holiday_days = pd.DatetimeIndex(np.concatenate(
        [pd.date_range(start, end).values for start, end in holiday_ranges])).unique()

    season_days = pd.DatetimeIndex(np.concatenate(
        [pd.date_range(pd.Timestamp(start) - pd.Timedelta(days=days_before),
                       pd.Timestamp(end) + pd.Timedelta(days=days_after)).values
         for start, end in holiday_ranges])).unique()

    return holiday_days, season_days


def assign_category(description, category_keywords):
    """Map a product description to the first category whose keywords it matches.

    The taxonomy is hand-built and fixed, so it carries no leakage, but it is an
    approximation: a product is assigned to the first match, not the best one.
    """
    text = str(description).upper()
    for category, keywords in category_keywords.items():
        if any(word in text for word in keywords):
            return category
    return 'unknown'


def add_transaction_features(frame, holiday_days, season_days, bulk_quantity,
                             category_keywords):
    """Add every row-level feature, each derived from one original column."""
    frame['is_return'] = (frame['invoice'].astype(str).str.strip().str.upper()
                          .str.startswith('C')).fillna(False).astype(bool)
    frame['invoice_line_count'] = frame.groupby('invoice')['invoice'].transform('size')

    frame['total_price'] = frame['quantity'] * frame['price']
    frame['is_bulk'] = frame['quantity'] >= bulk_quantity

    frame['year'] = frame['invoice_date'].dt.year
    frame['quarter'] = frame['invoice_date'].dt.quarter
    frame['month'] = frame['invoice_date'].dt.month
    frame['week'] = frame['invoice_date'].dt.isocalendar().week.astype(int)
    frame['day_of_month'] = frame['invoice_date'].dt.day
    frame['day_of_week'] = frame['invoice_date'].dt.day_name()
    frame['hour'] = frame['invoice_date'].dt.hour
    frame['minute'] = frame['invoice_date'].dt.minute
    frame['is_weekend'] = frame['invoice_date'].dt.dayofweek >= 5

    day = frame['invoice_date'].dt.normalize()
    frame['is_holiday'] = day.isin(holiday_days)
    frame['is_holiday_season'] = day.isin(season_days)

    frame['is_uk'] = frame['country'] == 'United Kingdom'

    frame['product_category'] = frame['description'].map(
        lambda text: assign_category(text, category_keywords))
    frame['description_words'] = frame['description'].astype(str).str.split().str.len()

    return frame


# ---------------------------------------------------------------------------
# customer level
# ---------------------------------------------------------------------------

def fit_reference(train, discount_threshold):
    """Learn the item price map and the two smoothing priors from the training window.

    The price map is the median unit price per stock code, taken over purchase
    rows only: on a return the price reflects the original sale.
    """
    purchases = train[~train['is_return']]
    price_map = purchases.groupby('stock_code')['price'].median()

    reference = purchases['stock_code'].map(price_map)
    discounted = purchases['total_price'].where(
        purchases['price'] < discount_threshold * reference, 0.0)

    discount_prior = discounted.sum() / purchases['total_price'].sum()
    return_prior = (train[train['is_return']]['total_price'].abs().sum()
                    / purchases['total_price'].sum())

    categories = sorted(train['product_category'].unique())
    return price_map, discount_prior, return_prior, categories


def interval_std_days(dates):
    """Standard deviation of gaps between unique purchase days; zero below two days.

    Days are normalised first, so several invoices on one day count as a single
    purchase event.
    """
    days = np.sort(dates.dt.normalize().unique())
    if len(days) < 2:
        return 0.0
    gaps = np.diff(days).astype('timedelta64[D]').astype(float)
    return float(gaps.std(ddof=0))


def build_customer_table(transactions, window, price_map, discount_prior, return_prior,
                         categories, alpha, days_per_month, min_frequency,
                         discount_threshold):
    """Aggregate one window of transactions into one row per customer.

    Volume features are divided by the window length in months, so the same
    behaviour yields the same value in windows of different length. Point-in-time
    features and ratios are left as they are.
    """
    window_start, window_end = window
    months = (window_end - window_start).days / days_per_month

    purchases = transactions[~transactions['is_return']].copy()
    reference = purchases['stock_code'].map(price_map)
    purchases['discount_spend'] = purchases['total_price'].where(
        purchases['price'] < discount_threshold * reference, 0.0)

    customer = purchases.groupby('customer_id').agg(
        first_purchase=('invoice_date', 'min'),
        last_purchase=('invoice_date', 'max'),
        n_invoices=('invoice', 'nunique'),
        n_items=('stock_code', 'nunique'),
        n_lines=('invoice', 'size'),
        total_quantity=('quantity', 'sum'),
        median_quantity=('quantity', 'median'),
        median_price=('price', 'median'),
        spend_positive=('total_price', 'sum'),
        discount_spend=('discount_spend', 'sum'),
        bulk_share=('is_bulk', 'mean'),
        uk_share=('is_uk', 'mean'),
        weekend_share=('is_weekend', 'mean'),
        holiday_share=('is_holiday_season', 'mean'),
        median_hour=('hour', 'median'),
        interval_std=('invoice_date', interval_std_days),
    )

    customer['spend_net'] = transactions.groupby('customer_id')['total_price'].sum()
    returns = transactions[transactions['is_return']].groupby('customer_id')['total_price'].sum().abs()
    customer['return_value'] = returns.reindex(customer.index).fillna(0.0)

    customer['recency'] = (window_end - customer['last_purchase']).dt.days
    customer['tenure'] = (window_end - customer['first_purchase']).dt.days
    customer['frequency'] = customer['n_invoices'] / months
    customer['monetary'] = customer['spend_net'] / months
    customer['depth'] = customer['n_items'] / customer['n_invoices']
    customer['basket_value'] = customer['spend_positive'] / customer['n_invoices']
    customer['campaign_pct'] = ((customer['discount_spend'] + alpha * discount_prior)
                                / (customer['spend_positive'] + alpha))
    customer['return_rate'] = ((customer['return_value'] + alpha * return_prior)
                               / (customer['spend_positive'] + alpha))
    customer['repeat_ratio'] = 1 - customer['n_items'] / customer['n_lines']

    category_spend = pd.crosstab(purchases['customer_id'], purchases['product_category'],
                                 values=purchases['total_price'], aggfunc='sum')
    category_share = category_spend.div(category_spend.sum(axis=1), axis=0).fillna(0.0)
    for category in categories:
        customer[f'cat_{category}'] = (category_share[category].reindex(customer.index).fillna(0.0)
                                       if category in category_share.columns else 0.0)

    customer = customer[(customer['n_invoices'] >= min_frequency) & (customer['spend_net'] > 0)]
    customer.index.name = 'customer_id'
    return customer


def attach_segments(transactions, segments):
    """Map customer-level segment labels back onto the transaction rows."""
    labelled = transactions.copy()
    labelled['segment'] = labelled['customer_id'].map(segments)
    return labelled


# ---------------------------------------------------------------------------
# feature space audits
# ---------------------------------------------------------------------------

def feature_families(customer):
    numeric = [c for c in customer.columns if customer[c].dtype.kind in 'ifb']

    families = {
        'volume': ['n_invoices', 'n_items', 'n_lines', 'total_quantity',
                   'spend_positive', 'spend_net', 'discount_spend', 'return_value'],
        'rate': ['frequency', 'monetary', 'basket_value', 'depth', 'median_price',
                 'median_quantity', 'interval_std'],
        'timing': ['recency', 'tenure', 'median_hour'],
        'ratio': ['bulk_share', 'uk_share', 'weekend_share', 'holiday_share',
                  'campaign_pct', 'return_rate', 'repeat_ratio'],
        'category': [c for c in numeric if c.startswith('cat_')],
    }

    assigned = [c for group in families.values() for c in group]
    families['unassigned'] = [c for c in numeric if c not in assigned]
    return {k: [c for c in v if c in numeric] for k, v in families.items() if v}


def feature_audit(customer, families):
    family_of = {c: name for name, cols in families.items() for c in cols}
    numeric = [c for c in customer.columns if customer[c].dtype.kind in 'ifb']

    audit = pd.DataFrame({
        'family': [family_of.get(c, 'other') for c in numeric],
        'median': customer[numeric].median(),
        'mean': customer[numeric].mean(),
        'max': customer[numeric].max(),
        'skew': customer[numeric].skew(),
        'zeros_pct': 100 * (customer[numeric] == 0).mean(),
        'nunique': customer[numeric].nunique(),
    })

    return audit.sort_values(['family', 'skew'], ascending=[True, False]).round(2)


def duplicate_pairs(customer, threshold=0.75, method='spearman'):
    numeric = [c for c in customer.columns if customer[c].dtype.kind in 'ifb']
    corr = customer[numeric].corr(method=method).abs()

    pairs = [{'feature_a': a, 'feature_b': b, 'correlation': round(corr.loc[a, b], 2)}
             for i, a in enumerate(numeric) for b in numeric[i + 1:]
             if corr.loc[a, b] > threshold]

    return pd.DataFrame(pairs).sort_values('correlation', ascending=False)


def effective_dimensions(customer, columns):
    scaled = MinMaxScaler().fit_transform(customer[columns])
    pca = PCA().fit(scaled)
    cumulative = np.cumsum(pca.explained_variance_ratio_)

    return pd.DataFrame({
        'component': range(1, len(cumulative) + 1),
        'explained': pca.explained_variance_ratio_.round(3),
        'cumulative': cumulative.round(3),
    }).head(15)


# ---------------------------------------------------------------------------
# scaling
# ---------------------------------------------------------------------------

def fit_transformer(customer, features):
    """Fit a Yeo-Johnson power transform followed by min-max scaling.

    Yeo-Johnson searches for the power that minimises skew per feature, so it
    adapts to each one instead of applying a fixed function. It also handles
    zeros and left tails, which a log does not.
    """
    power = PowerTransformer(method='yeo-johnson', standardize=False).fit(customer[features])
    scaler = MinMaxScaler().fit(power.transform(customer[features]))
    return {'features': features, 'power': power, 'scaler': scaler}


def apply_transformer(customer, transformer):
    """Apply a fitted transformer to any window."""
    powered = transformer['power'].transform(customer[transformer['features']])
    scaled = transformer['scaler'].transform(powered)
    return pd.DataFrame(scaled, columns=transformer['features'], index=customer.index)


def transform_effect(customer, scaled, features):
    baseline = pd.DataFrame(MinMaxScaler().fit_transform(customer[features]),
                            columns=features, index=customer.index)

    def spread(frame):
        return ((frame.quantile(0.99) - frame.min()) / (frame.max() - frame.min())).round(3)

    def point_mass(frame):
        return frame.round(3).apply(lambda s: s.value_counts(normalize=True).iloc[0]).round(3)

    return pd.DataFrame({
        'skew_before': customer[features].skew().round(2),
        'skew_after': scaled.skew().round(2),
        'spread_before': spread(baseline),
        'spread_after': spread(scaled),
        'point_mass': point_mass(scaled),
    }).sort_values('skew_before', ascending=False)
