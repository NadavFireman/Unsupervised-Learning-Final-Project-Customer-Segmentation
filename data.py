"""Loading, temporal splitting and cleaning of the raw transaction file.

Every step is local to a row and learns nothing, with one exception: the
description map, which is learned from the training window only.

Constants are not defined here. They live in the notebook and are passed in as
required arguments, so a mismatch between the two raises TypeError rather than
returning a quietly different answer.
"""

import re

import pandas as pd


def load_transactions(file_path):
    """Read the raw CSV and normalise column names to snake_case."""
    frame = pd.read_csv(file_path, parse_dates=['InvoiceDate'])
    frame.columns = [re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', column)
                     .strip().replace(' ', '_').lower()
                     for column in frame.columns]
    return frame


def temporal_split(frame, ratios):
    """Split transactions into three consecutive time windows by row mass.

    Cut points are rounded up to the following midnight, so every window is made
    of whole trading days and no invoice is divided between two windows.

    Returns the three frames and a dict of window boundaries.
    """
    frame = frame.sort_values('invoice_date', kind='stable').reset_index(drop=True)

    first = frame.loc[int(len(frame) * ratios[0]), 'invoice_date'].ceil('D')
    second = frame.loc[int(len(frame) * ratios[1]), 'invoice_date'].ceil('D')

    train = frame[frame['invoice_date'] < first].copy()
    val = frame[(frame['invoice_date'] >= first) & (frame['invoice_date'] < second)].copy()
    test = frame[frame['invoice_date'] >= second].copy()

    day_zero = frame['invoice_date'].min().normalize()
    day_end = frame['invoice_date'].max().normalize() + pd.Timedelta(days=1)
    windows = {'train': (day_zero, first), 'val': (first, second), 'test': (second, day_end)}

    return train, val, test, windows


def fill_missing_descriptions(train, val, test):
    """Fill missing product descriptions from the most common one per stock code.

    The map is learned from the training window only and applied to all three.
    Returns the map so it can be inspected.
    """
    known = train[train['description'].notna()]
    description_map = known.groupby('stock_code')['description'].agg(
        lambda values: values.value_counts().index[0])

    for frame in (train, val, test):
        frame['description'] = frame['description'].fillna(
            frame['stock_code'].map(description_map))

    return description_map


def drop_unidentified_customers(train, val, test):
    """Remove guest purchases, which carry no customer and cannot be aggregated."""
    frames = []
    for frame in (train, val, test):
        kept = frame[frame['customer_id'].notna()].copy()
        kept['customer_id'] = kept['customer_id'].astype(int)
        frames.append(kept)
    return tuple(frames)


def build_clean(frame, stock_code_pattern):
    """Drop exact duplicates, non-product stock codes and non-positive prices.

    Cancellation rows are kept on purpose: they offset net revenue at customer
    level and feed the return-rate feature.
    """
    frame = frame.drop_duplicates()
    frame = frame[frame['stock_code'].astype(str).str.match(stock_code_pattern)]
    frame = frame[frame['price'] > 0].copy()
    return frame
