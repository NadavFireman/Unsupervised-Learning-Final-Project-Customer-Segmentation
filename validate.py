"""End-to-end check that the five modules reproduce the notebook's published figures.

The notebook imports its functions from these modules but declares every modelling
constant itself, so the constants below are a hand-kept mirror. This script rebuilds
the customer tables from the raw CSV using only the modules, then asserts the figures
the notebook publishes. A constant that drifts out of step makes an assertion fail
here rather than passing silently in the notebook.

Module functions take the constants as required arguments, so a missing one raises
TypeError instead of falling back to a default.

    python validate.py <path-to-online_retail_II.csv>
"""

import sys
import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from clustering import assign_unseen, separation_scores
from data import (build_clean, drop_unidentified_customers, fill_missing_descriptions,
                  load_transactions, temporal_split)
from features import (add_transaction_features, apply_transformer, build_customer_table,
                      build_holiday_calendar, fit_reference, fit_transformer)

warnings.filterwarnings('ignore')

# --- constants, mirroring the notebook ------------------------------------

SEED = 42
SPLIT_RATIOS = (0.6, 0.8)
STOCK_CODE_PATTERN = r'^\d{5}'

MIN_FREQUENCY = 2
DAYS_PER_MONTH = 30.44
ALPHA = 100.0
DISCOUNT_THRESHOLD = 0.9
BULK_QUANTITY = 12

SEASON_DAYS_BEFORE = 21
SEASON_DAYS_AFTER = 7

HOLIDAY_RANGES = [
    ('2009-12-24', '2009-12-28'), ('2009-12-31', '2010-01-01'),
    ('2010-02-14', '2010-02-14'), ('2010-03-14', '2010-03-14'),
    ('2010-04-02', '2010-04-05'), ('2010-05-03', '2010-05-03'),
    ('2010-05-31', '2010-05-31'), ('2010-06-20', '2010-06-20'),
    ('2010-08-30', '2010-08-30'), ('2010-10-31', '2010-10-31'),
    ('2010-11-05', '2010-11-05'), ('2010-12-24', '2010-12-28'),
    ('2010-12-31', '2011-01-03'), ('2011-02-14', '2011-02-14'),
    ('2011-04-03', '2011-04-03'), ('2011-04-22', '2011-04-25'),
    ('2011-04-29', '2011-04-29'), ('2011-05-02', '2011-05-02'),
    ('2011-05-30', '2011-05-30'), ('2011-06-19', '2011-06-19'),
    ('2011-08-29', '2011-08-29'), ('2011-10-31', '2011-10-31'),
    ('2011-11-05', '2011-11-05'), ('2011-12-24', '2011-12-27'),
]

CATEGORY_KEYWORDS = {
    'holidays': ['CHRISTMAS', 'XMAS', 'SANTA', 'REINDEER', 'ADVENT', 'ORNAMENT', 'WREATH',
                 'EASTER', 'HALLOWEEN', 'VALENTINE', 'GIFT', 'WRAP', 'GARLAND'],
    'kitchen': ['MUG', 'CUP', 'TEA', 'COFFEE', 'PLATE', 'BOWL', 'SPOON', 'FORK', 'KNIFE',
                'JUG', 'BAKING', 'CAKE', 'KITCHEN', 'JAM', 'APRON', 'PANTRY', 'BOTTLE'],
    'storage': ['BOX', 'BASKET', 'TIN', 'STORAGE', 'BAG', 'CASE', 'CRATE', 'JAR', 'HOLDER'],
    'toys': ['TOY', 'DOLL', 'TEDDY', 'PUZZLE', 'GAME', 'PLAY', 'BUNNY', 'CHILD', 'CRAFT',
             'STICKER', 'BALLOON', 'MAGIC', 'DINOSAUR'],
    'garden': ['GARDEN', 'PLANT', 'SEED', 'OUTDOOR', 'PICNIC', 'BEACH', 'LANTERN', 'BIRD',
               'WATERING', 'HERB', 'PARASOL'],
    'party': ['PARTY', 'BIRTHDAY', 'CELEBRATION', 'BUNTING', 'CONFETTI', 'RIBBON'],
    'fashion': ['NECKLACE', 'BRACELET', 'EARRING', 'PURSE', 'SCARF', 'HAT', 'GLOVE',
                'JEWELLERY', 'CHARM', 'PENDANT', 'UMBRELLA'],
    'beauty': ['SOAP', 'LOTION', 'PERFUME', 'CREAM', 'BATH', 'TOWEL', 'MIRROR', 'COMB'],
    'decor': ['CANDLE', 'FRAME', 'HEART', 'VASE', 'PHOTO', 'DECORATION', 'HANGING',
              'LIGHT', 'CUSHION', 'CLOCK', 'SIGN', 'STAR'],
    'stationery': ['PENCIL', 'PEN', 'NOTEBOOK', 'CARD', 'JOURNAL', 'CHALK', 'PAPER',
                   'ENVELOPE', 'STATIONERY'],
    'household': ['HOOK', 'HANGER', 'RACK', 'SHELF', 'TRAY', 'DOORMAT', 'NAPKIN', 'PEG',
                  'SEWING', 'CURTAIN', 'DRAWER', 'KEY'],
}

FEATURES_FULL = ['frequency', 'monetary', 'basket_value', 'depth', 'median_price',
                 'interval_std', 'recency', 'tenure', 'bulk_share', 'campaign_pct',
                 'return_rate', 'repeat_ratio', 'holiday_share',
                 'cat_storage', 'cat_kitchen']

SELECTED_FEATURE_COUNT = 9
SELECTED_K = 3

# --- the figures the notebook publishes -----------------------------------

EXPECTED = {
    'train_customers': 3057,
    'val_customers': 1511,
    'test_customers': 1398,
    'silhouette_train': 0.208,
    'silhouette_val': 0.188,
    'segment_sizes': [1251, 1122, 684],
    'feature_order': ['interval_std', 'frequency', 'campaign_pct', 'repeat_ratio',
                      'monetary', 'holiday_share', 'tenure', 'bulk_share', 'recency'],
}


def run_pipeline(csv_path):
    """Rebuild everything from the raw CSV using the modules only."""
    frame = load_transactions(csv_path)
    train, val, test, windows = temporal_split(frame, SPLIT_RATIOS)

    fill_missing_descriptions(train, val, test)
    train, val, test = drop_unidentified_customers(train, val, test)
    train = build_clean(train, STOCK_CODE_PATTERN)
    val = build_clean(val, STOCK_CODE_PATTERN)
    test = build_clean(test, STOCK_CODE_PATTERN)

    holiday_days, season_days = build_holiday_calendar(
        HOLIDAY_RANGES, SEASON_DAYS_BEFORE, SEASON_DAYS_AFTER)
    for window in (train, val, test):
        add_transaction_features(window, holiday_days, season_days,
                                 BULK_QUANTITY, CATEGORY_KEYWORDS)

    price_map, discount_prior, return_prior, categories = fit_reference(
        train, DISCOUNT_THRESHOLD)

    tables = {}
    for name, window in [('train', train), ('val', val), ('test', test)]:
        tables[name] = build_customer_table(
            window, windows[name], price_map, discount_prior, return_prior,
            categories, ALPHA, DAYS_PER_MONTH, MIN_FREQUENCY, DISCOUNT_THRESHOLD)

    wide = apply_transformer(tables['train'], fit_transformer(tables['train'], FEATURES_FULL))
    order = list(separation_scores(wide, SEED).index)
    selected = order[:SELECTED_FEATURE_COUNT]

    transformer = fit_transformer(tables['train'], selected)
    train_values = apply_transformer(tables['train'], transformer)
    val_values = apply_transformer(tables['val'], transformer)

    model = KMeans(n_clusters=SELECTED_K, n_init=10, random_state=SEED).fit(train_values.values)
    val_labels = assign_unseen(val_values.values, model.cluster_centers_)

    return {
        'train_customers': len(tables['train']),
        'val_customers': len(tables['val']),
        'test_customers': len(tables['test']),
        'silhouette_train': round(silhouette_score(train_values.values, model.labels_), 3),
        'silhouette_val': round(silhouette_score(val_values.values, val_labels), 3),
        'segment_sizes': sorted(np.bincount(model.labels_).tolist(), reverse=True),
        'feature_order': selected,
    }


def main(csv_path):
    """Run the pipeline and compare every figure against the notebook."""
    actual = run_pipeline(csv_path)

    print(f"{'check':<20} {'expected':>22} {'actual':>22}   result")
    print('-' * 74)

    failures = 0
    for key, expected in EXPECTED.items():
        got = actual[key]
        passed = got == expected
        failures += not passed
        if isinstance(expected, list) and len(str(expected)) > 22:
            print(f'{key:<20} {"see below":>22} {"":>22}   {"PASS" if passed else "FAIL"}')
        else:
            print(f'{key:<20} {str(expected):>22} {str(got):>22}   {"PASS" if passed else "FAIL"}')

    print('-' * 74)
    print(f'feature order: {actual["feature_order"]}')
    print(f'\n{len(EXPECTED) - failures} of {len(EXPECTED)} checks passed')
    return failures


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/online_retail_II.csv'
    sys.exit(main(path))
