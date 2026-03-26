"""
data_loader.py — AHS data loading and codebook-informed preprocessing.
DO NOT MODIFY. Controlled by config/experiment.yaml.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

BUILT_MIDPOINTS = {
    1919: 1910, 1920: 1925, 1930: 1935, 1940: 1945, 1950: 1955,
    1960: 1965, 1970: 1972, 1975: 1977, 1980: 1982, 1985: 1987,
    1990: 1992, 1995: 1997
}

BINARY_12_COLS = ['TUB', 'PORCH', 'AIR', 'AIRSYS', 'DISH', 'COOK', 'EVROD', 'EROACH',
                  'CRACKS', 'HOLES', 'ELEV', 'WINTERNONE', 'INCP', 'EBAR',
                  'FRSTOC', 'QSS', 'QSELF', 'QRENT', 'QRETIR', 'ZINCH', 'IFFEE']

PRICE_CAT = ['REGION', 'METRO', 'CONDO', 'TYPE', 'CELLAR', 'GARAGE', 'FRSTOC']
PRICE_NUM_BASE = ['BATHS', 'UNITSF', 'LOT', 'ROOMS', 'DINING', 'FLOORS', 'NUNITS',
                  'TUB', 'PORCH', 'AIR', 'AIRSYS', 'DISH', 'COOK', 'DISPL',
                  'EVROD', 'EROACH', 'CRACKS', 'HOLES', 'WINTERNONE', 'EBAR']

INS_CAT = ['REGION', 'METRO3', 'CONDO', 'TYPE', 'CELLAR', 'HHSEX']
INS_NUM_BASE = ['CONFEE', 'ZSMHC', 'IFFEE', 'HHAGE', 'ZINC2', 'ZINC', 'ZINCN',
                'QSS', 'QSELF', 'QRENT', 'QRETIR', 'ZINCH', 'VALUE', 'UNITSF',
                'LOT', 'ROOMS', 'BUILT', 'CLIMB', 'FRSTOC', 'EVROD', 'EROACH',
                'CRACKS', 'HOLES', 'WINTERNONE', 'AIR', 'AIRSYS']


def _clean_survey_codes(df, cfg):
    """Replace AHS missing codes with NaN, handle -6 per variable."""
    codes = cfg['data'].get('survey_codes_to_nan', [-9, -8, -7])
    for col in df.columns:
        df[col] = df[col].replace(codes, np.nan)

    na_handling = cfg['data'].get('na_code_handling', {})
    for col in df.columns:
        if col in na_handling:
            df[col] = df[col].replace(-6, na_handling[col])
        elif na_handling.get('default') == 'nan':
            df[col] = df[col].replace(-6, np.nan)
    return df


def _recode_binary(df, cfg):
    """Recode AHS binary 1=Yes/2=No to 1/0."""
    if not cfg['preprocessing'].get('binary_recode', True):
        return df
    for col in BINARY_12_COLS:
        if col in df.columns:
            df[col] = df[col].map({1: 1, 2: 0})
    if cfg['preprocessing'].get('condo_recode', True) and 'CONDO' in df.columns:
        df['CONDO'] = df['CONDO'].map({1: 1, 3: 0})
    if 'HHSEX' in df.columns:
        df['HHSEX'] = df['HHSEX'].map({1: 1, 2: 0})
    if 'GARAGE' in df.columns:
        df['GARAGE'] = df['GARAGE'].map({1: 1, 2: 0})
    if 'DISPL' in df.columns:
        df['DISPL'] = df['DISPL'].map({1: 1, 2: 0})
    return df


def _round_imputed(df):
    """Round Census-imputed fractional values to nearest int."""
    for col in ['GARAGE', 'FRSTOC', 'WINTERNONE', 'DISPL']:
        if col in df.columns:
            df[col] = df[col].round(0)
    return df


def _add_engineered_features(df, feature_list, is_insurance=False):
    """Add engineered features based on config list."""
    if 'AGE' in feature_list and 'BUILT' in df.columns:
        df['BUILT_ADJ'] = df['BUILT'].map(lambda x: BUILT_MIDPOINTS.get(x, x))
        df['AGE'] = (2011 - df['BUILT_ADJ']).clip(lower=0)
    if 'LOG_UNITSF' in feature_list and 'UNITSF' in df.columns:
        df['LOG_UNITSF'] = np.log1p(df['UNITSF'])
    if 'LOG_LOT' in feature_list and 'LOT' in df.columns:
        df['LOG_LOT'] = np.log1p(df['LOT'])
    if 'ROOMS_PER_BATH' in feature_list and 'ROOMS' in df.columns and 'BATHS' in df.columns:
        df['ROOMS_PER_BATH'] = df['ROOMS'] / df['BATHS'].replace(0, 0.5)
    if 'TOTAL_PROBLEMS' in feature_list:
        df['TOTAL_PROBLEMS'] = sum(df[c].fillna(0) for c in ['EVROD','EROACH','CRACKS','HOLES'] if c in df.columns)
    if 'HAS_AMENITIES' in feature_list:
        df['HAS_AMENITIES'] = sum(df[c].fillna(0) for c in ['DISH','AIRSYS','GARAGE','PORCH','DISPL'] if c in df.columns)
    if 'HAS_BASEMENT' in feature_list and 'CELLAR' in df.columns:
        df['HAS_BASEMENT'] = df['CELLAR'].isin([1, 2]).astype(int)
    if is_insurance:
        if 'HOUSING_BURDEN' in feature_list and 'ZSMHC' in df.columns:
            df['HOUSING_BURDEN'] = (df['ZSMHC'] / (df['ZINC2'].replace(0, 1) / 12)).clip(-1, 5)
        if 'INCOME_SOURCES' in feature_list:
            df['INCOME_SOURCES'] = sum(df[c].fillna(0) for c in ['QSS','QSELF','QRENT','QRETIR'] if c in df.columns)
        if 'LOG_ZINC2' in feature_list and 'ZINC2' in df.columns:
            df['LOG_ZINC2'] = np.log1p(df['ZINC2'].clip(lower=0))
        if 'LOG_VALUE' in feature_list and 'VALUE' in df.columns:
            df['LOG_VALUE'] = np.log1p(df['VALUE'].clip(lower=0))
    return df


def load_and_preprocess(cfg):
    """Load and preprocess both datasets. Returns dicts with train/val/hold splits."""
    seed = cfg['random_seed']

    # ========== PRICE ==========
    df_p = pd.read_excel(cfg['data']['price_file'])
    df_p = _round_imputed(df_p)
    df_p = _clean_survey_codes(df_p, cfg)
    df_p = _recode_binary(df_p, cfg)

    # Drop high-NaN and leakage columns
    drop_cols = cfg['features'].get('drop_high_nan', []) + ['LOGVALUE']
    df_p.drop(columns=[c for c in drop_cols if c in df_p.columns], inplace=True, errors='ignore')

    # Engineered features
    eng_features = cfg['features'].get('enabled_engineered', [])
    df_p = _add_engineered_features(df_p, eng_features, is_insurance=False)

    # Target
    y_price = np.log(df_p['VALUE'].clip(lower=1))

    # Baseline formula
    df_raw = pd.read_excel(cfg['data']['price_file']).replace([-9,-8,-7,-6], np.nan)
    baseline_pred = (12.751 + 22.5*(df_raw['UNITSF'].fillna(0)/1e6)
                     - 78.9*(df_raw['LOT'].fillna(0)/1e9)
                     + 0.136*df_raw['ROOMS'].fillna(0) - df_raw['BUILT'].fillna(2000)/1000
                     + 0.34*df_raw['BATHS'].fillna(0) + 0.027*df_raw['CLIMB'].fillna(0))

    df_p.drop(columns=['VALUE', 'BUILT', 'BUILT_ADJ'] if 'BUILT_ADJ' in df_p.columns
              else ['VALUE', 'BUILT'], inplace=True, errors='ignore')

    # Feature columns
    cat_p = [c for c in PRICE_CAT if c in df_p.columns]
    num_p = [c for c in PRICE_NUM_BASE + eng_features if c in df_p.columns and c not in cat_p]
    num_p = list(dict.fromkeys(num_p))  # dedupe

    X_p = df_p[cat_p + num_p].copy()
    for c in cat_p: X_p[c] = X_p[c].fillna('missing').astype(str)
    for c in num_p: X_p[c] = X_p[c].fillna(X_p[c].median())

    # Split
    ts = cfg['data']['test_size']
    vs = cfg['data']['validation_size']
    X_tmp, X_hold, y_tmp, y_hold = train_test_split(X_p, y_price, test_size=ts, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=vs/(1-ts), random_state=seed)
    bl_hold = baseline_pred.loc[y_hold.index]

    # Outlier removal on train thresholds
    pcts = cfg['data'].get('outlier_percentiles', [0.01, 0.99])
    q_lo, q_hi = y_train.quantile(pcts[0]), y_train.quantile(pcts[1])
    m_tr = (y_train >= q_lo) & (y_train <= q_hi)
    m_va = (y_val >= q_lo) & (y_val <= q_hi)
    m_ho = (y_hold >= q_lo) & (y_hold <= q_hi)
    X_train, y_train = X_train[m_tr], y_train[m_tr]
    X_val, y_val = X_val[m_va], y_val[m_va]
    X_hold, y_hold = X_hold[m_ho], y_hold[m_ho]
    bl_hold = bl_hold[m_ho]

    # Preprocessing pipeline
    pre_p = ColumnTransformer([
        ('num', StandardScaler(), num_p),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_p)
    ])
    X_tr_t = pre_p.fit_transform(X_train)
    X_va_t = pre_p.transform(X_val)
    X_ho_t = pre_p.transform(X_hold)
    fn_p = list(pre_p.get_feature_names_out())

    price_data = {
        'X_train': X_tr_t, 'X_val': X_va_t, 'X_hold': X_ho_t,
        'y_train': y_train.values, 'y_val': y_val.values, 'y_hold': y_hold.values,
        'baseline_pred': bl_hold.values, 'feature_names': fn_p
    }

    # ========== INSURANCE ==========
    df_i = pd.read_excel(cfg['data']['insurance_file'])
    df_i = _round_imputed(df_i)

    amti_raw = df_i['AMTI'].copy().replace([-9,-8,-7], np.nan)
    amti_raw.loc[amti_raw == -6] = np.nan

    for col in df_i.columns:
        if col not in ['BUYI', 'AMTI']:
            df_i[col] = df_i[col].replace([-9,-8,-7], np.nan)
    df_i['CELLAR'] = df_i['CELLAR'].replace(-6, 0)
    df_i['CLIMB'] = df_i['CLIMB'].replace(-6, 0)
    df_i['LOT'] = df_i['LOT'].replace(-6, np.nan)
    df_i['CONFEE'] = df_i['CONFEE'].replace(-6, 0).replace(-9, np.nan)
    df_i['ZINCN'] = df_i['ZINCN'].replace(-6, np.nan)
    df_i['UNITSF'] = df_i['UNITSF'].replace([-8,-7], np.nan)

    df_i = _recode_binary(df_i, cfg)

    # Engineered features
    ins_features = eng_features + cfg['features'].get('insurance_extra', [])
    df_i['AGE_INS'] = (2011 - df_i['BUILT']).clip(lower=0)
    df_i = _add_engineered_features(df_i, ins_features, is_insurance=True)

    y_ins = df_i['BUYI'].copy()
    drop_leak = cfg['features'].get('drop_leakage', []) + cfg['features'].get('drop_high_nan', [])
    drop_leak = list(set(drop_leak + ['BUYI', 'AMTI', 'MOBILTYP']))
    df_i.drop(columns=[c for c in drop_leak if c in df_i.columns], inplace=True, errors='ignore')

    cat_i = [c for c in INS_CAT if c in df_i.columns]
    num_i = [c for c in INS_NUM_BASE + ins_features if c in df_i.columns and c not in cat_i]
    num_i = list(dict.fromkeys(num_i))

    X_i = df_i[cat_i + num_i].copy()
    for c in cat_i: X_i[c] = X_i[c].fillna('missing').astype(str)
    for c in num_i: X_i[c] = X_i[c].fillna(X_i[c].median())

    X_tmp_i, X_hold_i, y_tmp_i, y_hold_i = train_test_split(
        X_i, y_ins, test_size=ts, random_state=seed, stratify=y_ins)
    X_train_i, X_val_i, y_train_i, y_val_i = train_test_split(
        X_tmp_i, y_tmp_i, test_size=vs/(1-ts), random_state=seed, stratify=y_tmp_i)

    amti_val = amti_raw.loc[y_val_i.index]
    amti_hold = amti_raw.loc[y_hold_i.index]

    pre_i = ColumnTransformer([
        ('num', StandardScaler(), num_i),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_i)
    ])
    X_tr_it = pre_i.fit_transform(X_train_i)
    X_va_it = pre_i.transform(X_val_i)
    X_ho_it = pre_i.transform(X_hold_i)
    fn_i = list(pre_i.get_feature_names_out())

    # Resampling
    X_resampled, y_resampled = None, None
    samp_cfg = cfg['classification'].get('sampling', {})
    strategy = samp_cfg.get('strategy', 'none')
    if strategy == 'smote':
        ratio = samp_cfg.get('smote_ratio', 0.3)
        k = samp_cfg.get('smote_k_neighbors', 5)
        sm = SMOTE(random_state=seed, sampling_strategy=ratio, k_neighbors=k)
        X_resampled, y_resampled = sm.fit_resample(X_tr_it, y_train_i)
    elif strategy == 'undersample':
        rus = RandomUnderSampler(random_state=seed, sampling_strategy=0.5)
        X_resampled, y_resampled = rus.fit_resample(X_tr_it, y_train_i)

    insurance_data = {
        'X_train': X_tr_it, 'X_val': X_va_it, 'X_hold': X_ho_it,
        'y_train': y_train_i.values, 'y_val': y_val_i.values, 'y_hold': y_hold_i.values,
        'amti_val': amti_val, 'amti_hold': amti_hold,
        'feature_names': fn_i,
        'X_resampled': X_resampled,
        'y_resampled': y_resampled.values if y_resampled is not None else None,
    }

    return price_data, insurance_data
