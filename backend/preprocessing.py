"""
preprocessing.py — v3.1
Loads AI4I 2020 Predictive Maintenance Dataset.

Keeps original 5 features (matching research paper):
  Air Temp, Process Temp, Speed, Torque, Tool Wear

VERSION 3.1 UPGRADE:
  SMOTE oversampling on training set only
  → fixes Warning (1.4%) and Critical (2.1%) class imbalance
  → expected minority class recall improvement: +20-30%

Health Classes:
  0 = Normal   — No Failure
  1 = Warning  — Power Failure / Tool Wear Failure
  2 = Critical — Heat Dissipation / Overstrain / Random

SCL + PAG AAT | Sem 6 AIML | BMSCE
Team: Ravva Nagarjun
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from backend.logger import get_logger
from backend.constants import FEATURE_COLS, FEATURE_NAMES, LABEL_MAP_DICT

logger = get_logger(__name__)

def _map_label(row) -> int:
    ft = str(row.get('Failure Type', 'No Failure')).strip()
    if ft == 'No Failure':
        return 0
    elif ft in ['Power Failure', 'Tool Wear Failure']:
        return 1
    else:
        return 2


def load_and_preprocess(csv_path: str, use_smote: bool = True):
    """
    Load AI4I dataset and return train/test splits.
    5 features only (matching research paper).
    SMOTE applied to training set to fix class imbalance.
    """
    logger.info(f"Loading dataset: {csv_path}")
    logger.info(f"[1/5] Loading dataset from {csv_path} ...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError as e:
        logger.error(f"Dataset not found: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"Dataset is empty: {e}")
        raise
        
    logger.debug(f"Raw shape : {df.shape}")
    logger.debug(f"Columns   : {list(df.columns)}")

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        logger.error(f"Missing columns: {missing}")
        raise ValueError(f"Missing columns: {missing}")

    logger.info("[2/5] Cleaning data ...")
    df = df.dropna(subset=FEATURE_COLS + ['Failure Type'])
    logger.debug(f"Shape after dropna: {df.shape}")

    logger.info("[3/5] Mapping failure types to health classes ...")
    df['label'] = df.apply(_map_label, axis=1)

    X_raw = df[FEATURE_COLS].values
    y     = df['label'].values

    logger.info("[4/5] Normalising features (MinMaxScaler) ...")
    logger.info("[5/5] Stratified 80/20 train/test split ...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler  = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)

    # ── SMOTE on training set only ─────────────────────────────────────────────
    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            before = dict(zip(*np.unique(y_train, return_counts=True)))
            sm = SMOTE(random_state=42, k_neighbors=3)
            X_train, y_train = sm.fit_resample(X_train, y_train)
            after  = dict(zip(*np.unique(y_train, return_counts=True)))
            logger.info("SMOTE applied.")
            logger.info(f"Before: {before}")
            logger.info(f"After : {after}")
        except ImportError:
            logger.warning("imbalanced-learn not found. Run: pip install imbalanced-learn")

    # ── Summary ────────────────────────────────────────────────────────────────
    unique, counts = np.unique(y, return_counts=True)
    logger.info("── Dataset Summary ──────────────────────────────────────")
    logger.info(f"Train samples : {len(X_train)}")
    logger.info(f"Test  samples : {len(X_test)}")
    logger.info(f"Features      : {FEATURE_NAMES}")
    for k, v in zip(unique, counts):
        logger.info(f"{LABEL_MAP_DICT[k]:10s} : {v:>5} samples ({v/len(y)*100:.1f}%)")
    logger.info("─────────────────────────────────────────────────────────")

    df_clean         = df[FEATURE_COLS + ['label']].copy()
    df_clean.columns = FEATURE_NAMES + ['label']
    df_clean['health'] = df_clean['label'].map(LABEL_MAP_DICT)

    return X_train, X_test, y_train, y_test, FEATURE_NAMES, scaler, df_clean
