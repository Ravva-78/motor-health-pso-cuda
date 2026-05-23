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
Team: Ravva Nagarjun, Bharath Kumar T K, Fasi Owaiz Ahmed, Ahas Kaushik
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]',
]

RENAME_MAP = {
    'Air temperature [K]':      'temperature_air',
    'Process temperature [K]':  'temperature_process',
    'Rotational speed [rpm]':   'speed_rpm',
    'Torque [Nm]':              'torque',
    'Tool wear [min]':          'tool_wear',
}

FEATURE_NAMES = ['temperature_air', 'temperature_process',
                 'speed_rpm', 'torque', 'tool_wear']

LABEL_NAMES = {0: 'Normal', 1: 'Warning', 2: 'Critical'}


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
    print(f"\n[1/5] Loading dataset from {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"      Raw shape : {df.shape}")
    print(f"      Columns   : {list(df.columns)}")

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print("[2/5] Cleaning data ...")
    df = df.dropna(subset=FEATURE_COLS + ['Failure Type'])
    print(f"      Shape after dropna: {df.shape}")

    print("[3/5] Mapping failure types to health classes ...")
    df['label'] = df.apply(_map_label, axis=1)

    X_raw = df[FEATURE_COLS].values
    y     = df['label'].values

    print("[4/5] Normalising features (MinMaxScaler) ...")
    print("[5/5] Stratified 80/20 train/test split ...")
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
            print(f"\n  ✅ SMOTE applied:")
            print(f"     Before: {before}")
            print(f"     After : {after}")
        except ImportError:
            print("  ⚠️  imbalanced-learn not found. Run: pip install imbalanced-learn")

    # ── Summary ────────────────────────────────────────────────────────────────
    unique, counts = np.unique(y, return_counts=True)
    print("\n── Dataset Summary ──────────────────────────────────────")
    print(f"  Train samples : {len(X_train)}")
    print(f"  Test  samples : {len(X_test)}")
    print(f"  Features      : {FEATURE_NAMES}")
    print(f"  Label counts  :")
    for k, v in zip(unique, counts):
        print(f"    {LABEL_NAMES[k]:10s} : {v:>5} samples ({v/len(y)*100:.1f}%)")
    print("─────────────────────────────────────────────────────────")

    df_clean         = df[FEATURE_COLS + ['label']].copy()
    df_clean.columns = FEATURE_NAMES + ['label']
    df_clean['health'] = df_clean['label'].map(LABEL_NAMES)

    return X_train, X_test, y_train, y_test, FEATURE_NAMES, scaler, df_clean
