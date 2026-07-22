"""
model.py — v3.1
Train, evaluate and persist the PSO-optimised Random Forest.

VERSION 3.1:
  - predict_single computes derived features from 5 raw inputs automatically
  - class_weight='balanced' retained
  - XGBoost + K-Fold CV retained

SCL + PAG AAT | Sem 6 AIML | BMSCE
Team: Ravva Nagarjun, Bharath Kumar T K, Fasi Owaiz Ahmed, Ahas Kaushik
"""

import numpy as np
import pandas as pd
import pickle
import os
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

try:
    from xgboost import XGBClassifier
    _XGB = True
except ImportError:
    _XGB = False

from backend.cuda_module import gpu_accuracy
from backend.logger import get_logger
from backend.constants import LABEL_NAMES_LIST

logger = get_logger(__name__)


def train_baselines(X_train, X_test, y_train, y_test,
                    X_all=None, y_all=None, cv_folds=5):
    cv  = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    Xc  = X_all if X_all is not None else X_train
    yc  = y_all if y_all is not None else y_train
    bas = {}

    for name, clf in [
        ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42)),
        ('SVM',                 SVC(random_state=42, probability=True)),
        ('RF Baseline',         RandomForestClassifier(random_state=42, n_jobs=-1,
                                                       class_weight='balanced')),
    ] + ([('XGBoost', XGBClassifier(n_estimators=100, max_depth=6,
                                     random_state=42, eval_metric='mlogloss',
                                     use_label_encoder=False, verbosity=0))]
         if _XGB else []):
        logger.info(f"Training {name} ...")
        clf.fit(X_train, y_train)
        preds  = clf.predict(X_test)
        cv_sc  = cross_val_score(clf, Xc, yc, cv=cv, scoring='accuracy')
        bas[name] = {
            'model':    clf,
            'acc':      accuracy_score(y_test, preds),
            'cv_mean':  float(cv_sc.mean()),
            'cv_std':   float(cv_sc.std()),
            'preds':    preds,
        }
        logger.info(f"{name:25s}: {bas[name]['acc']*100:.2f}% (CV: {bas[name]['cv_mean']*100:.2f}% ± {bas[name]['cv_std']*100:.2f}%)")
    return bas


def train_final(best_params, X_train, X_test, y_train, y_test,
                X_scaled, y, feature_names, cv_folds=5):
    # class_weight='balanced' fixes class imbalance (Normal 96.5%, Warning 1.4%)
    model = RandomForestClassifier(**best_params, random_state=42,
                                   n_jobs=-1, class_weight='balanced')
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc  = gpu_accuracy(y_test.astype(np.int32), preds.astype(np.int32))
    prec = precision_score(y_test, preds, average='weighted', zero_division=0)
    rec  = recall_score   (y_test, preds, average='weighted', zero_division=0)
    f1   = f1_score       (y_test, preds, average='weighted', zero_division=0)
    cm   = confusion_matrix(y_test, preds)
    rep  = classification_report(y_test, preds, target_names=LABEL_NAMES_LIST)
    pcr  = recall_score(y_test, preds, average=None, zero_division=0)

    cv_strat  = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv_strat, scoring='accuracy')
    importance = dict(zip(feature_names, model.feature_importances_.tolist()))

    results = {
        'model':              model,
        'preds':              preds,
        'accuracy':           float(acc),
        'precision':          float(prec),
        'recall':             float(rec),
        'f1':                 float(f1),
        'confusion_matrix':   cm.tolist(),
        'report':             rep,
        'per_class_recall':   pcr.tolist(),
        'cv_scores':          cv_scores.tolist(),
        'cv_mean':            float(cv_scores.mean()),
        'cv_std':             float(cv_scores.std()),
        'feature_importance': importance,
        'best_params':        best_params,
    }

    logger.info("FINAL MODEL SUMMARY — PSO-Optimised Random Forest")
    logger.info(f"PSO Params     : {best_params}")
    logger.info(f"Accuracy       : {acc*100:.2f}%")
    logger.info(f"Precision      : {prec*100:.2f}%")
    logger.info(f"Recall         : {rec*100:.2f}%")
    logger.info(f"F1 Score       : {f1*100:.2f}%")
    logger.info(f"{cv_folds}-Fold CV     : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
    logger.info("Per-Class Recall:")
    for name, val in zip(LABEL_NAMES_LIST, pcr):
        logger.info(f"{name:10s}: {val*100:.1f}%")
    logger.info(f"\n{rep}")
    return results


def export_metrics_csv(results, baselines, output_path='results/metrics_export.csv'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rows = []
    for name, res in baselines.items():
        rows.append({
            'Model':     name,
            'Accuracy':  f"{res.get('acc',0)*100:.2f}%",
            'CV Mean':   f"{res.get('cv_mean',0)*100:.2f}%",
            'CV Std':    f"±{res.get('cv_std',0)*100:.2f}%",
            'Precision': 'N/A', 'Recall': 'N/A', 'F1': 'N/A',
        })
    r = results
    rows.append({
        'Model':     'PSO-Optimised RF (Ours)',
        'Accuracy':  f"{r['accuracy']*100:.2f}%",
        'CV Mean':   f"{r['cv_mean']*100:.2f}%",
        'CV Std':    f"±{r['cv_std']*100:.2f}%",
        'Precision': f"{r['precision']*100:.2f}%",
        'Recall':    f"{r['recall']*100:.2f}%",
        'F1':        f"{r['f1']*100:.2f}%",
    })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Metrics exported → {output_path}")
    logger.debug(df.to_string(index=False))
    return df


def save_model(model, scaler, paths=None):
    mp = (paths or {}).get('model',  'backend/saved_model.pkl')
    sp = (paths or {}).get('scaler', 'backend/scaler.pkl')
    os.makedirs(os.path.dirname(mp), exist_ok=True)
    try:
        with open(mp, 'wb') as f:
            pickle.dump(model, f)
        with open(sp, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info(f'Model saved  → {mp}')
        logger.info(f'Scaler saved → {sp}')
    except IOError as e:
        logger.error(f"Failed to save model or scaler: {e}")
        raise


def load_model(paths=None):
    mp = (paths or {}).get('model',  'backend/saved_model.pkl')
    sp = (paths or {}).get('scaler', 'backend/scaler.pkl')
    try:
        with open(mp, 'rb') as f:
            model = pickle.load(f)
        with open(sp, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError as e:
        logger.error(f"Could not load model or scaler: {e}")
        raise



def predict_single(model, scaler, input_dict: dict) -> dict:
    """
    Predict motor health from 5 raw sensor readings.
    Uses 5 features matching the scaler trained in preprocessing.py.
    """
    FEATURE_ORDER = [
        'temperature_air', 'temperature_process',
        'speed_rpm', 'torque', 'tool_wear'
    ]
    x        = np.array([[input_dict[k] for k in FEATURE_ORDER]])
    x_scaled = scaler.transform(x)
    label    = int(model.predict(x_scaled)[0])
    proba    = model.predict_proba(x_scaled)[0].tolist()
    health   = LABEL_NAMES_LIST[label]

    return {
        'label':         label,
        'health':        health,
        'probabilities': proba,
    }
