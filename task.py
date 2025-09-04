# task.py
# Uses your functions from fraud_pipeline.py to prepare federated data,
# plus helper utilities for XGBoost model serialization/evaluation.

import os
from typing import List, Tuple
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, accuracy_score

from fraud_pipeline import (
    load_data as user_load_data,
    check_for_balance,
    take_sample,
    feature_engineering,
    encode_categorical_features,
    feature_selection,
    resampling,
    scale_features,
    split_data,
    load_model as user_load_model,
)

def prepare_federated_data(num_clients: int = 2):
    # Run your full preprocessing pipeline once (simulation mode)
    df_train, df_test = user_load_data()

    # Your original sequence
    non_fraud, fraud_num = check_for_balance(df_train)
    df_train = take_sample(df_train)
    df_train, df_test = feature_engineering(df_train, df_test)
    df_train_encoded, df_test_encoded = encode_categorical_features(df_train, df_test)
    df_train_encoded, df_test_encoded = feature_selection(df_train_encoded, df_test_encoded)
    data_train, df_test_encoded = resampling(df_train_encoded, df_test_encoded)
    data_train_scaled, df_test_scaled = scale_features(data_train, df_test_encoded)
    X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled = split_data(data_train_scaled, df_test_scaled)

    X_train_scaled = np.asarray(X_train_scaled, dtype=np.float32)
    y_train_scaled = np.asarray(y_train_scaled, dtype=np.int32)
    X_test_scaled = np.asarray(X_test_scaled, dtype=np.float32)
    y_test_scaled = np.asarray(y_test_scaled, dtype=np.int32)

    # Simple IID partition
    def iid_partition(X: np.ndarray, y: np.ndarray, n: int):
        part_size = len(X) // n
        parts = []
        for i in range(n):
            s = i * part_size
            e = (i + 1) * part_size if i != n - 1 else len(X)
            parts.append((X[s:e], y[s:e]))
        return parts

    client_parts = iid_partition(X_train_scaled, y_train_scaled, num_clients)

    # Also compute a suggested scale_pos_weight from the *global* train split
    pos = float((y_train_scaled == 1).sum())
    neg = float((y_train_scaled == 0).sum())
    scale_pos_weight = max(1.0, neg / max(1.0, pos))

    return client_parts, X_test_scaled, y_test_scaled, scale_pos_weight

# --------------- XGBoost (de)serialization helpers ---------------
def xgb_to_bytes(model: xgb.XGBClassifier) -> bytes:
    booster = model.get_booster()
    return booster.save_raw()

def bytes_to_xgb(raw_bytes: bytes, model: xgb.XGBClassifier = None) -> xgb.XGBClassifier:
    booster = xgb.Booster()
    try:
        booster.load_model(bytearray(raw_bytes))
    except Exception:
        import tempfile
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(raw_bytes); tmp.flush(); tmp.close()
        booster = xgb.Booster(); booster.load_model(tmp.name)
        os.unlink(tmp.name)
    if model is None:
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model._Booster = booster
    model._le = None
    return model

def evaluate_model(model: xgb.XGBClassifier, X: np.ndarray, y: np.ndarray):
    if len(X) == 0:
        return float("nan"), float("nan")
    y_prob = model.predict_proba(X)[:, 1]
    loss = float(log_loss(y, np.vstack([1 - y_prob, y_prob]).T))
    y_pred = model.predict(X)
    acc = float(accuracy_score(y, y_pred))
    # evalueate other metrics as needed
    return loss, acc
