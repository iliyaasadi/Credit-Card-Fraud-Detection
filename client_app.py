# client_app.py
import argparse
import numpy as np
import flwr as fl
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report

from task import (
    prepare_federated_data,
    xgb_to_bytes,
    bytes_to_xgb,
    evaluate_model,
)
from fraud_pipeline import load_model as user_load_model

class XGBNumPyClient(fl.client.NumPyClient):
    def __init__(self, model: xgb.XGBClassifier, X_train, y_train, X_val=None, y_val=None):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        # Ensure model is initialized with a few samples
        if len(self.y_train) > 0:
            try:
                _ = self.model.get_booster()
            except Exception:
                n = min(10, len(self.y_train))
                self.model.fit(self.X_train[:n], self.y_train[:n])

    def get_parameters(self, config=None):
        try:
            raw = xgb_to_bytes(self.model)
            arr = np.frombuffer(raw, dtype=np.uint8)
        except Exception:
            arr = np.array([], dtype=np.uint8)
        return [arr]

    def fit(self, parameters, config=None):
        if parameters and len(parameters) > 0 and parameters[0].size > 0:
            try:
                self.model = bytes_to_xgb(parameters[0].tobytes(), model=self.model)
            except Exception:
                pass
        self.model.fit(self.X_train, self.y_train)
        raw = xgb_to_bytes(self.model)
        arr = np.frombuffer(raw, dtype=np.uint8)
        return [arr], len(self.X_train), {}

    def evaluate(self, parameters, config=None):
        if parameters and len(parameters) > 0 and parameters[0].size > 0:
            try:
                self.model = bytes_to_xgb(parameters[0].tobytes(), model=self.model)
            except Exception:
                pass
        X_eval = self.X_val if self.X_val is not None else self.X_train
        y_eval = self.y_val if self.y_val is not None else self.y_train

        preds = self.model.predict(X_eval)
        acc = accuracy_score(y_eval, preds)
        f1 = f1_score(y_eval, preds)
        precision = precision_score(y_eval, preds, zero_division=0)
        recall = recall_score(y_eval, preds, zero_division=0)

        print(f"Evaluation: Accuracy={acc}, Precision={precision}, Recall={recall}, F1={f1}")
        print(f"Confusion Matrix:\n{confusion_matrix(y_eval, preds)}")
        print(f"Classification Report:\n{classification_report(y_eval, preds)}")    

        loss, acc = evaluate_model(self.model, X_eval, y_eval)
        return float(loss), len(X_eval), {
            "accuracy": float(acc),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall)
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", type=int, required=True)
    parser.add_argument("--n-clients", type=int, default=2)
    parser.add_argument("--server-address", type=str, default="localhost:8080")
    args = parser.parse_args()

    client_parts, X_test, y_test, scale_pos_weight = prepare_federated_data(num_clients=args.n_clients)
    X_part, y_part = client_parts[args.client_id]

    X_train, X_val, y_train, y_val = train_test_split(X_part, y_part, test_size=0.2, random_state=42)

    # Load model
    model = user_load_model()
    if getattr(model, "scale_pos_weight", None) in (None, 0):
        try:
            model.set_params(scale_pos_weight=scale_pos_weight)
        except Exception:
            pass

    # Initialize with one sample from each class to avoid base_score issues
    if len(y_train) > 0:
        init_idx = np.concatenate([
            np.where(y_train == 0)[0][:1],
            np.where(y_train == 1)[0][:1]
        ])
        model.fit(X_train[init_idx], y_train[init_idx])

    client = XGBNumPyClient(model=model, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

    # --- NEW WAY: use .to_client() and start_client() ---
    fl.client.start_client(
        server_address=args.server_address,
        client=client.to_client()
    )

if __name__ == "__main__":
    main()