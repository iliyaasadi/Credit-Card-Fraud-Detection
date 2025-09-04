# server_app.py
import argparse
import flwr as fl
import time
from flwr.server.strategy import FedAvg
import numpy as np
from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays

class SafeFedAvg(FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        if not results:
            return None, {}

        # Use the first client's weights as reference
        ref_ndarrays = parameters_to_ndarrays(results[0][1].parameters)
        ref_shapes = [arr.shape for arr in ref_ndarrays]

        valid_results = []
        for client, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            shapes = [arr.shape for arr in ndarrays]
            if shapes == ref_shapes:
                valid_results.append((client, fit_res))
            else:
                print(f"Skipping client with mismatched weights: {shapes}")

        return super().aggregate_fit(rnd, valid_results, failures)




class CollectModelsStrategy(SafeFedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.collected = []

    def aggregate_fit(self, rnd, results, failures):
        aggregated, metrics = super().aggregate_fit(rnd, results, failures)

        for client, fit_res in results:
            try:
                nds = fl.common.parameters_to_ndarrays(fit_res.parameters)
                if nds and nds[0].size > 0:
                    self.collected.append(nds[0].copy())

                    with open(f"client_{client.cid}_round_{rnd}.bin", "wb") as f:
                        f.write(nds[0].tobytes())
                    print(f"[Round {rnd}] Saved client {client.cid} model to disk.")
            except Exception as e:
                print(f"Error saving model from client {client.cid}: {e}")

        return aggregated, metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-rounds", type=int, default=3)
    parser.add_argument("--server-address", type=str, default="0.0.0.0:8080")
    args = parser.parse_args()

    strategy = CollectModelsStrategy(
        fraction_fit=1.0,           # همه کلاینت‌ها در fit شرکت کنند
        min_fit_clients=1,           # حداقل یک کلاینت برای fit
        min_evaluate_clients=1,      # حداقل یک کلاینت برای evaluate
        min_available_clients=1      # حداقل یک کلاینت متصل باشد
    )
    
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )

    print("All rounds completed. Server will stay alive for new connections...")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Server stopped manually.")

if __name__ == "__main__":
    main()
