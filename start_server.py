import argparse
import flwr as fl
from server import get_strategy, get_client_manager
from save_metrics import save_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, required=True, help="Dirichlet alpha used for this run")
    parser.add_argument("--rounds", type=int, default=10, help="Number of training rounds")
    args = parser.parse_args()

    strategy = get_strategy()
    client_manager = get_client_manager()

    print(f"🚀 Starting server with alpha={args.alpha} for {args.rounds} rounds...")

    # Start the server and capture the training history
    history = fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_manager=client_manager,
    )

    print("✅ Training complete. Saving metrics...")
    save_metrics(history, args.alpha, f"results/fedavg_alpha_{args.alpha}.json")

from save_metrics import save_metrics
save_metrics(history, args.alpha, f"results_json/alpha_{args.alpha}.json")
