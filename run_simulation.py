import flwr as fl
from server import get_strategy, get_client_manager
from data_utils import generate_distributed_data
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    generate_distributed_data(num_clients=args.clients, alpha=args.alpha, save_dir="distributed_data")

    strategy = get_strategy()
    client_manager = get_client_manager()

    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_manager=client_manager,
    )
