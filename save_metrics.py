import json
import os
from flwr.server.history import History

def save_metrics(history: History, alpha: float, output: str):
    """
    Save training and evaluation metrics from Flower server history to a JSON file.
    """
    data = {
        "alpha": alpha,
        "eval_loss": history.losses_distributed,  
        "fit_loss": history.metrics_distributed_fit.get("loss", []),  
        "eval_accuracy": history.metrics_distributed.get("accuracy", []),
        "fit_accuracy": history.metrics_distributed_fit.get("accuracy", [])
    }

    os.makedirs(os.path.dirname(output), exist_ok=True)

    with open(output, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Metrics saved to {output}")
