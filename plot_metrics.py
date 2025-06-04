import json
import matplotlib.pyplot as plt
import sys
import os

def plot_metrics(filename, save_prefix=None):
    with open(filename, "r") as f:
        data = json.load(f)

    rounds = range(1, len(data["eval_accuracy"]) + 1)

    # --- Accuracy Plot ---
    eval_acc = [acc for _, acc in data["eval_accuracy"]]
    fit_acc = [acc for _, acc in data["fit_accuracy"]]

    plt.figure()
    plt.plot(rounds, eval_acc, label="Evaluation Accuracy")
    if fit_acc and any(acc > 0 for acc in fit_acc):
        plt.plot(rounds, fit_acc, label="Fit Accuracy")
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Round")
    plt.legend()
    plt.grid(True)

    if save_prefix:
        acc_path = f"{save_prefix}_accuracy.png"
        plt.savefig(acc_path)
        print(f" Accuracy plot saved to {acc_path}")
    else:
        plt.show()

    # --- Loss Plot ---
    # Support both "loss" and "eval_loss" formats
    eval_loss_data = data.get("eval_loss", data.get("loss", []))
    fit_loss_data = data.get("fit_loss", [])

    eval_loss = [loss for _, loss in eval_loss_data]
    fit_loss = [loss for _, loss in fit_loss_data] if fit_loss_data else []

    if eval_loss:
        plt.figure()
        plt.plot(rounds, eval_loss, label="Evaluation Loss")
        if fit_loss and any(loss > 0 for loss in fit_loss):
            plt.plot(rounds, fit_loss, label="Fit Loss")
        plt.xlabel("Rounds")
        plt.ylabel("Loss")
        plt.title("Loss per Round")
        plt.legend()
        plt.grid(True)

        if save_prefix:
            loss_path = f"{save_prefix}_loss.png"
            plt.savefig(loss_path)
            print(f" Loss plot saved to {loss_path}")
        else:
            plt.show()
    else:
        print(" No loss data found in JSON. Skipping loss plot.")

    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_metrics.py <input_json> [output_prefix]")
    else:
        filename = sys.argv[1]
        save_prefix = sys.argv[2] if len(sys.argv) > 2 else None
        plot_metrics(filename, save_prefix)
